#!/usr/bin/env python3
"""Small authenticated API for one ephemeral CiCy Koubo GPU instance."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import queue
import re
import signal
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field

from flask import Flask, Response, jsonify, request, send_file

API_VERSION = "v1"
CAPABILITIES_VERSION = "2026.07.30-multilingual-1"
COSYVOICE_SCRIPT = pathlib.Path("/content/cosy/cosyvoice_tts.py")
DIALECT_LANGUAGES = {
    "zh-CN", "zh-yue", "zh-minnan", "zh-sichuan", "zh-dongbei",
    "zh-shanghai", "zh-tianjin", "zh-shandong", "zh-shaanxi", "zh-shanxi",
    "en", "ko",
}
JOB_ID_RE = re.compile(r"^gpu_[0-9a-f]{32}$")
MAX_UPLOAD_BYTES = int(os.environ.get("KOUBO_MAX_UPLOAD_BYTES", str(5 * 1024**3)))
ROOT = pathlib.Path(os.environ.get("KOUBO_API_STATE_DIR", "/var/lib/cicy-koubo-api"))
JOBS_DIR = ROOT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> int:
    return int(time.time())


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Job:
    id: str
    kind: str
    engine: str
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    error: str = ""
    created_at: int = field(default_factory=_now)
    updated_at: int = field(default_factory=_now)
    result_name: str = ""
    result_sha256: str = ""
    result_size: int = 0
    result_url: str = ""
    cancel_requested: bool = False

    @property
    def directory(self) -> pathlib.Path:
        return JOBS_DIR / self.id

    def public(self) -> dict:
        result = asdict(self)
        result.pop("cancel_requested", None)
        result["result_ready"] = self.status == "succeeded"
        return result


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        for manifest in JOBS_DIR.glob("gpu_*/job.json"):
            try:
                job = Job(**json.loads(manifest.read_text(encoding="utf-8")))
                if job.status in {"running", "queued"}:
                    job.status = "failed"
                    job.stage = "interrupted"
                    job.error = "worker_restarted"
                    self.save(job)
                self._jobs[job.id] = job
            except Exception:
                continue

    def create(self, kind: str, engine: str) -> Job:
        job = Job(id="gpu_" + uuid.uuid4().hex, kind=kind, engine=engine)
        job.directory.mkdir(mode=0o700)
        with self._lock:
            self._jobs[job.id] = job
            self.save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def save(self, job: Job) -> None:
        job.updated_at = _now()
        _atomic_json(job.directory / "job.json", asdict(job))


class Engine:
    name = ""
    kind = ""

    def run(self, job: Job, payload: dict[str, pathlib.Path | str]) -> pathlib.Path:
        raise NotImplementedError

    @staticmethod
    def command(job: Job, args: list[str], log_name: str) -> None:
        log_path = job.directory / log_name
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                if job.cancel_requested:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    raise RuntimeError("cancelled")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"engine_failed:{process.returncode}")


class MuseTalkEngine(Engine):
    name = "musetalk"
    kind = "video"

    def run(self, job: Job, payload: dict[str, pathlib.Path | str]) -> pathlib.Path:
        video = pathlib.Path(payload["video"])
        audio = pathlib.Path(payload["audio"])
        bbox = str(payload.get("bbox") or "0")
        normalized = job.directory / "base_25fps.mp4"
        result = job.directory / "result.mp4"
        job.stage, job.progress = "normalizing", 10
        STORE.save(job)
        self.command(job, [
            "ffmpeg", "-v", "error", "-y", "-i", str(video), "-r", "25",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
            str(normalized),
        ], "ffmpeg.log")
        job.stage, job.progress = "lipsync", 25
        STORE.save(job)
        self.command(job, [
            "bash", "/content/mt/synthesize.sh", str(normalized), str(audio),
            str(result), bbox,
        ], "musetalk.log")
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError("missing_result")
        return result


class CosyVoiceEngine(Engine):
    name = "cosyvoice"
    kind = "tts"

    def run(self, job: Job, payload: dict[str, pathlib.Path | str]) -> pathlib.Path:
        reference = payload.get("reference")
        if payload.get("reference_url"):
            job.stage, job.progress = "downloading_reference", 5
            STORE.save(job)
            reference = _download_oss_url(
                str(payload["reference_url"]), job.directory / "reference.wav",
            )
        if not reference:
            raise RuntimeError("missing_reference")
        result = job.directory / "result.wav"
        args = [
            "/content/cosy/env/bin/python", "/content/cosy/cosyvoice_tts.py",
            "--ref", str(reference),
            "--ref-text", str(payload["reference_text"]),
            "--text", str(payload["text"]),
            "--language", str(payload.get("language") or "zh-CN"),
            "--speed", str(payload.get("speed") or "1.15"),
            "--out", str(result),
        ]
        job.stage, job.progress = "synthesizing", 20
        STORE.save(job)
        self.command(job, args, "cosyvoice.log")
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError("missing_result")
        return result


EDGE_VOICES = {
    "fr": "fr-FR-DeniseNeural", "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural", "it": "it-IT-ElsaNeural",
    "vi": "vi-VN-HoaiMyNeural", "id": "id-ID-GadisNeural",
    "ms": "ms-MY-YasminNeural", "th": "th-TH-PremwadeeNeural",
    "ru": "ru-RU-SvetlanaNeural", "ar": "ar-SA-ZariyahNeural",
    "km": "km-KH-SreymomNeural", "lo": "lo-LA-KeomanyNeural",
}
SUPPORTED_TTS_LANGUAGES = DIALECT_LANGUAGES | set(EDGE_VOICES)


class EdgeTtsEngine(Engine):
    name = "edge-tts"
    kind = "tts"

    def run(self, job: Job, payload: dict[str, pathlib.Path | str]) -> pathlib.Path:
        language = str(payload.get("language") or "")
        voice = EDGE_VOICES.get(language)
        if not voice:
            raise RuntimeError("unsupported_tts_language")
        media = job.directory / "edge.mp3"
        result = job.directory / "result.wav"
        job.stage, job.progress = "synthesizing_multilingual", 20
        STORE.save(job)
        self.command(job, [
            "/content/cosy/env/bin/edge-tts", "--voice", voice,
            "--text", str(payload["text"]), "--write-media", str(media),
        ], "edge-tts.log")
        self.command(job, [
            "ffmpeg", "-v", "error", "-y", "-i", str(media),
            "-ar", "24000", "-ac", "1", str(result),
        ], "edge-tts-convert.log")
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError("missing_result")
        return result


ENGINES: dict[tuple[str, str], Engine] = {
    ("video", "musetalk"): MuseTalkEngine(),
    ("tts", "cosyvoice"): CosyVoiceEngine(),
    ("tts", "edge-tts"): EdgeTtsEngine(),
}
STORE = JobStore()
WORK_QUEUE: queue.Queue[tuple[Job, Engine, dict]] = queue.Queue()


def _finish(job: Job, result: pathlib.Path) -> None:
    digest = hashlib.sha256()
    with result.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    job.result_name = result.name
    job.result_size = result.stat().st_size
    job.result_sha256 = digest.hexdigest()
    if job.kind == "video":
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(result),
            "-frames:v", "1", str(job.directory / "cover.jpg"),
        ], check=False, timeout=60, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
    job.status, job.stage, job.progress = "succeeded", "complete", 100
    STORE.save(job)


def _upload_oss_url(raw_url: str, source: pathlib.Path) -> None:
    if len(raw_url) > 4096:
        raise ValueError("result_upload_url_too_long")
    parsed = urllib.parse.urlsplit(raw_url)
    if (
        parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment
        or not parsed.hostname or not parsed.hostname.endswith(".aliyuncs.com")
    ):
        raise ValueError("invalid_result_upload_url")
    request_ = urllib.request.Request(
        raw_url, data=source.read_bytes(), method="PUT",
        headers={"Content-Type": "video/mp4"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request_, timeout=300):
        pass


def _worker() -> None:
    while True:
        job, engine, payload = WORK_QUEUE.get()
        try:
            if job.cancel_requested:
                job.status, job.stage = "cancelled", "cancelled"
            else:
                job.status, job.stage = "running", "starting"
                STORE.save(job)
                result = engine.run(job, payload)
                upload_url = str(payload.get("result_upload_url") or "")
                if upload_url:
                    job.stage, job.progress = "uploading_result", 96
                    STORE.save(job)
                    _upload_oss_url(upload_url, result)
                    job.result_url = str(payload.get("result_download_url") or "")
                _finish(job, result)
        except Exception as exc:
            if str(exc) == "cancelled":
                job.status, job.stage = "cancelled", "cancelled"
            else:
                job.status, job.stage, job.error = "failed", "failed", str(exc)[:500]
            STORE.save(job)
        finally:
            WORK_QUEUE.task_done()


threading.Thread(target=_worker, name="gpu-job-worker", daemon=True).start()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _authorized() -> bool:
    expected = os.environ.get("CICY_KOUBO_ACCESS_TOKEN", "")
    header = request.headers.get("Authorization", "")
    supplied = header[7:] if header.startswith("Bearer ") else ""
    return bool(expected and supplied and hmac.compare_digest(expected, supplied))


@app.before_request
def authenticate():
    if request.path == "/live":
        return None
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    length = request.content_length
    if length is not None and length > MAX_UPLOAD_BYTES:
        return jsonify({"error": "payload_too_large"}), 413
    return None


@app.get("/live")
def live():
    return jsonify({"ok": True, "api_version": API_VERSION})


@app.get("/v1/health")
def health():
    gpu = ""
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu = (probe.stdout or "").strip().splitlines()[0]
    except Exception:
        pass
    engines = {
        "musetalk": pathlib.Path("/content/mt/READY").is_file(),
        "cosyvoice": pathlib.Path("/content/cosy/COSY_READY").is_file(),
        "edge_tts": pathlib.Path("/content/cosy/env/bin/edge-tts").is_file(),
    }
    tts_script_sha256 = ""
    try:
        tts_script_sha256 = hashlib.sha256(COSYVOICE_SCRIPT.read_bytes()).hexdigest()
    except OSError:
        pass
    return jsonify({
        "ok": bool(gpu and engines["musetalk"]),
        "api_version": API_VERSION,
        "capabilities_version": CAPABILITIES_VERSION,
        "gpu": gpu,
        "engines": engines,
        "tts_languages": sorted(SUPPORTED_TTS_LANGUAGES),
        "tts_script_sha256": tts_script_sha256,
        "queue_depth": WORK_QUEUE.qsize(),
    })


def _save_upload(name: str, directory: pathlib.Path) -> pathlib.Path:
    upload = request.files.get(name)
    if not upload or not upload.filename:
        raise ValueError(f"{name}_required")
    suffix = pathlib.Path(upload.filename).suffix.lower()[:10]
    target = directory / f"{name}{suffix}"
    upload.save(target)
    return target


def _download_oss_url(raw_url: str, target: pathlib.Path) -> pathlib.Path:
    if len(raw_url) > 4096:
        raise ValueError("asset_url_too_long")
    parsed = urllib.parse.urlsplit(raw_url)
    if (
        parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment
        or not parsed.hostname or not parsed.hostname.endswith(".aliyuncs.com")
    ):
        raise ValueError("invalid_asset_url")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(raw_url, timeout=120) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError("empty_asset")
    return target


@app.post("/v1/video-jobs")
def create_video_job():
    payload_json = request.get_json(silent=True) or {}
    engine_name = str(payload_json.get("engine") or request.form.get("engine", "musetalk"))
    engine = ENGINES.get(("video", engine_name))
    if not engine:
        return jsonify({"error": "unsupported_engine"}), 400
    job = STORE.create("video", engine_name)
    try:
        video_url = str(payload_json.get("video_url") or "")
        audio_url = str(payload_json.get("audio_url") or "")
        payload = {
            "video": (
                _download_oss_url(video_url, job.directory / "video.mp4")
                if video_url else _save_upload("video", job.directory)
            ),
            "audio": (
                _download_oss_url(audio_url, job.directory / "audio.wav")
                if audio_url else _save_upload("audio", job.directory)
            ),
            "bbox": str(payload_json.get("bbox") or request.form.get("bbox", "0")),
            "result_upload_url": str(payload_json.get("result_upload_url") or ""),
            "result_download_url": str(payload_json.get("result_download_url") or ""),
        }
    except ValueError as exc:
        shutil.rmtree(job.directory, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400
    WORK_QUEUE.put((job, engine, payload))
    return jsonify(job.public()), 202


@app.post("/v1/tts-jobs")
def create_tts_job():
    payload_json = request.get_json(silent=True) or {}
    language = str(payload_json.get("language") or request.form.get("language", "zh-CN"))
    if language not in SUPPORTED_TTS_LANGUAGES:
        return jsonify({
            "error": "unsupported_tts_language",
            "language": language,
            "supported_languages": sorted(SUPPORTED_TTS_LANGUAGES),
        }), 400
    if language in DIALECT_LANGUAGES and not COSYVOICE_SCRIPT.is_file():
        return jsonify({
            "error": "multilingual_tts_script_missing",
            "language": language,
            "capabilities_version": CAPABILITIES_VERSION,
        }), 503
    engine = ENGINES[("tts", "edge-tts" if language in EDGE_VOICES else "cosyvoice")]
    text = str(payload_json.get("text") or request.form.get("text", "")).strip()
    reference_text = str(
        payload_json.get("reference_text") or request.form.get("reference_text", "")
    ).strip()
    if not text or len(text) > 20_000:
        return jsonify({"error": "invalid_text"}), 400
    job = STORE.create("tts", engine.name)
    try:
        reference_url = str(payload_json.get("reference_url") or "")
        reference = None if reference_url else _save_upload("reference", job.directory)
    except ValueError as exc:
        shutil.rmtree(job.directory, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400
    WORK_QUEUE.put((job, engine, {
        "reference": reference, "reference_text": reference_text, "text": text,
        "reference_url": reference_url,
        "language": language,
        "speed": str(payload_json.get("speed") or request.form.get("speed", "1.15")),
    }))
    return jsonify(job.public()), 202


def _job_or_error(job_id: str):
    if not JOB_ID_RE.fullmatch(job_id):
        return None, (jsonify({"error": "invalid_job_id"}), 400)
    job = STORE.get(job_id)
    if not job:
        return None, (jsonify({"error": "job_not_found"}), 404)
    return job, None


def _job_runtime_payload(job: Job) -> dict:
    payload = job.public()
    log_name = (
        "musetalk.log" if job.kind == "video"
        else ("edge-tts.log" if job.engine == "edge-tts" else "cosyvoice.log")
    )
    log_path = job.directory / log_name
    lines: list[str] = []
    if log_path.is_file():
        raw_lines = log_path.read_text(
            encoding="utf-8", errors="replace",
        ).replace("\r", "\n").splitlines()
        if job.kind == "video":
            lines = [line.strip()[:1000] for line in raw_lines if line.strip()][-40:]
        else:
            lines = [
                line.strip()[:1000] for line in raw_lines
                if line.strip() and (
                    re.match(r"^\[\d\d:\d\d:\d\d\]", line.strip())
                    or " INFO synthesis text " in line
                    or " INFO yield speech len " in line
                    or "Downloading Model to directory:" in line
                )
            ][-40:]
    payload["log"] = lines
    if job.status == "running" and job.kind == "tts":
        joined = "\n".join(lines)
        total_match = re.findall(r"分句结果:\s*(\d+)\s*段", joined)
        completed = len(re.findall(r"段\d+\s+合成完成:", joined))
        if total_match:
            total = max(1, int(total_match[-1]))
            completed = min(completed, total)
            payload["progress"] = min(95, 25 + round(70 * completed / total))
            payload["stage"] = (
                f"synthesizing_segment_{min(completed + 1, total)}_of_{total}"
                if completed < total else "finalizing_audio"
            )
            payload["segments_total"] = total
            payload["segments_completed"] = completed
    return payload


@app.get("/v1/jobs/<job_id>")
def get_job(job_id: str):
    job, error = _job_or_error(job_id)
    return error or jsonify(_job_runtime_payload(job))


@app.get("/v1/jobs/<job_id>/result")
def get_result(job_id: str):
    job, error = _job_or_error(job_id)
    if error:
        return error
    if job.status != "succeeded":
        return jsonify({"error": "result_not_ready", "status": job.status}), 409
    path = job.directory / job.result_name
    response = send_file(path, conditional=True, as_attachment=True)
    response.headers["X-Checksum-SHA256"] = job.result_sha256
    return response


@app.delete("/v1/jobs/<job_id>")
def cancel_job(job_id: str):
    job, error = _job_or_error(job_id)
    if error:
        return error
    if job.status in {"succeeded", "failed", "cancelled"}:
        return jsonify(job.public())
    job.cancel_requested = True
    job.stage = "cancelling"
    STORE.save(job)
    return jsonify(job.public()), 202


# Compatibility surface used by the local cicy-koubo backend. The browser
# continues to call localhost only; the local backend holds the temporary ga_
# token and proxies these routes to the ephemeral GPU instance.
@app.post("/api/generate-video")
def compat_generate_video():
    response = create_video_job()
    if isinstance(response, tuple):
        flask_response, status = response
        if status >= 300:
            return response
        payload = flask_response.get_json()
        return jsonify({"job_id": payload["id"]}), status
    payload = response.get_json()
    return jsonify({"job_id": payload["id"]}), response.status_code


@app.get("/api/job/<job_id>")
def compat_job(job_id: str):
    job, error = _job_or_error(job_id)
    if error:
        return error
    runtime = _job_runtime_payload(job)
    return jsonify({
        "stage": runtime["stage"],
        "progress": runtime["progress"],
        "log": runtime["log"],
        "result": job.id if job.status == "succeeded" else None,
        "error": job.error or None,
        "status": job.status,
    })


@app.get("/api/result/<job_id>")
def compat_result(job_id: str):
    return get_result(job_id)


@app.get("/api/cover/<job_id>")
def compat_cover(job_id: str):
    job, error = _job_or_error(job_id)
    if error:
        return error
    path = job.directory / "cover.jpg"
    if not path.is_file():
        return jsonify({"error": "cover_not_ready"}), 404
    return send_file(path, mimetype="image/jpeg", conditional=True)


def main() -> None:
    app.run(
        host=os.environ.get("KOUBO_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("KOUBO_API_PORT", "8770")),
        threaded=True,
    )


if __name__ == "__main__":
    main()
