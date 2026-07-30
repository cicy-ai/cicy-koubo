#!/usr/bin/env python3
"""爆款口播视频制作 — 本地后端(Flask)。
真实功能:
  - 生成视频:上传底板视频+音频 → 归一 25fps → scp 到 Colab → MuseTalk 对口型 → 取回
  - 剪辑:FFmpeg 烧录字幕(SRT/纯文本)
  - 音频生成:CosyVoice(装好后接;当前返回 not-ready)
  - 状态:Colab 隧道 / MuseTalk 环境 / CosyVoice
运行:python3 app.py  → http://127.0.0.1:8770
"""
import json
import datetime
import html
import os
import pathlib
import re
import secrets
import shlex
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid

from flask import Flask, request, jsonify, send_file, Response

_edit_lock = threading.Lock()
_tts_lock = threading.Lock()

SRC_DIR = pathlib.Path(__file__).resolve().parent      # src/
ROOT = SRC_DIR.parent / "app"                            # 项目根下的 app/
APP_DIR = ROOT
WORK = APP_DIR / "jobs"
WORK.mkdir(parents=True, exist_ok=True)
COSY_VENV = pathlib.Path.home() / "cosyvoice-venv/bin/python"
COSY_MODEL = pathlib.Path.home() / "CosyVoice/pretrained_models/CosyVoice2-0.5B/llm.pt"

GLOBAL_CFG = pathlib.Path.home() / "cicy-ai/global.json"

def load_global_cfg():
    """读取 cicy-ai 的唯一全局配置源。"""
    try:
        return json.load(open(GLOBAL_CFG))
    except FileNotFoundError:
        return {}

def save_global_cfg(cfg):
    """原子写回 global.json；口播只维护其中的 koubo 节点。"""
    GLOBAL_CFG.parent.mkdir(parents=True, exist_ok=True)
    tmp = GLOBAL_CFG.with_suffix(f".json.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, GLOBAL_CFG)

PORT = int(os.environ.get("KOUBO_PORT", "8770"))
def _gpu_ssh_conf():
    """远程 GPU 的 SSH 端点:环境变量 KOUBO_GPU_SSH(user@host)/KOUBO_GPU_SSH_PORT,
    否则读本机私有配置(不入库、不进代码)。都没有→占位值,远程操作自然失败,
    本机 GPU 模式(LOCAL_GPU)不受影响。"""
    host = os.environ.get("KOUBO_GPU_SSH", "")
    port = os.environ.get("KOUBO_GPU_SSH_PORT", "")
    if not host:
        try:
            cf = json.load(open(pathlib.Path.home() / "cicy-ai/db/colab-frp-ssh.json"))
            h = cf.get("gateway", {}).get("host", "")
            if h:
                host = "root@" + h
                port = port or str(cf.get("basePort") or "")
        except Exception:
            pass
    return host or "root@gpu-unconfigured", port or "22"


REMOTE, _GPU_PORT = _gpu_ssh_conf()
_KA = ["-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=12"]
SSH = ["ssh", "-p", _GPU_PORT, "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
       "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       *_KA, REMOTE]
SCP = ["scp", "-P", _GPU_PORT, "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null", *_KA]

# 本机 GPU 模式:app 直接跑在 Colab/GPU 机上(如 npx cicy-koubo on Colab)。
# 此时不走 frp/SSH——所有"远程"命令在 run() 里被改写为本地执行。
import shutil as _sht
LOCAL_GPU = os.environ.get("KOUBO_LOCAL_GPU", "") == "1" or (
    os.environ.get("KOUBO_LOCAL_GPU", "") != "0"
    and pathlib.Path("/content").exists() and _sht.which("nvidia-smi") is not None)


def _rewrite_local(cmd):
    if not LOCAL_GPU or not isinstance(cmd, list) or not cmd:
        return cmd
    if cmd[:len(SSH)] == SSH:                      # ssh <remote_cmd> → bash -lc <cmd>
        return ["bash", "-lc", " ".join(cmd[len(SSH):])]
    if cmd[:len(SCP)] == SCP:                      # scp a b host:dst → cp -f a b dst
        rest = [a.replace(REMOTE + ":", "") for a in cmd[len(SCP):]]
        return ["cp", "-f"] + rest
    return cmd


# ═══════════ GPU 模式自动检测 ═══════════
import platform as _platform

def _get_gpu_memory_mb():
    """返回 GPU 显存(MB)，无 GPU 返回 0"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return int((r.stdout or "").strip().splitlines()[0])
    except Exception:
        return 0


def _is_wsl():
    try:
        return "microsoft" in pathlib.Path("/proc/version").read_text(
            encoding="utf-8", errors="ignore",
        ).lower()
    except OSError:
        return False

def detect_gpu_mode():
    """自动检测 GPU 模式:
      macOS → colab_cli
      Linux + GPU ≥ 8GB → local
      Linux + GPU < 8GB → colab_cli
      无 GPU → colab_cli
    """
    if os.environ.get("KOUBO_GPU_MODE"):
        return os.environ["KOUBO_GPU_MODE"]
    configured = (load_global_cfg().get("koubo", {}).get("gpu", {}).get("provider") or "").lower()
    if configured in {"cicy_gpu", "local"}:
        return configured
    if configured in {"colab", "colab_cli"}:
        return "colab_cli"
    # 强制本地模式(跑在 Colab 上)
    if LOCAL_GPU:
        return "local"
    sys_name = _platform.system()
    if sys_name == "Darwin":
        return "colab_cli"
    # Linux
    gpu_mb = _get_gpu_memory_mb()
    if gpu_mb >= 8192:
        return "local"
    return "colab_cli"

GPU_MODE = detect_gpu_mode()

# CiCy GPU control-plane sessions are deliberately process-local.  The
# long-lived Gateway key stays in global.json; per-job GPU authorization never
# gets written to disk or exposed to the browser UI.
_CICY_GPU_SESSIONS = {}
_CICY_GPU_ACTIVE_JOB_ID = ""
_CICY_GPU_PREPARED_REFERENCES = {}
_CICY_GPU_RESULT_ASSETS = {}
_CICY_GPU_VIDEO_LOG_LINES = {}
_CICY_GPU_VIDEO_LAST_STATE = {}

def _local_gpu_conf():
    """Read the token written by the Windows installer without exposing it."""
    endpoint = os.environ.get("KOUBO_LOCAL_GPU_ENDPOINT", "http://cicy-koubo-gpu:8771").rstrip("/")
    candidates = []
    configured = os.environ.get("CICY_KOUBO_LOCAL_CONFIG", "")
    if configured:
        candidates.append(pathlib.Path(configured))
    candidates.extend(pathlib.Path("/mnt").glob("*/CiCy/koubo-local/config.json"))
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            token = str(data.get("token") or "")
            if token:
                return endpoint, token
        except (OSError, ValueError):
            continue
    return endpoint, ""

def _local_gpu_install_root():
    mounts = []
    for candidate in pathlib.Path("/mnt").glob("*"):
        try:
            if candidate.is_dir():
                mounts.append((_sht.disk_usage(candidate).free, candidate))
        except OSError:
            continue
    base = max(mounts, default=(0, pathlib.Path.home()), key=lambda item: item[0])[1]
    return base / "CiCy" / "koubo-local"


def _local_gpu_docker_status():
    endpoint, token = _local_gpu_conf()
    root = _local_gpu_install_root()
    for candidate in pathlib.Path("/mnt").glob("*/CiCy/koubo-local/config.json"):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if data.get("token"):
                root = candidate.parent
                break
        except (OSError, ValueError):
            continue
    docker_ok = _sht.which("docker") is not None
    status_text = "not_installed"
    logs = ""
    if docker_ok:
        probe = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", "cicy-koubo-gpu"],
            capture_output=True, text=True, timeout=10,
        )
        if probe.returncode == 0:
            status_text = probe.stdout.strip() or "stopped"
            if status_text != "running":
                log_result = subprocess.run(
                    ["docker", "logs", "--tail", "30", "cicy-koubo-gpu"],
                    capture_output=True, text=True, timeout=10,
                )
                logs = (log_result.stdout + log_result.stderr)[-4000:]
        elif token:
            status_text = "stopped"
    health = {}
    if status_text == "running" and token:
        try:
            health = _local_gpu_request("GET", "/v1/health", token, timeout=8)
        except Exception as exc:
            health = {"ok": False, "error": str(exc)}
    return {
        "installed": bool(token),
        "docker_available": docker_ok,
        "status": status_text,
        "running": status_text == "running" and bool(health.get("ok")),
        "endpoint": endpoint,
        "root": str(root),
        "health": health,
        "logs": logs,
    }


def _local_gpu_request(method, route, token, payload=None, timeout=120):
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "cicy-koubo/0.1.8"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request_obj = urllib.request.Request(
        _local_gpu_conf()[0] + route, data=data, headers=headers, method=method,
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request_obj, timeout=timeout,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _local_gpu_multipart(route, token, fields, files, timeout=600):
    """Stream local files with curl; credentials are supplied via stdin, not argv."""
    endpoint = _local_gpu_conf()[0] + route
    args = ["curl", "-fsS", "--max-time", str(timeout), "--config", "-"]
    for name, value in fields.items():
        args.extend(["-F", f"{name}={value}"])
    for name, file_path in files.items():
        args.extend(["-F", f"{name}=@{file_path}"])
    config = f'header = "Authorization: Bearer {token}"\n'
    result = subprocess.run(args + [endpoint], input=config, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("本地 GPU 上传失败：" + (result.stderr or "")[-300:])
    return json.loads(result.stdout)


def _local_gpu_wait(job_id, token, timeout_seconds=1800):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        job = _local_gpu_request(
            "GET", "/v1/jobs/" + urllib.parse.quote(job_id, safe=""), token, timeout=30,
        )
        if job.get("status") == "succeeded":
            return job
        if job.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(job.get("error") or f"本地 GPU {job.get('status')}")
        time.sleep(2)
    raise RuntimeError("本地 GPU 任务超时")


def _local_gpu_download(job_id, token, destination):
    request_obj = urllib.request.Request(
        _local_gpu_conf()[0] + "/v1/jobs/" + urllib.parse.quote(job_id, safe="") + "/result",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "cicy-koubo/0.1.8"},
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request_obj, timeout=600,
    ) as response, open(destination, "wb") as output:
        _sht.copyfileobj(response, output)


def _local_gpu_tts():
    endpoint, token = _local_gpu_conf()
    if not token:
        return jsonify({"error": "本地 GPU 尚未安装或访问 Token 不可用"}), 503
    text = (request.form.get("text") or ((request.get_json(silent=True) or {}).get("text")) or "").strip()
    if not text:
        return jsonify({"error": "没有要配音的文案"}), 400
    ref_id = (request.form.get("ref_id") or "").strip()
    if ref_id.startswith("voice-sample-") and (ROOT / "assets" / ref_id).is_file():
        ref = ROOT / "assets" / ref_id
    elif "ref" in request.files and request.files["ref"].filename:
        ref = WORK / (uuid.uuid4().hex[:8] + "_ref" + pathlib.Path(request.files["ref"].filename).suffix)
        request.files["ref"].save(ref)
    else:
        samples = sorted((ROOT / "assets").glob("voice-sample-*.wav"))
        if not samples:
            return jsonify({"error": "没有参考音色"}), 400
        ref = samples[-1]
    reference_text = _transcribe(str(ref), (request.form.get("stt_provider") or "auto").strip())
    if not reference_text:
        return jsonify({"error": "参考音频转写失败"}), 400
    try:
        created = _local_gpu_multipart("/v1/tts-jobs", token, {
            "text": text, "reference_text": reference_text,
        }, {"reference": ref})
        _local_gpu_wait(created["id"], token)
        jid = uuid.uuid4().hex[:10]
        dst = MEDIA_DIR / f"voice_{jid}.wav"
        _local_gpu_download(created["id"], token, dst)
        _normalize_voice_output(dst)
        duration = _ffdur(dst)
        _media_add({"id": jid, "type": "voice", "file": dst.name, "text": text[:100],
                    "chars": len(text), "duration": duration, "ts": time.strftime("%m-%d %H:%M")})
        return jsonify({"id": jid, "url": f"/api/media/{jid}/file",
                        "duration": duration, "chars": len(text)})
    except Exception as exc:
        return jsonify({"error": str(exc), "code": "local_gpu_failed"}), 502


def _local_gpu_video():
    endpoint, token = _local_gpu_conf()
    if not token:
        return jsonify({"error": "本地 GPU 尚未安装或访问 Token 不可用"}), 503
    audio_id = (request.form.get("audio_id") or "").strip()
    audio_entry = next((x for x in _media_list() if x.get("id") == audio_id), None)
    audio = MEDIA_DIR / audio_entry["file"] if audio_entry else None
    base_id = (request.form.get("base_id") or "").strip()
    video = ROOT / "assets" / base_id if base_id.startswith("base-video-") else None
    if not audio or not audio.is_file() or not video or not video.is_file():
        return jsonify({"error": "缺少本地底板或配音素材"}), 400
    try:
        created = _local_gpu_multipart("/v1/video-jobs", token, {
            "engine": request.form.get("engine") or "musetalk",
            "bbox": request.form.get("bbox") or "0",
        }, {"video": video, "audio": audio})
        job_id = created["id"]
        JOBS[job_id] = {"stage": "本地 GPU 处理中", "log": [], "result": None, "error": None}
        def finish_local():
            try:
                _local_gpu_wait(job_id, token)
                dst_dir = WORK / job_id
                dst_dir.mkdir(exist_ok=True)
                _local_gpu_download(job_id, token, dst_dir / "result.mp4")
                JOBS[job_id].update(stage="完成", result=job_id)
            except Exception as exc:
                JOBS[job_id].update(stage="失败", error=str(exc))
        threading.Thread(target=finish_local, daemon=True).start()
        return jsonify({"job_id": job_id})
    except Exception as exc:
        return jsonify({"error": str(exc), "code": "local_gpu_failed"}), 502


def _cicy_gateway_conf():
    cfg = load_global_cfg()
    for provider in cfg.get("providers", {}).get("items", []):
        url = str(provider.get("url") or "").rstrip("/")
        key = str(provider.get("apiKey") or "")
        if "gateway.cicy-ai.com" in url and key.startswith("sk-cicy-"):
            return url, key
    return "", ""


def _public_direct_opener():
    """Direct HTTPS opener isolated from desktop proxy/SSL_CERT_FILE state."""
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError):
        context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )


def _curl_upload_signed(path, upload_url, content_type, log_prefix, timeout=900):
    """Upload through curl/proxy and emit progress without exposing signed URL."""
    config = (
        f'url = "{upload_url}"\n'
        f'upload-file = "{path}"\n'
        f'header = "Content-Type: {content_type}"\n'
        'header = "Expect:"\n'
    )
    started = time.monotonic()
    last_bucket = -1
    with tempfile.TemporaryFile(mode="w+b") as progress:
        process = subprocess.Popen(
            [
                "curl", "--fail", "--show-error", "--progress-bar",
                "--retry", "3", "--retry-all-errors",
                "--noproxy", "*",
                "--connect-timeout", "20", "--max-time", str(timeout),
                "--speed-limit", "1024", "--speed-time", "30",
                "--config", "-",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=progress,
            text=True,
        )
        process.stdin.write(config)
        process.stdin.close()
        while process.poll() is None:
            time.sleep(1)
            progress.flush()
            progress.seek(0)
            report = progress.read().decode("utf-8", errors="replace")
            percentages = re.findall(r"(\d+(?:\.\d+)?)%", report)
            if percentages:
                percent = min(100, int(float(percentages[-1])))
                bucket = percent // 10
                if bucket > last_bucket:
                    last_bucket = bucket
                    sent = round(path.stat().st_size * percent / 100)
                    plog(
                        f"{log_prefix} 上传进度 {percent}% · "
                        f"{sent}/{path.stat().st_size} bytes · "
                        f"{time.monotonic() - started:.1f}s"
                    )
        progress.seek(0)
        report = progress.read().decode("utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(
            f"OSS 上传失败（curl {process.returncode}）：{report[-500:]}"
        )
    plog(
        f"{log_prefix} 上传完成 · {path.stat().st_size} bytes · "
        f"{time.monotonic() - started:.2f}s"
    )


def _cicy_gateway_request(method, path, payload=None, idempotency_key=""):
    base_url, api_key = _cicy_gateway_conf()
    if not base_url or not api_key:
        raise RuntimeError("global.json 中没有可用的 CiCy Gateway Key")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        # Cloudflare Browser Integrity rejects urllib's default
        # "Python-urllib/*" signature with Error 1010.
        "User-Agent": "cicy-koubo/0.1.8",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    # The desktop may set SSL_CERT_FILE to a private MITM bundle which does
    # not contain the public WebPKI chain.  Keep it for the system-proxy
    # attempt, but use certifi explicitly for the direct fallback.
    # CiCy Cloud/OSS must bypass the desktop proxy.  The system route can
    # block well beyond the UI timeout when its injected CA/proxy is stale.
    openers = (("direct", _public_direct_opener()),)
    last_error = None
    for attempt in range(3):
        route_name, opener = openers[0]
        req = urllib.request.Request(
            base_url + path, data=data, headers=headers, method=method,
        )
        try:
            with opener.open(req, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                result = json.loads(raw)
            except Exception:
                result = {"success": False, "error": raw or f"HTTP {exc.code}"}
            return exc.code, result
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, ConnectionError) as exc:
            last_error = exc
            plog(
                f"[CiCy Cloud] 请求连接失败，正在重试 · "
                f"path={path} route={route_name} attempt={attempt + 1}/3 "
                f"error={type(exc).__name__}"
            )
            if attempt < 2:
                time.sleep(attempt + 1)
    raise last_error or RuntimeError("CiCy Cloud 请求失败")


def _cicy_gpu_proxy(path, complete_on_success=False):
    session = _CICY_GPU_SESSIONS.get(_CICY_GPU_ACTIVE_JOB_ID) or {}
    endpoint = str(session.get("endpoint") or "").rstrip("/")
    authorization = str(session.get("authorization_token") or "")
    if not endpoint or not authorization:
        return jsonify({
            "error": "CiCy GPU 尚未就绪，请先申请 GPU 并等待临时 Endpoint",
            "code": "cicy_gpu_not_ready",
        }), 503
    headers = {
        "Authorization": f"Bearer {authorization}",
        "Accept": request.headers.get("Accept", "application/json"),
    }
    if request.content_type:
        headers["Content-Type"] = request.content_type
    data = request.get_data(cache=True) if request.method not in {"GET", "HEAD"} else None
    query = ("?" + request.query_string.decode()) if request.query_string else ""
    upstream = urllib.request.Request(
        endpoint + path + query, data=data, headers=headers, method=request.method,
    )
    try:
        # The ephemeral GPU endpoint is a direct public IP. Do not send its
        # authenticated upload/download traffic through the user's HTTP proxy.
        response = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
        ).open(upstream, timeout=7200)
        status = response.status
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        control_job_id = _CICY_GPU_ACTIVE_JOB_ID

        def stream():
            completed = False
            try:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        completed = True
                        break
                    yield chunk
            finally:
                response.close()
                if completed and complete_on_success and control_job_id:
                    try:
                        safe_id = urllib.parse.quote(control_job_id, safe="")
                        _cicy_gateway_request(
                            "POST", f"/api/koubo/jobs/{safe_id}/complete", {},
                        )
                        _CICY_GPU_SESSIONS.pop(control_job_id, None)
                    except Exception as exc:
                        plog(f"[CiCy GPU] 成品已下载，但释放通知失败: {exc}")

        return Response(stream(), status=status, content_type=content_type)
    except urllib.error.HTTPError as exc:
        return Response(
            exc.read(), status=exc.code,
            content_type=exc.headers.get("Content-Type", "application/json"),
        )
    except urllib.error.URLError as exc:
        return jsonify({
            "error": f"CiCy GPU 连接失败：{exc.reason}",
            "code": "cicy_gpu_connection_failed",
        }), 502
    except TimeoutError:
        return jsonify({
            "error": "CiCy GPU 请求超时，请重试；已有远程任务会自动清理",
            "code": "cicy_gpu_timeout",
        }), 504


def _cicy_gpu_video_status(job_id):
    """Fetch a video job and mirror new remote model output into the UI log."""
    session = _CICY_GPU_SESSIONS.get(_CICY_GPU_ACTIVE_JOB_ID) or {}
    endpoint = str(session.get("endpoint") or "").rstrip("/")
    authorization = str(session.get("authorization_token") or "")
    if not endpoint or not authorization:
        return jsonify({"error": "CiCy GPU 尚未就绪"}), 503
    safe_id = urllib.parse.quote(job_id, safe="")
    upstream = urllib.request.Request(
        endpoint + f"/api/job/{safe_id}",
        headers={"Authorization": f"Bearer {authorization}", "Accept": "application/json"},
    )
    try:
        with _public_direct_opener().open(upstream, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        seen = _CICY_GPU_VIDEO_LOG_LINES.setdefault(job_id, set())
        for raw_line in payload.get("log") or []:
            line = str(raw_line).strip()
            if line and line not in seen:
                seen.add(line)
                plog(f"[CiCy GPU 出片][模型] {line[:1000]}")
        if payload.get("status") == "running":
            progress_rows = re.findall(
                r"(\d{1,3})%\|[^|]*\|\s*(\d+)/(\d+)",
                "\n".join(str(line) for line in payload.get("log") or []),
            )
            if progress_rows:
                percent, completed, total = progress_rows[-1]
                payload["progress"] = min(99, int(percent))
                payload["stage"] = f"lipsync_frame_{completed}_of_{total}"
                payload["frames_completed"] = int(completed)
                payload["frames_total"] = int(total)
        state = (
            str(payload.get("status") or ""),
            str(payload.get("stage") or ""),
            int(payload.get("progress") or 0),
        )
        if _CICY_GPU_VIDEO_LAST_STATE.get(job_id) != state:
            _CICY_GPU_VIDEO_LAST_STATE[job_id] = state
            plog(
                f"[CiCy GPU 出片] 远程进度 · status={state[0]} "
                f"stage={state[1]} progress={state[2]}%"
            )
        return jsonify(payload)
    except urllib.error.HTTPError as exc:
        return Response(
            exc.read(), status=exc.code,
            content_type=exc.headers.get("Content-Type", "application/json"),
        )
    except Exception as exc:
        return jsonify({
            "error": f"CiCy GPU 状态读取失败：{type(exc).__name__}: {exc}",
            "code": "cicy_gpu_status_failed",
        }), 502


def _cicy_gpu_tts():
    """Upload the local reference voice, run remote TTS, and keep the result local."""
    session = _CICY_GPU_SESSIONS.get(_CICY_GPU_ACTIVE_JOB_ID) or {}
    endpoint = str(session.get("endpoint") or "").rstrip("/")
    authorization = str(session.get("authorization_token") or "")
    if not endpoint or not authorization:
        return jsonify({
            "error": "CiCy GPU 尚未就绪，请先申请 GPU 并等待临时 Endpoint",
            "code": "cicy_gpu_not_ready",
        }), 503

    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "没有要配音的文案"}), 400
    try:
        speed = max(0.5, min(1.5, float(request.form.get("speed", "1.15"))))
    except ValueError:
        speed = 1.15
    language = (request.form.get("language") or "zh-CN").strip()
    language_labels = {
        "zh-CN": "普通话", "zh-yue": "粤语", "zh-minnan": "闽南语",
        "zh-sichuan": "四川话", "zh-dongbei": "东北话",
        "zh-shanghai": "上海话", "zh-tianjin": "天津话",
        "zh-shandong": "山东话", "zh-shaanxi": "陕西话",
        "zh-shanxi": "山西话", "en": "English", "fr": "Français",
        "de": "Deutsch", "es": "Español", "it": "Italiano",
        "vi": "Tiếng Việt", "id": "Bahasa Indonesia",
        "ms": "Bahasa Melayu", "th": "ไทย", "ko": "한국어",
        "ru": "Русский", "ar": "العربية", "km": "ខ្មែរ", "lo": "ລາວ",
    }

    gpu_opener = _public_direct_opener()
    oss_opener = _public_direct_opener()
    gpu_headers = {
        "Authorization": f"Bearer {authorization}",
        "Content-Type": "application/json",
        "User-Agent": "cicy-koubo/0.1.8",
    }
    try:
        health_req = urllib.request.Request(
            endpoint + "/v1/health", headers=gpu_headers,
        )
        health = None
        health_error = None
        for attempt in range(3):
            try:
                with gpu_opener.open(health_req, timeout=15) as response:
                    health = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                health_error = exc
                plog(
                    f"[CiCy GPU 配音] 能力检查连接失败，重试 "
                    f"{attempt + 1}/3 · {type(exc).__name__}"
                )
                if attempt < 2:
                    time.sleep(attempt + 1)
        if health is None:
            raise health_error or RuntimeError("GPU 能力检查无响应")
        supported = set(health.get("tts_languages") or [])
        capability = str(health.get("capabilities_version") or "")
        if language not in supported or not capability.startswith("2026.07.30-multilingual-"):
            return jsonify({
                "error": (
                    f"当前 GPU 镜像不支持 {language_labels.get(language, language)}，"
                    "请销毁后重新启动新版 GPU"
                ),
                "code": "gpu_multilingual_capability_missing",
                "language": language,
                "capabilities_version": capability,
            }), 409
        plog(
            f"[CiCy GPU 配音] 目标语言确认 · "
            f"{language_labels.get(language, language)} ({language}) · capability={capability}"
        )
    except urllib.error.HTTPError as exc:
        return jsonify({
            "error": f"GPU 多语言能力校验失败：HTTP {exc.code}",
            "code": "gpu_capability_check_failed",
        }), 502
    except Exception as exc:
        return jsonify({
            "error": f"GPU 多语言能力校验失败：{exc}",
            "code": "gpu_capability_check_failed",
        }), 502
    prepared_id = (request.form.get("prepared_reference_id") or "").strip()
    prepared = _CICY_GPU_PREPARED_REFERENCES.pop(prepared_id, None)
    if prepared and prepared.get("expires_at", 0) > time.time():
        signed = prepared["signed"]
        reference_text = prepared["reference_text"]
        plog("[CiCy GPU 配音] 5/6 参考音频已就绪，提交远程配音任务")
    else:
        ref_id = (request.form.get("ref_id") or "").strip()
        if ref_id and ref_id.startswith("voice-sample-") and (ROOT / "assets" / ref_id).is_file():
            ref = ROOT / "assets" / ref_id
        elif "ref" in request.files and request.files["ref"].filename:
            ref = WORK / (uuid.uuid4().hex[:8] + "_ref" + pathlib.Path(request.files["ref"].filename).suffix)
            request.files["ref"].save(ref)
        else:
            samples = sorted((ROOT / "assets").glob("voice-sample-*.wav"))
            if not samples:
                return jsonify({"error": "没有参考音色,请先选择/上传一段人声样本"}), 400
            ref = samples[-1]
        stt_provider = (request.form.get("stt_provider") or "auto").strip()
        try:
            reference_text = _transcribe(str(ref), stt_provider)
        except GroqTranscriptionError as exc:
            return jsonify({"error": str(exc), "code": "groq_stt_failed"}), 502
        if not reference_text:
            return jsonify({"error": f"参考音频转写失败（{stt_provider}）"}), 400
        sign_status, signed = _cicy_gateway_request("POST", "/api/koubo/assets/sign", {
            "region_id": session.get("region_id") or "cn-hangzhou",
            "purpose": "reference",
            "content_type": "audio/wav",
            "extension": "wav",
        })
        if sign_status >= 300:
            return jsonify({"error": signed.get("error") or "OSS 上传签名失败"}), sign_status
        upload_req = urllib.request.Request(
            signed["upload_url"], data=ref.read_bytes(),
            headers={"Content-Type": "audio/wav", "User-Agent": "cicy-koubo/0.1.8"},
            method="PUT",
        )
        try:
            with oss_opener.open(upload_req, timeout=120):
                pass
        except urllib.error.URLError as exc:
            return jsonify({"error": f"OSS 上传失败：{exc.reason}"}), 502
    try:
        create_req = urllib.request.Request(
            endpoint + "/v1/tts-jobs",
            data=json.dumps({
                "text": text,
                "language": language,
                "speed": speed,
                "reference_text": reference_text,
                "reference_url": signed["download_url"],
            }, ensure_ascii=False).encode("utf-8"),
            headers=gpu_headers, method="POST",
        )
        with gpu_opener.open(create_req, timeout=120) as response:
            create_status = response.status
            create_raw = response.read().decode("utf-8")
        created = json.loads(create_raw)
        remote_job = str(created.get("id") or created.get("job_id") or "")
        if not remote_job:
            plog(
                f"[CiCy GPU 配音] 创建响应异常 · HTTP {create_status} · "
                f"keys={sorted(created.keys()) if isinstance(created, dict) else type(created).__name__}"
            )
            return jsonify({
                "error": "GPU 创建任务响应缺少任务 ID",
                "code": "gpu_create_response_invalid",
            }), 502
        plog(f"[CiCy GPU 配音] 远程任务已创建 · job={remote_job}")
        try:
            oss_opener.open(
                urllib.request.Request(signed["delete_url"], method="DELETE"), timeout=30,
            ).close()
        except Exception:
            pass
        last_remote_state = ("", "", -1)
        seen_remote_logs: set[str] = set()
        last_wait_log_at = 0.0
        for _ in range(300):
            status_req = urllib.request.Request(
                endpoint + "/v1/jobs/" + urllib.parse.quote(remote_job, safe=""),
                headers=gpu_headers,
            )
            try:
                with gpu_opener.open(status_req, timeout=30) as response:
                    remote = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
                continue
            remote_stage = str(remote.get("stage") or remote.get("status") or "")
            remote_state = (
                str(remote.get("status") or ""),
                remote_stage,
                int(remote.get("progress") or 0),
            )
            if remote_stage and remote_state != last_remote_state:
                last_remote_state = remote_state
                plog(
                    f"[CiCy GPU 配音] 远程进度 · status={remote.get('status')} "
                    f"stage={remote_stage} progress={remote.get('progress', 0)}%"
                )
            for raw_line in remote.get("log") or []:
                line = str(raw_line).strip()
                if not line or line in seen_remote_logs:
                    continue
                seen_remote_logs.add(line)
                plog(f"[CiCy GPU 配音][模型] {line[:1000]}")
            now = time.monotonic()
            if (
                remote.get("status") == "running"
                and now - last_wait_log_at >= 15
            ):
                last_wait_log_at = now
                plog(
                    f"[CiCy GPU 配音] 模型仍在运行 · stage={remote_stage} "
                    f"progress={remote.get('progress', 0)}%"
                )
            if remote.get("status") == "succeeded":
                break
            if remote.get("status") in {"failed", "cancelled"}:
                return jsonify({"error": remote.get("error") or "远程配音失败"}), 500
            time.sleep(2)
        else:
            return jsonify({"error": "远程配音超时"}), 504

        result_req = urllib.request.Request(
            endpoint + "/v1/jobs/" + urllib.parse.quote(remote_job, safe="") + "/result",
            headers=gpu_headers,
        )
        jid = uuid.uuid4().hex[:10]
        dst = MEDIA_DIR / f"voice_{jid}.wav"
        with gpu_opener.open(result_req, timeout=120) as response, open(dst, "wb") as output:
            _sht.copyfileobj(response, output)
        plog(f"[CiCy GPU 配音] 6/6 结果下载完成 · file={dst.name} bytes={dst.stat().st_size}")
        _normalize_voice_output(dst)
        duration = _ffdur(dst)
        _media_add({
            "id": jid, "type": "voice", "file": dst.name,
            "text": text[:100], "chars": len(text), "speed": speed,
            "duration": duration, "ts": time.strftime("%m-%d %H:%M"),
        })
        return jsonify({
            "id": jid, "url": f"/api/media/{jid}/file",
            "duration": duration, "chars": len(text),
        })
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw).get("error") or raw
        except Exception:
            detail = raw
        return jsonify({"error": f"远程配音失败：{detail or exc.code}"}), exc.code
    except urllib.error.URLError as exc:
        return jsonify({
            "error": f"CiCy GPU 连接失败：{exc.reason}",
            "code": "cicy_gpu_connection_failed",
        }), 502
    except Exception as exc:
        plog(
            f"[CiCy GPU 配音] 本地同步异常 · "
            f"{type(exc).__name__}: {str(exc)[:500]}"
        )
        return jsonify({
            "error": f"CiCy GPU 本地结果同步失败：{type(exc).__name__}: {exc}",
            "code": "cicy_gpu_local_sync_failed",
        }), 502


def _cicy_gpu_video():
    session = _CICY_GPU_SESSIONS.get(_CICY_GPU_ACTIVE_JOB_ID) or {}
    endpoint = str(session.get("endpoint") or "").rstrip("/")
    authorization = str(session.get("authorization_token") or "")
    if not endpoint or not authorization:
        return jsonify({"error": "CiCy GPU 尚未就绪"}), 503
    audio_id = (request.form.get("audio_id") or "").strip()
    audio_entry = next((x for x in _media_list() if x.get("id") == audio_id), None)
    audio = MEDIA_DIR / audio_entry["file"] if audio_entry else None
    base_id = (request.form.get("base_id") or "").strip()
    video = ROOT / "assets" / base_id if base_id.startswith("base-video-") else None
    if not audio or not audio.is_file() or not video or not video.is_file():
        return jsonify({"error": "缺少本地底板或配音素材"}), 400
    region = session.get("region_id") or "cn-hangzhou"
    plog(
        f"[CiCy GPU 出片] 1/7 素材确认 · "
        f"video={video.name} {video.stat().st_size} bytes · "
        f"audio={audio.name} {audio.stat().st_size} bytes"
    )
    signed_assets = []
    for asset_index, (purpose, path, content_type, extension) in enumerate((
        ("video", video, "video/mp4", "mp4"),
        ("audio", audio, "audio/wav", "wav"),
    ), start=2):
        plog(f"[CiCy GPU 出片] {asset_index}/7 获取 {purpose} OSS 签名")
        status, signed = _cicy_gateway_request("POST", "/api/koubo/assets/sign", {
            "region_id": region, "purpose": purpose,
            "content_type": content_type, "extension": extension,
        })
        if status >= 300:
            return jsonify({"error": signed.get("error") or "OSS 上传签名失败"}), status
        plog(
            f"[CiCy GPU 出片] {asset_index}/7 {purpose} 签名完成，开始上传 · "
            f"{path.stat().st_size} bytes"
        )
        _curl_upload_signed(
            path, signed["upload_url"], content_type,
            f"[CiCy GPU 出片] {asset_index}/7 {purpose}",
        )
        signed_assets.append(signed)
    plog("[CiCy GPU 出片] 4/7 获取成片回传签名")
    status, result_signed = _cicy_gateway_request("POST", "/api/koubo/assets/sign", {
        "region_id": region, "purpose": "result",
        "content_type": "video/mp4", "extension": "mp4",
    })
    if status >= 300:
        return jsonify({"error": result_signed.get("error") or "OSS 成片签名失败"}), status
    gpu_headers = {
        "Authorization": f"Bearer {authorization}",
        "Content-Type": "application/json",
        "User-Agent": "cicy-koubo/0.1.8",
    }
    opener = _public_direct_opener()
    payload = {
        "video_url": signed_assets[0]["download_url"],
        "audio_url": signed_assets[1]["download_url"],
        "engine": "musetalk",
        "bbox": "0",
        "result_upload_url": result_signed["upload_url"],
        "result_download_url": result_signed["download_url"],
    }
    try:
        plog("[CiCy GPU 出片] 5/7 提交 MuseTalk 远程任务")
        req = urllib.request.Request(
            endpoint + "/v1/video-jobs",
            data=json.dumps(payload).encode(), headers=gpu_headers, method="POST",
        )
        with opener.open(req, timeout=600) as response:
            remote = json.loads(response.read().decode())
        remote_job = str(remote.get("id") or remote.get("job_id") or "")
        if not remote_job:
            raise RuntimeError("GPU 出片响应缺少任务 ID")
        plog(f"[CiCy GPU 出片] 6/7 远程任务已创建 · job={remote_job}")
        _CICY_GPU_RESULT_ASSETS[remote_job] = result_signed
        for signed in signed_assets:
            try:
                _public_direct_opener().open(
                    urllib.request.Request(signed["delete_url"], method="DELETE"), timeout=30,
                ).close()
            except Exception:
                pass
        return jsonify({"job_id": remote_job}), 202
    except urllib.error.HTTPError as exc:
        return Response(exc.read(), status=exc.code, content_type="application/json")
    except Exception as exc:
        plog(f"[CiCy GPU 出片] ❌ 提交失败 · {type(exc).__name__}: {str(exc)[:500]}")
        return jsonify({
            "error": f"CiCy GPU 出片提交失败：{type(exc).__name__}: {exc}",
            "code": "cicy_gpu_video_submit_failed",
        }), 502

def _colab_profiles_cfg():
    """读取 Colab CLI 配置档；凭据文件由用户在本机单独准备。"""
    cfg = load_global_cfg()
    colab = cfg.setdefault("koubo", {}).setdefault("colab", {})
    profiles = colab.setdefault("profiles", [])
    if not profiles:
        profiles.append({
            "id": "default",
            "name": "默认 Google 账号",
            "auth": "oauth2",
            "credentials_path": "",
            "session_config": str(pathlib.Path.home() / ".config/colab-cli/sessions.json"),
            "session": "koubo",
            "gpu": "T4",
        })
        colab["active"] = "default"
    colab.setdefault("active", profiles[0]["id"])
    return cfg, colab


def _active_colab_profile():
    _, colab = _colab_profiles_cfg()
    active = colab.get("active")
    profile = next((p for p in colab["profiles"] if p.get("id") == active), colab["profiles"][0])
    return {
        "id": profile.get("id") or "default",
        "name": profile.get("name") or "Google 账号",
        "email": (profile.get("email") or "").strip(),
        "auth": (profile.get("auth") or "oauth2").lower(),
        "credentials_path": os.path.expanduser(profile.get("credentials_path") or ""),
        "session_config": os.path.expanduser(profile.get("session_config") or
                                             "~/.config/colab-cli/sessions.json"),
        "session": profile.get("session") or "koubo",
        "gpu": (profile.get("gpu") or "T4").upper(),
    }


def _colab_base_args(profile):
    colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
    return [colab_bin, "--auth", profile["auth"], "--config", profile["session_config"]]

_COLAB_SESSION_CACHE = {}


def _colab_resolve_session(profile, requested=None, force=False):
    """Resolve a session alias from this profile's local CLI state."""
    configured = requested or profile["session"]
    cache_key = (profile["id"], profile["session_config"], configured)
    cached = _COLAB_SESSION_CACHE.get(cache_key)
    if not force and cached and time.time() - cached[0] < 30:
        return cached[1]
    try:
        # The state file is keyed by the CLI alias accepted by exec/upload.
        # `colab sessions` prints the runtime endpoint instead, which is only a
        # display value and produces "Session not found" when passed to -s.
        state_path = pathlib.Path(profile["session_config"])
        if state_path.is_file():
            state = json.loads(state_path.read_text())
            if isinstance(state, dict) and configured in state:
                _COLAB_SESSION_CACHE[cache_key] = (time.time(), configured)
                return configured
            if isinstance(state, dict):
                names = [str(name).strip() for name in state if str(name).strip()]
            else:
                names = []
        else:
            names = []
        # colab-cli 0.6 can keep a live assignment even when its JSON state was
        # cleared. Only an authorized profile may perform this network lookup;
        # unauthorised profiles used to hang here waiting for OAuth input.
        if not names and _colab_oauth_token_path(profile).is_file():
            result = subprocess.run(
                _colab_base_args(profile) + ["sessions"],
                env=_colab_env(profile), capture_output=True, text=True, timeout=10,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            names = [
                name.strip()
                for name in re.findall(r"^\s*\[[^\]]*\]\s+([^|\r\n]+?)\s*\|", output, re.M)
                if name.strip()
            ]
        resolved = configured if configured in names else (names[0] if names else configured)
        _COLAB_SESSION_CACHE[cache_key] = (time.time(), resolved)
        if resolved != configured:
            plog(
                f"[Colab 会话自动切换] profile={profile['id']}, "
                f"configured={configured}, active={resolved}"
            )
        return resolved
    except Exception as exc:
        plog(f"[Colab 会话解析失败] profile={profile['id']}, error={exc}")
        return configured


def _colab_env(profile):
    env = os.environ.copy()
    # colab-cli 0.6 stores OAuth at ~/.config/colab-cli/token.json. Give every
    # additional Google profile its own HOME so credentials never bleed across
    # accounts. Keep the legacy default profile on the user's existing token.
    if profile.get("auth") == "oauth2" and profile.get("id") != "default":
        auth_home = pathlib.Path.home() / ".config/cicy-koubo/colab-oauth" / profile["id"]
        auth_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(auth_home)
    if profile.get("auth") == "adc" and profile.get("credentials_path"):
        env["GOOGLE_APPLICATION_CREDENTIALS"] = profile["credentials_path"]
    return env


def _colab_oauth_token_path(profile):
    if profile.get("id") == "default":
        return pathlib.Path.home() / ".config/colab-cli/token.json"
    return pathlib.Path.home() / ".config/cicy-koubo/colab-oauth" / profile["id"] / ".config/colab-cli/token.json"


def _colab_exec(cmd_str, session=None, timeout=300, profile=None):
    """通过 colab CLI 执行远程命令，返回 (returncode, stdout)"""
    import tempfile
    colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
    profile = profile or _active_colab_profile()
    # 写一个 Python 脚本来执行命令并打印输出
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(f"import subprocess,sys,json\n"
                f"r=subprocess.run({repr(cmd_str)},shell=True,capture_output=True,text=True,timeout={timeout})\n"
                f"sys.stdout.write(r.stdout)\n"
                f"sys.stdout.write(json.dumps({{'rc':r.returncode}}))\n")
        tmp = f.name
    resolved_session = _colab_resolve_session(profile, session)
    args = _colab_base_args(profile) + ["exec", "-s", resolved_session, "-f", tmp, "--timeout", str(timeout)]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 30,
                           env=_colab_env(profile))
    except FileNotFoundError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return 127, "Colab CLI 未安装或命令路径已失效"
    try:
        os.unlink(tmp)
    except Exception:
        pass
    # 解析输出: stdout + rc json
    output = r.stdout + r.stderr
    rc = r.returncode
    # 尝试从输出末尾提取 rc
    import re
    m = re.search(r'\{"rc":\s*(\d+)\}[\d]*\s*$', output)
    if m:
        rc = int(m.group(1))
        output = output[:m.start()].rstrip()
    return rc, output

def _colab_upload(local_path, remote_path, session=None, profile=None):
    """上传文件到 Colab；自动纠正失效会话名并重试瞬时网络错误。"""
    profile = profile or _active_colab_profile()
    resolved_session = _colab_resolve_session(profile, session)
    r = None
    for attempt in range(1, 4):
        r = subprocess.run(
            _colab_base_args(profile) + ["upload", "-s", resolved_session,
                         str(local_path), remote_path],
            env=_colab_env(profile), capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0:
            if attempt > 1:
                plog(f"[Colab 上传恢复] remote={remote_path}, attempt={attempt}/3")
            return True
        detail_raw = (r.stderr or r.stdout or "")
        if "not found" in detail_raw.lower():
            resolved_session = _colab_resolve_session(profile, session, force=True)
        if attempt < 3:
            plog(f"[Colab 上传重试] remote={remote_path}, attempt={attempt}/3")
            time.sleep(attempt * 2)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "unknown error").strip().replace("\n", " ")
        detail = re.sub(
            r"([?&](?:colab-runtime-proxy-token|token)=)[^&\s]+",
            r"\1<redacted>",
            detail,
            flags=re.I,
        )
        plog(
            f"[Colab 上传失败] profile={profile['id']}, session={resolved_session}, "
            f"local={pathlib.Path(local_path).name}, remote={remote_path}, "
            f"rc={r.returncode}, error={detail[:500]}"
        )
    return False


def _colab_session_active():
    """Check the active profile/session without requiring Whisper to be installed."""
    profile = _active_colab_profile()
    colab_bin = pathlib.Path(_colab_base_args(profile)[0])
    if not colab_bin.exists() and not _sht.which(str(colab_bin)):
        return False, f"Colab CLI 未安装（当前配置：{profile['name']}）"
    try:
        result = subprocess.run(
            _colab_base_args(profile) + ["sessions"],
            env=_colab_env(profile),
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        active = result.returncode == 0 and "No active" not in output
        if active:
            resolved = _colab_resolve_session(profile)
            return True, f"{profile['name']} · 会话 {resolved} 已连接"
        return False, f"{profile['name']} 没有活跃 Colab GPU 会话"
    except Exception as exc:
        return False, f"检查 Colab 会话失败：{exc}"

def _colab_download(remote_path, local_path, session=None, profile=None):
    """从 Colab 下载文件"""
    profile = profile or _active_colab_profile()
    resolved_session = _colab_resolve_session(profile, session)
    r = subprocess.run(_colab_base_args(profile) + ["download", "-s", resolved_session,
                        remote_path, str(local_path)], env=_colab_env(profile),
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0

app = Flask(__name__)
JOBS = {}  # id -> {stage, log[], result, error}

# ===== 唯一日志:所有动作都写这一个文件,页面日志面板也读它 =====
PLOG = ROOT / "pipeline.log"


def plog(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        with open(PLOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


@app.get("/api/logs")
def api_logs():
    try:
        lines = open(PLOG, encoding="utf-8").readlines()[-300:]
        content = "".join(lines)
    except Exception:
        content = "(暂无日志)"
    if request.args.get("raw") == "1":
        response = Response(content, mimetype="text/plain; charset=utf-8")
        response.headers["Cache-Control"] = "no-store"
        return response
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='color-scheme' content='light'>"
        "<title>cicy-koubo 日志</title>"
        "<style>html,body{margin:0;min-height:100%;background:#fff;color:#17212b}"
        "body{box-sizing:border-box;padding:20px}"
        "pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;"
        "font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}</style>"
        "</head><body><pre id='log'>" + html.escape(content) + "</pre>"
        "<script>"
        "const el=document.getElementById('log');let last=el.textContent;"
        "async function refresh(){try{"
        "const r=await fetch('/api/logs?raw=1&t='+Date.now(),{cache:'no-store'});"
        "const text=await r.text();"
        "if(text!==last){const follow=innerHeight+scrollY>=document.body.scrollHeight-80;"
        "el.textContent=text;last=text;if(follow)scrollTo(0,document.body.scrollHeight);}"
        "}catch(_){}}"
        "setInterval(refresh,1000);scrollTo(0,document.body.scrollHeight);"
        "</script></body></html>"
    )
    response = Response(page, mimetype="text/html; charset=utf-8")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/ui-log")
def api_ui_log():
    """Persist transient UI errors without storing form contents or secrets."""
    body = request.get_json(silent=True) or {}
    level = str(body.get("level") or "error").lower()
    if level not in {"error", "warn", "info"}:
        level = "error"
    message = " ".join(str(body.get("message") or "").split())[:1000]
    page = str(body.get("page") or "/").split("?", 1)[0][:200]
    if message:
        plog(f"[UI {level.upper()}] {page} · {message}")
    return jsonify({"ok": True})


# 持久化各行指纹，避免重启后重复输出
_PROV_FP = APP_DIR / "prov_seen.json"

def _prov_seen_load():
    try:
        return set(json.loads(open(_PROV_FP).read())) if _PROV_FP.exists() else set()
    except Exception:
        return set()

def _prov_seen_save(s):
    try:
        json.dump(list(s), open(_PROV_FP, "w"))
    except Exception:
        pass

def _provision_log_pump():
    """2s 轮询本地 provision.log，新行汇入 pipeline.log。"""
    seen = _prov_seen_load()
    while True:
        try:
            for logf in sorted(pathlib.Path("/content").glob("*/provision.log")):
                key = logf.parent.name
                if key not in ENGINES:
                    continue
                try:
                    for line in logf.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        fp = f"{key}:{hash(line)}"
                        if fp not in seen:
                            seen.add(fp)
                            plog(f"[{key}] {line}")
                            _prov_seen_save(seen)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(2)


threading.Thread(target=_provision_log_pump, daemon=True).start()


def run(cmd, timeout=1200):
    return subprocess.run(_rewrite_local(cmd), capture_output=True, text=True, timeout=timeout)


class _FakeCP:
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def run_retry(cmd, timeout=600, tries=4):
    """扛 frpc 隧道抖动:失败/超时(含 Connection closed)重试几次。"""
    last = _FakeCP(1, "", "no attempt")
    for _ in range(tries):
        try:
            last = run(cmd, timeout=timeout)
            if last.returncode == 0:
                return last
        except Exception as e:  # noqa: BLE001 (TimeoutExpired 等)
            last = _FakeCP(1, "", str(e))
        time.sleep(3)
    return last


def cosy_ready():
    if not COSY_MODEL.exists():
        return False
    # LFS 指针文件很小;真权重是几百 MB
    return COSY_MODEL.stat().st_size > 1_000_000


def _ssh_check(remote_cmd, needle, tries=3):
    """带重试的 SSH 布尔探测:任一次拿到 needle 即 True;隧道抖动不误判。"""
    for _ in range(tries):
        try:
            r = run(SSH + [remote_cmd], timeout=15)
            if needle in (r.stdout or ""):
                return True
            if r.returncode == 0:
                return False  # 连上了但确实没有 → 明确 False
        except Exception:
            pass
        time.sleep(2)
    return False


def tunnel_status():
    if GPU_MODE == "cicy_gpu":
        # CiCy GPU is provisioned per job, so an absent tunnel is the normal
        # idle state rather than a service failure.
        return "ready", "CiCy GPU · 按需启动"
    elif GPU_MODE == "colab_cli":
        # colab_cli 模式:检查是否有活跃 session
        colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
        try:
            r = subprocess.run([colab_bin, "sessions"], capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and "No active" not in (r.stdout or ""):
                # 有 session，检查环境是否就绪
                check_rc, check_out = _colab_exec("cat /content/mt/READY 2>/dev/null && echo OK || echo NO", timeout=15)
                if "OK" in check_out:
                    return "ready", "Colab CLI · 环境就绪"
                return "provisioning", "Colab CLI · 环境部署中"
            return "offline", "Colab CLI · 无活跃 session，需 colab new --gpu T4"
        except Exception:
            return "offline", "Colab CLI · 连接失败"
    elif GPU_MODE == "local" and not LOCAL_GPU:
        endpoint, token = _local_gpu_conf()
        try:
            health = _local_gpu_request("GET", "/v1/health", token, timeout=10)
            tstate = "ready" if health.get("ok") else "offline"
            tnote = f"本地 GPU Docker · {health.get('gpu') or '未就绪'}"
            audio = "ready" if health.get("engines", {}).get("cosyvoice") else "down"
            video = "ready" if health.get("engines", {}).get("musetalk") else "down"
        except Exception:
            tstate, tnote, audio, video = "offline", "本地 GPU Docker 未启动", "down", "down"
    elif GPU_MODE == "local":
        gpu_mb = _get_gpu_memory_mb()
        if gpu_mb < 8192:
            return "offline", "本机未检测到可用的 NVIDIA GPU（至少需要 8GB 显存）"
        return "ready", f"本机 NVIDIA GPU · {gpu_mb}MB 显存"
    # SSH 模式(原有逻辑)
    for _ in range(3):
        try:
            r = run(SSH + ["cat /content/mt/READY 2>/dev/null && echo OK || echo NOREADY"], timeout=15)
            if r.returncode == 0:
                return ("ready", "环境就绪,可出片") if "OK" in r.stdout else ("provisioning", "环境部署中")
        except Exception:
            pass
        time.sleep(2)
    return "offline", "Colab 隧道断开,重跑引导 cell 恢复"


@app.get("/api/status")
def status():
    # This is the single global health snapshot used by the header. In Colab
    # mode every probe must use the active Google profile; bare `colab
    # sessions` reads the default account and can incorrectly report offline.
    now = time.monotonic()
    cached = getattr(status, "_cache", None)
    force = request.args.get("refresh") == "1"
    if not force and cached and now - cached[0] < 20:
        return jsonify(cached[1])
    if GPU_MODE == "cicy_gpu":
        active_session = _CICY_GPU_SESSIONS.get(_CICY_GPU_ACTIVE_JOB_ID) or {}
        if active_session.get("endpoint"):
            tstate, tnote = "ready", "CiCy GPU · 实例在线"
            audio = video = "ready"
        elif _CICY_GPU_ACTIVE_JOB_ID:
            tstate, tnote = "provisioning", "CiCy GPU · 实例启动中"
            audio = video = "installing"
        else:
            tstate, tnote = "idle", "CiCy GPU · 未启动，处理任务时按需创建"
            audio = video = "idle"
    elif GPU_MODE == "colab_cli":
        active, tnote = _colab_session_active()
        tstate = "ready" if active else "offline"
        mt_ready = cosy_is_ready = False
        if active:
            _, ready_out = _colab_exec(
                "test -f /content/mt/READY && echo __MT_READY__; "
                "test -f /content/cosy/COSY_READY && echo __COSY_READY__",
                timeout=30,
            )
            mt_ready = "__MT_READY__" in ready_out
            cosy_is_ready = "__COSY_READY__" in ready_out
        audio = "ready" if cosy_is_ready else ("installing" if active else "down")
        video = "ready" if mt_ready else ("installing" if active else "down")
    elif GPU_MODE == "local":
        gpu_mb = _get_gpu_memory_mb()
        tstate = "ready" if gpu_mb >= 8192 else "offline"
        tnote = (
            f"本机 NVIDIA GPU · {gpu_mb}MB 显存"
            if tstate == "ready"
            else "本机未检测到可用的 NVIDIA GPU（至少需要 8GB 显存）"
        )
        audio = "ready" if tstate == "ready" and cosy_ready() else "down"
        video = "ready" if tstate == "ready" else "down"
    else:
        tstate, tnote = tunnel_status()
        audio = "ready" if (tstate == "ready" and _cosy_ready_remote()) else \
                ("installing" if tstate == "ready" else "down")
        video = "ready" if tstate == "ready" else "down"
    overall = "ready" if tstate == "ready" and audio == "ready" and video == "ready" else \
              ("idle" if tstate == "idle" else ("partial" if tstate in {"ready", "provisioning"} else "down"))
    payload = {
        "tunnel": tstate, "tunnel_note": tnote,
        "gpu_mode": GPU_MODE,
        "audio_service": audio,
        "video_service": video,
        "overall": overall,
    }
    status._cache = (now, payload)
    return jsonify(payload)


def _cosy_ready_remote():
    return _ssh_check("cat /content/cosy/COSY_READY 2>/dev/null && echo OK", "OK")


def do_generate(job_id, video_path, audio_path, bbox, opts=None):
    opts = opts or {}
    j = JOBS[job_id]
    def log(m):
        j["log"].append(m)
        plog(f"[出片 {job_id}] {m}")
    try:
        colab_profile = None
        colab_session = None
        if GPU_MODE == "colab_cli":
            colab_profile = _active_colab_profile()
            colab_session = _colab_resolve_session(colab_profile, force=True)
            log(f"绑定 Colab 会话: {colab_session}")
        jd = WORK / job_id
        jd.mkdir(exist_ok=True)
        j["stage"] = "归一底板 25fps"
        log("normalize base video → 25fps CFR")
        norm = jd / "base_25fps.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y"]
        if opts.get("randstart"):
            import random
            dur = _ffdur(video_path)
            if dur > 2:
                ss = round(random.uniform(0, dur / 2), 2)
                cmd += ["-ss", str(ss)]
                log(f"动作随机: 底板从 {ss}s 开始")
        cmd += ["-i", str(video_path), "-r", "25",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an", str(norm)]
        r = run(cmd)
        if r.returncode != 0:
            raise RuntimeError("ffmpeg normalize failed: " + r.stderr[:300])

        result = jd / "result.mp4"
        if opts.get("mode") == "simple":
            # 仅底板合成:本机 ffmpeg,不对口型、不需要 GPU
            if audio_path is None:
                # 无配音:底板(静音)直接作为成片,供测字幕/BGM
                j["stage"] = "本机合成(纯底板,无配音)"
                log("simple compose: base only, no audio")
                import shutil as _sh
                _sh.copy(norm, result)
            else:
                j["stage"] = "本机合成(底板+配音)"
                adur = _ffdur(audio_path)
                log(f"simple compose: loop base to {adur}s, mux audio")
                r = run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", str(norm),
                         "-i", str(audio_path), "-map", "0:v", "-map", "1:a",
                         "-t", str(adur), "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                         "-c:a", "aac", str(result)])
                if r.returncode != 0 or not result.exists():
                    raise RuntimeError("simple compose failed: " + r.stderr[:300])
        else:
            j["stage"] = "准备素材"
            if GPU_MODE == "colab_cli":
                log("upload base + audio → Colab")
                upload_started = time.monotonic()
                if not _colab_upload(
                    norm, f"/content/{norm.name}",
                    session=colab_session, profile=colab_profile,
                ):
                    raise RuntimeError("colab upload base failed")
                log(f"底板上传完成 {time.monotonic()-upload_started:.1f}s")
                audio_upload_started = time.monotonic()
                if not _colab_upload(
                    audio_path, f"/content/{pathlib.Path(audio_path).name}",
                    session=colab_session, profile=colab_profile,
                ):
                    raise RuntimeError("colab upload audio failed")
                log(f"音频上传完成 {time.monotonic()-audio_upload_started:.1f}s")
            else:
                log("copy base + audio → 本地 GPU")
                r = run(SCP + [str(norm), str(audio_path), REMOTE + ":/content/"], timeout=180)
                if r.returncode != 0:
                    raise RuntimeError("copy failed: " + r.stderr[:300])

            engine = opts.get("engine") or "musetalk"
            # 兼容前端 engine key
            engine = {"mt": "musetalk", "hg": "heygem"}.get(engine, engine)
            j["stage"] = ("HeyGem" if engine == "heygem" else "MuseTalk") + " 对口型(数分钟)"
            log(f"run {engine} on GPU")
            rv, ra = norm.name, pathlib.Path(audio_path).name
            out_remote = f"/content/out_{job_id}.mp4"
            if engine == "heygem":
                cmd = f"bash /content/hg/synthesize.sh /content/{shlex.quote(rv)} /content/{shlex.quote(ra)} {out_remote}"
            else:
                cmd = f"bash /content/mt/synthesize.sh /content/{shlex.quote(rv)} /content/{shlex.quote(ra)} {out_remote} {int(bbox)}"

            if GPU_MODE == "colab_cli":
                remote_log = f"/content/lipsync_{job_id}.log"
                remote_done = f"/content/lipsync_{job_id}.done"
                remote_rc = f"/content/lipsync_{job_id}.rc"
                inner = (
                    f"{cmd}; rc=$?; echo $rc > {remote_rc}; "
                    f"touch {remote_done}; exit $rc"
                )
                launch = (
                    f"rm -f {remote_done} {remote_rc}; "
                    f"nohup sh -c {shlex.quote(inner)} > {remote_log} 2>&1 </dev/null &"
                )
                log("colab background: " + cmd[:80])
                launch_rc, launch_out = _colab_exec(
                    launch, session=colab_session, timeout=30, profile=colab_profile,
                )
                if launch_rc != 0:
                    raise RuntimeError("MuseTalk launch failed via colab CLI: " + launch_out[-200:])
                remote_log_seen = ""
                rc = 124
                inference_started = time.monotonic()
                for _ in range(450):  # 最长 15 分钟
                    time.sleep(2)
                    poll_rc, poll_out = _colab_exec(
                        f"tail -c 30000 {remote_log} 2>/dev/null; "
                        f"echo __CICY_STATE__; "
                        f"if [ -f {remote_done} ]; then cat {remote_rc}; else echo RUNNING; fi",
                        session=colab_session, timeout=20, profile=colab_profile,
                    )
                    if poll_rc != 0 and any(
                        marker in poll_out.lower()
                        for marker in ("session", "lost", "not found", "404/401")
                    ):
                        raise RuntimeError("Colab 会话在对口型过程中断开或被 Google 回收，请重新启动会话后重试")
                    output, _, state = poll_out.rpartition("__CICY_STATE__")
                    output = output.lstrip("\r\n").replace("\r", "\n")
                    if output != remote_log_seen:
                        new_output = (
                            output[len(remote_log_seen):]
                            if output.startswith(remote_log_seen) else output
                        )
                        for line in new_output.splitlines():
                            if line.strip():
                                log(f"[{engine}] {line[:500]}")
                        remote_log_seen = output
                    state = state.strip()
                    if state != "RUNNING":
                        try:
                            rc = int(state.splitlines()[-1])
                        except Exception:
                            rc = 1
                        log(f"{engine} GPU 推理结束: rc={rc}, 耗时 {time.monotonic()-inference_started:.1f}s")
                        break
                if rc != 0:
                    raise RuntimeError(f"MuseTalk failed via colab CLI (rc={rc})")
            else:
                # 实时日志:stream 行写入 j["log"] 供前端轮询
                local_cmd = _rewrite_local(SSH + [cmd])
                log_file = jd / "synthesize.log"
                import subprocess as _sp
                with open(str(log_file), "w") as lf:
                    p = _sp.Popen(local_cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT,
                                  text=True, start_new_session=True)
                    for line in p.stdout:
                        line = line.rstrip('\n').rstrip('\r')
                        if line:
                            log(f"[MuseTalk] {line[:120]}")
                        lf.write(line + "\n")
                    p.wait()
                    rc = p.returncode
                if rc != 0:
                    raise RuntimeError("MuseTalk failed, check log")

            # 验证输出
            if GPU_MODE == "colab_cli":
                check_rc, check_out = _colab_exec(
                    f"test -f {out_remote} && echo OK || echo MISSING",
                    session=colab_session, timeout=30, profile=colab_profile,
                )
                if "OK" not in check_out:
                    raise RuntimeError("MuseTalk did not produce output")
            else:
                r2 = _sp.run(["bash", "-lc", f"test -f {out_remote} && echo OK || echo MISSING"],
                             capture_output=True, text=True)
                if "OK" not in (r2.stdout or ""):
                    raise RuntimeError("MuseTalk did not produce output")
            log("done")

            j["stage"] = "取回成片"
            if GPU_MODE == "colab_cli":
                download_started = time.monotonic()
                log("下载 GPU 成片")
                if not _colab_download(
                    out_remote, result, session=colab_session, profile=colab_profile,
                ):
                    raise RuntimeError("colab download failed")
                log(f"成片下载完成 {time.monotonic()-download_started:.1f}s")
            else:
                r = run(SCP + [REMOTE + ":" + out_remote, str(result)], timeout=180)
                if r.returncode != 0 or not result.exists():
                    raise RuntimeError("copy download failed: " + r.stderr[:300])

        if opts.get("sharp"):
            j["stage"] = "牙齿高清(锐化增强)"
            log("unsharp enhance")
            sharp = jd / "result_sharp.mp4"
            r = run(["ffmpeg", "-v", "error", "-y", "-i", str(result),
                     "-vf", "unsharp=5:5:0.8:5:5:0.4", "-c:a", "copy", str(sharp)])
            if r.returncode == 0 and sharp.exists():
                sharp.replace(result)

        run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(result),
             "-frames:v", "1", str(jd / "cover.jpg")])
        # 成片入媒体库(文件+meta)
        import shutil
        dur = _ffdur(result)
        dst = MEDIA_DIR / f"video_{job_id}.mp4"
        shutil.copy(result, dst)
        note = "MuseTalk 成片" if opts.get("mode") != "simple" else \
               ("底板成片(无配音)" if audio_path is None else "快速成片(不对口型)")
        _media_add({"id": job_id, "type": "video", "file": dst.name,
                    "duration": dur, "note": note,
                    "ts": time.strftime("%m-%d %H:%M")})
        j["stage"] = "完成"
        plog(f"[出片 {job_id}] ✅ 完成 {dur}s → media/video_{job_id}.mp4")
        j["result"] = job_id
        log("done")
    except Exception as e:  # noqa: BLE001
        j["stage"] = "失败"
        plog(f"[出片 {job_id}] ❌ 失败: {e}")
        j["error"] = str(e)
        log("ERROR " + str(e))


@app.post("/api/generate-video")
def generate_video():
    if GPU_MODE == "cicy_gpu":
        return _cicy_gpu_video()
    if GPU_MODE == "local" and not LOCAL_GPU:
        return _local_gpu_video()
    has_audio = "audio" in request.files and request.files["audio"].filename
    audio_media = None
    if not has_audio:
        aid = (request.form.get("audio_id") or "").strip()
        if aid:
            e = next((x for x in _media_list() if x.get("id") == aid), None)
            if e and (MEDIA_DIR / e.get("file", "")).exists():
                audio_media = MEDIA_DIR / e["file"]
                has_audio = True
    mode = request.form.get("mode") or "lipsync"
    if not has_audio:
        mode = "simple"  # 无配音只能仅底板合成(对口型必须有音频)
    use_default = "video" not in request.files or not request.files["video"].filename
    if use_default:
        # 底板选择:base_id(底板库)> state.json 默认
        bid = (request.form.get("base_id") or "").strip()
        if bid and bid.startswith("base-video-") and (ROOT / "assets" / bid).exists():
            default_video = ROOT / "assets" / bid
        else:
            try:
                st = json.load(open(ROOT / "state.json"))
            except Exception:
                st = {}
            bv = st.get("assets", {}).get("base_video")
            default_video = (ROOT / bv) if bv else None
        if not default_video or not default_video.exists():
            return jsonify({"error": "没有底板视频:请选择文件,或先通过飞书发一条底板视频"}), 400
    if mode != "simple":  # 仅底板合成是纯本机操作,不需要 GPU
        if GPU_MODE == "colab_cli":
            ready_rc, ready_out = _colab_exec(
                "test -f /content/mt/READY && nvidia-smi --query-gpu=name --format=csv,noheader",
                timeout=20,
            )
            if ready_rc != 0 or not ready_out.strip():
                return jsonify({"error": "Colab GPU 或 MuseTalk 未就绪，请先启动会话并安装 MuseTalk"}), 503
        else:
            tstate, _ = tunnel_status()
            if tstate != "ready":
                return jsonify({"error": "Colab GPU 未就绪,请先在 Colab 跑引导 cell(或切「仅底板合成」模式)"}), 503
    job_id = uuid.uuid4().hex[:10]
    jd = WORK / job_id
    jd.mkdir(exist_ok=True)
    if use_default:
        vp = jd / ("base" + default_video.suffix)
        import shutil
        shutil.copy(default_video, vp)
    else:
        vp = jd / ("base" + pathlib.Path(request.files["video"].filename).suffix)
        request.files["video"].save(vp)
    if has_audio:
        if audio_media is not None:
            import shutil
            ap = jd / ("audio" + audio_media.suffix)
            shutil.copy(audio_media, ap)
        else:
            ap = jd / ("audio" + pathlib.Path(request.files["audio"].filename).suffix)
            request.files["audio"].save(ap)
    else:
        ap = None
    bbox = request.form.get("bbox", "0")
    engine = request.form.get("engine") or "musetalk"
    # 兼容前端 engine key
    engine = {"mt": "musetalk", "hg": "heygem"}.get(engine, engine)
    if mode != "simple" and engine == "heygem" and \
            not _ssh_check("test -f /content/hg/HG_READY && echo OK", "OK"):
        return jsonify({"error": "HeyGem 引擎未安装:在 Colab notebook 里运行「可选:安装 HeyGem」cell(需 ≥16GB 显存)"}), 503
    opts = {"sharp": request.form.get("sharp") == "1",       # 牙齿高清=成片锐化
            "randstart": request.form.get("randstart") == "1",  # 动作随机=底板随机起点
            "mode": mode, "engine": engine}
    JOBS[job_id] = {"stage": "排队", "log": [], "result": None, "error": None}
    threading.Thread(target=do_generate, args=(job_id, vp, ap, bbox, opts), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>", methods=["GET", "DELETE"])
def job_status(job_id):
    if request.method == "DELETE":
        if GPU_MODE != "cicy_gpu":
            return jsonify({"error": "remote cancellation is only available in CiCy GPU mode"}), 400
        return _cicy_gpu_proxy(
            f"/v1/jobs/{urllib.parse.quote(job_id, safe='')}",
        )
    recovered = WORK / job_id / "result.mp4"
    if recovered.is_file():
        return jsonify({
            "stage": "完成",
            "log": ["CiCy Cloud 已取回成片"],
            "result": job_id,
            "error": None,
        })
    if GPU_MODE == "cicy_gpu":
        return _cicy_gpu_video_status(job_id)
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"stage": j["stage"], "log": j["log"][-6:],
                    "result": j["result"], "error": j["error"]})


@app.get("/api/result/<job_id>")
def result(job_id):
    recovered = WORK / job_id / "result.mp4"
    if recovered.is_file():
        return send_file(recovered, mimetype="video/mp4")
    result_asset = _CICY_GPU_RESULT_ASSETS.get(job_id)
    if result_asset:
        upstream = urllib.request.Request(result_asset["download_url"], method="GET")
        response = _public_direct_opener().open(upstream, timeout=300)

        def stream_result():
            completed = False
            try:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        completed = True
                        break
                    yield chunk
            finally:
                response.close()
                if completed:
                    try:
                        _public_direct_opener().open(
                            urllib.request.Request(result_asset["delete_url"], method="DELETE"),
                            timeout=30,
                        ).close()
                        _CICY_GPU_RESULT_ASSETS.pop(job_id, None)
                        plog(f"[CiCy GPU] 成片已拉取本地，OSS 临时对象已删除: {job_id}")
                    except Exception as exc:
                        plog(f"[CiCy GPU] OSS 临时对象删除失败，等待生命周期清理: {job_id} · {exc}")

        return Response(stream_result(), content_type="video/mp4")
    if GPU_MODE == "cicy_gpu":
        return _cicy_gpu_proxy(
            f"/api/result/{urllib.parse.quote(job_id, safe='')}",
            complete_on_success=False,
        )
    f = WORK / job_id / "result.mp4"
    if not f.exists():
        return jsonify({"error": "no result"}), 404
    return send_file(f, mimetype="video/mp4")


@app.get("/api/cover/<job_id>")
def cover(job_id):
    if GPU_MODE == "cicy_gpu":
        return _cicy_gpu_proxy(f"/api/cover/{urllib.parse.quote(job_id, safe='')}")
    f = WORK / job_id / "cover.jpg"
    return send_file(f, mimetype="image/jpeg") if f.exists() else ("", 404)


class GroqTranscriptionError(RuntimeError):
    """Groq STT request failed with an actionable user-facing reason."""


def _transcribe(path, provider="auto"):
    """转写参考音频 → 文字。优先 Groq，回退 Colab / 本机 Whisper。"""
    if provider in {"auto", "groq"}:
        groq_text = _groq_transcribe(path, raise_errors=(provider == "groq"))
        if groq_text:
            plog(f"[参考音频转写] Groq 成功: {len(groq_text)}字")
            return groq_text
        if provider == "groq":
            return ""
    if provider in {"auto", "colab"} and GPU_MODE == "colab_cli":
        # 通过 colab CLI 远程转写
        remote_audio = f"/content/_ref_{pathlib.Path(path).name}"
        if not _colab_upload(path, remote_audio):
            return ""
        script = (
            "import whisper,sys,json; "
            "m=whisper.load_model('medium',device='cuda'); "
            f"r=m.transcribe('{remote_audio}',fp16=True,language='zh'); "
            "sys.stdout.write(r['text'].strip())"
        )
        rc, out = _colab_exec(f"cd /content/cosy && env/bin/python -c {shlex.quote(script)}", timeout=300)
        return out.strip() if rc == 0 else ""
    if provider == "colab":
        return ""
    # 本地 Colab 环境
    cosy_py = pathlib.Path("/content/cosy/env/bin/python")
    if not cosy_py.exists():
        return ""
    try:
        r = subprocess.run([str(cosy_py), "-c", (
            "import whisper,sys; m=whisper.load_model('medium',device='cuda'); "
            "r=m.transcribe(sys.argv[1],fp16=True,language='zh'); "
            "print(r['text'].strip())"
        ), str(path)], capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _groq_transcribe(path, raise_errors=False):
    """Fast cloud STT for short media. Returns empty text on any failure."""
    import requests
    gk = _groq_key()
    if not gk:
        return ""
    try:
        session = requests.Session()
        with open(path, "rb") as audio_file:
            response = session.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {gk}", "User-Agent": "cicy-koubo/0.1"},
                files={"file": (pathlib.Path(path).name, audio_file, "audio/wav")},
                data={"model": "whisper-large-v3-turbo", "language": "zh"},
                timeout=(8, 60),
            )
        if not response.ok:
            detail = ""
            try:
                detail = str((response.json().get("error") or {}).get("message") or "").strip()
            except Exception:
                detail = ""
            messages = {
                401: "Groq API Key 无效或已失效，请打开供应商设置重新配置",
                403: "Groq 拒绝了转写请求，请检查账号权限或当前网络出口",
                429: "Groq 请求过于频繁或额度已用尽，请稍后重试",
            }
            reason = messages.get(
                response.status_code,
                f"Groq 转写服务返回 HTTP {response.status_code}",
            )
            if detail and detail.lower() not in {"forbidden", "unauthorized"}:
                reason += f"：{detail[:180]}"
            raise GroqTranscriptionError(reason)
        return response.json().get("text", "").strip()
    except requests.Timeout as exc:
        error = GroqTranscriptionError("Groq 转写超时，请检查网络后重试")
        plog(f"[参考音频转写] Groq 失败: Timeout: {exc}")
        if raise_errors:
            raise error from exc
        return ""
    except requests.RequestException as exc:
        error = GroqTranscriptionError(f"无法连接 Groq 转写服务：{exc}")
        plog(f"[参考音频转写] Groq 失败: {type(exc).__name__}: {exc}")
        if raise_errors:
            raise error from exc
        return ""
    except GroqTranscriptionError as exc:
        plog(f"[参考音频转写] Groq 失败: {exc}")
        if raise_errors:
            raise
        return ""
    except Exception as exc:
        plog(f"[参考音频转写] Groq 失败: {type(exc).__name__}: {exc}")
        if raise_errors:
            raise GroqTranscriptionError(f"Groq 转写失败：{exc}") from exc
        return ""




def _pull_colab_log(jid):
    """把 Colab 上这次配音的详细日志拉回来合并进 pipeline.log。"""
    try:
        r = run_retry(SSH + ["cat /content/cosy/last_tts.log 2>/dev/null"], timeout=20, tries=2)
        for ln in (r.stdout or "").splitlines():
            plog(f"  [colab] {ln}")
    except Exception:
        pass

@app.post("/api/tts")
def tts():
    if not _tts_lock.acquire(blocking=False):
        plog("[配音] 忽略重复请求：已有配音任务正在运行")
        return jsonify({
            "error": "已有配音任务正在进行，请等待完成，不要重复点击",
            "code": "tts_already_running",
        }), 409
    try:
        return _tts_impl()
    finally:
        _tts_lock.release()


def _tts_impl():
    """文案 + 参考音色 → CosyVoice zero-shot 克隆配音(本机 GPU 直接跑)。"""
    if GPU_MODE == "cicy_gpu":
        return _cicy_gpu_tts()
    if GPU_MODE == "local" and not LOCAL_GPU:
        return _local_gpu_tts()
    text = ""
    if request.is_json:
        text = (request.json or {}).get("text", "").strip()
    else:
        text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "没有要配音的文案"}), 400
    try:
        speed = max(0.5, min(1.5, float(request.form.get("speed", "1.15"))))
    except ValueError:
        speed = 1.15

    jid = uuid.uuid4().hex[:10]
    started = time.monotonic()
    whole = request.form.get("mode") == "whole"
    plog(f"[配音 {jid}] 开始: {len(text)}字 语速{speed} 模式{'整段' if whole else '分段'}")

    # Groq/Whisper only transcribes the reference. CosyVoice still needs the
    # configured GPU runtime, so fail early with an actionable message.
    if GPU_MODE == "colab_cli":
        plog(f"[配音 {jid}] 1/6 检查 Colab 会话")
        session_active, session_hint = _colab_session_active()
        if not session_active:
            plog(f"[配音 {jid}] COLAB_NOT_READY · {session_hint}")
            return jsonify({
                "error": (
                    f"{session_hint}。Groq 仅负责参考音频转写；"
                    "CosyVoice 配音仍需 Colab GPU，请先在系统设置中启动 Colab 会话后重试"
                ),
                "code": "colab_session_inactive",
            }), 503

    # 参考音色
    ref_id = (request.form.get("ref_id") or "").strip()
    if ref_id and ref_id.startswith("voice-sample-") and (ROOT / "assets" / ref_id).exists():
        ref = ROOT / "assets" / ref_id
    elif "ref" in request.files and request.files["ref"].filename:
        ref = WORK / (uuid.uuid4().hex[:8] + "_ref" + pathlib.Path(request.files["ref"].filename).suffix)
        request.files["ref"].save(ref)
    else:
        samples = sorted((ROOT / "assets").glob("voice-sample-*.wav"))
        if not samples:
            return jsonify({"error": "没有参考音色,请先选择/上传一段人声样本"}), 400
        ref = samples[-1]

    # 裁到 ~10s 16kHz mono
    plog(f"[配音 {jid}] 2/6 准备参考音色")
    ref_trim = WORK / f"ref_trim_{jid}.wav"
    tr = run(["ffmpeg", "-v", "error", "-y", "-i", str(ref), "-t", "6",
              "-ar", "16000", "-ac", "1", str(ref_trim)])
    if tr.returncode == 0 and ref_trim.exists():
        ref = ref_trim
    stt_provider = (request.form.get("stt_provider") or "auto").strip()
    if stt_provider == "groq" and not _groq_key():
        return jsonify({
            "error": "Groq Whisper 未配置 API Key，请在 cicy-ai/global.json 的 groqStt 中配置，或切换到 Colab Whisper"
        }), 400
    if stt_provider == "colab":
        cli_installed, session_active, whisper_installed = _colab_whisper_status()
        if not cli_installed:
            return jsonify({
                "error": "Colab Whisper 不可用，请先安装 Colab CLI，或切换到已配置的 Whisper API"
            }), 400
        if not session_active:
            return jsonify({
                "error": "Colab CLI 已安装，但没有活跃 GPU 会话，请先启动会话"
            }), 400
        if not whisper_installed:
            return jsonify({
                "error": "当前 Colab 会话尚未安装 Whisper，请先完成环境安装"
            }), 400
    try:
        transcribe_started = time.monotonic()
        plog(f"[配音 {jid}] 3/6 转写参考音频 provider={stt_provider}")
        ref_text = _transcribe(str(ref), stt_provider)
    except GroqTranscriptionError as exc:
        return jsonify({"error": str(exc), "code": "groq_stt_failed"}), 502
    if not ref_text:
        return jsonify({"error": f"参考音频转写失败（{stt_provider}），请检查服务状态或更换清晰中文人声音频"}), 400
    plog(
        f"[配音 {jid}] 3/6 转写完成 {time.monotonic()-transcribe_started:.1f}s "
        f"→ {len(ref_text)}字"
    )

    # 本机直接跑 CosyVoice
    cosy_dir = pathlib.Path("/content/cosy")
    cosy_py = cosy_dir / "env/bin/python"
    cosy_script = cosy_dir / "cosyvoice_tts.py"

    import base64
    t_b64 = base64.b64encode(text.encode()).decode()
    rt_b64 = base64.b64encode(ref_text.encode()).decode()

    if GPU_MODE == "colab_cli":
        # 上传 ref 音频到 Colab
        remote_ref = f"/content/_ref_{jid}.wav"
        upload_started = time.monotonic()
        plog(f"[配音 {jid}] 4/6 上传参考音频到 Colab")
        if not _colab_upload(ref, remote_ref):
            return jsonify({"error": "上传参考音频到 Colab 失败"}), 500
        plog(f"[配音 {jid}] 4/6 上传完成 {time.monotonic()-upload_started:.1f}s")
        # 远程执行 CosyVoice TTS
        tts_inner = (
            f"env/bin/python cosyvoice_tts.py "
            f"--ref {remote_ref} --ref-text-b64 {rt_b64} "
            f"--text-b64 {t_b64} --speed {speed}{' --whole' if whole else ''} "
            f"--out /content/tts_{jid}.wav; "
            f"rc=$?; if [ $rc -eq 0 ]; then echo done > /content/tts_{jid}.done; fi; exit $rc"
        )
        cmd = (
            "export MPLBACKEND=Agg LD_LIBRARY_PATH=/usr/lib64-nvidia; "
            f"cd /content/cosy && nohup sh -c {shlex.quote(tts_inner)} "
            f"> /content/tts_{jid}.log 2>&1 </dev/null &"
        )
        plog(f"[配音 {jid}] 5/6 启动 CosyVoice GPU 推理")
        launch_rc, launch_out = _colab_exec(cmd, timeout=10)
        if launch_rc != 0:
            plog(f"[配音 {jid}] 5/6 启动失败: {launch_out[-300:]}")
            return jsonify({"error": "CosyVoice 推理进程启动失败"}), 500
        # 轮询完成
        inference_started = time.monotonic()
        remote_log_seen = ""
        for poll_index in range(120):
            time.sleep(3)
            check_rc, check_out = _colab_exec(
                f"(test -f /content/tts_{jid}.done && echo DONE || true); "
                f"echo __KOUBO_TTS_LOG__; tail -c 20000 /content/tts_{jid}.log 2>/dev/null || true",
                timeout=10,
            )
            state_text, _, remote_log = check_out.partition("__KOUBO_TTS_LOG__")
            remote_log = remote_log.lstrip("\r\n").replace("\r", "\n")
            if remote_log != remote_log_seen:
                if remote_log.startswith(remote_log_seen):
                    new_log = remote_log[len(remote_log_seen):]
                else:
                    new_log = remote_log
                for line in new_log.splitlines():
                    if line.strip():
                        plog(f"[配音 {jid}][CosyVoice] {line}")
                remote_log_seen = remote_log
            if "DONE" in state_text:
                plog(f"[配音 {jid}] 5/6 GPU 推理完成 {time.monotonic()-inference_started:.1f}s")
                break
            if poll_index and poll_index % 10 == 0:
                plog(f"[配音 {jid}] 5/6 GPU 推理中 {time.monotonic()-inference_started:.0f}s")
        # 下载结果
        result_wav = WORK / f"tts_{jid}.wav"
        download_started = time.monotonic()
        plog(f"[配音 {jid}] 6/6 下载并保存成品")
        if not _colab_download(f"/content/tts_{jid}.wav", result_wav):
            log_out = ""
            _, log_out = _colab_exec(f"tail -5 /content/tts_{jid}.log 2>/dev/null", timeout=10)
            return jsonify({"error": "CosyVoice TTS 失败: " + log_out[:200]}), 500
        _normalize_voice_output(result_wav)
        # 入媒体库，并与本机分支保持相同 JSON 协议。
        import shutil as _sh
        dur = _ffdur(result_wav)
        dst = MEDIA_DIR / f"voice_{jid}.wav"
        _sh.copy(result_wav, dst)
        _media_add({"id": jid, "type": "voice", "file": dst.name,
                    "text": text[:100], "chars": len(text), "speed": speed,
                    "duration": dur, "ts": time.strftime("%m-%d %H:%M")})
        plog(
            f"[配音 {jid}] 6/6 保存完成 {time.monotonic()-download_started:.1f}s · "
            f"总耗时 {time.monotonic()-started:.1f}s"
        )
        plog(f"[配音 {jid}] ✅ 完成 {dur}s → media/voice_{jid}.wav")
        return jsonify({"id": jid, "url": f"/api/media/{jid}/file",
                        "duration": dur, "chars": len(text)})

    # 本机 Colab 环境
    if not cosy_script.exists():
        return jsonify({"error": "CosyVoice 未安装,请先在安装管理中安装"}), 503

    out_wav = cosy_dir / f"tts_{jid}.wav"
    done_file = cosy_dir / f"tts_{jid}.done"
    log_file = cosy_dir / f"tts_{jid}.log"

    # 拷贝 ref 到 cosy 目录
    ref_dst = cosy_dir / f"ref_{jid}.wav"
    import shutil as _sh
    _sh.copy(ref, ref_dst)

    local_inner = (
        f"{cosy_py} {cosy_script} --ref {ref_dst} --ref-text-b64 {rt_b64} "
        f"--text-b64 {t_b64} --speed {speed}{' --whole' if whole else ''} --out {out_wav}; "
        f"rc=$?; if [ $rc -eq 0 ]; then echo done > {done_file}; fi; exit $rc"
    )
    launch = (f"export MPLBACKEND=Agg LD_LIBRARY_PATH=/usr/lib64-nvidia; "
              f"cd {cosy_dir} && nohup bash -lc {shlex.quote(local_inner)} "
              f"> {log_file} 2>&1 </dev/null &")
    subprocess.Popen(["bash", "-lc", launch], start_new_session=True)

    # 轮询 done
    for _ in range(120):
        time.sleep(3)
        if done_file.exists():
            break

    if not done_file.exists():
        tail = ""
        try:
            tail = log_file.read_text()[-300:] if log_file.exists() else ""
        except Exception:
            pass
        plog(f"[配音 {jid}] ❌ 超时: {tail}")
        return jsonify({"error": "配音超时: " + tail[-200:]}), 500

    if not out_wav.exists():
        return jsonify({"error": "配音完成后未生成音频文件"}), 500

    _normalize_voice_output(out_wav)
    # 入媒体库
    dur = _ffdur(out_wav)
    dst = MEDIA_DIR / f"voice_{jid}.wav"
    _sh.copy(out_wav, dst)
    _media_add({"id": jid, "type": "voice", "file": dst.name,
                "text": text[:100], "chars": len(text), "speed": speed,
                "duration": dur, "ts": time.strftime("%m-%d %H:%M")})
    plog(f"[配音 {jid}] ✅ 完成 {dur}s → media/voice_{jid}.wav")
    return jsonify({"id": jid, "url": f"/api/media/{jid}/file",
                    "duration": dur, "chars": len(text)})


@app.post("/api/edit")
def edit():
    """烧录字幕:输入 job_id(取该 job 成片)或上传 video,text=字幕文本,返回带字幕 mp4。"""
    if not _edit_lock.acquire(blocking=False):
        plog("[剪辑] 忽略重复请求：已有剪辑任务正在烧录")
        return jsonify({"error": "已有剪辑任务正在进行，请等待完成，不要重复点击"}), 409
    try:
        return _edit_impl()
    finally:
        _edit_lock.release()


def _edit_impl():
    """执行单个剪辑任务；由 edit() 保证进程内不并发烧录。"""
    job_id = request.form.get("job_id")
    src = None
    # 只从「原始未加工成片」剪辑,绝不拿剪辑产物再剪(否则字幕/BGM 会叠加)
    if job_id and (WORK / job_id / "result.mp4").exists():
        src = WORK / job_id / "result.mp4"          # 出片任务的干净原片(烧录不改动它)
    if src is None and "video" in request.files and request.files["video"].filename:
        tmp = WORK / (uuid.uuid4().hex[:8] + ".mp4")
        request.files["video"].save(tmp)
        src = tmp
    if src is None:
        # 兜底:历史里最新的「原始成片」(note 不是"剪辑成片"的那种),对应 jobs/<id>/result.mp4
        for e in _media_list():
            if e.get("type") == "video" and "剪辑" not in (e.get("note") or ""):
                cand = WORK / e["id"] / "result.mp4"
                if cand.exists():
                    src = cand
                    break
    if src is None or not src.exists():
        return jsonify({"error": "没有可剪辑的原始成片:先生成视频"}), 400
    # 每次剪辑都输出到独立文件,原片 result.mp4 永远保持干净
    out = WORK / (uuid.uuid4().hex[:8] + "_edit.mp4")
    text = (request.form.get("text") or "").strip()
    style = {
        "fontsize": int(request.form.get("fontsize") or 14),
        "color": request.form.get("color") or "#FFFFFF",
        "outline": request.form.get("outline") or "#000000",
        "mb": int(request.form.get("mb") or 300),
        "font_id": request.form.get("font") or "heavy",
    }
    bgm_path = None
    bgm_id = (request.form.get("bgm_id") or "").strip()
    if bgm_id and (ROOT / "assets/bgm" / bgm_id).exists() and ".." not in bgm_id:
        bgm_path = ROOT / "assets/bgm" / bgm_id
    elif "bgm" in request.files and request.files["bgm"].filename:
        bgm_path = src.with_name("bgm" + pathlib.Path(request.files["bgm"].filename).suffix)
        request.files["bgm"].save(bgm_path)
    try:
        bgm_vol = max(0.0, min(1.0, float(request.form.get("bgm_vol") or 0.15)))
    except ValueError:
        bgm_vol = 0.15
    plog(f"[剪辑] 开始: 源片={src.parent.name}/{src.name} 字幕{len(text)}字符 BGM={bgm_path.name if bgm_path else '无'} 音量{bgm_vol}")
    if not text and not bgm_path:
        return send_file(src, mimetype="video/mp4")
    # 本机 ffmpeg 无 drawtext/subtitles 滤镜;用 PIL 渲染字幕 PNG 再 overlay(SRT 按时间轴逐句)
    try:
        dims = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=p=0", str(src)])
        w, h = [int(x) for x in dims.stdout.strip().split(",")[:2]]
        dur_probe = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "csv=p=0", str(src)])
        src_dur = float(dur_probe.stdout.strip())
        inputs = ["-i", str(src)]
        fc_parts = []
        last_v = "0:v"
        idx = 1
        segs = _parse_srt(text) if text else []
        if segs:  # SRT 逐句:每句一张 PNG,按时间窗 overlay
            for k, (t0, t1, line) in enumerate(segs[:80]):
                png = src.with_name(f"sub{k}.png")
                _render_caption(line, w, h, png, **style)
                inputs += ["-i", str(png)]
                fc_parts.append(f"[{last_v}][{idx}:v]overlay=0:0:enable='between(t,{t0},{t1})'[v{idx}]")
                last_v = f"v{idx}"
                idx += 1
        elif text:  # 无时间轴的纯文本:整程显示
            png = src.with_name("sub.png")
            _render_caption(text, w, h, png, **style)
            inputs += ["-i", str(png)]
            fc_parts.append(f"[{last_v}][{idx}:v]overlay=0:0[v{idx}]")
            last_v = f"v{idx}"
            idx += 1
        # BGM 混音(循环铺满,音量 bgm_vol);源片无音轨时 BGM 直接作为唯一音轨
        probe_a = run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                       "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(src)])
        has_audio = bool((probe_a.stdout or "").strip())
        amap = ["-map", "0:a", "-c:a", "copy"] if has_audio else []
        if bgm_path:
            inputs += ["-stream_loop", "-1", "-i", str(bgm_path)]
            if has_audio:
                # normalize=0:禁用 amix 自动归一化,否则 volume 设置会被拉平失效
                fc_parts.append(f"[0:a]volume=1.0[a0];[{idx}:a]volume={bgm_vol}[bg];"
                                f"[a0][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")
            else:
                fc_parts.append(f"[{idx}:a]volume={bgm_vol}[aout]")
            amap = ["-map", "[aout]"]
            idx += 1
        vmap = ["-map", f"[{last_v}]"] if last_v != "0:v" else ["-map", "0:v"]
        cmd = (["ffmpeg", "-v", "error", "-y"] + inputs +
               (["-filter_complex", ";".join(fc_parts)] if fc_parts else []) + vmap + amap +
               ["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-t", str(src_dur), str(out)])
        r = run(cmd, timeout=600)
        if r.returncode != 0:
            plog(f"[剪辑] ❌ ffmpeg 失败: {r.stderr[:150]}")
            return jsonify({"error": "ffmpeg: " + r.stderr[:300]}), 500
        # 剪辑结果也入媒体库,历史/下载拿到的才是带字幕/BGM 的版本
        import shutil
        eid = "edit" + uuid.uuid4().hex[:8]
        dst = MEDIA_DIR / f"video_{eid}.mp4"
        shutil.copy(out, dst)
        note = f"剪辑成片({'字幕' if text else ''}{'+' if text and bgm_path else ''}{'BGM' + str(int(bgm_vol * 100)) + '%' if bgm_path else ''})"
        _media_add({"id": eid, "type": "video", "file": dst.name,
                    "duration": _ffdur(out), "note": note, "ts": time.strftime("%m-%d %H:%M")})
        plog(f"[剪辑] ✅ 完成 → media/video_{eid}.mp4 ({note})")
        return send_file(out, mimetype="video/mp4")
    except Exception as e:  # noqa: BLE001
        plog(f"[剪辑] ❌ 失败: {e}")
        return jsonify({"error": "subtitle render: " + str(e)}), 500


def _parse_srt(text):
    """解析 SRT → [(start_s, end_s, line)];非 SRT 格式返回 []。"""
    import re as _re
    pat = _re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
    out = []
    blocks = _re.split(r"\n\s*\n", text.strip())
    for b in blocks:
        lines = [x for x in b.strip().splitlines() if x.strip()]
        for i, ln in enumerate(lines):
            mm = pat.search(ln)
            if mm:
                g = [int(x) for x in mm.groups()]
                t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                content = " ".join(lines[i + 1:]).strip()
                if content:
                    out.append((t0, t1, content))
                break
    return out


FONTS = {
    "soft": ("/System/Library/Fonts/PingFang.ttc", "柔和无衬线（多语种推荐）"),
    "heavy": (str(ROOT / "assets/fonts/SourceHanSansCN-Heavy.otf"), "抖音口播·特粗（推荐）"),
    "bold": (str(ROOT / "assets/fonts/SourceHanSansCN-Bold.otf"), "抖音口播·粗体"),
    "heiti": ("/System/Library/Fonts/STHeiti Medium.ttc", "系统黑体"),
    "pingfang": ("/System/Library/Fonts/PingFang.ttc", "苹方"),
    "songti": ("/System/Library/Fonts/Supplemental/Songti.ttc", "宋体"),
}


def _font_path(fid):
    p = FONTS.get(fid or "heavy", FONTS["heavy"])[0]
    if os.path.exists(p):
        return p
    for c in (FONTS["heiti"][0],                                   # macOS
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",   # Linux/Colab Noto CJK
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/content/SourceHanSansCN-Heavy.otf",                    # 用户上传字体(Colab /content)
              "/content/SourceHanSansCN-Bold.otf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(c):
            return c
    return p


@app.get("/api/fonts")
def fonts():
    # 返回回退链中实际可用的字体
    FALLBACK_FONTS = [
        ("heavy", "思源黑体·特粗(抖音风,推荐)", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ("bold", "思源黑体·粗", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ("heavy", "思源黑体·特粗(用户上传)", "/content/SourceHanSansCN-Heavy.otf"),
        ("bold", "思源黑体·粗(用户上传)", "/content/SourceHanSansCN-Bold.otf"),
    ]
    out = []
    seen = set()
    for k, v in FONTS.items():
        if os.path.exists(v[0]):
            out.append({"id": k, "name": v[1]})
            seen.add(k)
    for fid, name, path in FALLBACK_FONTS:
        if fid not in seen and os.path.exists(path):
            out.append({"id": fid, "name": name})
            seen.add(fid)
    return jsonify(out)


def _hex_rgba(s, default):
    s = (s or "").lstrip("#")
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    except Exception:
        return default


def _load_font(fid, size, text=""):
    """加载字体。遍历回退链直到找到可用的字体（PIL path/bytes 双模式）。"""
    from PIL import ImageFont
    script_candidates = []
    if any("\u0600" <= ch <= "\u06ff" for ch in text):
        script_candidates += ["/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"]
    if any("\u0e00" <= ch <= "\u0e7f" for ch in text):
        script_candidates += ["/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"]
    if any("\u1780" <= ch <= "\u17ff" for ch in text):
        script_candidates += ["/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"]
    if any("\u0e80" <= ch <= "\u0eff" for ch in text):
        script_candidates += ["/usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf"]
    if any("\uac00" <= ch <= "\ud7af" for ch in text):
        script_candidates += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    candidates = script_candidates + [
        FONTS.get(fid or "soft", FONTS["soft"])[0],                  # 用户选择
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        FONTS["heiti"][0],                                           # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",       # Linux/Colab Noto CJK
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/content/SourceHanSansCN-Heavy.otf",                        # 用户上传(Colab /content)
        "/content/SourceHanSansCN-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",      # 最后后备
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            try:
                with open(p, "rb") as ff:
                    return ImageFont.truetype(ff, size)
            except Exception:
                continue
    # 全部失败，用第一个非空路径硬试
    p = _font_path(fid)
    return ImageFont.truetype(p, size)


def _render_caption(text, w, h, png_path, fontsize=0, color="#FFFFFF", outline="#000000", mb=60, font_id="heavy"):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # UI 字号按手机端约 160px 的视觉基准等比换算到成片。
    # 默认 14 在 1080p 竖屏中约为 95px，接近大字抖音口播字幕。
    ui_fs = max(8, fontsize or 14)
    fs = max(8, min(180, round(ui_fs * w / 160)))
    fg = _hex_rgba(color, (255, 255, 255, 255))
    og = _hex_rgba(outline, (0, 0, 0, 255))
    font = _load_font(font_id, fs, text)
    # 大字口播字幕每行约 10 个中文字；同时保留 12% 的左右安全区。
    maxw = w * 0.88
    max_chars = 10
    lines, cur = [], ""
    for ch in text:
        if (d.textlength(cur + ch, font=font) > maxw or len(cur) >= max_chars) and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    lh = round(fs * 1.10)
    y = h - lh * len(lines) - max(10, mb)
    for ln in lines:
        tw = d.textlength(ln, font=font)
        x = (w - tw) / 2
        stroke = max(2, round(fs * 0.06))
        for dx in (-stroke, stroke):
            for dy in (-stroke, stroke):
                d.text((x + dx, y + dy), ln, font=font, fill=og)
        d.text((x, y), ln, font=font, fill=fg)
        y += lh
    img.save(png_path)


def _douyin_dl():
    """douyin-dl 提取脚本:环境变量 > 包内 vendor 副本 > 本机 skill。找不到返回 None。"""
    for c in (os.environ.get("KOUBO_DOUYIN_DL", ""),
              SRC_DIR / "vendor/douyin-dl/douyin-dl",
              pathlib.Path.home() / ".claude/skills/douyin-dl/bin/douyin-dl"):
        if c and pathlib.Path(c).exists():
            return pathlib.Path(c)
    return None
SCRIPTS_CACHE = APP_DIR / "scripts.json"
MEDIA_DIR = APP_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)
MEDIA_REG = APP_DIR / "media.json"


def _media_list():
    if MEDIA_REG.exists():
        try:
            return json.load(open(MEDIA_REG, encoding="utf-8"))
        except Exception:
            return []
    return []


def _media_add(entry):
    data = _media_list()
    data.insert(0, entry)
    json.dump(data[:300], open(MEDIA_REG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _ffdur(path):
    try:
        r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)], timeout=20)
        return round(float(r.stdout.strip()), 2)
    except Exception:
        return 0.0


PROBE_CACHE = APP_DIR / "probe_cache.json"


def _probe(path):
    """时长/分辨率/大小/时间,按 (文件名, mtime) 缓存,避免每次列表都跑 ffprobe。"""
    try:
        st = path.stat()
    except OSError:
        return {}
    key = f"{path.name}:{int(st.st_mtime)}"
    try:
        cache = json.load(open(PROBE_CACHE, encoding="utf-8")) if PROBE_CACHE.exists() else {}
    except Exception:
        cache = {}
    if key in cache:
        return cache[key]
    info = {"size": st.st_size,
            "ts": time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime))}
    try:
        r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-show_entries", "stream=width,height", "-of", "json", str(path)], timeout=20)
        d = json.loads(r.stdout)
        info["duration"] = round(float(d.get("format", {}).get("duration", 0) or 0), 2)
        for s in d.get("streams", []):
            if s.get("width"):
                info["w"], info["h"] = s["width"], s["height"]
    except Exception:
        pass
    cache[key] = info
    json.dump(dict(list(cache.items())[-500:]), open(PROBE_CACHE, "w", encoding="utf-8"))
    return info


THUMBS = APP_DIR / "thumbs"
THUMBS.mkdir(exist_ok=True)


@app.get("/api/thumb/<kind>/<path:aid>")
def thumb(kind, aid):
    """视频缩略图(base=底板库 / media=成品库),按 mtime 缓存。"""
    if ".." in aid:
        return ("", 404)
    if kind == "base":
        src = ROOT / "assets" / aid
    elif kind == "media":
        e = next((x for x in _media_list() if x.get("id") == aid), None)
        src = MEDIA_DIR / e["file"] if e else None
    else:
        return ("", 404)
    if not src or not src.exists():
        return ("", 404)
    out = THUMBS / f"{kind}_{aid.replace('/', '_')}_{int(src.stat().st_mtime)}.jpg"
    if not out.exists():
        run(["ffmpeg", "-v", "error", "-y", "-ss", "0.3", "-i", str(src),
             "-frames:v", "1", "-vf", "scale=360:-2", str(out)], timeout=30)
    return send_file(out, mimetype="image/jpeg") if out.exists() else ("", 404)


@app.get("/api/voices")
def voices():
    """参考音色库:assets 里的人声样本。"""
    out = []
    for f in sorted((ROOT / "assets").glob("voice-sample-*.wav")):
        p = _probe(f)
        out.append({"id": f.name,
                    "name": _meta_name("voice", f.name, f.stem.replace("voice-sample-", "音色样本")),
                    "duration": p.get("duration", 0), "size": p.get("size", 0), "ts": p.get("ts", "")})
    return jsonify(out)


@app.get("/api/voices/<vid>")
def voice_file(vid):
    f = ROOT / "assets" / vid
    if not vid.startswith("voice-sample-") or not f.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(f, mimetype="audio/wav")


@app.get("/api/basevideos")
def basevideos():
    """底板库:assets 里的底板视频(优先 25fps 归一版)。"""
    seen, out = set(), []
    files = sorted((ROOT / "assets").glob("base-video-*.mp4"), reverse=True)
    for f in files:
        key = f.name.replace("-25fps", "")
        if key in seen:
            continue
        seen.add(key)
        pref = f.with_name(key.replace(".mp4", "-25fps.mp4"))
        use = pref if pref.exists() else f
        p = _probe(use)
        out.append({"id": use.name,
                    "name": _meta_name("base", key.replace(".mp4", ""), key.replace(".mp4", "")),
                    "duration": p.get("duration", 0), "size": p.get("size", 0),
                    "w": p.get("w"), "h": p.get("h"), "ts": p.get("ts", "")})
    return jsonify(out)


@app.get("/api/basevideos/<bid>")
def basevideo_file(bid):
    f = ROOT / "assets" / bid
    if not bid.startswith("base-video-") or not f.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(f, mimetype="video/mp4")


@app.post("/api/cover")
def gen_cover():
    """封面 = 底板抽帧 + 蒙版 + 大字标题 → jpg。tpl: dark1 / dark2 / none
    底板来源优先级:上传 video > base_id(底板库) > 当前任务原片 > 默认底板。"""
    title = (request.form.get("title") or "").strip()
    tpl = request.form.get("tpl") or "dark1"
    src = None
    if "video" in request.files and request.files["video"].filename:
        src = WORK / (uuid.uuid4().hex[:8] + "_cov.mp4")
        request.files["video"].save(src)
    else:
        bid = (request.form.get("base_id") or "").strip()
        if bid and bid.startswith("base-video-") and (ROOT / "assets" / bid).exists():
            src = ROOT / "assets" / bid
        else:
            job_id = (request.form.get("job_id") or "").strip()
            if job_id and (WORK / job_id / "base_25fps.mp4").exists():
                src = WORK / job_id / "base_25fps.mp4"     # 该任务归一后的底板
            else:
                try:
                    st = json.load(open(ROOT / "state.json"))
                except Exception:
                    st = {}
                bv = st.get("assets", {}).get("base_video")
                if bv and (ROOT / bv).exists():
                    src = ROOT / bv
    if not src or not src.exists():
        return jsonify({"error": "没有底板可做封面,先选/传底板视频"}), 400
    try:
        frame = WORK / (uuid.uuid4().hex[:8] + "_cover_frame.jpg")
        r = run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(src),
                 "-frames:v", "1", str(frame)])
        if r.returncode != 0:
            return jsonify({"error": "抽帧失败"}), 500
        from PIL import Image, ImageDraw, ImageFont
        img = Image.open(frame).convert("RGBA")
        w, h = img.size
        if tpl in ("dark1", "dark2"):
            alpha = 110 if tpl == "dark1" else 165
            img = Image.alpha_composite(img, Image.new("RGBA", (w, h), (0, 0, 0, alpha)))
        if title:
            d = ImageDraw.Draw(img)
            # Render as a 16px visual size in the 320px-wide UI preview while
            # preserving equivalent proportions in the full-resolution cover.
            fs = max(16, round(w * 16 / 320))
            font = _load_font("heavy", fs)
            maxw = w * 0.88
            lines, cur = [], ""
            for ch in title:
                if d.textlength(cur + ch, font=font) > maxw and cur:
                    lines.append(cur); cur = ch
                else:
                    cur += ch
            if cur:
                lines.append(cur)
            lh = fs + max(4, round(fs * 0.22))
            y = (h - lh * len(lines)) / 2
            for ln in lines:
                x = (w - d.textlength(ln, font=font)) / 2
                stroke = max(1, round(fs * 0.035))
                for dx in (-stroke, stroke):
                    for dy in (-stroke, stroke):
                        d.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0, 255))
                d.text((x, y), ln, font=font, fill=(255, 255, 255, 255))
                y += lh
        out = frame.with_name(frame.stem + "_done.jpg")
        img.convert("RGB").save(out, quality=92)
        frame.unlink(missing_ok=True)
        return send_file(out, mimetype="image/jpeg")
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.get("/api/bgms")
def bgms():
    out = []
    for f in sorted((ROOT / "assets/bgm").glob("*.mp3")):
        p = _probe(f)
        out.append({"id": f.name, "name": _meta_name("bgm", f.name, f.stem.split("-")[0]),
                    "duration": p.get("duration", 0), "size": p.get("size", 0), "ts": p.get("ts", "")})
    return jsonify(out)


@app.get("/api/bgms/<bid>")
def bgm_file(bid):
    f = ROOT / "assets/bgm" / bid
    if ".." in bid or not f.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(f, mimetype="audio/mpeg")


@app.post("/api/srt")
def gen_srt():
    """音频 → Groq whisper(verbose_json)或本机 whisper → SRT 文本(逐句时间轴)。"""
    if "audio" not in request.files:
        return jsonify({"error": "缺音频"}), 400
    tmp = WORK / (uuid.uuid4().hex[:8] + ".wav")
    request.files["audio"].save(tmp)
    gk = _groq_key()

    def _ts(t):
        h, rem = divmod(t, 3600)
        mnt, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(mnt):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"

    def _to_srt(s):
        lines = []
        for i, s in enumerate(s, 1):
            lines.append(f"{i}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text'].strip()}\n")
        return "\n".join(lines)

    # 策略 1: 云端 Groq whisper（快，~10s）
    if gk:
        proxy = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))
        cmd = ["curl", "-s", "https://api.groq.com/openai/v1/audio/transcriptions",
               "-H", f"Authorization: Bearer {gk}", "-F", f"file=@{tmp}",
               "-F", "model=whisper-large-v3-turbo", "-F", "language=zh",
               "-F", "response_format=verbose_json"]
        if proxy:
            cmd = cmd[:1] + ["--proxy", proxy] + cmd[1:]
        try:
            r = run(cmd, timeout=120)
            data = json.loads(r.stdout)
            segs = data.get("segments") or []
            if segs:
                return jsonify({"srt": _to_srt(segs), "segments": len(segs), "provider": "groq"})
        except Exception:
            pass  # Groq 失败，回退到本机

    # 策略 2: Colab whisper medium（GPU，~30-60s）
    if GPU_MODE == "colab_cli":
        remote_audio = f"/content/_srt_{tmp.name}"
        if _colab_upload(tmp, remote_audio):
            script = (
                "import sys, os, warnings, json; "
                "warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; "
                "import whisper; m=whisper.load_model('medium',device='cuda'); "
                f"r=m.transcribe('{remote_audio}',language='zh',verbose=False); "
                "sys.stdout.write(json.dumps(r['segments'],ensure_ascii=False))"
            )
            rc, out = _colab_exec(f"cd /content/cosy && env/bin/python -c {shlex.quote(script)}", timeout=300)
            if rc == 0:
                try:
                    segs = json.loads(out)
                    if segs:
                        return jsonify({"srt": _to_srt(segs), "segments": len(segs), "provider": "colab-whisper"})
                except Exception:
                    pass
    # 本机 CosyVoice 环境 whisper medium
    cosy_dir = pathlib.Path("/content/cosy")
    if cosy_dir.exists():
        try:
            script = (
                "import sys, os, warnings; "
                "warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; "
                "import whisper; m=whisper.load_model('medium',device='cuda'); "
                f"r=m.transcribe('{tmp}',language='zh',verbose=False); "
                "import json; sys.stdout.write(json.dumps(r['segments'],ensure_ascii=False))"
            )
            cosy_python = str(cosy_dir / "env" / "bin" / "python")
            r = subprocess.run([cosy_python, "-c", script], capture_output=True, text=True, timeout=300, cwd=str(cosy_dir))
            segs = json.loads(r.stdout)
            if segs:
                return jsonify({"srt": _to_srt(segs), "segments": len(segs), "provider": "local-whisper"})
        except Exception as e:
            return jsonify({"error": f"本机识别失败: {e}"}), 500
    return jsonify({"error": "缺转写凭据，且无 whisper 环境(Colab/本地)"}), 500


@app.get("/api/media")
def media_list():
    out = []
    for e in _media_list():
        f = MEDIA_DIR / e.get("file", "")
        e = dict(e)
        e["size"] = f.stat().st_size if f.exists() else 0
        out.append(e)
    return jsonify(out)


@app.get("/api/media/<mid>/file")
def media_file(mid):
    e = next((x for x in _media_list() if x.get("id") == mid), None)
    if not e:
        return jsonify({"error": "not found"}), 404
    f = MEDIA_DIR / e["file"]
    if not f.exists():
        return jsonify({"error": "file missing"}), 404
    mt = "audio/wav" if e.get("type") == "voice" else "video/mp4"
    return send_file(f, mimetype=mt)


@app.post("/api/media/delete")
def media_delete():
    body = request.json or {}
    data = _media_list()
    if body.get("all"):
        for e in data:
            (MEDIA_DIR / e.get("file", "")).unlink(missing_ok=True)
        data = []
    else:
        mid = body.get("id")
        keep = []
        for e in data:
            if e.get("id") == mid:
                (MEDIA_DIR / e.get("file", "")).unlink(missing_ok=True)
            else:
                keep.append(e)
        data = keep
    json.dump(data, open(MEDIA_REG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return jsonify({"ok": True, "remaining": len(data)})


def _load_scripts():
    if SCRIPTS_CACHE.exists():
        try:
            return json.load(open(SCRIPTS_CACHE, encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_script(entry):
    data = _load_scripts()
    data = [e for e in data if e.get("id") != entry["id"]]  # 去重,新的覆盖
    data.insert(0, entry)
    json.dump(data[:200], open(SCRIPTS_CACHE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def _resolve_vid(url):
    """短链 302 → .../video/<id>,只做一次 HEAD,失败返回空。"""
    import re
    proxy = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{url_effective}", "-L", "-I", url]
    if proxy:
        cmd = cmd[:1] + ["--proxy", proxy] + cmd[1:]
    try:
        r = run(cmd, timeout=20)
        m = re.search(r"/video/(\d+)", r.stdout or "")
        return m.group(1) if m else ""
    except Exception:
        return ""


def _json_data(output):
    payload = json.loads(output or "{}")
    return payload.get("data", payload)


def _extract_log(trace_id, started_at, stage, detail=""):
    elapsed = time.monotonic() - started_at
    suffix = f" · {detail}" if detail else ""
    plog(f"[文案提取 {trace_id}] +{elapsed:.2f}s {stage}{suffix}")


def _electron_media_download(url, job_dir, trace_id, started_at, cached_video_ids=None):
    """Resolve and download Douyin media only through Electron profile 0."""
    electron = _sht.which("agent-electron")
    desktop = _sht.which("agent-desktop")
    if not electron or not desktop:
        raise RuntimeError("缺少 agent-electron / agent-desktop，无法打开抖音页面")

    _extract_log(trace_id, started_at, "ELECTRON_OPEN_START", "profile=0, 独立 BrowserWindow")
    open_started = time.monotonic()
    opened = run(
        [electron, "open", url, "--idx", "0", "--no-reuse", "--json"],
        timeout=30,
    )
    if opened.returncode != 0:
        raise RuntimeError("Electron profile 0 打开失败: " + (opened.stderr or "")[-300:])
    win_id = int(_json_data(opened.stdout).get("winId") or 0)
    if not win_id:
        raise RuntimeError("Electron 未返回 BrowserWindow ID")
    _extract_log(
        trace_id, started_at, "ELECTRON_OPEN_DONE",
        f"window={win_id}, command={time.monotonic() - open_started:.2f}s",
    )

    expression = r"""(() => {
      const match = location.pathname.match(/\/video\/(\d+)/);
      const vid = match ? match[1] : "";
      const el = document.querySelector("#RENDER_DATA");
      const pageVideos = Array.from(document.querySelectorAll("video"));
      const directVideo = pageVideos.find(node => {
        const src = node.currentSrc || node.src || "";
        return /^https?:/.test(src) && (!vid || src.includes(vid) || src.includes("__vid="));
      }) || pageVideos.find(node => /^https?:/.test(node.currentSrc || node.src || ""));
      if (vid && directVideo) {
        const directUrl = directVideo.currentSrc || directVideo.src;
        return JSON.stringify({
          vid, ready: document.readyState, renderData: !!el,
          candidates: 1, matchedVideoId: vid,
          duration: Number.isFinite(directVideo.duration) ? directVideo.duration * 1000 : 0,
          media: {url: directUrl, size: 0, bitrate: 0},
          variants: 1, fallback: "video-currentSrc"
        });
      }
      if (!vid || !el) return JSON.stringify({
        vid, ready: document.readyState, renderData: !!el,
        candidates: 0, media: null
      });
      let root;
      try { root = JSON.parse(decodeURIComponent(el.textContent)); } catch (_) { return ""; }
      const seen = new Set(), found = [];
      function walk(value) {
        if (!value) return;
        if (typeof value === "string" && /^[\[{]/.test(value.trim())) {
          try { walk(JSON.parse(value)); } catch (_) {}
          return;
        }
        if (typeof value !== "object" || seen.has(value)) return;
        seen.add(value);
        if (value.video) {
          const candidateId = String(value.awemeId || value.aweme_id || value.id || "");
          const candidateVideo = value.video;
          const hasMedia = (candidateVideo.playAddr && candidateVideo.playAddr.length) ||
            (candidateVideo.bitRateList && candidateVideo.bitRateList.length) ||
            (candidateVideo.bit_rate && candidateVideo.bit_rate.length);
          if (hasMedia) found.push({ value, candidateId });
        }
        Object.values(value).forEach(walk);
      }
      walk(root);
      const selected = found.find(item => item.candidateId === vid) || found[0];
      const video = selected && selected.value.video;
      if (!video) {
        const raw = decodeURIComponent(el.textContent);
        const idAt = raw.indexOf(vid);
        const searchFrom = idAt >= 0 ? idAt : 0;
        const playAt = raw.indexOf('"playAddr"', searchFrom);
        const block = playAt >= 0 ? raw.slice(playAt, playAt + 12000) : "";
        const srcMatch = block.match(/"src":"((?:\\.|[^"])*)"/);
        let fallbackUrl = "";
        if (srcMatch) {
          try { fallbackUrl = JSON.parse('"' + srcMatch[1] + '"'); } catch (_) {}
        }
        const durationBlock = raw.slice(Math.max(0, playAt - 500), playAt + 500);
        const durationMatch = durationBlock.match(/"duration":(\d+)/);
        const sizeMatch = block.match(/"playAddrSize":(\d+)/);
        return JSON.stringify({
          vid, ready: document.readyState, renderData: true,
          candidates: found.length, matchedVideoId: "",
          duration: durationMatch ? Number(durationMatch[1]) : 0,
          media: /^https?:/.test(fallbackUrl) ? {
            url: fallbackUrl, size: sizeMatch ? Number(sizeMatch[1]) : 0, bitrate: 0
          } : null,
          variants: fallbackUrl ? 1 : 0, fallback: "playAddr-regex"
        });
      }
      const rates = video.bitRateList || video.bit_rate || [];
      const variants = rates.map(item => ({
        url: (item.playAddr?.src || item.play_addr?.url_list || [])[0] || "",
        size: item.playAddr?.dataSize || item.play_addr?.data_size || 0,
        bitrate: item.bitRate || item.bit_rate || 0
      })).filter(item => /^https?:/.test(item.url));
      const base = (video.playAddr || []).map(item => ({
        url: item.src || "", size: video.playAddrSize || video.dataSize || 0, bitrate: 0
      })).filter(item => /^https?:/.test(item.url));
      const choices = variants.length ? variants : base;
      choices.sort((a, b) => (a.size || a.bitrate) - (b.size || b.bitrate));
      return JSON.stringify({
        vid, ready: document.readyState, renderData: true,
        candidates: found.length, matchedVideoId: selected?.candidateId || "",
        duration: video.duration || 0, media: choices[0] || null, variants: choices.length
      });
    })()"""

    completed = False
    try:
        media_info = None
        discovery_started = time.monotonic()
        _extract_log(trace_id, started_at, "MEDIA_DISCOVERY_START", f"window={win_id}")
        attempts = 0
        for attempts in range(1, 25):
            evaluated = run([
                electron, "cdp", str(win_id), "Runtime.evaluate",
                json.dumps({"expression": expression, "returnByValue": True}),
            ], timeout=20)
            if evaluated.returncode == 0:
                try:
                    value = json.loads(evaluated.stdout).get("result", {}).get("value")
                    if value:
                        media_info = json.loads(value)
                        if (media_info.get("media") or {}).get("url"):
                            break
                except Exception:
                    pass
            if attempts in {1, 6, 12, 18, 24}:
                diagnostic = media_info or {}
                _extract_log(
                    trace_id, started_at, "MEDIA_DISCOVERY_WAIT",
                    f"window={win_id}, attempt={attempts}/24, "
                    f"ready={diagnostic.get('ready', '-')}, "
                    f"render_data={diagnostic.get('renderData', False)}, "
                    f"candidates={diagnostic.get('candidates', 0)}",
                )
            time.sleep(0.5)
        if not media_info or not (media_info.get("media") or {}).get("url"):
            raise RuntimeError("Electron 页面已打开，但未发现可下载 media")

        vid = media_info["vid"]
        if vid and vid in (cached_video_ids or set()):
            completed = True
            _extract_log(
                trace_id, started_at, "MEDIA_DOWNLOAD_SKIPPED",
                f"window={win_id}, video={vid}, reason=已有文案缓存",
            )
            return None, vid
        media_path = job_dir / f"dy_{vid}.mp4"
        media_url = media_info["media"]["url"]
        selected_size = int((media_info.get("media") or {}).get("size") or 0)
        page_duration = float(media_info.get("duration") or 0) / 1000
        _extract_log(
            trace_id, started_at, "MEDIA_DISCOVERY_DONE",
            f"window={win_id}, video={vid}, attempts={attempts}, "
            f"discovery={time.monotonic() - discovery_started:.2f}s, "
            f"duration={page_duration:.1f}s, variants={media_info.get('variants', 0)}, "
            f"selected={selected_size / 1024 / 1024:.1f}MB",
        )
        _extract_log(trace_id, started_at, "SESSION_DOWNLOAD_START", f"window={win_id}, video={vid}")
        download_started = time.monotonic()
        download_env = os.environ.copy()
        # agent-desktop defaults to a 30s RPC ACK timeout. Large Douyin media
        # routinely needs longer even though session_download_url itself is
        # configured for five minutes.
        download_env["CICY_AGENT_TIMEOUT_MS"] = "330000"
        downloaded = subprocess.run(_rewrite_local([
            desktop, "rpc", "session_download_url",
            json.dumps({
                "win_id": win_id, "url": media_url,
                "save_path": str(media_path), "timeout": 300000,
            }),
            "--json",
        ]), capture_output=True, text=True, timeout=340, env=download_env)
        if downloaded.returncode != 0:
            raise RuntimeError("Electron Session 下载失败: " + (downloaded.stderr or downloaded.stdout)[-500:])
        if not media_path.exists() or media_path.stat().st_size < 10_000:
            raise RuntimeError("Electron 报告下载完成，但媒体文件不存在或不完整")
        duration = _ffdur(str(media_path))
        if duration <= 0:
            raise RuntimeError("媒体下载完成，但 ffprobe 校验失败")
        completed = True
        _extract_log(
            trace_id, started_at, "SESSION_DOWNLOAD_DONE",
            f"window={win_id}, download={time.monotonic() - download_started:.2f}s, "
            f"file={media_path.stat().st_size / 1024 / 1024:.1f}MB",
        )
        _extract_log(
            trace_id, started_at, "MEDIA_VERIFY_DONE",
            f"ffprobe_duration={duration:.1f}s, path={media_path.name}",
        )
        return str(media_path), vid
    finally:
        if completed:
            _extract_log(trace_id, started_at, "ELECTRON_CLOSE_START", f"window={win_id}")
            closed = run([electron, "close", str(win_id)], timeout=20)
            _extract_log(
                trace_id, started_at, "ELECTRON_CLOSE_DONE",
                f"window={win_id}, rc={closed.returncode}, reason=下载完成",
            )
        else:
            # Keep the failed page visible for diagnosis. The user can inspect
            # the exact BrowserWindow instead of losing the failure state.
            focused = run([
                desktop, "rpc", "control_electron_BrowserWindow",
                json.dumps({
                    "win_id": win_id,
                    "code": "(win.isMinimized()&&win.restore(),win.show(),win.focus(),true)",
                }),
                "--json",
            ], timeout=20)
            _extract_log(
                trace_id, started_at, "ELECTRON_KEPT_OPEN",
                f"window={win_id}, focus_rc={focused.returncode}, reason=失败现场保留",
            )


def _is_text_llm_provider(item):
    """AI rewrite accepts text LLMs only, never STT or vision-only models."""
    if (item.get("protocol") or "").lower() != "openai":
        return False
    key = (item.get("key") or "").lower()
    models = [str(model).lower() for model in (item.get("models") or [])]
    default_model = str(item.get("defaultModel") or "").lower()
    if key.startswith("custom_") and not (item.get("apiKey") or "").strip():
        return False
    if key in {"groqstt", "zhipu"} or "stt" in key:
        return False
    if "whisper" in default_model or any("whisper" in model for model in models):
        return False
    return True


def _openai_provider(provider_key="defaultOpenAi"):
    """按 key 使用 global.json 中的 OpenAI 协议服务。"""
    try:
        providers = load_global_cfg().get("providers", {})
        items = providers.get("items") or []
        p = next((item for item in items if item.get("key") == provider_key), None)
        if p and _is_text_llm_provider(p) and (p.get("apiKey") or "").strip():
            return p
    except Exception:
        pass
    return None


def _chat(messages, model=None, max_tokens=1200, timeout=90, provider_key="defaultOpenAi"):
    import requests
    p = _openai_provider(provider_key)
    if not p:
        raise RuntimeError(f"global.json 中供应商 {provider_key} 不可用")
    mdl = model or p.get("defaultModel") or "deepseek-v4-flash"
    endpoint = (p.get("url") or "").rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions"
    # OpenCode Zen's public models authenticate with the literal public token.
    # The configured provider key is used by cicy's local gateway and is not a
    # valid direct Zen bearer credential.
    bearer_token = "public" if provider_key == "opencodeZen" else p["apiKey"]
    headers = {
        "Authorization": "Bearer " + bearer_token,
        "Content-Type": "application/json",
        "User-Agent": "cicy-koubo/0.1",
    }
    payload = {"model": mdl, "messages": messages,
               "max_tokens": max_tokens, "temperature": 0.9}
    request_timeout = (8, min(float(timeout), 90))
    session = requests.Session()
    # The desktop browsing proxy may inject a self-signed certificate.
    # LLM API requests use a direct connection instead.
    session.trust_env = False
    try:
        response = session.post(endpoint, headers=headers, json=payload, timeout=request_timeout)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise RuntimeError(f"{provider_key} 返回了空内容，请切换模型后重试")
        return content.strip()
    except (requests.ConnectionError, requests.Timeout) as exc:
        last_error = exc
    if isinstance(last_error, requests.Timeout):
        raise RuntimeError(f"{provider_key} 响应超时，请稍后重试或切换模型")
    raise RuntimeError(f"{provider_key} 连接失败: {last_error}")


DEFAULT_REWRITE_PROMPT = (
    "你是抖音爆款口播文案专家。把用户给的对标文案仿写成一条全新的口播稿:"
    "保留原文的钩子结构和节奏,但更换具体行业/场景/数字/案例,做到不搬运、可直接口播。"
    "要求:开头3秒强钩子抓停留;中间给足干货或情绪;结尾引导关注。"
    "只输出改写后的正文,不要解释、不要小标题、不要序号。"
)
DEFAULT_TITLE_PROMPT = (
    "你是短视频内容策划专家。根据用户提供的口播文案，生成3个适合发布的高点击标题，"
    "每个标题带1到2个相关话题标签；再生成一句简洁有力的封面文案。"
    "直接输出结果，结构清晰，不要解释创作过程。"
)


@app.get("/api/rewrite-prompt")
def rewrite_prompt_default():
    return jsonify({"prompt": DEFAULT_REWRITE_PROMPT})


@app.get("/api/title-prompt")
def title_prompt_default():
    return jsonify({"prompt": DEFAULT_TITLE_PROMPT})


@app.get("/api/llm-options")
def llm_options():
    """Return safe provider/model metadata; never expose API keys."""
    items = load_global_cfg().get("providers", {}).get("items") or []
    providers = []
    seen_endpoints = set()
    for item in items:
        if _is_text_llm_provider(item):
            endpoint = (item.get("url") or "").strip().rstrip("/").lower()
            if endpoint and endpoint in seen_endpoints:
                continue
            if endpoint:
                seen_endpoints.add(endpoint)
            default_model = (item.get("defaultModel") or "").strip()
            models = item.get("models") or ([default_model] if default_model else [])
            providers.append({
                "key": item.get("key"),
                "name": item.get("name") or item.get("key"),
                "configured": bool((item.get("apiKey") or "").strip()),
                "defaultModel": default_model,
                "models": [str(model) for model in models if model],
            })
    return jsonify({"providers": providers, "defaultProvider": "defaultOpenAi"})


def _colab_whisper_status():
    colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
    if not pathlib.Path(colab_bin).exists():
        return False, False, False
    try:
        sessions = subprocess.run(
            [colab_bin, "sessions"], capture_output=True, text=True, timeout=15
        )
        active = sessions.returncode == 0 and "No active" not in (sessions.stdout or "")
    except Exception:
        active = False
    if not active:
        return True, False, False
    try:
        rc, _ = _colab_exec(
            "test -x /content/cosy/env/bin/python && "
            "/content/cosy/env/bin/python -c 'import whisper'",
            timeout=20,
        )
        return True, True, rc == 0
    except Exception:
        return True, True, False


@app.get("/api/stt-options")
def stt_options():
    items = load_global_cfg().get("providers", {}).get("items") or []
    groq = next((item for item in items if item.get("key") in {"groqStt", "groq_stt"}), None)
    cli_installed, session_active, whisper_installed = _colab_whisper_status()
    if not cli_installed:
        colab_hint = "Colab CLI 未安装"
    elif not session_active:
        colab_hint = "Colab CLI 已安装，但没有活跃 GPU 会话"
    elif not whisper_installed:
        colab_hint = "Colab 会话可用，但 Whisper 尚未安装"
    else:
        colab_hint = "Colab CLI、GPU 会话和 Whisper 均已就绪"
    return jsonify({
        "options": [
            {
                "key": "groq", "name": "Groq Whisper",
                "available": bool(groq and (groq.get("apiKey") or "").strip()),
                "hint": "读取 global.json 的 groqStt",
            },
            {
                "key": "colab", "name": "Colab Whisper",
                "available": cli_installed and session_active and whisper_installed,
                "hint": colab_hint,
                "cliInstalled": cli_installed,
                "sessionActive": session_active,
                "whisperInstalled": whisper_installed,
            },
        ],
        "default": "groq",
    })


def _desktop_rpc(tool, args, timeout=15, connect_retries=6):
    """Call one cicy-desktop tool and unwrap its occasionally nested JSON."""
    desktop = _sht.which("agent-desktop")
    if not desktop:
        raise RuntimeError("未安装 agent-desktop，无法打开 cicy-code")
    payload = {}
    for attempt in range(connect_retries):
        result = run(
            [desktop, "rpc", tool, json.dumps(args, ensure_ascii=False), "--json"],
            timeout=timeout,
        )
        raw = result.stdout or result.stderr or ""
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            payload = {}
        disconnected = (
            "no cicy-desktop client connected" in raw
            or "no cicy-desktop client connected" in str(payload)
        )
        if disconnected and attempt + 1 < connect_retries:
            time.sleep(0.6)
            continue
        if disconnected:
            raise RuntimeError("cicy-desktop 当前未连接，请确认桌面端正在运行后重试")
        if result.returncode != 0:
            raise RuntimeError((raw or "cicy-desktop 调用失败")[-500:])
        break
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "cicy-desktop 调用失败"))
    data = payload.get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
    return data


@app.post("/api/open-provider-settings")
def open_provider_settings():
    """Activate cicy-code's profile-0 tab and click its provider button."""
    target_hosts = {"localhost:8008", "127.0.0.1:8008"}
    def find_cicy_code_tab():
        tabs_data = _desktop_rpc("electron_tabs", {"accountIdx": 0})
        tabs = tabs_data.get("tabs", []) if isinstance(tabs_data, dict) else []
        return next((
            item for item in tabs
            if urllib.parse.urlsplit(item.get("url") or "").netloc in target_hosts
        ), None)

    try:
        tab = find_cicy_code_tab()
        if tab:
            web_contents_id = int(tab["webContentsId"])
            _desktop_rpc("electron_tab_activate", {"webContentsId": web_contents_id})
        else:
            _desktop_rpc("electron_tab_open", {
                "accountIdx": 0,
                "url": "http://127.0.0.1:8008/",
                "trusted": True,
                "activate": True,
            })
            # Different desktop versions return different shapes from tab_open.
            # Query the authoritative tab list instead of trusting that response.
            tab = None
            for _ in range(20):
                tab = find_cicy_code_tab()
                if tab:
                    break
                time.sleep(0.25)
            if not tab:
                raise RuntimeError("已请求打开 cicy-code，但 profile 0 中未找到对应 tab")
            web_contents_id = int(tab["webContentsId"])
            _desktop_rpc("electron_tab_activate", {"webContentsId": web_contents_id})

        click_code = """new Promise(resolve => {
          const deadline = Date.now() + 10000;
          const click = () => {
            const button = document.querySelector('[data-id="btn-providers"]');
            if (button) {
              button.click();
              resolve({clicked: true, title: document.title});
            } else if (Date.now() >= deadline) {
              resolve({clicked: false, title: document.title});
            } else {
              setTimeout(click, 250);
            }
          };
          click();
        })"""
        clicked = _desktop_rpc("electron_tab_eval", {
            "webContentsId": web_contents_id,
            "code": click_code,
        }, timeout=15)
        if isinstance(clicked, dict):
            clicked = clicked.get("result", clicked)
        if isinstance(clicked, dict) and not clicked.get("clicked", True):
            raise RuntimeError("cicy-code 已打开，但没有找到供应商设置按钮")
        plog(f"[模型配置] 已激活 cicy-code profile=0 tab={web_contents_id} 并打开供应商设置")
        return jsonify({"ok": True, "webContentsId": web_contents_id})
    except Exception as exc:
        plog(f"[模型配置] 打开失败: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.post("/api/rewrite")
def rewrite():
    body = request.json or {}
    src = (body.get("text") or "").strip()
    if not src:
        return jsonify({"error": "没有可改写的文案"}), 400
    sys = (body.get("system") or "").strip() or DEFAULT_REWRITE_PROMPT
    provider_key = (body.get("provider") or "defaultOpenAi").strip()
    model = (body.get("model") or "").strip() or None
    plog(f"[AI仿写] REQUEST · provider={provider_key}, model={model or 'default'}")
    if not _openai_provider(provider_key):
        return jsonify({
            "error": f"供应商 {provider_key} 未配置 API Key，请在 cicy-ai/global.json 中完成配置"
        }), 400
    language_code = (body.get("outputLanguage") or "").strip()
    language_names = {
        "zh-CN": "简体中文", "zh-TW": "繁体中文", "en": "英语",
        "ja": "日语", "ko": "韩语", "es": "西班牙语", "fr": "法语",
        "de": "德语", "th": "泰语", "vi": "越南语", "id": "印度尼西亚语",
    }
    if language_code in language_names:
        sys += (
            "\n\n输出语言要求：请将最终口播文案翻译并输出为"
            f"{language_names[language_code]}（{language_code}）。"
            "只输出该语言的最终文案，不要附带原文、翻译说明或语言标签。"
        )
    try:
        out = _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": src}],
            model=model, provider_key=provider_key,
        )
        plog(f"[AI仿写] SUCCESS · provider={provider_key}, model={model or 'default'}, chars={len(out)}")
        return jsonify({"text": out})
    except Exception as e:  # noqa: BLE001
        plog(f"[AI仿写] FAILED · provider={provider_key}, model={model or 'default'}, error={type(e).__name__}: {e}")
        return jsonify({"error": "改写失败: " + str(e)}), 500


@app.post("/api/title")
def title():
    body = request.json or {}
    src = (body.get("text") or "").strip()
    if not src:
        return jsonify({"error": "没有文案"}), 400
    provider_key = (body.get("provider") or "defaultOpenAi").strip()
    model = (body.get("model") or "").strip() or None
    system_prompt = (body.get("system") or "").strip() or DEFAULT_TITLE_PROMPT
    language_code = (body.get("outputLanguage") or "").strip()
    language_names = {
        "zh-CN": "简体中文", "zh-TW": "繁体中文", "en": "英语",
        "ja": "日语", "ko": "韩语", "es": "西班牙语", "fr": "法语",
        "de": "德语", "th": "泰语", "vi": "越南语", "id": "印度尼西亚语",
    }
    if language_code in language_names:
        system_prompt += (
            f"\n\n输出语言要求：全部内容必须使用{language_names[language_code]}"
            f"（{language_code}）输出，不要附带原文或翻译说明。"
        )
    system_prompt += (
        '\n\n输出格式要求：只返回合法 JSON 对象，不要使用 Markdown 代码块。'
        '对象必须且只能包含三个字段：'
        '"title"（3个候选标题组成的数组）、'
        '"tags"（话题标签字符串数组，不带#）、'
        '"coverText"（一句封面文案字符串）。'
    )
    plog(f"[标题文案] REQUEST · provider={provider_key}, model={model or 'default'}, chars={len(src)}")
    try:
        out = _chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": src}],
            model=model, max_tokens=1600, provider_key=provider_key,
        )
        raw = out.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        titles = result.get("title") or []
        if isinstance(titles, str):
            titles = [titles]
        tags = result.get("tags") or []
        if isinstance(tags, str):
            tags = [tag.strip().lstrip("#") for tag in tags.replace("，", ",").split(",") if tag.strip()]
        cover_text = str(result.get("coverText") or "").strip()
        if not titles or not cover_text:
            raise RuntimeError("模型返回的标题文案结构不完整")
        payload = {
            "title": [str(title).strip() for title in titles if str(title).strip()][:3],
            "tags": [str(tag).strip().lstrip("#") for tag in tags if str(tag).strip()],
            "coverText": cover_text,
        }
        plog(f"[标题文案] SUCCESS · provider={provider_key}, model={model or 'default'}, titles={len(payload['title'])}, tags={len(payload['tags'])}")
        return jsonify(payload)
    except Exception as e:  # noqa: BLE001
        plog(f"[标题文案] FAILED · provider={provider_key}, model={model or 'default'}, error={type(e).__name__}: {e}")
        return jsonify({"error": str(e)}), 500


def _groq_key():
    """只从 global.json（或显式环境变量）读取 Groq STT key。"""
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    try:
        gj = load_global_cfg()
        for x in (gj.get("providers", {}).get("items") or []):
            if x.get("key") in ("groqStt", "groq_stt"):
                k = (x.get("apiKey") or "").strip()
                if k:
                    return k
    except Exception:
        pass
    return ""


@app.post("/api/extract")
def extract():
    """抖音链接 → 下载音频 → Groq 转写 → 返回文案。"""
    started_at = time.monotonic()
    body = request.json or {}
    url = body.get("url", "").strip()
    force = body.get("force", False)
    trace_id = "".join(ch for ch in str(body.get("trace_id") or "") if ch.isalnum() or ch in "-_")[:32]
    trace_id = trace_id or uuid.uuid4().hex[:10]
    _extract_log(trace_id, started_at, "REQUEST_RECEIVED", f"force={bool(force)}")
    if "douyin.com" not in url:
        _extract_log(trace_id, started_at, "REQUEST_REJECTED", "未识别到 douyin.com")
        return jsonify({"error": "请粘贴抖音分享链接(含 v.douyin.com)"}), 400
    jd = WORK / ("dy_" + uuid.uuid4().hex[:8])
    jd.mkdir(exist_ok=True)
    _extract_log(trace_id, started_at, "JOB_CREATED", f"dir={jd.name}")
    try:
        # 第一层:按链接字符串直接查缓存，零网络秒回。
        import re
        code = ""
        cm = re.search(r"douyin\.com/(?:video/)?([A-Za-z0-9_]+)", url)
        if cm:
            code = cm.group(1)
        scripts = _load_scripts()
        _extract_log(trace_id, started_at, "CACHE_CHECK_START", f"entries={len(scripts)}")
        if not force:
            for e in scripts:
                eu = e.get("url", "")
                if url == eu or (code and code in eu) or (code and code == e.get("id")):
                    _extract_log(trace_id, started_at, "CACHE_HIT", f"video={e.get('id') or 'share-link'}")
                    _extract_log(trace_id, started_at, "RESPONSE_SUCCESS", "cached=true")
                    return jsonify({
                        "text": e["text"], "audio": e.get("audio"), "cached": True,
                        "trace_id": trace_id, "elapsed": round(time.monotonic() - started_at, 1),
                    })
        _extract_log(trace_id, started_at, "CACHE_MISS")
        # 唯一路径:Electron profile 0 BrowserWindow + 同 Session 下载。
        # 不调用 yt-dlp，也不回退 Chrome CDP。
        media, vid = _electron_media_download(
            url, jd, trace_id, started_at,
            cached_video_ids={str(e.get("id")) for e in scripts if e.get("id")},
        )
        # 命中缓存(同一视频)直接返回,除非 force;下载失败也回退到缓存
        if vid and not force:
            _extract_log(trace_id, started_at, "VIDEO_CACHE_CHECK_START", f"video={vid}")
            for e in scripts:
                if e.get("id") == vid:
                    # Remember the share URL as an alias. The next click can
                    # return before opening Electron or downloading media.
                    if e.get("url") != url:
                        e["url"] = url
                        _save_script(e)
                    _extract_log(trace_id, started_at, "VIDEO_CACHE_HIT", f"video={vid}")
                    _extract_log(trace_id, started_at, "RESPONSE_SUCCESS", "cached=true, media下载已跳过")
                    return jsonify({
                        "text": e["text"], "audio": e.get("audio"), "cached": True,
                        "trace_id": trace_id, "elapsed": round(time.monotonic() - started_at, 1),
                    })
        if not media or not os.path.exists(media):
            _extract_log(trace_id, started_at, "MEDIA_MISSING", f"video={vid or '-'}")
            for e in _load_scripts():  # 下载失败但曾缓存过 → 用缓存
                if url in (e.get("url") or "") or (vid and e.get("id") == vid):
                    _extract_log(trace_id, started_at, "CACHE_FALLBACK", "下载失败，返回历史缓存")
                    return jsonify({
                        "text": e["text"], "cached": True, "note": "下载失败,返回历史缓存",
                        "trace_id": trace_id, "elapsed": round(time.monotonic() - started_at, 1),
                    })
            return jsonify({"error": "Electron 下载失败，请查看日志"}), 502
        # 短视频优先走 Groq。先只提取低码率单声道音频，上传更小、更快；
        # 长视频不占用云端 STT，直接回退 Colab / 本机 Whisper。
        text = ""
        provider = ""
        duration = _ffdur(media)
        _extract_log(
            trace_id, started_at, "STT_ROUTE",
            f"duration={duration:.1f}s, groq_configured={bool(_groq_key())}, gpu_mode={GPU_MODE}",
        )
        if 0 < duration <= 600 and _groq_key():
            stt_audio = jd / f"dy_{vid or 'audio'}.mp3"
            _extract_log(trace_id, started_at, "AUDIO_COMPRESS_START", "mono=1, rate=16kHz, bitrate=32k")
            compress_started = time.monotonic()
            audio_result = run([
                "ffmpeg", "-v", "error", "-y", "-i", media,
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(stt_audio),
            ], timeout=120)
            if audio_result.returncode == 0 and stt_audio.exists():
                _extract_log(
                    trace_id, started_at, "AUDIO_COMPRESS_DONE",
                    f"elapsed={time.monotonic() - compress_started:.2f}s, "
                    f"file={stt_audio.stat().st_size / 1024 / 1024:.1f}MB",
                )
                _extract_log(trace_id, started_at, "GROQ_STT_START", "model=whisper-large-v3-turbo")
                stt_started = time.monotonic()
                text = _groq_transcribe(str(stt_audio))
                if text:
                    provider = "groq"
                    _extract_log(
                        trace_id, started_at, "GROQ_STT_DONE",
                        f"elapsed={time.monotonic() - stt_started:.2f}s, chars={len(text)}",
                    )
                else:
                    _extract_log(trace_id, started_at, "GROQ_STT_EMPTY", "准备回退")
            else:
                _extract_log(
                    trace_id, started_at, "AUDIO_COMPRESS_FAILED",
                    f"rc={audio_result.returncode}",
                )

        # Groq 不可用、请求失败或长视频:回退 Colab whisper。
        if GPU_MODE == "colab_cli":
            stt_input = str(stt_audio) if 'stt_audio' in locals() and stt_audio.exists() else media
            remote_audio = f"/content/_dy_{pathlib.Path(stt_input).name}"
            if not text:
                _extract_log(trace_id, started_at, "COLAB_UPLOAD_START", f"file={pathlib.Path(stt_input).name}")
            if not text and _colab_upload(stt_input, remote_audio):
                _extract_log(trace_id, started_at, "COLAB_UPLOAD_DONE")
                _extract_log(trace_id, started_at, "COLAB_STT_START", "model=whisper-medium")
                colab_started = time.monotonic()
                script = (
                    "import whisper,sys,json; "
                    "m=whisper.load_model('medium',device='cuda'); "
                    f"r=m.transcribe('{remote_audio}',language='zh',verbose=False); "
                    "sys.stdout.write(r['text'].strip())"
                )
                rc, out = _colab_exec(f"cd /content/cosy && env/bin/python -c {shlex.quote(script)}", timeout=300)
                if rc == 0:
                    text = out.strip()
                    if text:
                        provider = "colab-whisper"
                        _extract_log(
                            trace_id, started_at, "COLAB_STT_DONE",
                            f"elapsed={time.monotonic() - colab_started:.2f}s, chars={len(text)}",
                        )
        if not text:
            # fallback: 本机 CosyVoice 环境
            cosy_py = pathlib.Path("/content/cosy/env/bin/python")
            if cosy_py.exists():
                _extract_log(trace_id, started_at, "LOCAL_STT_START", "model=whisper-medium")
                try:
                    r = subprocess.run([str(cosy_py), "-c", (
                        "import whisper,sys; m=whisper.load_model('medium',device='cuda'); "
                        f"r=m.transcribe('{media}',language='zh'); "
                        "print(r['text'].strip())"
                    )], capture_output=True, text=True, timeout=300)
                    text = r.stdout.strip() if r.returncode == 0 else ""
                    if text:
                        provider = "local-whisper"
                        _extract_log(trace_id, started_at, "LOCAL_STT_DONE", f"chars={len(text)}")
                except Exception as local_error:
                    _extract_log(trace_id, started_at, "LOCAL_STT_FAILED", str(local_error)[:300])
        if not text:
            _extract_log(trace_id, started_at, "STT_FAILED", "所有转写路径均不可用")
            return jsonify({"error": "转写失败:无可用 whisper 环境(Colab/本地)"}), 500
        _extract_log(trace_id, started_at, "SCRIPT_SAVE_START", f"video={vid}, chars={len(text)}")
        _save_script({"id": vid, "url": url, "text": text, "audio": media,
                      "ts": time.strftime("%Y-%m-%d %H:%M")})
        _extract_log(trace_id, started_at, "SCRIPT_SAVE_DONE", f"video={vid}")
        _extract_log(
            trace_id, started_at, "RESPONSE_SUCCESS",
            f"cached=false, provider={provider}, total={time.monotonic() - started_at:.2f}s",
        )
        return jsonify({
            "text": text, "audio": media, "cached": False,
            "provider": provider, "elapsed": round(time.monotonic() - started_at, 1),
            "trace_id": trace_id,
        })
    except Exception as e:  # noqa: BLE001
        _extract_log(
            trace_id, started_at, "REQUEST_FAILED",
            f"{type(e).__name__}: {str(e)[:500]}",
        )
        return jsonify({"error": str(e)}), 500


@app.get("/api/scripts")
def list_scripts():
    return jsonify(_load_scripts())


@app.post("/api/scripts/delete")
def delete_script():
    body = request.json or {}
    if body.get("all"):
        json.dump([], open(SCRIPTS_CACHE, "w", encoding="utf-8"))
        return jsonify({"ok": True, "remaining": 0})
    sid = body.get("id")
    data = [e for e in _load_scripts() if e.get("id") != sid]
    json.dump(data, open(SCRIPTS_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return jsonify({"ok": True, "remaining": len(data)})


# ---------------- 素材库 CRUD(音色 / 底板 / BGM 的上传·重命名·删除) ----------------
ASSETS_META = APP_DIR / "assets_meta.json"


def _meta_all():
    if ASSETS_META.exists():
        try:
            return json.load(open(ASSETS_META, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _meta_name(kind, aid, default):
    return _meta_all().get(kind, {}).get(aid) or default


def _meta_set(kind, aid, name=None, remove=False):
    data = _meta_all()
    data.setdefault(kind, {})
    if remove:
        data[kind].pop(aid, None)
    else:
        data[kind][aid] = name
    json.dump(data, open(ASSETS_META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _next_free(pattern, fmt):
    n = 1
    while (ROOT / "assets" / fmt.format(n)).exists() or \
          (ROOT / "assets" / fmt.format(n).replace(".mp4", "-25fps.mp4")).exists():
        n += 1
    return fmt.format(n)


def _process_reference_voice(raw, dst, mode):
    """Convert a reference recording while preserving its original upload."""
    filters = {
        "raw": "anull",
        "auto": "highpass=f=70,loudnorm=I=-20:LRA=7:TP=-2",
        "denasal": (
            "highpass=f=70,"
            "equalizer=f=850:t=q:w=1.1:g=-2.5,"
            "equalizer=f=1400:t=q:w=1.2:g=-1.5,"
            "loudnorm=I=-20:LRA=7:TP=-2"
        ),
    }
    mode = mode if mode in filters else "auto"
    original_dir = ROOT / "assets" / "originals"
    original_dir.mkdir(parents=True, exist_ok=True)
    original = original_dir / dst.name
    original_cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(raw),
        "-ar", "16000", "-ac", "1", str(original),
    ]
    converted = run(original_cmd, timeout=120)
    if converted.returncode != 0 or not original.exists():
        return converted, mode
    processed = run([
        "ffmpeg", "-v", "error", "-y", "-i", str(original),
        "-af", filters[mode], "-ar", "16000", "-ac", "1", str(dst),
    ], timeout=120)
    return processed, mode


def _normalize_voice_output(path):
    """Normalize generated speech for short-video playback without changing speed."""
    src = pathlib.Path(path)
    normalized = src.with_name(src.stem + "_normalized.wav")
    result = run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-af", "loudnorm=I=-16:LRA=7:TP=-1.5",
        "-ar", "24000", "-ac", "1", str(normalized),
    ], timeout=120)
    if result.returncode == 0 and normalized.exists() and normalized.stat().st_size > 1024:
        normalized.replace(src)
        return True
    normalized.unlink(missing_ok=True)
    plog(f"[配音响度标准化] 跳过: {result.stderr[-200:]}")
    return False


@app.post("/api/voices/upload")
def voice_upload():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "缺文件"}), 400
    (ROOT / "assets").mkdir(parents=True, exist_ok=True)
    raw = WORK / (uuid.uuid4().hex[:8] + "_vup" + pathlib.Path(request.files["file"].filename).suffix)
    request.files["file"].save(raw)
    vid = _next_free("voice", "voice-sample-{:02d}.wav")
    dst = ROOT / "assets" / vid
    r, optimize = _process_reference_voice(
        raw, dst, (request.form.get("optimize") or "auto").strip().lower()
    )
    raw.unlink(missing_ok=True)
    if r.returncode != 0 or not dst.exists():
        return jsonify({"error": "音频转换失败: " + r.stderr[:200]}), 400
    # Reject recordings that contain no usable microphone signal. Electron
    # may grant getUserMedia while the selected input device still yields
    # digital silence.
    try:
        import array
        import wave
        with wave.open(str(dst), "rb") as wav:
            samples = array.array("h", wav.readframes(wav.getnframes()))
        peak = max((abs(sample) for sample in samples), default=0)
        if peak < 200:
            dst.unlink(missing_ok=True)
            return jsonify({
                "error": "录音中未检测到声音，请检查麦克风权限和系统输入设备后重试"
            }), 400
    except Exception:
        pass
    name = (request.form.get("name") or "").strip()
    if name:
        import datetime
        name = f"{name}-{datetime.datetime.now().strftime('%m%d%H%M')}"
        _meta_set("voice", vid, name)
    return jsonify({
        "id": vid, "name": name or vid, "duration": _ffdur(dst),
        "optimize": optimize,
    })


@app.post("/api/voices/delete")
def voice_delete():
    vid = (request.json or {}).get("id") or ""
    if not vid.startswith("voice-sample-") or ".." in vid:
        return jsonify({"error": "bad id"}), 400
    (ROOT / "assets" / vid).unlink(missing_ok=True)
    (ROOT / "assets" / "originals" / vid).unlink(missing_ok=True)
    _meta_set("voice", vid, remove=True)
    return jsonify({"ok": True})


@app.post("/api/basevideos/upload")
def basevideo_upload():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "缺文件"}), 400
    (ROOT / "assets").mkdir(parents=True, exist_ok=True)
    raw = WORK / (uuid.uuid4().hex[:8] + "_bup" + pathlib.Path(request.files["file"].filename).suffix)
    request.files["file"].save(raw)
    base = _next_free("base", "base-video-{}.mp4")
    orig = ROOT / "assets" / base
    norm = ROOT / "assets" / base.replace(".mp4", "-25fps.mp4")
    import shutil
    shutil.move(str(raw), str(orig))
    # 归一 25fps CFR + 去音轨(出片要求),失败则退回用原文件
    r = run(["ffmpeg", "-v", "error", "-y", "-i", str(orig), "-r", "25",
             "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an", str(norm)], timeout=600)
    use = norm if (r.returncode == 0 and norm.exists()) else orig
    name = (request.form.get("name") or "").strip()
    if name:
        import datetime
        name = f"{name}-{datetime.datetime.now().strftime('%m%d%H%M')}"
        _meta_set("base", base.replace(".mp4", ""), name)
    return jsonify({"id": use.name, "name": name or base.replace(".mp4", ""), "duration": _ffdur(use)})


@app.post("/api/basevideos/delete")
def basevideo_delete():
    bid = (request.json or {}).get("id") or ""
    if not bid.startswith("base-video-") or ".." in bid:
        return jsonify({"error": "bad id"}), 400
    key = bid.replace("-25fps", "")
    (ROOT / "assets" / key).unlink(missing_ok=True)
    (ROOT / "assets" / key.replace(".mp4", "-25fps.mp4")).unlink(missing_ok=True)
    _meta_set("base", key.replace(".mp4", ""), remove=True)
    return jsonify({"ok": True})


@app.post("/api/bgms/upload")
def bgm_upload():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "缺文件"}), 400
    src_name = pathlib.Path(request.files["file"].filename)
    raw = WORK / (uuid.uuid4().hex[:8] + "_gup" + src_name.suffix)
    request.files["file"].save(raw)
    stem = "".join(c for c in src_name.stem if c.isalnum() or c in "-_")[:40] or uuid.uuid4().hex[:8]
    dst = ROOT / "assets/bgm" / (stem + ".mp3")
    n = 1
    while dst.exists():
        dst = ROOT / "assets/bgm" / f"{stem}-{n}.mp3"
        n += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vn", "-b:a", "192k", str(dst)], timeout=300)
    raw.unlink(missing_ok=True)
    if r.returncode != 0 or not dst.exists():
        return jsonify({"error": "音频转换失败: " + r.stderr[:200]}), 400
    name = (request.form.get("name") or "").strip()
    if name:
        _meta_set("bgm", dst.name, name)
    return jsonify({"id": dst.name, "name": name or dst.stem, "duration": _ffdur(dst)})


@app.post("/api/bgms/delete")
def bgm_delete():
    bid = (request.json or {}).get("id") or ""
    f = ROOT / "assets/bgm" / bid
    if ".." in bid or "/" in bid or not f.exists():
        return jsonify({"error": "not found"}), 404
    f.unlink(missing_ok=True)
    _meta_set("bgm", bid, remove=True)
    return jsonify({"ok": True})


@app.post("/api/assets/rename")
def asset_rename():
    body = request.json or {}
    kind, aid, name = body.get("kind"), body.get("id"), (body.get("name") or "").strip()
    if kind not in ("voice", "base", "bgm") or not aid or not name:
        return jsonify({"error": "参数不全"}), 400
    if kind == "base":
        aid = aid.replace("-25fps", "").replace(".mp4", "")
    _meta_set(kind, aid, name)
    return jsonify({"ok": True})


@app.post("/api/scripts/save")
def script_save():
    """文案 新建/更新:{id?, text, title?}。无 id 则新建 manual 条目。"""
    body = request.json or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "文案为空"}), 400
    sid = body.get("id") or ("manual-" + uuid.uuid4().hex[:8])
    entry = next((e for e in _load_scripts() if e.get("id") == sid), None) or {"id": sid, "url": ""}
    entry["text"] = text
    if body.get("title"):
        entry["title"] = body["title"].strip()
    entry["ts"] = time.strftime("%m-%d %H:%M")
    _save_script(entry)
    return jsonify({"ok": True, "id": sid})


@app.post("/api/media/update")
def media_update():
    body = request.json or {}
    mid, note = body.get("id"), (body.get("note") or "").strip()
    data = _media_list()
    hit = False
    for e in data:
        if e.get("id") == mid:
            e["note"] = note
            hit = True
    if not hit:
        return jsonify({"error": "not found"}), 404
    json.dump(data, open(MEDIA_REG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return jsonify({"ok": True})


def _is_installing(key):
    """检测 /content/<key>/provision.sh 是否在运行。匹配进程命令行里的路径。"""
    if GPU_MODE == "colab_cli":
        try:
            rc, out = _colab_exec(
                f"pgrep -f '/content/{key}/[p]rovision.sh' >/dev/null && echo YES || echo NO",
                timeout=15,
            )
            return rc == 0 and out.strip().endswith("YES")
        except Exception:
            return False
    try:
        r = subprocess.run(["pgrep", "-f", f"/content/{key}/provision.sh"],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# 启动时确保中文字体可用（字幕渲染 PIL 需要）
def _ensure_cjk_font():
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"):
        if os.path.exists(f):
            return
    # 只有 Colab/Linux 需要装，macOS 自带
    if sys.platform == "linux":
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
_ensure_cjk_font()


# ═══════════ 安装管理 ═══════════
ENGINES = {
    "mt":       {
        "name": "MuseTalk",
        "ready": "/content/mt/READY",
        "dir": "/content/mt",
        "install_hint": "制作数字人口型视频需要；只做配音时不用安装",
    },
    "cosy":     {
        "name": "CosyVoice",
        "ready": "/content/cosy/COSY_READY",
        "dir": "/content/cosy",
        "install_hint": "配音和参考音频转写需要；安装后同时提供 Whisper",
    },
    "whisper":  {"name": "Whisper",   "ready": "/content/cosy/COSY_READY", "dir": "/content/cosy", "builtin": True},
    "hg":       {
        "name": "HeyGem",
        "ready": "/content/hg/HG_READY",
        "dir": "/content/hg",
        "install_hint": "仅选择 HeyGem 数字人时安装；其他流程不用安装",
    },
}

COLAB_GPUS = ("T4", "L4", "G4", "H100", "A100")
_COLAB_OAUTH_PROCESSES = {}
_COLAB_INSTALL_LOG_BRIDGES = set()
_COLAB_ENGINE_STARTING = {}
_COLAB_ENGINE_INSTALL_INFLIGHT = set()
_COLAB_ENGINE_INSTALL_LOCK = threading.Lock()


def _start_colab_install_log_bridge(engine, profile):
    """Continuously copy a remote provision.log into the unified UI log."""
    bridge_key = (profile["id"], engine)
    if bridge_key in _COLAB_INSTALL_LOG_BRIDGES:
        return
    _COLAB_INSTALL_LOG_BRIDGES.add(bridge_key)
    cfg = ENGINES[engine]
    remote_log = shlex.quote(f"{cfg['dir']}/provision.log")
    process_pattern = shlex.quote(f"{cfg['dir']}/[p]rovision.sh")

    def run():
        offset = 0
        stopped_checks = 0
        try:
            while stopped_checks < 3:
                command = (
                    f"size=$(wc -c < {remote_log} 2>/dev/null || echo 0); "
                    f"if [ \"$size\" -lt {offset} ]; then start=1; "
                    f"else start={offset + 1}; fi; "
                    f"tail -c +$start {remote_log} 2>/dev/null || true; "
                    f"printf '\\n__KOUBO_LOG_META__%s|' \"$size\"; "
                    f"(pgrep -f {process_pattern} >/dev/null && echo YES || echo NO)"
                )
                rc, output = _colab_exec(command, timeout=20, profile=profile)
                body, marker, meta = output.rpartition("__KOUBO_LOG_META__")
                if rc == 0 and marker:
                    size_text, _, running_text = meta.strip().partition("|")
                    try:
                        new_offset = int(size_text)
                    except ValueError:
                        new_offset = offset
                    if new_offset < offset:
                        offset = 0
                    if new_offset > offset:
                        for line in body.replace("\r", "\n").splitlines():
                            if line.strip():
                                plog(f"[Colab 安装][{cfg['name']}] {line}")
                        offset = new_offset
                    stopped_checks = 0 if running_text.strip().endswith("YES") else stopped_checks + 1
                    if running_text.strip().endswith("YES"):
                        _COLAB_ENGINE_STARTING.pop(engine, None)
                else:
                    plog(f"[Colab 安装日志暂不可用][{cfg['name']}] 正在自动重试")
                    stopped_checks = 0
                time.sleep(2)
            plog(f"[Colab 安装][{cfg['name']}] 安装进程已结束")
        finally:
            _COLAB_ENGINE_STARTING.pop(engine, None)
            _COLAB_INSTALL_LOG_BRIDGES.discard(bridge_key)

    threading.Thread(
        target=run,
        name=f"colab-install-log-{engine}",
        daemon=True,
    ).start()


@app.get("/api/colab/profiles")
def api_colab_profiles():
    """列出可切换的 Google/Colab 配置档，不返回凭据内容。"""
    _, colab = _colab_profiles_cfg()
    items = []
    for raw in colab["profiles"]:
        p = dict(raw)
        p["auth"] = (p.get("auth") or "oauth2").lower()
        cred = pathlib.Path(os.path.expanduser(p.get("credentials_path") or ""))
        if p["auth"] == "oauth2":
            p["credentials_ready"] = _colab_oauth_token_path({
                "id": p.get("id") or "default"
            }).is_file()
        else:
            p["credentials_ready"] = bool(p.get("credentials_path")) and cred.is_file()
        items.append(p)
    return jsonify({"active": colab["active"], "profiles": items, "gpu_options": COLAB_GPUS})


@app.post("/api/colab/oauth/start")
def api_colab_oauth_start():
    """Start colab-cli's copy/paste OAuth flow and return its Google URL."""
    profile = _active_colab_profile()
    if profile["auth"] != "oauth2":
        return jsonify({"error": "当前账号不是 OAuth2 模式"}), 400
    old = _COLAB_OAUTH_PROCESSES.pop(profile["id"], None)
    if old and old.poll() is None:
        old.terminate()
    token_path = _colab_oauth_token_path(profile)
    # Never delete a working token merely because the user opened the OAuth
    # dialog. A failed/cancelled authorization must leave existing access intact.
    try:
        proc = subprocess.Popen(
            _colab_base_args(profile) + ["whoami"],
            env=_colab_env(profile), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False,
        )
        _COLAB_OAUTH_PROCESSES[profile["id"]] = proc
        import selectors
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        selector.register(proc.stderr, selectors.EVENT_READ)
        chunks = []
        deadline = time.time() + 15
        auth_url = ""
        while time.time() < deadline and proc.poll() is None:
            for key, _ in selector.select(timeout=0.5):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if chunk:
                    chunks.append(chunk.decode("utf-8", errors="replace"))
            output = "".join(chunks)
            match = re.search(r"https://accounts\.google\.com/[^\s]+", output)
            if match:
                auth_url = match.group(0)
                break
        selector.close()
        if not auth_url:
            output = "".join(chunks)
            if proc.poll() is None:
                proc.terminate()
            _COLAB_OAUTH_PROCESSES.pop(profile["id"], None)
            return jsonify({"error": "未能取得 Google 授权地址：" + output[-300:]}), 500
        plog(f"[Colab OAuth] profile={profile['id']} 等待用户授权")
        return jsonify({"ok": True, "auth_url": auth_url})
    except Exception as exc:
        return jsonify({"error": f"启动 OAuth2 失败：{exc}"}), 500


@app.post("/api/colab/oauth/complete")
def api_colab_oauth_complete():
    profile = _active_colab_profile()
    code = ((request.get_json(silent=True) or {}).get("code") or "").strip()
    proc = _COLAB_OAUTH_PROCESSES.get(profile["id"])
    if not code:
        return jsonify({"error": "请粘贴 Google 页面显示的授权码"}), 400
    if not proc or proc.poll() is not None:
        return jsonify({"error": "授权流程已失效，请重新点击授权"}), 409
    try:
        proc.stdin.write((code + "\n").encode())
        proc.stdin.flush()
        stdout, stderr = proc.communicate(timeout=90)
        output = (stdout + stderr).decode("utf-8", errors="replace")
        _COLAB_OAUTH_PROCESSES.pop(profile["id"], None)
        if proc.returncode != 0 or not _colab_oauth_token_path(profile).is_file():
            return jsonify({"error": "Google 授权失败，请重新授权：" + output[-300:]}), 400
        email_match = re.search(r"Email:\s+([^\s]+)", output)
        email = email_match.group(1) if email_match else ""
        if email:
            cfg, colab = _colab_profiles_cfg()
            current = next(
                (item for item in colab["profiles"] if item.get("id") == profile["id"]),
                None,
            )
            if current is not None:
                current["email"] = email
                current["authorized_at"] = time.time()
                save_global_cfg(cfg)
        plog(f"[Colab OAuth] profile={profile['id']} 授权成功 email={email or '-'}")
        return jsonify({"ok": True, "email": email})
    except subprocess.TimeoutExpired:
        proc.terminate()
        _COLAB_OAUTH_PROCESSES.pop(profile["id"], None)
        return jsonify({"error": "Google 授权超时，请重试"}), 408


@app.post("/api/colab/profiles")
def api_colab_profiles_save():
    data = request.get_json(force=True) or {}
    cfg, colab = _colab_profiles_cfg()
    if data.get("active"):
        wanted = data["active"]
        if not any(p.get("id") == wanted for p in colab["profiles"]):
            return jsonify({"error": "配置档不存在"}), 404
        colab["active"] = wanted
        save_global_cfg(cfg)
        return jsonify({"ok": True})
    profile = data.get("profile") or {}
    profile_id = "".join(c for c in (profile.get("id") or f"google-{int(time.time())}") if c.isalnum() or c in "-_")
    gpu = (profile.get("gpu") or "T4").upper()
    if not profile_id or gpu not in COLAB_GPUS:
        return jsonify({"error": "配置档 ID 或 GPU 类型无效"}), 400
    normalized = {
        "id": profile_id,
        "name": (profile.get("name") or profile_id).strip(),
        "auth": (profile.get("auth") or "oauth2").lower(),
        "credentials_path": os.path.expanduser((profile.get("credentials_path") or "").strip()),
        "session_config": os.path.expanduser((profile.get("session_config") or
                                               f"~/.config/colab-cli/{profile_id}-sessions.json").strip()),
        "session": (profile.get("session") or f"koubo-{profile_id}").strip(),
        "gpu": gpu,
        "email": (profile.get("email") or "").strip(),
    }
    if normalized["auth"] not in ("oauth2", "adc"):
        return jsonify({"error": "授权方式仅支持 oauth2 或 adc"}), 400
    if normalized["auth"] == "adc" and not normalized["credentials_path"]:
        return jsonify({"error": "请填写该 Google 账号的 ADC 凭据文件路径"}), 400
    existing = next((p for p in colab["profiles"] if p.get("id") == profile_id), None)
    if existing:
        if not normalized["email"]:
            normalized["email"] = (existing.get("email") or "").strip()
        existing.update(normalized)
    else:
        colab["profiles"].append(normalized)
    colab["active"] = profile_id
    pathlib.Path(normalized["session_config"]).parent.mkdir(parents=True, exist_ok=True)
    save_global_cfg(cfg)
    return jsonify({"ok": True, "profile": normalized})


@app.post("/api/colab/session/start")
def api_colab_session_start():
    profile = _active_colab_profile()
    plog(f"[Colab 会话] 1/3 校验授权 profile={profile['id']} gpu={profile['gpu']}")
    if profile["auth"] == "oauth2" and not _colab_oauth_token_path(profile).is_file():
        return jsonify({"error": "当前 Google 账号尚未授权", "code": "colab_auth_required"}), 401
    if profile["auth"] == "adc" and not pathlib.Path(profile["credentials_path"]).is_file():
        return jsonify({"error": "当前账号的 ADC 凭据文件不存在"}), 400
    args = _colab_base_args(profile) + ["new", "-s", profile["session"], "--gpu", profile["gpu"]]
    try:
        plog(f"[Colab 会话] 2/3 请求创建 session={profile['session']}")
        r = subprocess.run(args, capture_output=True, text=True, timeout=180,
                           env=_colab_env(profile))
        if r.returncode:
            raw_error = (r.stderr or r.stdout or "").strip()
            plog(
                f"[Colab 会话启动失败] profile={profile['id']}, "
                f"gpu={profile['gpu']}, error={raw_error[-1200:]}"
            )
            lowered = raw_error.lower()
            if "toomanyassignmentserror" in lowered or \
                    ("precondition failed" in lowered and "412" in lowered):
                return jsonify({
                    "error": (
                        "当前 Google 账号仍占用一个旧的 Colab GPU 会话。"
                        "请先点击“停止”释放旧会话，等待片刻后再启动"
                    ),
                    "code": "colab_assignment_conflict",
                }), 409
            if "service unavailable" in lowered or "503" in lowered:
                return jsonify({
                    "error": "Google Colab 暂时无法分配 GPU，请稍后重试",
                    "code": "colab_service_unavailable",
                }), 503
            if "unauthorized" in lowered or "authentication" in lowered or "401" in lowered:
                return jsonify({
                    "error": "Google 授权已失效，请重新授权后再启动会话",
                    "code": "colab_auth_expired",
                }), 401
            return jsonify({
                "error": "Colab 会话创建失败，请查看系统日志了解详细原因",
                "code": "colab_start_failed",
            }), 500
        cfg, colab = _colab_profiles_cfg()
        current = next((p for p in colab["profiles"] if p.get("id") == colab["active"]), None)
        if current is not None:
            current["started_at"] = time.time()
            save_global_cfg(cfg)
        plog(f"[Colab 会话] 3/3 启动成功 session={profile['session']} gpu={profile['gpu']}")
        return jsonify({"ok": True, "session": profile["session"], "gpu": profile["gpu"],
                        "output": (r.stdout or "").strip()[-500:]})
    except subprocess.TimeoutExpired:
        return jsonify({
            "error": "创建 Colab 会话超时，Google 暂未返回结果，请稍后重试",
            "code": "colab_start_timeout",
        }), 504
    except Exception as exc:
        plog(f"[Colab 会话启动异常] profile={profile['id']}, error={type(exc).__name__}: {exc}")
        return jsonify({
            "error": "Colab 会话启动异常，请查看系统日志了解详细原因",
            "code": "colab_start_failed",
        }), 500


@app.post("/api/colab/session/stop")
def api_colab_session_stop():
    profile = _active_colab_profile()
    resolved_session = _colab_resolve_session(profile, force=True)
    try:
        plog(f"[Colab 会话] 1/2 请求停止 session={resolved_session}")
        r = subprocess.run(_colab_base_args(profile) + ["stop", "-s", resolved_session],
                           capture_output=True, text=True, timeout=60, env=_colab_env(profile))
        if r.returncode:
            return jsonify({"error": (r.stderr or r.stdout)[-800:]}), 500
        cfg, colab = _colab_profiles_cfg()
        current = next((p for p in colab["profiles"] if p.get("id") == colab["active"]), None)
        if current is not None:
            current.pop("started_at", None)
            save_global_cfg(cfg)
        plog(f"[Colab 会话] 2/2 已停止 session={resolved_session}")
        return jsonify({"ok": True})
    except Exception as exc:
        plog(f"[Colab 会话停止失败] session={profile['session']} error={type(exc).__name__}: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.get("/api/colab/session")
def api_colab_session_status():
    """返回当前配置档的会话状态与运行时长，不暴露会话 token。"""
    profile = _active_colab_profile()
    credentials_ready = (
        _colab_oauth_token_path(profile).is_file()
        if profile["auth"] == "oauth2"
        else bool(profile["credentials_path"]) and pathlib.Path(profile["credentials_path"]).is_file()
    )
    state_path = pathlib.Path(profile["session_config"])
    try:
        sessions = json.loads(state_path.read_text()) if state_path.exists() else {}
        session = sessions.get(profile["session"])
    except Exception:
        session = None
    resolved_session = profile["session"]
    if credentials_ready and not session:
        resolved_session = _colab_resolve_session(profile, force=True)
        if resolved_session != profile["session"]:
            session = {"accelerator": profile["gpu"], "variant": "GPU"}
    account_email = ""
    if credentials_ready:
        try:
            who = subprocess.run(_colab_base_args(profile) + ["whoami"], capture_output=True,
                                 text=True, timeout=10, env=_colab_env(profile))
            for line in (who.stdout or who.stderr).splitlines():
                if line.startswith("Email:"):
                    account_email = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    common = {
        "profile_id": profile["id"],
        "session": resolved_session,
        "gpu": profile["gpu"],
        "account_email": account_email or profile.get("email") or "",
        "credentials_ready": credentials_ready,
        "plan": None,
        "compute_units": None,
        "manage_url": "https://colab.research.google.com/signup",
    }
    expected_email = profile.get("email", "").strip().lower()
    actual_email = account_email.strip().lower()
    if expected_email and actual_email and expected_email != actual_email:
        plog(
            f"[Colab 账号不匹配] profile={profile['id']} "
            f"expected={expected_email} actual={actual_email}"
        )
        return jsonify({
            "running": False,
            "account_mismatch": True,
            "expected_email": profile.get("email", ""),
            **common,
        })
    # A session JSON file is only a local cache. Without this profile's own
    # credentials it may be stale and must never be presented as a live session.
    if not credentials_ready or not session:
        return jsonify({"running": False, **common})
    endpoint = session.get("endpoint", "")
    started_at = None
    history = state_path.parent / "history" / f"{profile['session']}.jsonl"
    if history.exists():
        try:
            for line in history.read_text().splitlines():
                event = json.loads(line)
                if event.get("event_type") == "session_created" and (
                        not endpoint or event.get("endpoint") == endpoint):
                    started_at = datetime.datetime.fromisoformat(
                        event["timestamp"].replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    if started_at is None:
        _, colab = _colab_profiles_cfg()
        current = next((p for p in colab["profiles"] if p.get("id") == colab["active"]), {})
        started_at = current.get("started_at")
    now = time.time()
    gpu_usage = {}
    try:
        rc, raw_usage = _colab_exec(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw "
            "--format=csv,noheader,nounits | head -1",
            timeout=20,
        )
        parts = [part.strip() for part in raw_usage.splitlines()[-1].split(",")]
        if rc == 0 and len(parts) >= 4:
            gpu_usage = {
                "utilization_percent": float(parts[0]),
                "memory_used_mb": float(parts[1]),
                "memory_total_mb": float(parts[2]),
                "power_watts": float(parts[3]),
            }
    except Exception:
        pass
    return jsonify({
        "running": True,
        **common,
        "gpu": session.get("accelerator") or profile["gpu"],
        "variant": session.get("variant", ""),
        "started_at": started_at,
        "runtime_seconds": max(0, int(now - started_at)) if started_at else None,
        "last_execution": (session.get("last_execution") or [None, None, None])[-1],
        "gpu_usage": gpu_usage,
    })

# 启动时自动部署 provision 脚本到 /content/（Colab 重启后目录丢失）
if pathlib.Path("/content").exists():
    PROV_SRC = SRC_DIR.parent / "scripts/provision"
    if PROV_SRC.exists():
        for key, cfg in ENGINES.items():
            dst = pathlib.Path(cfg["dir"])
            dst.mkdir(parents=True, exist_ok=True)
            src_dir = PROV_SRC / key
            if not src_dir.exists():
                continue
            import shutil as _sh
            # 部署目录下所有文件(provision.sh + cosyvoice_tts.py 等)
            for sf in src_dir.iterdir():
                if sf.is_file():
                    df = dst / sf.name
                    _sh.copy(sf, df)
                    if sf.suffix == ".sh":
                        df.chmod(0o755)

# 启动时预置 BGM（从项目脚本目录 seed 到 assets/bgm/，不覆盖已存在的）
_BGM_SEED = SRC_DIR.parent / "scripts/provision/bgm-prebuild"
if _BGM_SEED.exists():
    _bgm_dir = ROOT / "assets/bgm"
    _bgm_dir.mkdir(parents=True, exist_ok=True)
    for _sf in _BGM_SEED.glob("*.mp3"):
        _df = _bgm_dir / _sf.name
        if not _df.exists():
            import shutil as _sh
            _sh.copy(_sf, _df)


@app.get("/api/cicy-gpu/regions")
def cicy_gpu_regions():
    try:
        status, result = _cicy_gateway_request("GET", "/api/koubo/regions")
        return jsonify(result), status
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.get("/api/cicy-gpu/billing")
def cicy_gpu_billing():
    try:
        status, result = _cicy_gateway_request("GET", "/api/koubo/billing")
        return jsonify(result), status
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.get("/api/cicy-gpu/balance")
def cicy_gpu_balance():
    try:
        status, result = _cicy_gateway_request("GET", "/api/balance")
        return jsonify(result), status
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.post("/api/cicy-gpu/references/prepare")
def cicy_gpu_prepare_reference():
    payload = request.get_json(silent=True) or {}
    ref_id = str(payload.get("ref_id") or "")
    region_id = str(payload.get("region_id") or "cn-hangzhou")
    stt_provider = str(payload.get("stt_provider") or "groq")
    if ref_id and ref_id.startswith("voice-sample-") and (ROOT / "assets" / ref_id).is_file():
        ref = ROOT / "assets" / ref_id
    else:
        samples = sorted((ROOT / "assets").glob("voice-sample-*.wav"))
        if not samples:
            return jsonify({"error": "没有参考音色,请先选择/上传一段人声样本"}), 400
        ref = samples[-1]
    try:
        plog(f"[CiCy GPU 配音] 1/6 参考音频转写 · file={ref.name} bytes={ref.stat().st_size}")
        reference_text = _transcribe(str(ref), stt_provider)
    except GroqTranscriptionError as exc:
        return jsonify({"error": str(exc), "code": "groq_stt_failed"}), 502
    if not reference_text:
        return jsonify({"error": f"参考音频转写失败（{stt_provider}）"}), 400
    plog(f"[CiCy GPU 配音] 2/6 获取 OSS 上传签名 · region={region_id}")
    status, signed = _cicy_gateway_request("POST", "/api/koubo/assets/sign", {
        "region_id": region_id,
        "purpose": "reference",
        "content_type": "audio/wav",
        "extension": "wav",
    })
    if status >= 300:
        return jsonify({"error": signed.get("error") or "OSS 上传签名失败"}), status
    payload_bytes = ref.read_bytes()
    plog(f"[CiCy GPU 配音] 3/6 上传参考音频 · bytes={len(payload_bytes)}")
    opener = _public_direct_opener()
    upload = urllib.request.Request(
        signed["upload_url"], data=payload_bytes,
        headers={"Content-Type": "audio/wav", "User-Agent": "cicy-koubo/0.1.8"},
        method="PUT",
    )
    try:
        upload_started = time.monotonic()
        with opener.open(upload, timeout=30) as response:
            plog(f"[CiCy GPU 配音] 4/6 OSS 上传完成 · HTTP {response.status} · {time.monotonic()-upload_started:.2f}s")
    except urllib.error.URLError as exc:
        return jsonify({"error": f"OSS 上传失败：{exc.reason}"}), 502
    prepared_id = "ref_" + uuid.uuid4().hex
    _CICY_GPU_PREPARED_REFERENCES[prepared_id] = {
        "signed": signed,
        "reference_text": reference_text,
        "expires_at": time.time() + 12 * 60,
    }
    return jsonify({"success": True, "prepared_reference_id": prepared_id})


@app.route("/api/gpu-provider", methods=["GET", "POST"])
def gpu_provider():
    global GPU_MODE
    cfg = load_global_cfg()
    gpu = cfg.setdefault("koubo", {}).setdefault("gpu", {})
    available = ["cicy_gpu", "colab"]
    if _platform.system() != "Darwin" and _get_gpu_memory_mb() >= 8192:
        available.append("local")
    if request.method == "GET":
        return jsonify({
            "provider": gpu.get("provider") or GPU_MODE,
            "region_id": gpu.get("region_id") or "cn-hongkong",
            "available_providers": available,
            "local_provider_label": "WSL2 Docker + NVIDIA GPU" if _is_wsl() else "本地 NVIDIA Docker",
        })
    payload = request.get_json(silent=True) or {}
    provider = str(payload.get("provider") or "")
    if provider not in available:
        return jsonify({"success": False, "error": "invalid_gpu_provider"}), 400
    gpu["provider"] = provider
    GPU_MODE = "colab_cli" if provider == "colab" else provider
    try:
        status._cache = None
    except Exception:
        pass
    if payload.get("region_id"):
        gpu["region_id"] = str(payload["region_id"])
    save_global_cfg(cfg)
    return jsonify({
        "success": True,
        "provider": provider,
        "region_id": gpu.get("region_id") or "cn-hongkong",
        "restart_required": False,
    })


@app.get("/api/local-gpu/status")
def local_gpu_status_api():
    return jsonify(_local_gpu_docker_status())


@app.post("/api/local-gpu/install")
def local_gpu_install_api():
    if not _sht.which("docker"):
        return jsonify({"success": False, "error": "Docker CLI 或 docker.sock 不可用"}), 503
    root = _local_gpu_install_root()
    plog(f"[本地 GPU 安装] 1/3 选择最大可用磁盘目录 {root}")
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps({
            "root": str(root), "image": "cicy-koubo-gpu:2026.07.29-api",
            "port": 8771, "token": secrets.token_urlsafe(32),
            "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2), encoding="utf-8")
        config_path.chmod(0o600)
    image = "cicy-koubo-gpu:2026.07.29-api"
    plog(f"[本地 GPU 安装] 2/3 检查镜像 {image}")
    check = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    if check.returncode:
        pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=1800)
        if pull.returncode:
            return jsonify({"success": False, "error": (pull.stderr or "GPU 镜像下载失败")[-500:]}), 502
    plog("[本地 GPU 安装] 3/3 安装完成")
    return jsonify({"success": True, **_local_gpu_docker_status()})


@app.post("/api/local-gpu/start")
def local_gpu_start_api():
    state = _local_gpu_docker_status()
    if not state["installed"]:
        return jsonify({"success": False, "error": "请先安装本地 GPU"}), 409
    _, token = _local_gpu_conf()
    image = "cicy-koubo-gpu:2026.07.29-api"
    plog("[本地 GPU 启动] 1/4 验证 NVIDIA Container Runtime")
    probe = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "sh", image,
         "-lc", "command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null"],
        capture_output=True, text=True, timeout=60,
    )
    if probe.returncode:
        return jsonify({"success": False, "error": "NVIDIA GPU 或 Container Runtime 不可用"}), 503
    subprocess.run(["docker", "rm", "-f", "cicy-koubo-gpu"], capture_output=True)
    plog("[本地 GPU 启动] 2/4 清理旧容器并创建私有网络")
    subprocess.run(["docker", "network", "create", "cicy-koubo-local"], capture_output=True)
    subprocess.run(["docker", "network", "connect", "cicy-koubo-local",
                    os.environ.get("HOSTNAME", "")], capture_output=True)
    plog("[本地 GPU 启动] 3/4 启动 GPU API，端口 8771")
    run_result = subprocess.run([
        "docker", "run", "-d", "--name", "cicy-koubo-gpu", "--restart", "unless-stopped",
        "--network", "cicy-koubo-local", "--gpus", "all",
        "-p", "127.0.0.1:8771:8771", "-e", "KOUBO_API_PORT=8771",
        "-e", f"CICY_KOUBO_ACCESS_TOKEN={token}",
        "-v", f"{state['root']}/state:/var/lib/cicy-koubo-api", image,
    ], capture_output=True, text=True, timeout=120)
    if run_result.returncode:
        plog(f"[本地 GPU 启动失败] {run_result.stderr[-500:]}")
        return jsonify({"success": False, "error": run_result.stderr[-500:]}), 502
    plog("[本地 GPU 启动] 4/4 容器已启动，等待健康检查")
    return jsonify({"success": True, **_local_gpu_docker_status()})


@app.post("/api/local-gpu/stop")
def local_gpu_stop_api():
    plog("[本地 GPU 停止] 1/2 请求停止容器")
    subprocess.run(["docker", "stop", "cicy-koubo-gpu"], capture_output=True, timeout=60)
    plog("[本地 GPU 停止] 2/2 容器已停止，数据保留")
    return jsonify({"success": True, **_local_gpu_docker_status()})


@app.post("/api/cicy-gpu/jobs")
def cicy_gpu_create_job():
    payload = request.get_json(silent=True) or {}
    region_id = str(payload.get("region_id") or "")
    instance_type = str(payload.get("instance_type") or "")
    if not region_id:
        return jsonify({"success": False, "error": "region_id_required"}), 400
    if not instance_type:
        return jsonify({"success": False, "error": "instance_type_required"}), 400
    client_request_id = str(payload.get("client_request_id") or uuid.uuid4())
    try:
        status, result = _cicy_gateway_request(
            "POST", "/api/koubo/jobs",
            {
                "region_id": region_id,
                "instance_type": instance_type,
                "client_request_id": client_request_id,
                "input": payload.get("input") or {},
            },
            idempotency_key=client_request_id,
        )
        job_id = str(result.get("job_id") or (result.get("job") or {}).get("id") or "")
        authorization = result.pop("authorization_token", "")
        if status < 300 and job_id:
            _CICY_GPU_SESSIONS[job_id] = {
                "authorization_token": authorization,
                "expires_at": result.get("authorization_expires_at", ""),
                "region_id": region_id,
                "instance_type": instance_type,
            }
        return jsonify(result), status
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.get("/api/cicy-gpu/jobs/<job_id>")
def cicy_gpu_job(job_id):
    global _CICY_GPU_ACTIVE_JOB_ID
    try:
        safe_job_id = urllib.parse.quote(job_id, safe="")
        status, result = _cicy_gateway_request("GET", f"/api/koubo/jobs/{safe_job_id}")
        job = result.get("job") or {}
        authorization = str(result.pop("authorization_token", "") or "")
        if status < 300 and job.get("endpoint") and authorization:
            _CICY_GPU_SESSIONS[job_id] = {
                "authorization_token": authorization,
                "expires_at": job.get("authorization_expires_at", ""),
                "region_id": job.get("region_id") or "cn-hangzhou",
                "instance_type": job.get("instance_type") or "",
                "endpoint": job["endpoint"],
            }
            _CICY_GPU_ACTIVE_JOB_ID = job_id
        result["local_authorization_loaded"] = job_id in _CICY_GPU_SESSIONS
        return jsonify(result), status
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.post("/api/cicy-gpu/jobs/<job_id>/cancel")
def cicy_gpu_cancel_job(job_id):
    try:
        safe_job_id = urllib.parse.quote(job_id, safe="")
        status, result = _cicy_gateway_request(
            "POST", f"/api/koubo/jobs/{safe_job_id}/cancel", {},
        )
        if status < 300:
            _CICY_GPU_SESSIONS.pop(job_id, None)
        return jsonify(result), status
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.get("/api/system")
def api_system():
    import shutil
    info = {"gpu_name": "", "gpu_memory_mb": 0, "gpu_free_mb": 0,
            "cpu_cores": os.cpu_count() or 0,
            "ram_gb": 0, "disk_total_gb": 0, "disk_free_gb": 0,
            "python_version": "", "cuda_version": "", "ffmpeg_version": "",
            "gpu_mode": GPU_MODE,
            "local_gpu_managed": GPU_MODE == "local" and not LOCAL_GPU,
            "is_colab": pathlib.Path("/content").exists(),
            "colab_cli_installed": False, "colab_cli_version": ""}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        parts = (r.stdout or "").strip().split(",")
        if len(parts) >= 3:
            info["gpu_name"] = parts[0].strip()
            info["gpu_memory_mb"] = int(parts[1].strip())
            info["gpu_free_mb"] = int(parts[2].strip())
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        for line in (r.stdout or "").splitlines():
            if "CUDA Version:" in line:
                info["cuda_version"] = line.split("CUDA Version:")[-1].strip().split()[0]
    except Exception:
        pass
    mem = shutil.disk_usage("/")
    info["disk_total_gb"] = round(mem.total / (1024**3), 1)
    info["disk_free_gb"] = round(mem.free / (1024**3), 1)
    try:
        if pathlib.Path("/proc/meminfo").exists():
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal:" in line:
                        info["ram_gb"] = round(int(line.split()[1]) / (1024**2), 1)
                        break
        else:
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                info["ram_gb"] = round(int(r.stdout.strip()) / (1024**3), 1)
    except Exception:
        pass
    try:
        import sys as _sys
        r = subprocess.run([_sys.executable, "-c", "import sys;print(sys.version.split()[0],end='')"], capture_output=True, text=True, timeout=5)
        info["python_version"] = (r.stdout or r.stderr).strip()
        if not info["python_version"] and r.returncode != 0:
            print(f"[api/system] python version failed rc={r.returncode} stderr={r.stderr[:80]}", file=_sys.stderr, flush=True)
    except Exception:
        pass
    # ffmpeg
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        first_line = (r.stdout or "").splitlines()[0] if r.stdout else ""
        # "ffmpeg version 7.1.1 Copyright ..."
        info["ffmpeg_version"] = first_line.split()[2] if len(first_line.split()) >= 3 else first_line
    except Exception:
        pass
    # whisper: CosyVoice 环境自带
    info["whisper_local"] = ""
    if GPU_MODE == "colab_cli":
        # Remote engine readiness is loaded independently by /api/engines.
        # Never block local system information on a slow or stale Colab session.
        info["whisper_local"] = "随 CosyVoice 远程提供"
    else:
        cosy_python = pathlib.Path.home() / "cosyvoice-venv/bin/python"
        if cosy_python.exists():
            try:
                r = subprocess.run([str(cosy_python), "-c",
                                    "import whisper; print(whisper.__version__, end='')"],
                                   capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    info["whisper_local"] = r.stdout.strip() or "installed"
            except Exception:
                pass
    # colab cli
    try:
        colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
        if pathlib.Path(colab_bin).exists():
            info["colab_cli_installed"] = True
            # 从 dist-info 读版本(uv tools 或 pip)
            import glob as _glob
            for pat in [str(pathlib.Path.home() / ".local/share/uv/tools/*/lib/python*/site-packages/google_colab_cli-*.dist-info"),
                        str(pathlib.Path.home() / ".local/lib/python*/site-packages/google_colab_cli-*.dist-info")]:
                for d in _glob.glob(pat):
                    ver = d.split("google_colab_cli-")[1].split(".dist-info")[0]
                    info["colab_cli_version"] = ver
                    break
                if info["colab_cli_version"]:
                    break
    except Exception:
        pass
    return jsonify(info)


@app.post("/api/install/ffmpeg")
def install_ffmpeg():
    """自动检测平台并安装 ffmpeg"""
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            r = subprocess.run(["brew", "install", "ffmpeg"], capture_output=True, text=True, timeout=600)
        elif system == "Linux":
            # 尝试 apt (Ubuntu/Debian) → yum (CentOS/RHEL) → apk (Alpine)
            for pkg_mgr, install_cmd in [
                ("apt-get", ["sudo", "apt-get", "update", "-qq", "&&", "sudo", "apt-get", "install", "-y", "ffmpeg"]),
                ("yum", ["sudo", "yum", "install", "-y", "epel-release", "&&", "sudo", "yum", "install", "-y", "ffmpeg"]),
                ("apk", ["sudo", "apk", "add", "ffmpeg"]),
            ]:
                if pathlib.Path(f"/usr/bin/{pkg_mgr}").exists() or pathlib.Path(f"/usr/bin/{pkg_mgr}-get").exists():
                    r = subprocess.run(" ".join(install_cmd), shell=True, capture_output=True, text=True, timeout=600)
                    break
            else:
                return jsonify({"error": "未找到包管理器(apt/yum/apk)，请手动安装 ffmpeg"}), 500
        else:
            return jsonify({"error": f"不支持的平台: {system}"}), 500
        if r.returncode == 0:
            # 验证安装
            vr = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            ver = (vr.stdout or "").splitlines()[0].split()[2] if len((vr.stdout or "").splitlines()[0].split()) >= 3 else "unknown"
            return jsonify({"ok": True, "version": ver})
        else:
            return jsonify({"error": r.stderr[:500]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/install/colab-cli")
def install_colab_cli():
    """安装官方 google-colab-cli；优先使用隔离的 uv tool。"""
    try:
        plog("[Colab CLI 安装] 1/3 选择隔离安装工具")
        if _sht.which("uv"):
            cmd = ["uv", "tool", "install", "--force", "google-colab-cli"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "--user", "--upgrade", "google-colab-cli"]
        plog(f"[Colab CLI 安装] 2/3 执行 {' '.join(cmd[:3])}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
            vr = subprocess.run([colab_bin, "version"], capture_output=True, text=True, timeout=15)
            version = (vr.stdout or vr.stderr).strip().splitlines()[-1] if (vr.stdout or vr.stderr).strip() else "installed"
            plog(f"[Colab CLI 安装] 3/3 安装并验证成功 version={version}")
            return jsonify({"ok": True, "version": version})
        else:
            plog(f"[Colab CLI 安装失败] {(r.stderr or r.stdout)[-800:]}")
            return jsonify({"error": (r.stderr or r.stdout)[-800:],
                            "command": " ".join(cmd[:3])}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/engines")
def api_engines():
    out = []
    is_remote = GPU_MODE == "colab_cli"
    # 远程检测: colab cli 模式下通过 exec 检查 READY 文件
    remote_status = {}
    if is_remote:
        checks = []
        for key, cfg in ENGINES.items():
            ready_remote = cfg["ready"]
            script = f"/content/{key}/[p]rovision.sh"
            checks.append(
                f"echo __KOUBO_ENGINE_{key}_BEGIN__; "
                f"(test -f {shlex.quote(ready_remote)} && "
                f"{{ echo __KOUBO_INSTALLED_YES__; cat {shlex.quote(ready_remote)}; }} || true); "
                f"echo __KOUBO_INSTALLING__; "
                f"(pgrep -f {shlex.quote(script)} >/dev/null && echo YES || echo NO); "
                f"echo __KOUBO_ENGINE_{key}_END__"
            )
        # A single Colab round-trip avoids four sequential 30-second checks.
        _, batch_out = _colab_exec("; ".join(checks), timeout=30)
        for key in ENGINES:
            begin = f"__KOUBO_ENGINE_{key}_BEGIN__"
            end = f"__KOUBO_ENGINE_{key}_END__"
            segment = (
                batch_out.split(begin, 1)[1].split(end, 1)[0]
                if begin in batch_out and end in batch_out else ""
            )
            installed_marker = "__KOUBO_INSTALLED_YES__"
            installed = installed_marker in segment
            version_and_state = segment.split(installed_marker, 1)[1] if installed else segment
            before, _, after = version_and_state.rpartition("__KOUBO_INSTALLING__")
            remote_status[key] = {
                "installed": installed,
                "version_text": before.strip() if installed else "",
                "installing": after.strip().endswith("YES"),
            }
    for key, cfg in ENGINES.items():
        if is_remote:
            rs = remote_status.get(key, {})
            installed = rs.get("installed", False)
            version_info = {}
            if installed and rs.get("version_text"):
                for line in rs["version_text"].splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        version_info[k.strip()] = v.strip()
        else:
            ready = pathlib.Path(cfg["ready"])
            installed = ready.exists()
            version_info = {}
            if installed:
                try:
                    for line in ready.read_text().strip().splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            version_info[k.strip()] = v.strip()
                except Exception:
                    pass
        starting_at = _COLAB_ENGINE_STARTING.get(key, 0)
        starting = bool(starting_at and time.time() - starting_at < 300)
        installing = (
            remote_status.get(key, {}).get("installing", False) or starting
            if is_remote else _is_installing(key)
        )
        if is_remote and installing and not cfg.get("builtin", False):
            _start_colab_install_log_bridge(key, _active_colab_profile())
        # In Colab mode the source script lives in this app and is uploaded on
        # demand. Checking /content on the local Mac incorrectly reports every
        # remote engine as “missing install script”.
        bundled_script = SRC_DIR.parent / "scripts/provision" / key / "provision.sh"
        deployed_script = pathlib.Path(cfg["dir"]) / "provision.sh"
        has_script = cfg.get("builtin", False) or bundled_script.is_file() or deployed_script.is_file()
        out.append({"key": key, "name": cfg["name"],
                    "installed": installed, "installing": installing,
                    "version_info": version_info, "has_script": has_script,
                    "remote": is_remote, "builtin": cfg.get("builtin", False),
                    "install_hint": cfg.get("install_hint", ""),
                    "note": cfg.get("note", "")})
    return jsonify({"engines": out})


@app.get("/api/engines/<engine>/log")
def api_engine_log(engine):
    if engine not in ENGINES:
        return jsonify({"error": "unknown engine"}), 404
    logf = pathlib.Path(ENGINES[engine]["dir"]) / "provision.log"
    tail = request.args.get("tail", 200, type=int)
    # colab cli 模式: 远程读日志
    if GPU_MODE == "colab_cli":
        script = f"/content/{engine}/[p]rovision.sh"
        rc, out = _colab_exec(
            f"tail -n {max(20, min(tail, 1000))} {logf} 2>/dev/null || true; "
            f"echo __KOUBO_INSTALLING__; "
            f"(pgrep -f '{script}' >/dev/null && echo YES || echo NO)",
            timeout=30,
        )
        text, _, state = out.rpartition("__KOUBO_INSTALLING__")
        return jsonify({"lines": text.splitlines(), "installing": state.strip().endswith("YES"), "remote": True})
    if not logf.exists():
        return jsonify({"lines": [], "installing": False})
    try:
        lines = logf.read_text().splitlines()[-tail:]
    except Exception:
        lines = []
    installing = _is_installing(engine)
    return jsonify({"lines": lines, "installing": installing})


@app.get("/api/engines/<engine>/stream")
def api_engine_stream(engine):
    """SSE 实时日志流——安装进度实时推送，不靠轮询。"""
    if engine not in ENGINES:
        return jsonify({"error": "unknown engine"}), 404
    logf = pathlib.Path(ENGINES[engine]["dir"]) / "provision.log"
    def generate():
        if GPU_MODE == "colab_cli":
            previous = None
            stopped_checks = 0
            import time as _time
            for _ in range(1800):  # 最长约 2 小时，每次远程读取后等待
                script = f"/content/{engine}/[p]rovision.sh"
                try:
                    rc, out = _colab_exec(
                        f"tail -c 30000 {logf} 2>/dev/null || true; "
                        f"echo __KOUBO_INSTALLING__; "
                        f"(pgrep -f '{script}' >/dev/null && echo YES || echo NO)",
                        timeout=20,
                    )
                    text, _, state = out.rpartition("__KOUBO_INSTALLING__")
                    text = text.replace("\r", "\n")
                    if text != previous:
                        yield f"data: {json.dumps({'replace': text, 'remote': True})}\n\n"
                        previous = text
                    else:
                        yield ": keepalive\n\n"
                    if state.strip().endswith("YES"):
                        stopped_checks = 0
                    else:
                        stopped_checks += 1
                    # Remote shells can briefly miss the process while bash
                    # replaces a child command. Require three consecutive
                    # stopped observations before ending the stream.
                    if stopped_checks >= 3:
                        yield f"data: {json.dumps({'done': True, 'remote': True})}\n\n"
                        return
                except Exception as exc:
                    yield f"data: {json.dumps({'warning': '远程日志暂时不可用，正在重试'})}\n\n"
                _time.sleep(2)
            return
        # 先推送已有日志
        if logf.exists():
            try:
                existing = logf.read_text().replace("\r", "\n")
                if existing.strip():
                    yield f"data: {json.dumps({'text': existing})}\n\n"
            except Exception:
                pass
        pos = logf.stat().st_size if logf.exists() else 0
        import time as _time
        for _ in range(7200):  # max 2h
            try:
                if logf.exists():
                    st = logf.stat()
                    if st.st_size > pos:
                        with open(str(logf), "r") as f:
                            f.seek(pos)
                            chunk = f.read()
                            pos = f.tell()
                        # \r → \n so progress bars don't overwrite
                        chunk = chunk.replace("\r", "\n")
                        yield f"data: {json.dumps({'text': chunk})}\n\n"
            except Exception:
                pass
            # check if still installing
            if not _is_installing(engine):
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            _time.sleep(0.8)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/engines/<engine>/install")
def api_engine_install(engine):
    if engine not in ENGINES:
        return jsonify({"error": "unknown engine"}), 404
    cfg = ENGINES[engine]
    src_dir = SRC_DIR.parent / "scripts/provision" / engine
    if not src_dir.exists():
        return jsonify({"error": f"{cfg['name']} 安装脚本目录不存在: {src_dir}"}), 404
    # colab cli 模式: 上传脚本到远程再执行
    if GPU_MODE == "colab_cli":
        profile = _active_colab_profile()
        task_key = (profile["id"], profile["session"], engine)
        with _COLAB_ENGINE_INSTALL_LOCK:
            if task_key in _COLAB_ENGINE_INSTALL_INFLIGHT:
                return jsonify({"ok": True, "already_running": True, "engine": engine}), 202
            _COLAB_ENGINE_INSTALL_INFLIGHT.add(task_key)
            _COLAB_ENGINE_STARTING[engine] = time.time()
        remote_dir = cfg["dir"]
        remote_log = f"{remote_dir}/provision.log"
        check_rc, check_out = _colab_exec(
            f"if test -f {shlex.quote(cfg['ready'])}; then echo READY; fi; "
            f"if pgrep -f {shlex.quote(f'{remote_dir}/[p]rovision.sh')} >/dev/null; "
            f"then echo RUNNING; fi",
            timeout=20,
            profile=profile,
        )
        if "READY" in check_out:
            _COLAB_ENGINE_INSTALL_INFLIGHT.discard(task_key)
            _COLAB_ENGINE_STARTING.pop(engine, None)
            return jsonify({"ok": True, "already_installed": True, "engine": engine})
        if "RUNNING" in check_out:
            _COLAB_ENGINE_INSTALL_INFLIGHT.discard(task_key)
            _COLAB_ENGINE_STARTING.pop(engine, None)
            _start_colab_install_log_bridge(engine, profile)
            return jsonify({"ok": True, "already_running": True, "engine": engine}), 202
        plog(f"[Colab 安装][{cfg['name']}] 1/3 创建远程目录")
        mkdir_rc, mkdir_out = _colab_exec(
            f"mkdir -p {shlex.quote(remote_dir)}",
            timeout=30,
        )
        if mkdir_rc != 0:
            _COLAB_ENGINE_INSTALL_INFLIGHT.discard(task_key)
            _COLAB_ENGINE_STARTING.pop(engine, None)
            return jsonify({
                "error": f"创建 Colab 安装目录失败：{mkdir_out[-300:] or remote_dir}"
            }), 500
        _colab_exec(
            f"printf '\\n[%s] 开始准备 {shlex.quote(cfg['name'])} 安装环境\\n' "
            f"\"$(date '+%H:%M:%S')\" >> {shlex.quote(remote_log)}",
            timeout=30,
        )
        # 上传整个目录
        upload_files = [sf for sf in src_dir.iterdir() if sf.is_file()]
        for index, sf in enumerate(upload_files, 1):
            plog(
                f"[Colab 安装][{cfg['name']}] 2/3 上传文件 "
                f"{index}/{len(upload_files)}: {sf.name}"
            )
            _colab_exec(
                f"printf '[%s] 上传文件 {index}/{len(upload_files)}: %s ... ' "
                f"\"$(date '+%H:%M:%S')\" {shlex.quote(sf.name)} >> {shlex.quote(remote_log)}",
                timeout=30,
            )
            if sf.is_file():
                ok = _colab_upload(sf, f"{remote_dir}/{sf.name}")
                if not ok:
                    _colab_exec(
                        f"printf '失败\\n' >> {shlex.quote(remote_log)}",
                        timeout=30,
                    )
                    _COLAB_ENGINE_INSTALL_INFLIGHT.discard(task_key)
                    _COLAB_ENGINE_STARTING.pop(engine, None)
                    return jsonify({"error": f"上传 {sf.name} 到 Colab 失败"}), 500
                _colab_exec(
                    f"printf '完成\\n' >> {shlex.quote(remote_log)}",
                    timeout=30,
                )
        # 远程执行 provision.sh
        script = f"{remote_dir}/provision.sh"
        plog(f"[Colab 安装][{cfg['name']}] 3/3 启动远程安装")
        _colab_exec(
            f"printf '[%s] 文件上传完成，启动安装脚本\\n' "
            f"\"$(date '+%H:%M:%S')\" >> {shlex.quote(remote_log)}",
            timeout=30,
        )
        rc, out = _colab_exec(
            f"chmod +x {shlex.quote(script)} && "
            f"nohup bash {shlex.quote(script)} >> {shlex.quote(remote_log)} 2>&1 &",
            timeout=30,
        )
        if rc != 0:
            _colab_exec(
                f"printf '[%s] 启动安装脚本失败\\n' "
                f"\"$(date '+%H:%M:%S')\" >> {shlex.quote(remote_log)}",
                timeout=30,
            )
            _COLAB_ENGINE_INSTALL_INFLIGHT.discard(task_key)
            _COLAB_ENGINE_STARTING.pop(engine, None)
            return jsonify({"error": f"启动 {cfg['name']} 安装失败：{out[-300:]}"}), 500
        _start_colab_install_log_bridge(engine, profile)
        _COLAB_ENGINE_INSTALL_INFLIGHT.discard(task_key)
        return jsonify({"ok": True, "started": True, "remote": True, "log_path": f"{remote_dir}/provision.log"})
    # 本机模式
    script = pathlib.Path(cfg["dir"]) / "provision.sh"
    if not script.exists():
        return jsonify({"error": f"{cfg['name']} 安装脚本不存在 — 请先准备 {script}"}), 404
    if _is_installing(engine):
        return jsonify({"ok": True, "already_running": True, "log_path": str(script.parent / "provision.log")})
    logf = script.parent / "provision.log"
    if not logf.exists():
        logf.write_text(f"=== install started at {time.strftime('%Y-%m-%dT%H:%M:%SZ')} ===\n")
    with open(str(logf), "a") as f:
        subprocess.Popen(["nohup", "bash", str(script)],
                         stdout=f, stderr=f,
                         start_new_session=True, cwd=str(script.parent))
    return jsonify({"ok": True, "started": True, "log_path": str(logf)})


@app.get("/")
def index():
    return Response((SRC_DIR / "index.html").read_text(encoding="utf-8"), mimetype="text/html")


if __name__ == "__main__":
    print(f"爆款口播视频制作 → http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
