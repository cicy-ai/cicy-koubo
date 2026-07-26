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
import os
import pathlib
import shlex
import subprocess
import threading
import time
import uuid

from flask import Flask, request, jsonify, send_file, Response

ROOT = pathlib.Path.home() / "projects/digital-human"
APP_DIR = ROOT / "kr-app"
WORK = APP_DIR / "jobs"
WORK.mkdir(parents=True, exist_ok=True)
COSY_VENV = ROOT.parent / "cosyvoice-venv/bin/python"
COSY_MODEL = ROOT.parent / "CosyVoice/pretrained_models/CosyVoice2-0.5B/llm.pt"

PORT = int(os.environ.get("KOUBO_PORT", "8770"))
SRC_DIR = pathlib.Path(__file__).resolve().parent   # 代码目录(index.html 随代码走,数据仍在 ROOT)
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
        return Response("".join(lines), mimetype="text/plain; charset=utf-8")
    except Exception:
        return Response("(暂无日志)", mimetype="text/plain; charset=utf-8")


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
    # 隧道通不通:能 SSH 上就算 up;READY 在则 ready
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
    tstate, tnote = tunnel_status()
    audio = "ready" if (tstate == "ready" and _cosy_ready_remote()) else \
            ("installing" if tstate == "ready" else "down")
    return jsonify({
        "tunnel": tstate, "tunnel_note": tnote,
        "audio_service": audio,
        "video_service": "ready" if tstate == "ready" else "down",
    })


def _cosy_ready_remote():
    return _ssh_check("cat /content/cosy/COSY_READY 2>/dev/null && echo OK", "OK")


def do_generate(job_id, video_path, audio_path, bbox, opts=None):
    opts = opts or {}
    j = JOBS[job_id]
    def log(m):
        j["log"].append(m)
        plog(f"[出片 {job_id}] {m}")
    try:
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
            j["stage"] = "上传素材到 GPU"
            log("scp base + audio → Colab")
            r = run(SCP + [str(norm), str(audio_path), REMOTE + ":/content/"], timeout=180)
            if r.returncode != 0:
                raise RuntimeError("scp upload failed: " + r.stderr[:300])

            engine = opts.get("engine") or "musetalk"
            j["stage"] = ("HeyGem" if engine == "heygem" else "MuseTalk") + " 对口型(数分钟)"
            log(f"run {engine} on GPU")
            rv, ra = norm.name, pathlib.Path(audio_path).name
            out_remote = f"/content/out_{job_id}.mp4"
            if engine == "heygem":
                cmd = f"bash /content/hg/synthesize.sh /content/{shlex.quote(rv)} /content/{shlex.quote(ra)} {out_remote}"
            else:
                cmd = f"bash /content/mt/synthesize.sh /content/{shlex.quote(rv)} /content/{shlex.quote(ra)} {out_remote} {int(bbox)}"
            r = run(SSH + [cmd], timeout=1800)
            log(r.stdout.strip()[-400:] if r.stdout else "")
            if r.returncode != 0 or "OK out=" not in (r.stdout or ""):
                raise RuntimeError("MuseTalk failed: " + (r.stderr or r.stdout)[-400:])

            j["stage"] = "取回成片"
            r = run(SCP + [REMOTE + ":" + out_remote, str(result)], timeout=180)
            if r.returncode != 0 or not result.exists():
                raise RuntimeError("scp download failed: " + r.stderr[:300])

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
    if mode != "simple" and engine == "heygem" and \
            not _ssh_check("test -f /content/hg/HG_READY && echo OK", "OK"):
        return jsonify({"error": "HeyGem 引擎未安装:在 Colab notebook 里运行「可选:安装 HeyGem」cell(需 ≥16GB 显存)"}), 503
    opts = {"sharp": request.form.get("sharp") == "1",       # 牙齿高清=成片锐化
            "randstart": request.form.get("randstart") == "1",  # 动作随机=底板随机起点
            "mode": mode, "engine": engine}
    JOBS[job_id] = {"stage": "排队", "log": [], "result": None, "error": None}
    threading.Thread(target=do_generate, args=(job_id, vp, ap, bbox, opts), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/job/<job_id>")
def job_status(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"stage": j["stage"], "log": j["log"][-6:],
                    "result": j["result"], "error": j["error"]})


@app.get("/api/result/<job_id>")
def result(job_id):
    f = WORK / job_id / "result.mp4"
    if not f.exists():
        return jsonify({"error": "no result"}), 404
    return send_file(f, mimetype="video/mp4")


@app.get("/api/cover/<job_id>")
def cover(job_id):
    f = WORK / job_id / "cover.jpg"
    return send_file(f, mimetype="image/jpeg") if f.exists() else ("", 404)


def _groq_transcribe(path):
    gk = _groq_key()
    if not gk:
        return ""
    proxy = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))
    cmd = ["curl", "-s", "https://api.groq.com/openai/v1/audio/transcriptions",
           "-H", f"Authorization: Bearer {gk}", "-F", f"file=@{path}",
           "-F", "model=whisper-large-v3-turbo", "-F", "language=zh"]
    if proxy:
        cmd = cmd[:1] + ["--proxy", proxy] + cmd[1:]
    try:
        r = run(cmd, timeout=120)
        return json.loads(r.stdout).get("text", "").strip()
    except Exception:
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
    """文案 + 参考音色 → CosyVoice zero-shot 克隆配音(本机 GPU 直接跑)。"""
    text = ""
    if request.is_json:
        text = (request.json or {}).get("text", "").strip()
    else:
        text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "没有要配音的文案"}), 400
    try:
        speed = max(0.5, min(1.5, float(request.form.get("speed", "1.0"))))
    except ValueError:
        speed = 1.0

    jid = uuid.uuid4().hex[:10]
    whole = request.form.get("mode") == "whole"
    plog(f"[配音 {jid}] 开始: {len(text)}字 语速{speed} 模式{'整段' if whole else '分段'}")

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
    ref_trim = WORK / f"ref_trim_{jid}.wav"
    tr = run(["ffmpeg", "-v", "error", "-y", "-i", str(ref), "-t", "10",
              "-ar", "16000", "-ac", "1", str(ref_trim)])
    if tr.returncode == 0 and ref_trim.exists():
        ref = ref_trim
    ref_text = _groq_transcribe(str(ref))

    # 本机直接跑 CosyVoice
    cosy_dir = pathlib.Path("/content/cosy")
    cosy_py = cosy_dir / "env/bin/python"
    cosy_script = cosy_dir / "cosyvoice_tts.py"
    if not cosy_script.exists():
        return jsonify({"error": "CosyVoice 未安装,请先在安装管理中安装"}), 503

    import base64
    t_b64 = base64.b64encode(text.encode()).decode()
    rt_b64 = base64.b64encode((ref_text or "参考声音").encode()).decode()

    out_wav = cosy_dir / f"tts_{jid}.wav"
    done_file = cosy_dir / f"tts_{jid}.done"
    log_file = cosy_dir / f"tts_{jid}.log"

    # 拷贝 ref 到 cosy 目录
    ref_dst = cosy_dir / f"ref_{jid}.wav"
    import shutil as _sh
    _sh.copy(ref, ref_dst)

    launch = (f"export MPLBACKEND=Agg LD_LIBRARY_PATH=/usr/lib64-nvidia; "
              f"cd {cosy_dir} && nohup {cosy_py} {cosy_script} "
              f"--ref {ref_dst} --ref-text-b64 {rt_b64} "
              f"--text-b64 {t_b64} --speed {speed}{' --whole' if whole else ''} --out {out_wav} "
              f"> {log_file} 2>&1; echo done > {done_file} &")
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
        "fontsize": int(request.form.get("fontsize") or 0),
        "color": request.form.get("color") or "#FFFFFF",
        "outline": request.form.get("outline") or "#000000",
        "mb": int(request.form.get("mb") or 60),
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
        amap = ["-map", "0:a?"]
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
               ["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-shortest", str(out)])
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
    "heavy": (str(ROOT / "assets/fonts/SourceHanSansCN-Heavy.otf"), "思源黑体·特粗(抖音风,推荐)"),
    "bold": (str(ROOT / "assets/fonts/SourceHanSansCN-Bold.otf"), "思源黑体·粗"),
    "heiti": ("/System/Library/Fonts/STHeiti Medium.ttc", "系统黑体"),
    "pingfang": ("/System/Library/Fonts/PingFang.ttc", "苹方"),
    "songti": ("/System/Library/Fonts/Supplemental/Songti.ttc", "宋体"),
}


def _font_path(fid):
    p = FONTS.get(fid or "heavy", FONTS["heavy"])[0]
    if os.path.exists(p):
        return p
    for c in (FONTS["heiti"][0],                                   # macOS
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",   # Linux/Colab
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(c):
            return c
    return p


@app.get("/api/fonts")
def fonts():
    return jsonify([{"id": k, "name": v[1]} for k, v in FONTS.items() if os.path.exists(v[0])])


def _hex_rgba(s, default):
    s = (s or "").lstrip("#")
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    except Exception:
        return default


def _render_caption(text, w, h, png_path, fontsize=0, color="#FFFFFF", outline="#000000", mb=60, font_id="heavy"):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fs = fontsize if fontsize and fontsize >= 16 else max(28, int(w * 0.06))
    fg = _hex_rgba(color, (255, 255, 255, 255))
    og = _hex_rgba(outline, (0, 0, 0, 255))
    font = ImageFont.truetype(_font_path(font_id), fs)
    maxw = w * 0.86
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=font) > maxw and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    lh = fs + 12
    y = h - lh * len(lines) - max(10, mb)
    for ln in lines:
        tw = d.textlength(ln, font=font)
        x = (w - tw) / 2
        for dx in (-2, 2):
            for dy in (-2, 2):
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
            fs = max(48, int(w * 0.09))
            font = ImageFont.truetype(_font_path("heavy"), fs)
            maxw = w * 0.88
            lines, cur = [], ""
            for ch in title:
                if d.textlength(cur + ch, font=font) > maxw and cur:
                    lines.append(cur); cur = ch
                else:
                    cur += ch
            if cur:
                lines.append(cur)
            lh = fs + 16
            y = h * 0.32 - lh * len(lines) / 2
            for ln in lines:
                x = (w - d.textlength(ln, font=font)) / 2
                for dx in (-3, 3):
                    for dy in (-3, 3):
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
    """音频 → Groq whisper(verbose_json)→ SRT 文本(逐句时间轴)。"""
    if "audio" not in request.files:
        return jsonify({"error": "缺音频"}), 400
    tmp = WORK / (uuid.uuid4().hex[:8] + ".wav")
    request.files["audio"].save(tmp)
    gk = _groq_key()
    if not gk:
        return jsonify({"error": "缺转写凭据"}), 500
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

        def _ts(t):
            h, rem = divmod(t, 3600)
            mnt, s = divmod(rem, 60)
            return f"{int(h):02d}:{int(mnt):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"

        lines = []
        for i, s in enumerate(segs, 1):
            lines.append(f"{i}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{s['text'].strip()}\n")
        return jsonify({"srt": "\n".join(lines), "segments": len(segs)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500
    finally:
        tmp.unlink(missing_ok=True)


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


def _openai_provider():
    """从 global.json 找第一个有 apiKey 的 openai/anthropic 协议 provider（跳过 voice/stt）。"""
    try:
        gj = json.load(open(pathlib.Path.home() / "cicy-ai/global.json"))
        defs = (gj.get("providers", {}).get("default") or {})
        items = gj.get("providers", {}).get("items") or []
        # 按默认 provider 顺序优先，然后补其余的
        order = [v for v in defs.values() if isinstance(v, str) and v not in ("groqStt","doubaoVoice","zhipuVision")]
        ordered = sorted(items, key=lambda x: order.index(x["key"]) if x["key"] in order else 999)
        for p in ordered:
            k = (p.get("apiKey") or "").strip()
            proto = (p.get("protocol") or "").lower()
            if k and proto in ("openai", "anthropic"):
                return p
    except Exception:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return {"url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
                "apiKey": os.environ["OPENAI_API_KEY"],
                "defaultModel": os.environ.get("OPENAI_MODEL", "gpt-4o-mini")}
    return None


def _chat(messages, model=None, max_tokens=1200, timeout=90):
    import urllib.request
    p = _openai_provider()
    if not p:
        raise RuntimeError("global.json 缺 defaultOpenAi")
    mdl = model or "deepseek-v4-flash"
    body = json.dumps({"model": mdl, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.9}).encode()
    req = urllib.request.Request(p["url"].rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + p["apiKey"],
                                          "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["choices"][0]["message"].get("content", "").strip()


DEFAULT_REWRITE_PROMPT = (
    "你是抖音爆款口播文案专家。把用户给的对标文案仿写成一条全新的口播稿:"
    "保留原文的钩子结构和节奏,但更换具体行业/场景/数字/案例,做到不搬运、可直接口播。"
    "要求:开头3秒强钩子抓停留;中间给足干货或情绪;结尾引导关注。"
    "只输出改写后的正文,不要解释、不要小标题、不要序号。"
)


@app.get("/api/rewrite-prompt")
def rewrite_prompt_default():
    return jsonify({"prompt": DEFAULT_REWRITE_PROMPT})


@app.post("/api/rewrite")
def rewrite():
    body = request.json or {}
    src = (body.get("text") or "").strip()
    style = (body.get("style") or "").strip()
    if not src:
        return jsonify({"error": "没有可改写的文案"}), 400
    sys = (body.get("system") or "").strip() or DEFAULT_REWRITE_PROMPT
    user = (f"对标文案:\n{src}\n\n" + (f"改写风格/行业要求:{style}\n\n" if style else "") +
            "请输出仿写后的口播文案。")
    try:
        out = _chat([{"role": "system", "content": sys}, {"role": "user", "content": user}])
        return jsonify({"text": out})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "改写失败: " + str(e)}), 500


@app.post("/api/title")
def title():
    src = ((request.json or {}).get("text") or "").strip()
    if not src:
        return jsonify({"error": "没有文案"}), 400
    try:
        out = _chat([{"role": "user", "content":
                      "根据这段口播文案,给出3个抖音爆款标题(带1-2个话题#)和一句封面文案。"
                      "直接列出,不要解释。\n\n" + src}], max_tokens=400)
        return jsonify({"text": out})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


def _groq_key():
    """优先用 groqStt 的 key，没有则回退到 _openai_provider 的 key。"""
    try:
        gj = json.load(open(pathlib.Path.home() / "cicy-ai/global.json"))
        for x in (gj.get("providers", {}).get("items") or []):
            if x.get("key") == "groqStt":
                k = (x.get("apiKey") or "").strip()
                if k:
                    return k
    except Exception:
        pass
    # 回退：用任意有效 openai provider 的 key
    p = _openai_provider()
    if p:
        return p.get("apiKey", "")
    return os.environ.get("GROQ_API_KEY", "")


@app.post("/api/extract")
def extract():
    """抖音链接 → 下载音频 → Groq 转写 → 返回文案。"""
    url = (request.json or {}).get("url", "").strip()
    force = (request.json or {}).get("force", False)
    if "douyin.com" not in url:
        return jsonify({"error": "请粘贴抖音分享链接(含 v.douyin.com)"}), 400
    jd = WORK / ("dy_" + uuid.uuid4().hex[:8])
    jd.mkdir(exist_ok=True)
    try:
        # 第一层:按链接字符串直接查缓存,零网络秒回(短链解析要 7s,能免则免)
        import re
        code = ""
        cm = re.search(r"douyin\.com/(?:video/)?([A-Za-z0-9_]+)", url)
        if cm:
            code = cm.group(1)
        if not force:
            for e in _load_scripts():
                eu = e.get("url", "")
                if url == eu or (code and code in eu) or (code and code == e.get("id")):
                    return jsonify({"text": e["text"], "audio": e.get("audio"), "cached": True})
        # 第二层:解析短链拿数字ID再查(处理换了短链但同一视频的情况)
        if not force:
            pre_vid = _resolve_vid(url)
            if pre_vid:
                for e in _load_scripts():
                    if e.get("id") == pre_vid:
                        return jsonify({"text": e["text"], "audio": e.get("audio"), "cached": True})
        dl = _douyin_dl()
        if not dl:
            return jsonify({"error": "本机没有 douyin-dl 提取组件(设 KOUBO_DOUYIN_DL 指向脚本,或升级 cicy-koubo)"}), 500
        dlcmd = [str(dl)] if os.access(dl, os.X_OK) else ["node", str(dl)]
        r = run(dlcmd + [url, "-o", str(jd)], timeout=180)
        media, vid = "", ""
        for line in (r.stdout or "").splitlines():
            if line.startswith("media="):
                media = line.split("=", 1)[1].strip()
                vid = pathlib.Path(media).stem.replace("dy_", "")
        # 命中缓存(同一视频)直接返回,除非 force;下载失败也回退到缓存
        if vid and not force:
            for e in _load_scripts():
                if e.get("id") == vid:
                    return jsonify({"text": e["text"], "audio": e.get("audio"), "cached": True})
        if not media or not os.path.exists(media):
            for e in _load_scripts():  # 下载失败但曾缓存过 → 用缓存
                if url in (e.get("url") or "") or (vid and e.get("id") == vid):
                    return jsonify({"text": e["text"], "cached": True, "note": "下载失败,返回历史缓存"})
            return jsonify({"error": "下载失败(抖音风控),稍后重试。日志:" + (r.stdout or r.stderr)[-200:]}), 502
        gk = _groq_key()
        if not gk:
            return jsonify({"error": "缺 Groq 转写凭据"}), 500
        proxy = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))
        cmd = ["curl", "-s", "https://api.groq.com/openai/v1/audio/transcriptions",
               "-H", f"Authorization: Bearer {gk}", "-F", f"file=@{media}",
               "-F", "model=whisper-large-v3-turbo", "-F", "language=zh"]
        if proxy:
            cmd = cmd[:1] + ["--proxy", proxy] + cmd[1:]
        tr = run(cmd, timeout=180)
        text = json.loads(tr.stdout).get("text", "").strip()
        _save_script({"id": vid, "url": url, "text": text, "audio": media,
                      "ts": time.strftime("%Y-%m-%d %H:%M")})
        return jsonify({"text": text, "audio": media, "cached": False})
    except Exception as e:  # noqa: BLE001
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


@app.post("/api/voices/upload")
def voice_upload():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "缺文件"}), 400
    (ROOT / "assets").mkdir(parents=True, exist_ok=True)
    raw = WORK / (uuid.uuid4().hex[:8] + "_vup" + pathlib.Path(request.files["file"].filename).suffix)
    request.files["file"].save(raw)
    vid = _next_free("voice", "voice-sample-{:02d}.wav")
    dst = ROOT / "assets" / vid
    r = run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ar", "16000", "-ac", "1", str(dst)], timeout=120)
    raw.unlink(missing_ok=True)
    if r.returncode != 0 or not dst.exists():
        return jsonify({"error": "音频转换失败: " + r.stderr[:200]}), 400
    name = (request.form.get("name") or "").strip()
    if name:
        _meta_set("voice", vid, name)
    return jsonify({"id": vid, "name": name or vid, "duration": _ffdur(dst)})


@app.post("/api/voices/delete")
def voice_delete():
    vid = (request.json or {}).get("id") or ""
    if not vid.startswith("voice-sample-") or ".." in vid:
        return jsonify({"error": "bad id"}), 400
    (ROOT / "assets" / vid).unlink(missing_ok=True)
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
    try:
        r = subprocess.run(["pgrep", "-f", f"/content/{key}/provision.sh"],
                           capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# ═══════════ 安装管理 ═══════════
ENGINES = {
    "hg":       {"name": "HeyGem",    "ready": "/content/hg/HG_READY",     "dir": "/content/hg"},
    "mt":       {"name": "MuseTalk",  "ready": "/content/mt/READY",        "dir": "/content/mt"},
    "cosy":     {"name": "CosyVoice", "ready": "/content/cosy/COSY_READY", "dir": "/content/cosy"},
}

# 启动时自动部署 provision 脚本到 /content/（Colab 重启后目录丢失）
if pathlib.Path("/content").exists():
    PROV_SRC = SRC_DIR.parent / "scripts" / "provision"
    if PROV_SRC.exists():
        for key, cfg in ENGINES.items():
            dst = pathlib.Path(cfg["dir"])
            dst.mkdir(parents=True, exist_ok=True)
            src_script = PROV_SRC / key / "provision.sh"
            dst_script = dst / "provision.sh"
            if src_script.exists():
                import shutil as _sh
                _sh.copy(src_script, dst_script)
                dst_script.chmod(0o755)


@app.get("/api/system")
def api_system():
    import shutil
    info = {"gpu_name": "", "gpu_memory_mb": 0, "gpu_free_mb": 0,
            "cpu_cores": os.cpu_count() or 0,
            "ram_gb": 0, "disk_total_gb": 0, "disk_free_gb": 0,
            "python_version": "", "cuda_version": "",
            "is_colab": pathlib.Path("/content").exists()}
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
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal:" in line:
                    info["ram_gb"] = round(int(line.split()[1]) / (1024**2), 1)
                    break
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
    return jsonify(info)


@app.get("/api/engines")
def api_engines():
    out = []
    for key, cfg in ENGINES.items():
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
        installing = _is_installing(key)
        has_script = (pathlib.Path(cfg["dir"]) / "provision.sh").exists()
        out.append({"key": key, "name": cfg["name"],
                    "installed": installed, "installing": installing,
                    "version_info": version_info, "has_script": has_script})
    return jsonify({"engines": out})


@app.get("/api/engines/<engine>/log")
def api_engine_log(engine):
    if engine not in ENGINES:
        return jsonify({"error": "unknown engine"}), 404
    logf = pathlib.Path(ENGINES[engine]["dir"]) / "provision.log"
    if not logf.exists():
        return jsonify({"lines": [], "installing": False})
    tail = request.args.get("tail", 200, type=int)
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
