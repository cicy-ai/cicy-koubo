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
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid

from flask import Flask, request, jsonify, send_file, Response

_edit_lock = threading.Lock()

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

def detect_gpu_mode():
    """自动检测 GPU 模式:
      macOS → colab_cli
      Linux + GPU ≥ 8GB → local
      Linux + GPU < 8GB → colab_cli
      无 GPU → colab_cli
    """
    if os.environ.get("KOUBO_GPU_MODE"):
        return os.environ["KOUBO_GPU_MODE"]
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
    """Resolve stale configured aliases to the active CLI session for this account."""
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
        result = subprocess.run(
            _colab_base_args(profile) + ["sessions"],
            env=_colab_env(profile), capture_output=True, text=True, timeout=20,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        names = re.findall(r"^\s*\[[^\]]*\]\s+([^|\r\n]+?)\s*\|", output, re.M)
        names = [name.strip() for name in names if name.strip()]
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


def _colab_exec(cmd_str, session=None, timeout=300):
    """通过 colab CLI 执行远程命令，返回 (returncode, stdout)"""
    import tempfile
    colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
    profile = _active_colab_profile()
    # 写一个 Python 脚本来执行命令并打印输出
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(f"import subprocess,sys,json\n"
                f"r=subprocess.run({repr(cmd_str)},shell=True,capture_output=True,text=True,timeout={timeout})\n"
                f"sys.stdout.write(r.stdout)\n"
                f"sys.stdout.write(json.dumps({{'rc':r.returncode}}))\n")
        tmp = f.name
    resolved_session = _colab_resolve_session(profile, session)
    args = _colab_base_args(profile) + ["exec", "-s", resolved_session, "-f", tmp, "--timeout", str(timeout)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 30,
                       env=_colab_env(profile))
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

def _colab_upload(local_path, remote_path, session=None):
    """上传文件到 Colab；自动纠正失效会话名并重试瞬时网络错误。"""
    profile = _active_colab_profile()
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

def _colab_download(remote_path, local_path, session=None):
    """从 Colab 下载文件"""
    profile = _active_colab_profile()
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
        return Response(content, mimetype="text/plain; charset=utf-8")
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='color-scheme' content='light'>"
        "<title>cicy-koubo 日志</title>"
        "<style>html,body{margin:0;min-height:100%;background:#fff;color:#17212b}"
        "body{box-sizing:border-box;padding:20px}"
        "pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;"
        "font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}</style>"
        "</head><body><pre>" + html.escape(content) + "</pre></body></html>"
    )
    return Response(page, mimetype="text/html; charset=utf-8")


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
    if GPU_MODE == "colab_cli":
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
    if GPU_MODE == "colab_cli":
        active, tnote = _colab_session_active()
        tstate = "ready" if active else "offline"
        mt_ready = cosy_ready = False
        if active:
            _, ready_out = _colab_exec(
                "test -f /content/mt/READY && echo __MT_READY__; "
                "test -f /content/cosy/COSY_READY && echo __COSY_READY__",
                timeout=30,
            )
            mt_ready = "__MT_READY__" in ready_out
            cosy_ready = "__COSY_READY__" in ready_out
        audio = "ready" if cosy_ready else ("installing" if active else "down")
        video = "ready" if mt_ready else ("installing" if active else "down")
    else:
        tstate, tnote = tunnel_status()
        audio = "ready" if (tstate == "ready" and _cosy_ready_remote()) else \
                ("installing" if tstate == "ready" else "down")
        video = "ready" if tstate == "ready" else "down"
    overall = "ready" if tstate == "ready" and audio == "ready" and video == "ready" else \
              ("partial" if tstate == "ready" else "down")
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
                if not _colab_upload(norm, f"/content/{norm.name}"):
                    raise RuntimeError("colab upload base failed")
                if not _colab_upload(audio_path, f"/content/{pathlib.Path(audio_path).name}"):
                    raise RuntimeError("colab upload audio failed")
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
                launch_rc, launch_out = _colab_exec(launch, timeout=30)
                if launch_rc != 0:
                    raise RuntimeError("MuseTalk launch failed via colab CLI: " + launch_out[-200:])
                seen_lines = 0
                rc = 124
                for _ in range(180):  # 最长 15 分钟
                    time.sleep(5)
                    poll_rc, poll_out = _colab_exec(
                        f"tail -n 80 {remote_log} 2>/dev/null; "
                        f"echo __CICY_STATE__; "
                        f"if [ -f {remote_done} ]; then cat {remote_rc}; else echo RUNNING; fi",
                        timeout=20,
                    )
                    if poll_rc != 0 and any(
                        marker in poll_out.lower()
                        for marker in ("session", "lost", "not found", "404/401")
                    ):
                        raise RuntimeError("Colab 会话在对口型过程中断开或被 Google 回收，请重新启动会话后重试")
                    output, _, state = poll_out.rpartition("__CICY_STATE__")
                    lines = [line for line in output.splitlines() if line.strip()]
                    for line in lines[seen_lines:]:
                        log(f"[MuseTalk] {line[:120]}")
                    seen_lines = len(lines)
                    state = state.strip()
                    if state != "RUNNING":
                        try:
                            rc = int(state.splitlines()[-1])
                        except Exception:
                            rc = 1
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
                check_rc, check_out = _colab_exec(f"test -f {out_remote} && echo OK || echo MISSING", timeout=30)
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
                if not _colab_download(out_remote, result):
                    raise RuntimeError("colab download failed")
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
    """文案 + 参考音色 → CosyVoice zero-shot 克隆配音(本机 GPU 直接跑)。"""
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
    whole = request.form.get("mode") == "whole"
    plog(f"[配音 {jid}] 开始: {len(text)}字 语速{speed} 模式{'整段' if whole else '分段'}")

    # Groq/Whisper only transcribes the reference. CosyVoice still needs the
    # configured GPU runtime, so fail early with an actionable message.
    if GPU_MODE == "colab_cli":
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
        ref_text = _transcribe(str(ref), stt_provider)
    except GroqTranscriptionError as exc:
        return jsonify({"error": str(exc), "code": "groq_stt_failed"}), 502
    if not ref_text:
        return jsonify({"error": f"参考音频转写失败（{stt_provider}），请检查服务状态或更换清晰中文人声音频"}), 400

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
        if not _colab_upload(ref, remote_ref):
            return jsonify({"error": "上传参考音频到 Colab 失败"}), 500
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
        _colab_exec(cmd, timeout=10)
        # 轮询完成
        for _ in range(120):
            time.sleep(3)
            check_rc, check_out = _colab_exec(f"cat /content/tts_{jid}.done 2>/dev/null && echo DONE", timeout=10)
            if "DONE" in check_out:
                break
        # 下载结果
        result_wav = WORK / f"tts_{jid}.wav"
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


def _load_font(fid, size):
    """加载字体。遍历回退链直到找到可用的字体（PIL path/bytes 双模式）。"""
    from PIL import ImageFont
    candidates = [
        FONTS[fid or "heavy"][0],                                   # ROOT 标准路径
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
    font = _load_font(font_id, fs)
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
        "install_hint": "首次安装需下载模型，耗时取决于 Colab 网络和缓存；请查看实时日志",
    },
    "cosy":     {
        "name": "CosyVoice",
        "ready": "/content/cosy/COSY_READY",
        "dir": "/content/cosy",
        "install_hint": "首次安装需下载模型，耗时取决于 Colab 网络和缓存；请查看实时日志",
    },
    "whisper":  {"name": "Whisper",   "ready": "/content/cosy/COSY_READY", "dir": "/content/cosy", "builtin": True},
    "hg":       {
        "name": "HeyGem",
        "ready": "/content/hg/HG_READY",
        "dir": "/content/hg",
        "install_hint": "首次安装需下载模型，耗时取决于 Colab 网络和缓存；请查看实时日志",
        "note": "可以先安装；生成时可能需要更多 GPU 显存，建议 ≥16GB，T4 可能 OOM",
    },
}

COLAB_GPUS = ("T4", "L4", "G4", "H100", "A100")
_COLAB_OAUTH_PROCESSES = {}


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
    token_path.unlink(missing_ok=True)  # “授权”也可用于安全地切换/重新授权账号
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
    if profile["auth"] == "adc" and not pathlib.Path(profile["credentials_path"]).is_file():
        return jsonify({"error": "当前账号的 ADC 凭据文件不存在"}), 400
    args = _colab_base_args(profile) + ["new", "-s", profile["session"], "--gpu", profile["gpu"]]
    try:
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
    try:
        r = subprocess.run(_colab_base_args(profile) + ["stop", "-s", profile["session"]],
                           capture_output=True, text=True, timeout=60, env=_colab_env(profile))
        if r.returncode:
            return jsonify({"error": (r.stderr or r.stdout)[-800:]}), 500
        cfg, colab = _colab_profiles_cfg()
        current = next((p for p in colab["profiles"] if p.get("id") == colab["active"]), None)
        if current is not None:
            current.pop("started_at", None)
            save_global_cfg(cfg)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/colab/session")
def api_colab_session_status():
    """返回当前配置档的会话状态与运行时长，不暴露会话 token。"""
    profile = _active_colab_profile()
    state_path = pathlib.Path(profile["session_config"])
    try:
        sessions = json.loads(state_path.read_text()) if state_path.exists() else {}
        session = sessions.get(profile["session"])
    except Exception:
        session = None
    account_email = ""
    try:
        who = subprocess.run(_colab_base_args(profile) + ["whoami"], capture_output=True,
                             text=True, timeout=30, env=_colab_env(profile))
        for line in (who.stdout or who.stderr).splitlines():
            if line.startswith("Email:"):
                account_email = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    common = {
        "session": profile["session"],
        "gpu": profile["gpu"],
        "account_email": account_email,
        "plan": None,
        "compute_units": None,
        "manage_url": "https://colab.research.google.com/signup",
    }
    if not session:
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


@app.get("/api/system")
def api_system():
    import shutil
    info = {"gpu_name": "", "gpu_memory_mb": 0, "gpu_free_mb": 0,
            "cpu_cores": os.cpu_count() or 0,
            "ram_gb": 0, "disk_total_gb": 0, "disk_free_gb": 0,
            "python_version": "", "cuda_version": "", "ffmpeg_version": "",
            "gpu_mode": GPU_MODE,
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
        if _sht.which("uv"):
            cmd = ["uv", "tool", "install", "--force", "google-colab-cli"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "--user", "--upgrade", "google-colab-cli"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            colab_bin = _sht.which("colab") or str(pathlib.Path.home() / ".local/bin/colab")
            vr = subprocess.run([colab_bin, "version"], capture_output=True, text=True, timeout=15)
            version = (vr.stdout or vr.stderr).strip().splitlines()[-1] if (vr.stdout or vr.stderr).strip() else "installed"
            return jsonify({"ok": True, "version": version})
        else:
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
        installing = remote_status.get(key, {}).get("installing", False) if is_remote else _is_installing(key)
        has_script = (pathlib.Path(cfg["dir"]) / "provision.sh").exists()
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
        remote_dir = cfg["dir"]
        remote_log = f"{remote_dir}/provision.log"
        mkdir_rc, mkdir_out = _colab_exec(
            f"mkdir -p {shlex.quote(remote_dir)}",
            timeout=30,
        )
        if mkdir_rc != 0:
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
                    return jsonify({"error": f"上传 {sf.name} 到 Colab 失败"}), 500
                _colab_exec(
                    f"printf '完成\\n' >> {shlex.quote(remote_log)}",
                    timeout=30,
                )
        # 远程执行 provision.sh
        script = f"{remote_dir}/provision.sh"
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
            return jsonify({"error": f"启动 {cfg['name']} 安装失败：{out[-300:]}"}), 500
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
