#!/usr/bin/env python3
"""CDP 嗅探抖音真实媒体地址(yt-dlp 被 fresh-cookies 风控拦时的兜底)。

原理:本机 agent-chrome profile(调试端口 11001/11002…)里的 Chrome 带着真实
浏览器指纹+cookies,新开一个 tab 导航到视频页,监听 Network 事件,抓
douyinvod CDN 的 media-audio / media-video 直链(这些直链用普通 UA+Referer
即可下载)。用完关掉 tab,不打扰原有页面。

用法: cdp_sniff.py <video_id> [port ...]     (默认探测 11001 11002)
输出: JSON {"audio": url|null, "video": url|null, "port": n}
依赖: pip install websocket-client
"""
import json, sys, time, urllib.request, urllib.parse

try:
    import websocket
except ImportError:
    print(json.dumps({"error": "缺 websocket-client(pip install websocket-client)"}))
    sys.exit(2)

vid = sys.argv[1]
ports = [int(p) for p in sys.argv[2:]] or [11001, 11002]
target_url = f"https://www.douyin.com/video/{vid}"


def sniff(port):
    base = f"http://127.0.0.1:{port}"
    try:
        urllib.request.urlopen(f"{base}/json", timeout=3)
    except Exception:
        return None
    # 新开 tab(PUT /json/new),避免打扰用户已开页面
    tab = None
    try:
        req = urllib.request.Request(f"{base}/json/new?{urllib.parse.quote(target_url, safe='')}", method="PUT")
        tab = json.load(urllib.request.urlopen(req, timeout=8))
    except Exception:
        # 老 Chrome 不支持 PUT;退回找现成抖音 tab / 任一页面 tab 导航
        tabs = json.load(urllib.request.urlopen(f"{base}/json", timeout=5))
        pages = [t for t in tabs if t.get("type") == "page"]
        tab = next((t for t in pages if "douyin.com" in t.get("url", "")), pages[0] if pages else None)
    if not tab:
        return None
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], max_size=None, timeout=45)
    mid = 0
    def send(m, p=None):
        nonlocal mid
        mid += 1
        ws.send(json.dumps({"id": mid, "method": m, "params": p or {}}))
    send("Network.enable")
    send("Page.navigate", {"url": target_url})
    found = {"audio": None, "video": None}
    t0 = time.time()
    while time.time() - t0 < 35 and not (found["audio"] and found["video"]):
        ws.settimeout(max(1, 35 - (time.time() - t0)))
        try:
            m = json.loads(ws.recv())
        except Exception:
            break
        if m.get("method") == "Network.responseReceived":
            r = m["params"]["response"]
            u, mt = r.get("url", ""), r.get("mimeType", "")
            if "douyinvod" in u or mt.startswith("video/"):
                if "media-audio" in u and not found["audio"]:
                    found["audio"] = u
                elif ("media-video" in u or mt.startswith("video/")) and not found["video"]:
                    if "media-audio" not in u:
                        found["video"] = u
    ws.close()
    try:  # 关掉我们开的 tab
        urllib.request.urlopen(f"{base}/json/close/{tab['id']}", timeout=3)
    except Exception:
        pass
    if found["audio"] or found["video"]:
        found["port"] = port
        return found
    return None


for p in ports:
    r = sniff(p)
    if r:
        print(json.dumps(r))
        sys.exit(0)
print(json.dumps({"error": "所有端口都没嗅到媒体地址(Chrome 没开?或页面被风控)"}))
sys.exit(1)
