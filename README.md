# cicy-koubo

爆款口播视频制作 — 一条命令启动本地数字人口播工作台。

```sh
npx cicy-koubo            # 本地启动 → http://127.0.0.1:8770(macOS 自动开浏览器)
npx cicy-koubo --cft      # 同时开 cloudflared 快速隧道,直接打印公网 https 地址
npx cicy-koubo --port 9000
```

功能:抖音对标文案提取 → AI 仿写/标题 → CosyVoice 声音克隆配音 → MuseTalk 数字人对口型 → 字幕/BGM 剪辑 → 封面生成;素材库(音色/底板/BGM/文案/成品)全 CRUD。

## 依赖

- Node ≥ 16(跑本启动器)
- python3 + flask + pillow(缺则自动 `pip install --user`)
- ffmpeg(剪辑/归一/封面必需)
- GPU 环节(配音/对口型)走 Colab:界面右上角「🚀 Colab 一键装」打开预置 notebook,全部运行即可
- `--cft` 需要 cloudflared(缺则自动下载到 `~/.cicy-koubo/bin/`)

数据目录固定在 `~/projects/digital-human/`(素材、成品、任务、缓存),升级/重装不丢。

## 开发

页面与后端的源头在 `~/projects/digital-human/kr-app/`,发版前 `npm run sync` 同步进包。
