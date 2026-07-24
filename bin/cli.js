#!/usr/bin/env node
/* cicy-koubo — 爆款口播视频制作工作台启动器
 *   npx cicy-koubo            本地启动 → http://127.0.0.1:8770
 *   npx cicy-koubo --cft      同时开 cloudflared 快速隧道,打印公网 URL
 *   npx cicy-koubo --port N   指定端口
 */
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const http = require('http');

const argv = process.argv.slice(2);
if (argv.includes('--help') || argv.includes('-h')) {
  console.log(`cicy-koubo — 数字人口播工作台
  npx cicy-koubo            本地启动(默认端口 8770)
  npx cicy-koubo --cft      额外开 cloudflared 快速隧道,输出公网 https 地址
  npx cicy-koubo --port N   指定端口`);
  process.exit(0);
}
const useCft = argv.includes('--cft');
const port = (() => {
  const i = argv.indexOf('--port');
  return i >= 0 && argv[i + 1] ? parseInt(argv[i + 1], 10) : 8770;
})();

const PKG = path.join(__dirname, '..');
const APP_PY = path.join(PKG, 'app', 'app.py');
const ROOT = path.join(os.homedir(), 'projects', 'digital-human');
const BIN_DIR = path.join(os.homedir(), '.cicy-koubo', 'bin');

const ok = (m) => console.log('  \x1b[32m✓\x1b[0m ' + m);
const warn = (m) => console.log('  \x1b[33m!\x1b[0m ' + m);
const die = (m) => { console.error('  \x1b[31m✗\x1b[0m ' + m); process.exit(1); };
const sh = (cmd, args, opts = {}) =>
  spawnSync(cmd, args, { encoding: 'utf8', timeout: opts.timeout || 120000, ...opts });

console.log('\n🎬 cicy-koubo — 爆款口播视频制作\n');

/* ---------- 1. 依赖自检 ---------- */
const py = ['python3', 'python'].find((p) => sh(p, ['--version']).status === 0);
if (!py) die('需要 python3(未找到)。macOS: brew install python3');
ok(`python: ${py}`);

if (sh(py, ['-c', 'import flask, PIL']).status !== 0) {
  warn('缺 flask/pillow,尝试自动安装…');
  let r = sh(py, ['-m', 'pip', 'install', '--user', 'flask', 'pillow'], { timeout: 300000 });
  if (r.status !== 0)
    r = sh(py, ['-m', 'pip', 'install', '--user', '--break-system-packages', 'flask', 'pillow'],
           { timeout: 300000 });
  if (r.status !== 0 || sh(py, ['-c', 'import flask, PIL']).status !== 0)
    die('flask/pillow 安装失败,请手动: pip3 install flask pillow');
  ok('flask + pillow 已安装');
} else ok('flask + pillow');

if (sh('ffmpeg', ['-version']).status === 0) ok('ffmpeg');
else warn('未找到 ffmpeg — 剪辑/归一/封面会失败。macOS: brew install ffmpeg');

/* ---------- 2. 数据目录引导 ---------- */
for (const d of ['assets/bgm', 'assets/fonts', 'kr-app'])
  fs.mkdirSync(path.join(ROOT, d), { recursive: true });
const stateFile = path.join(ROOT, 'state.json');
if (!fs.existsSync(stateFile))
  fs.writeFileSync(stateFile, JSON.stringify({ assets: {}, processed_ids: [] }, null, 2));
ok(`数据目录: ${ROOT}`);

// Linux(如 Colab)且无中文字体 → 下载思源黑体,字幕/封面才有中文
const fontFile = path.join(ROOT, 'assets', 'fonts', 'SourceHanSansCN-Heavy.otf');
if (process.platform === 'linux' && !fs.existsSync(fontFile) &&
    !fs.existsSync('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')) {
  warn('下载中文字体(约 8MB,一次性)…');
  const r = sh('curl', ['-fsSL', '-o', fontFile,
    'https://cdn.jsdelivr.net/gh/adobe-fonts/source-han-sans@release/OTF/SimplifiedChinese/SourceHanSansSC-Heavy.otf'],
    { timeout: 300000 });
  if (r.status === 0) ok('中文字体已就位'); else { warn('字体下载失败,字幕将用系统字体'); fs.rmSync(fontFile, { force: true }); }
}

/* ---------- 3. 起服务 ---------- */
const child = spawn(py, [APP_PY], {
  env: { ...process.env, KOUBO_PORT: String(port) },
  stdio: ['ignore', 'ignore', 'pipe'],
});
let errBuf = '';
child.stderr.on('data', (d) => { errBuf += d; });
child.on('exit', (code) => {
  if (code !== 0 && code !== null) {
    console.error('\n后端退出(code ' + code + '):\n' + errBuf.slice(-800));
    process.exit(1);
  }
});

function waitUp(tries = 60) {
  return new Promise((res, rej) => {
    const ping = (n) => {
      const rq = http.get({ host: '127.0.0.1', port, path: '/', timeout: 1500 },
        (r) => { r.resume(); r.statusCode === 200 ? res() : retry(n); });
      rq.on('error', () => retry(n));
      rq.on('timeout', () => { rq.destroy(); retry(n); });
    };
    const retry = (n) => (n <= 0 ? rej(new Error('后端 30 秒内未就绪')) : setTimeout(() => ping(n - 1), 500));
    ping(tries);
  });
}

/* ---------- 4. cloudflared 快速隧道 ---------- */
function ensureCloudflared() {
  if (sh('cloudflared', ['--version']).status === 0) return 'cloudflared';
  const local = path.join(BIN_DIR, 'cloudflared');
  if (fs.existsSync(local)) return local;
  warn('未找到 cloudflared,自动下载中…');
  fs.mkdirSync(BIN_DIR, { recursive: true });
  const arch = { x64: 'amd64', arm64: 'arm64' }[process.arch] || 'amd64';
  const base = 'https://github.com/cloudflare/cloudflared/releases/latest/download/';
  let r;
  if (process.platform === 'darwin') {
    const tgz = path.join(BIN_DIR, 'cf.tgz');
    r = sh('curl', ['-fsSL', '-o', tgz, `${base}cloudflared-darwin-${arch}.tgz`], { timeout: 300000 });
    if (r.status === 0) r = sh('tar', ['-xzf', tgz, '-C', BIN_DIR]);
    fs.rmSync(tgz, { force: true });
  } else {
    r = sh('curl', ['-fsSL', '-o', local, `${base}cloudflared-linux-${arch}`], { timeout: 300000 });
  }
  if (!fs.existsSync(local) || (r && r.status !== 0)) die('cloudflared 下载失败,请手动: brew install cloudflared');
  fs.chmodSync(local, 0o755);
  ok('cloudflared 已就位');
  return local;
}

function startTunnel() {
  const bin = ensureCloudflared();
  const tun = spawn(bin, ['tunnel', '--url', `http://127.0.0.1:${port}`, '--no-autoupdate'],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  let found = false, buf = '';
  const scan = (d) => {
    buf += d.toString();
    const m = buf.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/);
    if (m && !found) {
      found = true;
      console.log('\n──────────────────────────────────────────────');
      console.log('  🌐 公网地址: \x1b[1m\x1b[36m' + m[0] + '\x1b[0m');
      console.log('     (cloudflared 快速隧道,本进程存活期间有效;边缘节点生效约需 1 分钟)');
      console.log('──────────────────────────────────────────────\n');
    }
  };
  tun.stdout.on('data', scan);
  tun.stderr.on('data', scan);
  tun.on('exit', (c) => { if (!found) warn('cloudflared 退出(code ' + c + '),没有拿到公网地址'); });
  setTimeout(() => { if (!found) warn('20 秒未取到公网地址,cloudflared 仍在重试…'); }, 20000);
  return tun;
}

/* ---------- 5. 主流程 ---------- */
let tunProc = null;
waitUp().then(() => {
  console.log('\n──────────────────────────────────────────────');
  console.log('  ✅ 已启动: \x1b[1mhttp://127.0.0.1:' + port + '\x1b[0m');
  console.log('──────────────────────────────────────────────');
  if (useCft) tunProc = startTunnel();
  else console.log('  提示: 加 --cft 可得到一个公网 https 地址\n');
  if (process.platform === 'darwin' && !process.env.KOUBO_NO_OPEN)
    sh('open', ['http://127.0.0.1:' + port]);
}).catch((e) => {
  console.error('\n启动失败: ' + e.message + '\n' + errBuf.slice(-800));
  child.kill('SIGKILL');
  process.exit(1);
});

const bye = () => {
  if (tunProc) tunProc.kill('SIGTERM');
  child.kill('SIGTERM');
  setTimeout(() => process.exit(0), 300);
};
process.on('SIGINT', bye);
process.on('SIGTERM', bye);
