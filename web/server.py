"""最小可用的聊天前端（FastAPI 提供页面 + SSE 流式渲染）。

**这是"怎么把它用到真实界面里"的参考实现。**

为什么不用 Gradio 了？
Gradio 有两个问题，让它只适合录屏、不适合真用：
  1. 每次交互都要跑一次完整的事件往返，没法做真正的流式打字机效果
  2. 它的页面结构是固定的，你没法把它嵌进自己的产品里

所以真实项目里的结构应该是：

    ┌─────────────┐    HTTP+SSE    ┌──────────────┐    内部调用    ┌──────────────┐
    │  你的前端    │ ─────────────▶ │   中间件      │ ───────────▶ │  大/小模型    │
    │ (网页/App)  │ ◀───────────── │  (:8000)     │              │              │
    └─────────────┘   流式返回      └──────────────┘              └──────────────┘
             │
             └── 前端只做三件事：发消息、渲染流、显示 session_id

这个文件同时是两样东西：
  1. **可直接运行的演示**（比 Gradio 更接近真实产品的形态）
  2. **给前端同学的对接参考**（HTML/JS 部分可以直接抄）

启动：
    .venv\\Scripts\\python.exe -m web.server
然后打开 http://127.0.0.1:8080
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

# ---------------------------------------------------------------- 页面
#
# 这里故意用一个单文件 HTML + 原生 JS，不引任何框架。
# 原因：不管你的前端是 React / Vue / 小程序，核心逻辑都是这三步：
#   1. 用 fetch + ReadableStream 读 SSE（或 EventSource）
#   2. 逐块把文本追加到气泡里（打字机效果）
#   3. 把 session_id 存在前端（localStorage / 状态管理）
# 框架只是包装，逻辑是一样的。

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>中间件接入示例</title>
<style>
  :root { --bg:#f7f7f8; --fg:#1a1a1a; --muted:#8a8f98; --line:#e5e7eb; --user:#2563eb; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         background:var(--bg); color:var(--fg); }
  .wrap { height:100vh; display:flex; flex-direction:column; }
  header { padding:16px 20px; border-bottom:1px solid var(--line); background:#fff;
           display:flex; align-items:center; gap:12px; flex-shrink:0; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sid { margin-left:auto; font-size:12px; color:var(--muted); font-family:ui-monospace,monospace; }
  .msg { margin-bottom:18px; display:flex; flex-direction:column; }
  .msg.user { align-items:flex-end; }
  .bubble { max-width:78%; padding:11px 15px; border-radius:14px; white-space:pre-wrap;
            word-break:break-word; background:#fff; border:1px solid var(--line); }
  .msg.user .bubble { background:var(--user); color:#fff; border-color:var(--user); }
  .meta { margin-top:6px; font-size:12px; color:var(--muted); font-family:ui-monospace,monospace; }
  .msg.user .meta { text-align:right; }
  .meta b { color:#059669; font-weight:600; }

  /* ---- 右侧「记忆面板」---- */
  .app { display:flex; gap:16px; flex:1; min-height:0; padding:16px; max-width:1320px;
         margin:0 auto; width:100%; }
  .chatcol { flex:1; display:flex; flex-direction:column; min-width:0;
             background:#fff; border:1px solid var(--line); border-radius:14px; }
  #log { flex:1; overflow-y:auto; padding:18px; }
  #log:empty::after { content:'上方输入一句话开始对话。'; color:var(--muted); }
  .sidecol { width:340px; flex-shrink:0; display:none; flex-direction:column;
             background:#fff; border:1px solid var(--line); border-radius:14px;
             overflow:hidden; }
  .sidecol.show { display:flex; }
  .sidecol header { padding:12px 14px; border-bottom:1px solid var(--line);
                    font-size:14px; font-weight:600; }
  .sidebody { flex:1; overflow-y:auto; padding:14px; font-size:13px; line-height:1.7; }
  .sidebody h4 { margin:14px 0 6px; font-size:13px; color:#374151; }
  .sidebody h4:first-child { margin-top:0; }
  .sidebody ul { margin:0; padding-left:16px; }
  .sidebody li { margin-bottom:6px; word-break:break-word; }
  .sidebody .imp { font-family:ui-monospace,monospace; color:#059669; font-size:12px; }
  .sidebody .emo { color:#7c3aed; font-size:12px; }
  .sidebody .muted { color:var(--muted); }
  @media (max-width:900px) { .sidecol { display:none !important; } }

  footer { padding:14px 20px 20px; background:transparent; }
  .row { display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; padding:11px 13px; border:1px solid var(--line);
             border-radius:12px; font:inherit; outline:none; max-height:140px; }
  textarea:focus { border-color:var(--user); }
  button { padding:11px 20px; border:0; border-radius:12px; background:var(--user);
           color:#fff; font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .tools { display:flex; gap:8px; margin-top:9px; }
  .tools button { padding:7px 13px; font-size:13px; font-weight:500;
                  background:#eef2ff; color:#3730a3; }
  .cursor::after { content:'▋'; animation:blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  .hint { font-size:12px; color:var(--muted); margin-bottom:10px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>中间件接入示例</h1>
    <span class="sid" id="sidLabel">session: -</span>
  </header>

  <div class="app">
    <!-- 聊天区 -->
    <div class="chatcol">
      <div id="log"></div>
      <footer>
        <div class="hint">
          试试点「开新会话」再问「你还记得我是谁吗」——聊天记录清空了，但记忆还在中间件里。
        </div>
        <div class="row">
          <textarea id="input" rows="1" placeholder="说点什么…（Enter 发送，Shift+Enter 换行）"></textarea>
          <button id="send">发送</button>
        </div>
        <div class="tools">
          <button id="newSess">开新会话</button>
          <button id="showMemory">看它记住了什么</button>
          <button id="clear">清空显示</button>
          <button id="reset">重置记忆</button>
        </div>
      </footer>
    </div>

    <!-- 记忆面板：显示中间件里存了什么。点击「看它记住了什么」才出现 -->
    <div class="sidecol" id="sidecol">
      <header>
        中间件记住了什么
        <button id="closeSide"
                style="float:right;padding:2px 8px;font-size:12px;background:#f3f4f6;color:#374151">×</button>
      </header>
      <div class="sidebody" id="sidebody"><span class="muted">加载中…</span></div>
    </div>
  </div>
</div>

<script>
// ============================================================
// 前端要做的三件事（不管你用什么框架，逻辑都是这样）
// ============================================================

// ---------- ① session_id 存在前端 ----------
// 中间件不认识"登录态"，它只认 user_id 和 session_id。
// user_id 来自你的账号体系；session_id 由前端持有。
// 换会话 = 换 session_id；换设备也带着同一个 user_id，记忆就跟着走。
const USER_ID = 'web_demo_user';
let sessionId = localStorage.getItem('mw_session_id') || '';
let lastTraceId = '';

const log = document.getElementById('log');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const sidLabel = document.getElementById('sidLabel');

function updateSid() {
  sidLabel.textContent = 'session: ' + (sessionId || '(待创建)');
}

// ---------- ② 渲染消息气泡 ----------
function addMsg(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  const meta = document.createElement('div');
  meta.className = 'meta';
  wrap.appendChild(bubble);
  wrap.appendChild(meta);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return { bubble, meta };
}

// ---------- ③ 核心：读 SSE 流并逐块渲染 ----------
async function send() {
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;

  addMsg('user', text);
  const { bubble, meta } = addMsg('assistant', '');
  bubble.classList.add('cursor');

  const t0 = performance.now();
  let firstChunkAt = 0;

  try {
    // 关键点：用 fetch 而不是 EventSource
    // EventSource 只能发 GET 请求，带不了 JSON body。
    // fetch + ReadableStream 可以 POST，并且能拿到我们自定义的事件类型。
    const resp = await fetch('/v1/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: USER_ID,
        session_id: sessionId || null,
        message: text,
      }),
    });

    if (!resp.ok) throw new Error('HTTP ' + resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // SSE 的帧格式是 "data: {...}\n\n"，要按空行切
      const frames = buf.split('\n\n');
      buf = frames.pop();   // 最后一段可能不完整，留到下一轮

      for (const frame of frames) {
        const line = frame.trim();
        if (!line.startsWith('data:')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }

        if (evt.type === 'meta') {
          // 中间件告诉前端：这一轮参考了哪些历史记忆、用了多少 token
          sessionId = evt.session_id;
          lastTraceId = evt.trace_id;
          localStorage.setItem('mw_session_id', sessionId);
          updateSid();
          bubble.dataset.memories = evt.memories_used || 0;
        } else if (evt.type === 'delta') {
          if (!firstChunkAt) {
            firstChunkAt = performance.now() - t0;
            bubble.classList.remove('cursor');
          }
          bubble.textContent += evt.text;
          log.scrollTop = log.scrollHeight;
        } else if (evt.type === 'done') {
          const total = Math.round(performance.now() - t0);
          const mem = Number(bubble.dataset.memories) || 0;
          // 措辞说明：这里说的是"本轮参考了几条历史记忆"，
          // 不是"新写入了几条记忆"。新记忆是在后台异步抽取的，
          // 两者是不同的动作，所以用词必须能区分开。
          const memNote = mem > 0
            ? ' · <b>本轮参考了 ' + mem + ' 条历史记忆</b>'
            : ' · <span style="color:#9ca3af">本轮未参考历史记忆</span>';
          meta.innerHTML =
            '首字 ' + Math.round(firstChunkAt) + 'ms · 总 ' + total + 'ms' +
            memNote + ' · ' + lastTraceId;
        } else if (evt.type === 'error') {
          bubble.textContent = '出错了：' + evt.message;
        }
      }
    }
  } catch (e) {
    bubble.textContent = '请求失败：' + e.message;
  } finally {
    bubble.classList.remove('cursor');
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.onclick = send;
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});

// ---------- 辅助按钮 ----------
document.getElementById('newSess').onclick = () => {
  sessionId = '';                      // 换会话：只清 session，不动记忆
  localStorage.removeItem('mw_session_id');
  updateSid();
  addMsg('assistant', '已开启新会话。聊天记录清空了，但你的记忆还在——试着问「你还记得我是谁吗」。');
};

document.getElementById('clear').onclick = () => { log.innerHTML = ''; };

document.getElementById('reset').onclick = async () => {
  if (!confirm('确定要清空这个用户的所有记忆吗？')) return;
  const r = await fetch('/v1/memory/' + USER_ID, { method: 'DELETE' });
  const data = await r.json();
  log.innerHTML = '';
  addMsg('assistant',
    '已清空：记忆 ' + data.memories_deleted + ' 条、档案 ' + data.facts_deleted + ' 条。');
  if (sidecol.classList.contains('show')) showMemory();
};

// ============================================================
// 记忆面板：看中间件里到底存了什么
//
// 这个面板是"接入示例"之外额外加的，目的是让中间件的工作可见 ——
// 不然别人只会看到一个能聊天的机器人，和直接调 API 没区别。
// 真实产品里这些信息通常放在运营后台，不暴露给终端用户。
// ============================================================
const sidecol = document.getElementById('sidecol');
const sidebody = document.getElementById('sidebody');

document.getElementById('closeSide').onclick = () => sidecol.classList.remove('show');

const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const pct = (v) => Math.round((Number(v) || 0) * 100);

async function showMemory() {
  sidecol.classList.add('show');
  sidebody.innerHTML = '<span class="muted">加载中…</span>';
  try {
    const [mem, emo] = await Promise.all([
      fetch('/v1/memory/' + USER_ID).then(r => r.json()),
      fetch('/v1/emotion/' + USER_ID).then(r => r.json()),
    ]);

    let html = '';

    // ---- 用户档案 ----
    html += '<h4>用户档案</h4>';
    if (mem.profile && mem.profile.length) {
      html += '<ul>' + mem.profile.map(p =>
        '<li><b>' + esc(p.key) + '</b>：' + esc(p.value) + '</li>').join('') + '</ul>';
    } else {
      html += '<div class="muted">（还没有）</div>';
    }

    // ---- 长期记忆 ----
    html += '<h4>长期记忆（' + (mem.total || 0) + ' 条）</h4>';
    if (mem.memories && mem.memories.length) {
      html += '<ul>' + mem.memories.map(m => {
        const e = m.emotion_label
          ? ' <span class="emo">' + esc(m.emotion_label) + '</span>' : '';
        return '<li><span class="imp">' + pct(m.importance) + '%</span> '
             + esc(m.content) + e + '</li>';
      }).join('') + '</ul>';
    } else {
      html += '<div class="muted">（还没有。先聊几句自我介绍）</div>';
    }

    // ---- 情绪状态 ----
    html += '<h4>情绪状态</h4>';
    const c = emo.current || {};
    html += '<ul>'
      + '<li>当前：<b>' + esc(c.label) + '</b>（效价 ' + (c.valence ?? 0).toFixed(2)
      + '，唤醒 ' + (c.arousal ?? 0).toFixed(2) + '）</li>'
      + '<li>长期倾向：效价 ' + (c.stable_valence ?? 0).toFixed(2)
      + '　趋势 ' + esc(c.trend) + '</li>'
      + '<li class="muted">已更新 ' + (c.turns ?? 0) + ' 轮</li>'
      + '</ul>';

    // ---- 情绪轨迹 ----
    if (emo.points && emo.points.length) {
      html += '<h4>情绪轨迹</h4><ul>';
      emo.points.slice(-8).forEach(p => {
        html += '<li><span class="imp">' + (p.valence >= 0 ? '+' : '')
              + Number(p.valence).toFixed(2) + '</span> '
              + esc((p.content || '').slice(0, 28)) + '</li>';
      });
      html += '</ul>';
    }

    sidebody.innerHTML = html;
  } catch (e) {
    sidebody.innerHTML = '<span class="muted">读取失败：' + esc(e.message) + '</span>';
  }
}

document.getElementById('showMemory').onclick = showMemory;

updateSid();
fetch('/health').then(r => r.json()).then(d => {
  if (!d.has_api_key) {
    addMsg('assistant', '提示：服务端还没配置 LLM_API_KEY，聊天会失败。请检查 .env。');
  }
});
</script>
</body>
</html>
"""


def create_app() -> FastAPI:
    """把中间件的路由挂进来，再加上这个页面。

    真实项目里通常是**分开部署**的：
      中间件跑在 :8000，前端跑在别的域名，中间用 CORS 打通。
    这里为了演示方便，放在同一个进程里（前端同源，没有跨域问题）。
    """
    setup_logging()

    app = FastAPI(title="中间件接入示例", version="0.1.0")

    # 如果是分开部署（前端在别的域名），必须开 CORS。
    # 生产环境不要把 allow_origins 写成 ["*"]，要写具体的前端域名。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 把中间件本身的接口全部挂上（/v1/chat、/v1/chat/stream 等）
    from app.api import chat as chat_api
    from app.main import health

    app.include_router(chat_api.router)
    app.get("/health")(health)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE

    return app


app = create_app()


def main() -> int:
    import asyncio

    import uvicorn

    from app.memory.database import init_db

    asyncio.run(init_db())

    print("=" * 68)
    print("接入示例启动中……")
    print(f"  中间件模型：{settings.llm_model}")
    print("  打开浏览器：http://127.0.0.1:8080")
    print()
    print("  这个页面演示了前端接入中间件的最小实现：")
    print("    1. user_id 来自账号体系，session_id 由前端持有")
    print("    2. 用 fetch + ReadableStream 读 SSE（不是 EventSource）")
    print("    3. 逐块把文本追加到气泡，实现打字机效果")
    print("=" * 68)

    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
