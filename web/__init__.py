"""接入示例：一个可直接运行的聊天界面。

启动：
    .venv\\Scripts\\python.exe -m web.server
然后打开 http://127.0.0.1:8080

本目录三个文件的分工：

  server.py        服务端 + 页面。单文件 HTML + 原生 JS，零前端依赖。
                   除了能聊天，它同时是**接入中间件的参考实现** ——
                   前端要写的三步逻辑（存 session_id、读 SSE 流、渲染打字机）
                   都在里面的 <script> 段，写得很直白，任何框架都能照抄。
                   页面右侧有个「看它记住了什么」面板，用来让中间件的工作可见。

  verify.py        验证 SSE 链路。完全按浏览器的方式读流
                   （POST 拿流、按空行切帧、解析 meta/delta/done/error 四种事件），
                   需要 API Key。

  verify_panel.py  验证记忆面板依赖的两个接口返回结构正确，
                   不需要 API Key。

为什么不做成 React / Vue 应用？
因为要证明的是**中间件**，不是前端。用原生 JS 反而让"接入只需要 30 行"更清楚。
"""
