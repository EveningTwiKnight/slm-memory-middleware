"""测试与验证脚本。

这些不是 pytest 用例，而是可以直接运行的验证脚本 —— 每个都针对一个能力，
跑完会打印通过/失败项和实测数据。

为什么不用 pytest？
因为这些脚本要输出**给人看的实测数据**（相似度分布、token 曲线、延迟对比），
而不只是断言。pytest 的断言输出不适合展示这类信息。
不过断言本身是完整的，CI 里就是直接跑它们（见 .github/workflows/ci.yml）。

分两类：

  不需要 API Key（离线）
    test_api            接口冒烟
    test_memory         短期记忆
    test_extractor      记忆抽取（--offline 只跑解析逻辑）
    test_emotion        情感模块（--offline 只跑规则）
    test_budget         Token 预算与缓存（--offline）
    test_retrieval      语义检索与去重（--offline）
    test_fixes          档案拆分与知识门槛
    test_intent         检索意图判断

  需要 API Key（会真的调模型）
    test_connection     模型连通性
    test_e2e            端到端记忆
    test_injection      跨会话记忆
    test_memory_flow    异步抽取落库
    test_full_stack     记忆 + 知识库全链路
    test_model_comparison  小模型 vs 大模型对照
    test_latency        首字延迟对比

用法：
    .venv\\Scripts\\python.exe -m tests.test_memory
    .venv\\Scripts\\python.exe -m tests.test_budget --offline
"""
