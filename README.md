# 小模型增强的 RAG + 情感记忆维护中间件

[![CI](https://github.com/EveningTwiKnight/slm-memory-middleware/actions/workflows/ci.yml/badge.svg)](https://github.com/EveningTwiKnight/slm-memory-middleware/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

让 7B 级小模型在长对话里**记得住人、查得到资料、说话得体**的一层中间件。

传统做法是把对话历史全量塞进提示词，但这会带来三个问题：聊到第 30 轮时输入 token
涨到首轮的 3.1 倍、延迟超过 2 秒；而且小模型在长上下文里会**漏看关键信息**——
实测记忆类问题正确率只有 4/6，失败的题目里答案其实就在上下文中间。

本项目把"记住什么"从模型层剥离出来交给工程系统。实测小模型加上这层之后：

| 指标 | 改善 |
|---|---|
| 记忆类问题正确率 | 4/6 → **6/6** |
| 检索命中率 Recall@5 | 3/5 → **5/5** |
| 第 30 轮输入 token | 相比全量方案**降低 50%**，并到达平台期 |
| 首字延迟 | **167~298ms**（流式） |
| 情感姿态合规率 | 67% → **100%** |
| 与大模型对比 | 走同样中间件**效果打平**，但延迟低 2.5 倍 |
| 无关问题的 token 开销 | 意图判断后**降低 67%** |

> 所有指标均为**离线实测**，每条结论都有对应的验证脚本（见 tests/），可自行复现。

## 它解决什么

```
业务应用 ──▶ ★ 中间件 ★ ──▶ 小模型
                │
                ├─ 记得住人    跨会话记忆 + 用户档案 + 关键事实
                ├─ 查得到资料  知识库检索（RAG），答不出就说不知道
                ├─ 说话得体    识别情绪，调整回复姿态；情感加权召回
                └─ 不撑爆上下文 Token 预算调度 + 两级缓存 + 降级链
```

业务侧只需要调用一个对话接口 —— **接入成本就是改一个 URL**。

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/EveningTwiKnight/slm-memory-middleware.git
cd slm-memory-middleware

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

> 首次安装会下载 PyTorch（100MB+），耐心等一会儿。
> 国内网络建议加镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
>
> 首次运行时还会自动下载嵌入模型（约 100MB），之后缓存在本地，加载只要 1.5 秒。

### 2. 配置 API Key

中间件本身**不含模型**，它调用一个大模型服务。你需要自己提供一个 API Key
（一份 Key 可以用很久，阿里云百炼有免费额度）。

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Mac/Linux
```

然后编辑 `.env`，只需改 **一行** —— 把 Key 填在 `LLM_API_KEY=` 后面：

```
LLM_API_KEY=sk-你的key
```

**三家服务商可选，`.env.example` 里写了详细获取步骤：**

| 服务商 | 特点 | 需要改的配置 |
|---|---|---|
| **阿里云百炼**（默认） | 有免费额度，中文好 | 只填 Key 即可 |
| **智谱 BigModel** | 有免费额度 | 填 Key + 改 `LLM_BASE_URL` 和 `LLM_MODEL` |
| **本地 Ollama** | **完全免费，数据不出本机** | 改两行 + Key 随便填 |

> 如果没填 Key，启动时不会崩，但**聊天会返回一句明确的提示**告诉你该填什么。
> ⚠️ `qwen-turbo` 已于 2025-10-10 下线，不要再用。
> ⚠️ `.env` 已被 `.gitignore` 忽略，不会提交到 Git。

### 3. 自检

```bash
.venv\Scripts\python.exe -m tests.test_connection   # 模型连通性（需要 Key）
.venv\Scripts\python.exe -m tests.test_api          # 接口冒烟（不需要 Key）
```

**不想先搞 Key？** 大部分测试不需要 Key，可以直接跑：

```bash
.venv\Scripts\python.exe -m tests.test_memory             # 短期记忆，15 项
.venv\Scripts\python.exe -m tests.test_budget --offline    # 预算调度 + 缓存，26 项
.venv\Scripts\python.exe -m tests.test_emotion --offline   # 情感模块，28 项
.venv\Scripts\python.exe -m tests.test_intent              # 检索意图判断，21 条
.venv\Scripts\python.exe -m web.verify                     # 前端 SSE 链路（需要 Key）
.venv\Scripts\python.exe -m web.verify_panel               # 界面数据链路，22 项
```

### 4. 启动服务

**方式一：接入示例（推荐先跑这个）**

```bash
.venv\Scripts\python.exe -m web.server
```

然后浏览器打开 **http://127.0.0.1:8080**

这是一个最小可用的聊天界面（单文件 HTML + 原生 JS，零前端依赖）。
它同时是**接入中间件的参考实现** —— 前端要写的那 30 行代码都在里面可抄。
界面上有「看它记住了什么」按钮，能直接看到中间件存了哪些记忆和情绪。

> ⚠️ 关于地址：`127.0.0.1` 是"本机"的意思，**不是公网地址**。
> 所有 `127.0.0.1:xxxx` 的链接都需要你先把服务跑起来，才能在**你自己的浏览器**里打开。
> 别人访问不了你的 `127.0.0.1`，这个项目本身也没有线上部署。

**方式二：只看 HTTP 接口**

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

服务起来后，浏览器打开：

| 地址 | 是什么 |
|---|---|
| http://127.0.0.1:8000/docs | **交互式接口文档**（FastAPI 根据代码自动生成，可以直接在页面上点着调接口） |
| http://127.0.0.1:8000/health | 健康检查，返回服务状态与当前模型名 |
| `/v1/debug/prompt?user_id=xx&message=yy` | 看模型实际收到的完整提示词（调试用） |

> 上面这两个 `8000` 端口的地址只有在你用**方式二**启动时才是活的；
> 如果用方式一（8080），所有接口也在 8080 上，例如
> http://127.0.0.1:8080/docs 同样可用。

## ★ 用 5 分钟验证它真的有用

如果只是想快速确认"这东西在干什么"，按这个顺序走：

```bash
# ① 确认环境是好的（不需要 Key）
.venv\Scripts\python.exe -m tests.test_memory        # 应输出 15/15 通过

# ② 起服务
.venv\Scripts\python.exe -m web.server               # 打开 http://127.0.0.1:8080
```

然后在网页上做这四步：

| 步骤 | 操作 | 应该看到 |
|---|---|---|
| 1 | 说：`我叫林知远，是个算法工程师，在做推荐系统` | 正常回复；气泡下方显示耗时 |
| 2 | **等 3 秒**（后台在抽取记忆），点「看它记住了什么」 | 右侧弹出面板，列出刚提炼出的记忆和情绪 |
| 3 | 点「开新会话」 | 聊天框清空，右上角 session 变了 |
| 4 | 问：`你还记得我是谁吗？` | **它答出姓名和职业** |

**第 4 步就是核心**：聊天记录已经清空了，答案完全来自中间件注入的记忆。
第 2 步则证明"记忆不是凭空来的"——你能看到它实际存了什么。

再想验证数据确实存下来了：

```bash
.venv\Scripts\python.exe -m scripts.show_memory web_demo_user
```

想测知识库检索（RAG），先入库一份示例文档：

```bash
.venv\Scripts\python.exe -m scripts.ingest_kb --dir <放你文档的目录> --reset
```

## 工具脚本

```bash
# 看数据库里存了什么
.venv\Scripts\python.exe -m scripts.inspect_db

# 看某个用户被记住了什么
.venv\Scripts\python.exe -m scripts.show_memory <user_id>

# 清空数据（重新演示时用）
.venv\Scripts\python.exe -m scripts.clear_data --stats        # 先看有什么
.venv\Scripts\python.exe -m scripts.clear_data --all          # 清空（保留知识库）
.venv\Scripts\python.exe -m scripts.clear_data --user demo_user  # 只清某个用户

# 知识库入库
.venv\Scripts\python.exe -m scripts.ingest_kb --dir <目录> --reset
```

## 跑完整评估与消融实验

```bash
# 先知识库入库（评估里的知识类题目要用）
.venv\Scripts\python.exe -m scripts.ingest_kb --dir <放你文档的目录> --reset

# 跑评估（七种配置对照，会调模型，约 4 分钟）
.venv\Scripts\python.exe -m eval.run
```

报告输出到 `eval/reports/`（JSON + Markdown，该目录不入库）。

## Docker 部署

```bash
cp .env.example .env      # 填 Key
docker compose up -d
curl http://127.0.0.1:8000/health
```

## 项目结构

```
app/
├─ main.py                 # 服务入口（含嵌入模型预热）
├─ api/                    # L1 接入层
│  ├─ schemas.py           # 请求/响应数据模型
│  └─ chat.py              # 全部接口实现
├─ core/                   # 基础设施
│  ├─ config.py            # 配置中心（读 .env）
│  ├─ logging.py           # 日志 + trace_id
│  └─ tasks.py             # 后台任务（异步写记忆）
├─ llm/                    # L5 模型层（可插拔后端）
│  ├─ base.py              # 抽象接口
│  ├─ openai_compat.py     # OpenAI 兼容实现
│  └─ fake.py              # 假模型（无 Key 也能测）
├─ memory/                 # L2 状态层
│  ├─ models.py            # 五张表定义
│  ├─ database.py          # 连接管理
│  ├─ migrate.py           # 自动补列（表结构变更）
│  ├─ short_term.py        # 会话原文
│  ├─ long_term.py         # 结构化记忆 + 档案
│  ├─ extractor.py         # 记忆抽取器
│  ├─ embeddings.py        # 嵌入模型封装
│  ├─ vector_store.py      # 向量库（Chroma）
│  ├─ emotion.py           # 情绪打分 + 状态 + 姿态
│  ├─ emotion_store.py     # 情绪状态持久化
│  └─ media.py             # 多模态拓展点（预留未实现）
├─ retrieval/              # L3 检索层
│  └─ memory_retriever.py  # 复合打分检索 + 语义去重
└─ orchestration/          # L4 编排层
   ├─ prompt_builder.py    # 提示词组装
   ├─ intent.py            # 检索意图判断
   ├─ budget.py            # Token 预算调度
   └─ cache.py             # 两级回答缓存

web/      接入示例 + 可用的聊天界面（单文件，零前端依赖）
eval/     评估集定义与七配置对照执行器
tests/    15 个验证脚本（多数不需要 API Key）
scripts/  4 个工具脚本（入库 / 查看 / 清理）
```

## 设计要点

- **模型层做了抽象**：上层只依赖 `LLMClient` 接口，换服务商只改 `.env` 一行。
- **记忆分四类存储**：会话原文（连贯性）、结构化记忆（可检索）、用户档案（全量注入）、
  情绪状态（影响姿态与排序）。职责不同所以分开放。
- **检索是四项复合打分**：`0.45×相关性 + 0.20×时间衰减 + 0.15×重要度 + 0.20×情绪一致性`。
- **先判断意图再检索**：问"今天天气怎么样"这类与用户无关的问题时**不检索记忆**
  （省 67% prompt token，也避免干扰模型）。实测相关与无关问题的相似度区间完全重叠，
  用固定门槛切不开，所以改用规则判断意图。
- **阈值按实测数据定**：语义去重 0.80（同义句实测 0.82，无关句 0.54）；
  知识库门槛 0.45（相关问题 0.49~0.61，无关 0.19~0.21）。
- **每次请求打 trace_id**：一次对话经过情绪 → 检索 → 编排 → 模型四步，靠它把日志串起来。
- **每层都有降级**：7 类故障的兜底策略，任何单点失败都不中断对话。

## 已知局限

- **情绪识别用词典 + 规则实现**（零延迟、零成本）。它判不了反讽和隐晦表达，
  实测"真是太好了，又加班到十二点"会被判成正向、"最近睡得不太好"判成中性。
  设计上靠**低置信度兜底**（退回中性姿态）而不是硬判。
- **没有实现混合召回（BM25）与重排序模型**。检索是"向量召回 + 规则复合打分"，
  不是"向量 + 关键词混合召回 + 交叉编码器精排"。
- **没有微调过模型**。项目只做提示词工程与上下文治理，不涉及训练。
- **没有鉴权**。`/v1/memory/{user_id}` 目前可被任意访问，上线前必须补
  API Key / JWT，并且要校验调用方是否有权访问该 user_id。
- **数据自动迁移只处理"加列"**。删列、改类型、加约束需要人工处理，
  生产环境应换 Alembic。
- **单机架构**。SQLite + Chroma 本地文件 + 进程内缓存，多实例部署需要换
  PostgreSQL + Qdrant + Redis。
- **评估集只有 14 道题**，规模不大，靠标注质量而非数量；
  忠实度用 LLM-as-Judge，裁判本身也是模型，会有误差。
- **多模态（表情包识别）只开了接口**，识别逻辑未实现，见 `app/memory/media.py`。
- **所有指标均为离线实测**，无线上数据。
