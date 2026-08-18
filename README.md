# Policy RAG｜惠企金融政策智能问答系统

> 我的金融专业硕士毕业论文原型：把政策文本转化为可检索的向量与摘要索引，并通过 Agentic RAG 完成多轮问答、检索路由和证据回溯。

论文题目：**《基于大语言模型的惠企金融政策问答系统实现研究》**

## 30 秒了解项目

| 招聘官可能关心的问题 | 回答 |
| --- | --- |
| 解决什么问题？ | 惠企政策文本长、条款分散、检索门槛高；系统帮助企业按自然语言定位适用政策并获得可追溯回答。 |
| 我的工作是什么？ | 独立完成问题定义、文档清洗、文本切分、Embedding、摘要检索、FAQ 路由、多轮对话 Agent、RAG 链路、FastAPI 接口、前端衔接与评测。 |
| 文本如何转成向量？ | 使用 BAAI/bge-large-zh-v1.5 对政策摘要编码，Chroma 保存向量；MultiVectorRetriever 用摘要召回并返回绑定的原始政策段落。 |
| 为什么不只用普通向量检索？ | 系统同时比较 BM25、原文向量、混合检索和摘要检索，并增加 FAQ 快速通道与“是否重新检索”决策 Agent。 |
| 有什么量化结果？ | 论文的独立测试集含 238 个问答对；摘要检索 Context Recall 为 0.964。Hybrid RAG 的 Faithfulness 为 0.9600，Semantic Similarity 为 0.8280。 |
| 是否是生产系统？ | 这是论文研究与可运行原型，不代表已完成高并发、权限体系、长期监控和商业化运维。 |

## 系统流程

~~~mermaid
flowchart LR
    A["用户问题 + 企业信息 + 最近 5 轮对话"] --> B["检索决策 Agent"]
    B -->|高频标准问题| C["FAQ 快速匹配"]
    B -->|新主题| D["摘要向量检索"]
    B -->|追问| E["复用上一轮证据"]
    D --> F["MultiVectorRetriever"]
    F --> G["返回对应原始政策段落"]
    C --> H["答案"]
    E --> I["LLM 生成"]
    G --> I
    I --> H
    H --> J["政策名称 / 依据 / 原文摘录"]
~~~

## 我的核心工作

### 1. 政策文本结构化与向量化

- 清洗政策 Markdown，按标题层级切分，提取政策名称与政策依据。
- 为每个原始文档块生成摘要，使用中文 BGE Embedding 编码摘要。
- 通过文档 ID 将“摘要向量”与“原始政策段落”绑定：检索轻量摘要，回答时回到原文证据。

### 2. Hybrid / Agentic RAG

- FAQ Matcher：高频标准问题直接命中，减少无必要的生成调用。
- Retrieval Decision Agent：结合最近 5 轮对话判断当前输入是新主题、连续追问还是寒暄。
- 连续追问时复用上一轮检索证据；新主题触发重新检索，降低对话跳题造成的上下文污染。
- 将企业地区、行业、成立年限、员工规模和注册资本作为可选背景，支持更针对性的政策回答。

### 3. 可追溯回答与服务封装

- Prompt 要求回答严格基于检索内容，无法回答时明确拒答。
- API 返回答案、对话历史以及政策标题、政策依据和原文摘录。
- 使用 FastAPI 封装 `/api/policy-qa`，并挂载静态前端，形成端到端演示原型。
- 日志记录请求、路由判断、FAQ 命中、检索文档和最终回答，便于错误分析。

## 论文评测结果

评测使用 238 个相互隔离的问答对（20 个真实高频问题 + 218 个经筛选的合成问题），避免直接用构建知识库时的提示样本作测试。

### 检索层

| 方法 | 代表性结果 |
| --- | ---: |
| 摘要检索 Context Recall | **0.964** |
| 对比方法 | BM25、原文向量、Hybrid、摘要检索 |

### 端到端 Hybrid RAG

| 指标 | 结果 |
| --- | ---: |
| Factual Correctness | 0.5630 |
| Faithfulness | **0.9600** |
| Semantic Similarity | **0.8280** |

这些结果表明系统对检索证据的忠实度较高，但事实覆盖仍有提升空间；项目不把评测分数等同于真实生产准确率。

## 代码导航

- [FastAPI 服务入口](api.py)：请求模型、检索路由、缓存证据和来源返回。
- [RAG 主链](rag_chain/chain.py)：Prompt、检索、企业背景拼接与答案生成。
- [摘要向量检索](rag_chain/retriever_config.py)：BGE Embedding、Chroma 与 MultiVectorRetriever。
- [检索决策 Agent](rag_chain/retrieval_decision_agent.py)：新话题/追问路由。
- [FAQ 匹配](rag_chain/faq_matcher.py)：高频问题快速通道。
- [多轮对话记忆](rag_chain/memory.py)：历史消息构造与窗口控制。
- [日志模块](rag_chain/log_utils.py)：关键链路可观测性。
- [前端](frontend)：问答交互页面。

## 本地运行

推荐 Python 3.10。首次运行会下载 `BAAI/bge-large-zh-v1.5`，CPU 环境加载时间可能较长。

~~~bash
git clone https://github.com/Lagomfzx/rag-policy-demo.git
cd rag-policy-demo

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 在 .env 中填写 DEEPSEEK_API_KEY；不要把真实密钥提交到 GitHub

uvicorn api:app --host 127.0.0.1 --port 8001
~~~

浏览器访问 `http://127.0.0.1:8001/frontend/`，API 文档位于 `http://127.0.0.1:8001/docs`。

### API 示例

~~~bash
curl -X POST http://127.0.0.1:8001/api/policy-qa \
  -H "Content-Type: application/json" \
  -d '{
    "query": "武汉的小微企业可以申请哪些融资支持？",
    "history": [],
    "enterprise_info": {
      "region": "武汉",
      "industry": "软件服务",
      "years": 3,
      "employees": 25,
      "capital": "500万元"
    }
  }'
~~~

## 研究边界与后续工作

- 当前知识库覆盖特定地区和版本的政策文本，政策变化后需要重新清洗、索引和评测。
- 原型使用进程内缓存，不适合多用户并发；生产化需要会话隔离、持久化缓存和权限控制。
- 需要继续补充拒答/冲突政策测试、检索消融实验和线上反馈闭环。
- 仓库中的成本测算属于论文场景估计，不应解读为真实商业运营结果。

## 技术栈

Python · LangChain/LCEL · BGE Embedding · Chroma · MultiVectorRetriever · DeepSeek · FastAPI · RAG Evaluation · HTML/CSS/JavaScript
