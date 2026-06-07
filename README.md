# RAG Agent 项目说明

参考开源项目：https://github.com/icey1287/SuperMew 做的本地实现。

以下是Agent的项目记录，方便后续持续更新与展示。

<img width="2559" height="1410" alt="image" src="https://github.com/user-attachments/assets/bb0b13fc-5e39-4d53-b3d3-5ace3e8e7b2a" />

RAGAS评分

<img width="989" height="590" alt="ragas_scores" src="https://github.com/user-attachments/assets/1c7705b5-cf1e-453a-9ac1-97ddaf210091" />

## RAGAS评估步骤

1. 准备问答数据集eval_questions.csv。（列名：question, ground_truth）
2. uv run python backend/app.py开启RAG后端服务后，批量导入dify的workflow，生成result.csv
3. 运行parse_dify_result.py，生成eval_ready.csv
4. 运行 RAGAS 评测: RAGAS_evaluation_manual.ipynb
5. 得到ragas_scores_manual.csv和ragas_scores.png

## 本地部署
### 1) 环境准备
- Python `3.12+`
- 包管理建议：`uv`（也支持 `pip`）
- Docker / Docker Compose（用于启动 Milvus 依赖）

### 2) 使用 pyproject 安装依赖
在项目根目录执行：

```bash
# 推荐（uv）
uv sync

# 运行服务
uv run python backend/app.py
```

### 3) 创建 `.env` 文件
在项目根目录新建 `.env`，可直接使用下面模板：

```env
# ===== Model （API KEY直接输入环境变量名，本项目已经适配隐式传递API KEY）=====
ARK_API_KEY=your_ark_api_key
MODEL=your_model_name
BASE_URL=https://your-llm-endpoint/v1
EMBEDDER=your_embedding_model

# ===== Rerank (可选，不配则自动降级，本项目没有RERANK环节，适合小量数据) =====
RERANK_MODEL=your_rerank_model
RERANK_BINDING_HOST=https://your-rerank-host
RERANK_API_KEY=your_rerank_api_key

# ===== Milvus =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=embeddings_collection

# ===== Database / Cache =====
DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/langchain_app
REDIS_URL=redis://127.0.0.1:6379/0

# ===== Auth =====
JWT_SECRET_KEY=replace-with-strong-random-secret
ADMIN_INVITE_CODE=supermew-admin-2026
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
PASSWORD_PBKDF2_ROUNDS=310000

# ===== Tools （可选，这里的是查询天气功能）=====
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=your_amap_api_key

```

### 4) Docker 部署（数据库 + 缓存 + 向量库）
当前仓库的 `docker-compose.yml` 同时承载业务依赖与 Milvus 依赖：
- 业务依赖：`postgres`、`redis`
- 向量依赖：`etcd`、`minio`、`standalone`、`attu`

```bash
# 启动向量库依赖
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志（可选）
docker compose logs -f standalone
```

端口说明：
- PostgreSQL：`5432`
- Redis：`6379`
- Milvus：`19530`
- Milvus 健康检查：`9091`
- MinIO API：`9000`
- MinIO Console：`9001`
- Attu：`8080`

浏览器访问：
- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

## 项目概览
- **核心能力**：
  - LangChain Agent + 自定义工具。
  - 文档上传后执行三级滑动窗口分块，叶子分块向量化写入 Milvus，父级分块写入 PostgreSQL。
  - 用户注册/登录、JWT 鉴权、基于角色的 RBAC 权限控制（admin/user）。
  - 会话记忆与摘要，聊天与历史记录落地 PostgreSQL，并引入 Redis 缓存热点会话与父文档。
- **运行形态**：FastAPI 后端 + 纯前端（Vue 3 CDN 单页）+ Milvus 向量库。

## 关键创新点
- **混合检索落地**：稠密向量 + BM25 稀疏向量，Milvus Hybrid Search + RRF 排序，兼顾语义与词匹配。
- **Jina Rerank 接入**：Hybrid/Dense 召回后进行 API 级精排，支持返回 `rerank_score` 并在前端可视化。
- **双向降级**：稀疏生成或 Hybrid 调用失败时自动降级为纯稠密检索，提升稳定性。
- **流式输出（Streaming）**：后端基于 `agent.astream(stream_mode="messages")` 逐 token 推送，前端 SSE + ReadableStream 实现打字机效果。
- **实时 RAG 过程可视化**：检索过程在模型"思考中"阶段就开始展示，通过 `asyncio.Queue` + 后台任务架构实现工具执行期间的实时推送。
- **回答终止功能**：前端 `AbortController` + 后端 `StreamingResponse` 支持用户随时中断正在生成的回答。
- **会话摘要记忆**：自动摘要旧消息并注入系统提示，维持上下文且控制 token。
- **文档处理链路**：上传 → 切分 → 稠密/稀疏向量同步生成 → Milvus 入库，支持重复上传自动清理旧 chunk。
- **三级分块 + Auto-merging**：L1/L2/L3 三层滑窗切分；检索时优先召回 L3，满足阈值后自动合并到父块（L3->L2->L1）。
- **Leaf-only 向量化存储**：仅叶子分块写入 Milvus，父块写入 DocStore，减少向量冗余并保留上下文聚合能力。
- **工具可扩展**：天气查询示例 + 知识库检索，便于按需增添第三方 API 或企业数据源。
- **RAG 过程可观测**：记录检索、评分、重写与来源信息，前端可展开查看每一步细节。
- **查询重写体系**：Step-Back 与 HyDE 两种扩展方式 + 路由选择，必要时触发重写检索。
- **相关性评分门控**：基于结构化输出的 `grade_documents` 判断是否需要重写检索。
- **实时思考链路展示**：通过 `asyncio` 事件循环穿透技术，实现 Agent 在执行 RAG、评分、重写等同步工具时，实时向前端推送思考步骤（Searching -> Grading -> Rewriting），彻底解决"静默思考"问题。

## 目录与架构
- 后端：`backend/`
  - [app.py](backend/app.py)：FastAPI 入口、CORS、静态资源挂载。
  - [api.py](backend/api.py)：聊天、会话管理、文档管理接口。
  - [auth.py](backend/auth.py)：注册登录、JWT 鉴权、权限检查、密码哈希与校验。
  - [database.py](backend/database.py)：数据库引擎与会话工厂、建表入口。
  - [models.py](backend/models.py)：ORM 模型定义（用户、会话、消息、父文档）。
  - [cache.py](backend/cache.py)：Redis JSON 缓存封装。
  - [agent.py](backend/agent.py)：LangChain Agent、会话存储、摘要逻辑。
  - [tools.py](backend/tools.py)：天气查询、知识库检索工具。
  - [embedding.py](backend/embedding.py)：稠密向量 API 调用 + BM25 稀疏向量生成。
  - [document_loader.py](backend/document_loader.py)：PDF/Word 加载与分片。
  - [parent_chunk_store.py](backend/parent_chunk_store.py)：父级分块仓储（PostgreSQL + Redis，用于 Auto-merging 回取父块）。
  - [milvus_writer.py](backend/milvus_writer.py)：向量写入（稠密+稀疏）。
  - [milvus_client.py](backend/milvus_client.py)：Milvus 集合定义、混合检索。
  - [schemas.py](backend/schemas.py)：Pydantic 请求/响应模型。
- 前端：`frontend/`
  - [index.html](frontend/index.html) + [script.js](frontend/script.js) + [style.css](frontend/style.css)：Vue 3 + marked + highlight.js，提供聊天、历史会话、文档上传/删除界面。
- 数据：`data/`
  - `documents/`：上传文档原文件。
- 向量库：Milvus（可由 `docker-compose` 或自建服务提供）。

## 核心流程

### 1) 项目全链路（端到端）
1. 用户在前端输入问题，调用 `POST /chat/stream`（流式）。
2. FastAPI `api.py` 返回 `StreamingResponse(media_type="text/event-stream")`。
3. LangChain Agent 根据问题类型决定是否调用工具：
  - 天气问题 → `get_current_weather`
  - 知识问答 → `search_knowledge_base`
4. 若命中知识库工具，进入 `rag_pipeline.py` 执行检索工作流，各阶段通过 `emit_rag_step()` 实时推送到前端。
5. 检索结果与 RAG Trace 一起返回，Agent 流式生成最终回答（逐 token 推送）。
6. 前端 ReadableStream 逐块解析 SSE，打字机效果实时渲染。
7. 同时消息持久化到 PostgreSQL，并通过 Redis 缓存加速历史会话回放。

### 2) RAG 全链路（重点）
1. **初次召回**：`retrieve_initial`
  - 调用 `retrieve_documents`。
  - 先按 `chunk_level == 3` 执行 Milvus Hybrid 检索（Dense + Sparse + RRF）。
  - 取更大候选集后走 Jina Rerank 精排。
  - 对召回叶子块执行 Auto-merging（L3->L2->L1），父块从 DocStore 读取。
2. **相关性打分门控**：`grade_documents`
  - 使用结构化输出打分 `yes/no`。
  - `yes` 直接进入生成回答；`no` 进入重写阶段。
3. **查询重写路由**：`rewrite_question`
  - 在 `step_back / hyde / complex` 中选择策略。
  - 生成 `rewrite_query`、`step_back_question`、`hypothetical_doc` 等中间结果。
4. **二次召回**：`retrieve_expanded`
  - 对重写后的查询（或 HyDE 文档）再次检索。
  - 同样执行 L3 召回 + Auto-merging，结果去重后返回上下文。
5. **答案生成**：Agent 结合上下文生成最终回答。
6. **可观测追踪**：返回 `rag_trace`，包括
  - 评分结果与路由决策
  - 重写策略与重写内容
  - 初次/二次检索结果
  - 三级检索与合并信息（`leaf_retrieve_level`、`auto_merge_*`）
  - 检索分数 `score` 与精排分数 `rerank_score`

### 3) 文档入库链路
1. 前端上传 PDF/Word 到 `POST /documents/upload`。
2. `document_loader.py` 执行三级滑动窗口分块并写入层级元数据（chunk_id / parent_chunk_id / root_chunk_id / chunk_level）。
3. L1/L2 父级分块写入 `parent_chunk_store.py`（DocStore）。
4. L3 叶子分块进入 `embedding.py` 生成 Dense 向量与 BM25 Sparse 向量。
5. `milvus_writer.py` 仅将叶子块向量 + 元数据写入 Milvus。
5. 后续检索可直接利用新文档参与召回。

### 4) 会话记忆链路
1. 每轮问答按当前登录用户 + `session_id` 写入 PostgreSQL。
2. 当消息过长时触发摘要压缩，保留长期上下文。
3. Redis 缓存会话列表与会话消息，减少高频读取数据库压力。
4. 前端可通过会话接口读取、删除当前用户自己的历史对话。

## 技术栈
- 后端：FastAPI、LangChain Agents、Pydantic、Uvicorn、SQLAlchemy、PostgreSQL、Redis。
- 向量与检索：Milvus（HNSW 稠密索引 + SPARSE_INVERTED_INDEX 稀疏索引）、RRF 融合、Jina Rerank 精排。
- 嵌入与稀疏：自定义 API 调用获取稠密向量；BM25 手写稀疏向量；同时输出双塔特征。
- 前端：Vue 3 (CDN)、marked、highlight.js、纯静态部署。
- 工具链：dotenv 配置、requests、langchain_text_splitters、langchain_community.loaders。

## 环境变量
需在仓库根目录或运行环境配置：
- 模型相关：`ARK_API_KEY`、`MODEL`、`BASE_URL`、`EMBEDDER`
- Rerank 相关：`RERANK_MODEL`、`RERANK_BINDING_HOST`、`RERANK_API_KEY`
- Milvus：`MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION`
- 数据库缓存：`DATABASE_URL`、`REDIS_URL`
- 鉴权相关：`JWT_SECRET_KEY`、`ADMIN_INVITE_CODE`、`JWT_ALGORITHM`、`JWT_EXPIRE_MINUTES`
- 密码参数：`PASSWORD_PBKDF2_ROUNDS`
- Auto-merging：`AUTO_MERGE_ENABLED`、`AUTO_MERGE_THRESHOLD`、`LEAF_RETRIEVE_LEVEL`
- 工具：`AMAP_WEATHER_API`、`AMAP_API_KEY`

## API 速览
- 鉴权
  - `POST /auth/register`：注册（支持普通用户/管理员邀请码模式）。
  - `POST /auth/login`：登录，返回 Bearer Token。
  - `GET /auth/me`：获取当前登录用户信息。
- 聊天
  - `POST /chat`：聊天（非流式），入参 `message`、`session_id`。
  - `POST /chat/stream`：聊天（流式 SSE），入参同上，返回 `text/event-stream`。
- 会话（用户隔离）
  - `GET /sessions`：列出当前用户会话。
  - `GET /sessions/{session_id}`：拉取当前用户某会话消息。
  - `DELETE /sessions/{session_id}`：删除当前用户会话。
- 文档（管理员权限）
  - `GET /documents`：列出已入库文档及 chunk 数。
  - `POST /documents/upload`：上传并向量化 PDF/Word/Excel。
  - `DELETE /documents/{filename}`：删除指定文档向量数据。

## 整体架构

```
用户发送消息
    │
    ▼
POST /chat/stream → StreamingResponse(text/event-stream)
    │
    ▼
chat_with_agent_stream()
    │
    ├── 创建统一输出队列 (asyncio.Queue)
    ├── 设置 _RagStepProxy → emit_rag_step() 的输出直接入队
    ├── 启动 _agent_worker 后台任务 (asyncio.create_task)
    │     └── agent.astream(stream_mode="messages") 逐 token 产出
    │           ├── AIMessageChunk (文本) → {"type": "content"} 入队
    │           └── tool_call_chunks (工具调用) → 跳过
    │
    └── 主循环：await output_queue.get() → yield SSE
          ▲
          │ (并发) RAG 工具在线程池中执行
          │ emit_rag_step() → loop.call_soon_threadsafe → 入队
          │ {"type": "rag_step"} 立即从队列取出并推送到前端
```

## 学习日志

### 一、架构认知：从零到一的全局视角

学习这个项目首先让我建立了一个完整的系统架构认知。项目采用 **uv + Docker** 的双引擎架构：
- **uv**：负责 Python 运行环境，管理依赖和启动 FastAPI 后端。
- **Docker**：负责基础设施服务，包括 PostgreSQL、Redis、Milvus 等。

这种分离设计让我理解了"运行时代码"与"数据层服务"的职责边界，代码不再耦合具体的存储实现。

### 二、混合检索：理论与落地的桥梁

混合检索是本项目最核心的技术亮点之一，也是花费我最多时间理解的部分。

**1. 稠密向量 + BM25 稀疏向量的双塔结构**
- 稠密向量由通义千问 API 生成（1024 维），负责语义级别的匹配。
- BM25 稀疏向量基于 jieba 分词，负责关键词精确匹配。
- 两者互补：语义理解 + 词频匹配。

**2. RRF 融合（倒数排名融合）**
- 对两路召回结果按排名进行融合，k=60 是常用参数。
- 这让我理解了"先召回、后精排"的两阶段设计理念：Recall 阶段追求覆盖广度，Rerank 阶段追求排序精度。

**3. Jina Rerank 精排**
- 在混合检索后调用 Jina API 进行二次排序。
- 返回的 `rerank_score` 可视化到前端，让用户直观看到相关性。

### 三、三级分块与 Auto-merging：检索粒度的艺术

**三级滑动窗口分块（L1/L2/L3）** 是这个项目最巧妙的设计之一：
- 仅将 L3 叶子分块写入 Milvus（检索粒度），父块（L1/L2）存入 PostgreSQL（阅读粒度）。
- 这解决了 RAG 中"最优检索粒度"和"最优阅读粒度"不一致的问题。

**Auto-merging 检索策略**：
- 先用最小颗粒 L3 找到答案，再自动合并到父块补充上下文。
- 检索流程：L3 召回 → Auto-merging → 父块回取 → 上下文聚合。

### 四、查询重写与路由：让检索更聪明

**Step-Back 策略**：
- 将具体问题"退一步"抽象为通用问题，先获取背景知识，再辅助检索。
- 扩展后的查询包含：原始问题 + 抽象问题 + 背景答案。

**HyDE 假设文档生成**：
- 让 LLM 先写一段"像真实文档"的假设文本，再用这段文本去做检索。
- 本质是把 Query 翻译成"像文档一样的语言"。

**策略路由器**：
- 在 RAG 开始前判断"这次问题该走哪条检索路线"。
- 通过 `RAGState` 在 pipeline 各节点间传递状态，是 LangGraph 状态机式 RAG 的典型实现。

### 五、流式输出与实时思考可视化

**跨线程事件调度** 是这个功能的技术核心：
- 后台线程（LangChain 工具）执行检索时，通过 `loop.call_soon_threadsafe` 将事件入队到主线程。
- 主线程负责 SSE 推送，前端实时看到 "Searching → Grading → Rewriting" 的思考链路。

**SSE（Server-Sent Events）**：
- 后端不是一次性发送完数据，而是一小段一小段往外"滴"。
- 前端需要手动解析 SSE 事件，根据 `type` 分发处理。

### 六、高级特性：会话记忆与缓存策略

**会话摘要记忆**：
- 当聊天过长时，自动摘要旧消息并注入系统提示。
- 摘要是以 SystemMessage 形式加入的，对模型来说是"背景事实"。

**Redis 缓存策略**：
- 用户聊天记录、会话列表、父级分块内容都会先查缓存，缓存未命中再查数据库。
- 写入/删除时自动失效缓存，保证一致性。

### 七、代码改动：BM25 状态持久化升级

在检查改动部分，我深入理解了 BM25 从"进程内临时统计"到"可持久化状态"的升级：

**1. 本地模型切换**
- 原来调用阿里云 API 生成稠密向量。
- 改动后使用 `langchain_huggingface` 的 `HuggingFaceEmbeddings`（默认 BAAI/bge-m3），支持多语言和工业级质量。

**2. BM25 状态持久化**
- 新增 `_load_state()`、`_persist()`、`_persist_unlocked()` 方法，将 BM25 状态写入磁盘。
- 支持增量更新（`increment_add_documents`）和删除回滚（`increment_remove_documents`）。

**3. 并发安全**
- 使用 `threading.Lock` 保证多进程/多线程场景下 BM25 状态一致性。
- 导出全局单例 `embedding_service`，全进程共享一份状态。

### 八、学习总结

通过这个项目的学习，我掌握了：
- **RAG 系统的完整链路**：从文档上传、分块、向量化，到检索、评分、重写、生成。
- **混合检索的落地实现**：Dense + Sparse + RRF + Rerank 的两阶段设计。
- **LangChain Agent 的工程实践**：流式输出、工具调用、会话管理。
- **异步编程范式**：asyncio.Queue、call_soon_threadsafe、StreamingResponse。
- **状态持久化的工程思维**：从内存变量到磁盘存储，从单机到多进程。

这个项目是一个生产级的 RAG 系统，在检索精度、用户体验和系统稳定性方面都有完善的考虑，值得深入学习和持续跟进。
