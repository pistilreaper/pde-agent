# AGENTS.md — 神经算子PDE智能体竞赛项目

> 本文档面向 AI 编码智能体。阅读本文档前，默认你对本项目一无所知。本文所有信息均基于项目中的实际文件与规则，不做任何假设性推断。
>

---

## 1. 项目概述

本项目是 **“任务4：神经算子PDE智能体”** 的竞赛工程目录，属于 AI for Science（AI4S）场景下的科研工作流 Agent 竞赛。

**核心目标**：开发一个具备深度逻辑推理、架构演进与复杂代码工程能力的 LLM Agent，使其能够在零人工干预环境下，自主驱动“问题理解—模型改进—物理验证—数值复现”的完整科研闭环，针对 PDE 神经算子模型进行诊断、改进与验证。

本项目目前覆盖三类正式任务：

- **Task 1：固定物理环境 1D Burgers 短窗预测**，允许使用官方 Task 1 PDEBench checkpoint 微调。
- **Task 2：多粘性系数 1D Burgers 泛化预测**，测试时不提供粘性系数，必须从头训练。
- **Task 3：Kuramoto-Sivashinsky（KS）方程多参数长时预测**，测试时不提供 $\lambda_2$，必须仅基于官方 KS 训练数据从头训练。

比赛不是单纯提交模型权重，而是提交 Agent 的完整产出：预测结果、科研日志、源代码、耗时记录和方法说明。

**关键合规要求**：

- 提交目录 `code/` 中的**所有代码必须完全由 Agent 自主生成**，不得在 Agent session 结束后由人工编写或修改。
- **初始代码也必须由 Agent 在科研系统启动后自主完成**，不允许将人工预先编写的代码直接放入 `code/` 目录作为初始基线。
- 科研日志（`task{N}_logs.log`）必须完整记录代码从无到有的生成过程，评审会校验 log 中记录的代码生成过程与 `code/` 目录中代码的对应关系。
- 不得调用数值 PDE 求解器用于预测、伪标签、训练标签或额外数据生成；不得生成额外轨迹；不得使用外部数据集。
- Task 1 可使用官方 PDEBench checkpoint；Task 2 和 Task 3 必须从头训练，不得使用公开预训练权重、第三方 checkpoint 或其他任务迁移权重。
- 每个正式任务 session 的 wall-clock 时间必须控制在 12 小时内；每个提交任务的推理时间必须控制在 120 秒内。

---

## 2. 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch（CPU/CUDA 均可，正式训练应优先使用 CUDA） |
| 数据格式 | HDF5（通过 `h5py` 读写） |
| 数值计算 | NumPy、SciPy（不得用于生成额外 PDE 解轨迹） |
| 日志代理 | FastAPI + uvicorn + httpx（用于拦截并记录 LLM API 调用） |
| 编程语言 | Python 3 |

**注意**：本项目是以数据驱动实验为核心的研究型项目，依赖按需安装即可。若安装额外依赖，必须在 Agent 日志中体现安装命令和原因。

---

## 3. 目录结构

```text
.
├── docs/
│   ├── Background.md                          # 赛事背景、赛题介绍、评分规则（中文，必读）
│   ├── NEURAL_OPERATOR_PRINCIPLES.md          # 神经算子、Burgers 与 KS 方程技术原理文档（中文）
│   ├── AGENT_CODE_GUIDE.md                    # Agent 生成训练/推理代码的工程指南
│   ├── AGENTS.md                              # 本文件
│
├── code-ref/                              # 供 Agent 阅读理解的参考实现，只读，不可直接复制到 code/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   ├── utils.py
│   └── eval_checkpoint.py
│
├── baselines-repo/                         # 供 Agent 学习理解的基线算法的原始github仓库，只读，不可直接复制到 code/
│   ├── DeepONet/
│   ├── FNO/
│   ├── MarkovNeuralOperator/
│   ├── PI-DeepONet/
│   ├── PINO/
│   └── U-Net/
│
├── data/
│   ├── task1/               # Task 1 官方数据集及PDEBench训练数据集
│   │   ├── task1_test.hdf5	# Task1 测试集：1000样本 × 10时间步 × 256空间点
│   │   ├── task1_val.hdf5	# Task1 验证集：100样本 × 200时间步 × 256空间点
│   │   ├── 1D_Burgers_Sols_Nu0.001.hdf5	# PDEBench训练集：10000样本 × 200时间步 × 1024空间点
│   ├── task2/               # Task 2 官方数据集
│   │   ├── task2_part0_train.h5	# Task2 训练集Part0：1000样本 × 320时间步 × 256空间点
│   │   ├── task2_part1_train.h5	# Task2 训练集Part1：1000样本 × 320时间步 × 256空间点
│   │   ├── task2_part2_train.h5	# Task2 训练集Part2：1000样本 × 320时间步 × 256空间点
│   │   ├── task2_test.h5	# Task2 测试集：1000样本 × 10时间步 × 256空间点
│   │   └── task2_val.h5	# Task2 验证集：100样本 × 210时间步 × 256空间点
│   ├── task3/               # Task 3 官方数据集
│   │   ├── KS_test.hdf5	    # Task3 测试集：100样本 × 20时间步 × 256空间点
│   │   ├── KS_train.hdf5	    # Task3 训练集：2000样本 × 400时间步 × 256空间点
│   │   ├── KS_val.hdf5	    # Task3 验证集：100样本 × 400时间步 × 256空间点
├── sample_submission/ 	# 提交样例（必须严格参考其目录结构和文件格式）
│   ├── submission.json                # 提交元数据
│   ├── taskx_pred.hdf5                # Taskx 预测结果（样例）
│   ├── taskx_time.csv                 # Taskx 耗时记录（样例）
│   └── taskx_logs.log                 # Taskx 科研日志（样例）
│
├── code/                                  # Agent 运行时自主生成的正式提交代码
├── output/                                # 模型输出、检查点、预测结果（Agent 运行时生成）
└── task_log_sample/                       
	├── README.md                          # LLM调用日志规范说明（必读）
    ├── task1_logs.log                     # 样例日志文件（JSON Lines格式）
    ├── task2_logs.log                     # 样例日志文件（JSON Lines格式）
    └── openai-log/                        # 本地日志代理工具
        ├── proxy.py                       # FastAPI代理服务器，自动记录LLM调用
        └── requirements.txt               # 代理工具依赖：fastapi, uvicorn, httpx
```

### 3.1 `code-ref/` 参考代码库的使用规范

`code-ref/` 目录中存放的是**供 Agent 阅读理解的参考实现**，其目的是帮助 Agent 快速理解数据加载、模型架构、训练流程、评分计算等核心模块的工程化写法。

**Agent 对 `code-ref/` 的使用必须遵循以下原则**：

1. **允许阅读与学习**：Agent 可以读取 `code-ref/` 中的文件，理解其中的设计思路、API 接口、算法逻辑和工程技巧。
2. **禁止直接复制**：Agent **不得**将 `code-ref/` 中的代码直接复制或稍作修改后放入 `code/` 目录作为提交代码。`code/` 目录中的每一行代码都必须是 Agent 在科研迭代过程中**自主构思并生成**的。
3. **鼓励消化吸收后重构**：Agent 应在理解参考代码核心思想的基础上，结合自己的科研假设和实验需求，重新组织代码结构、重新命名变量、重新设计接口，写出符合当前实验目标的实现。
4. **日志必须体现自主生成过程**：科研日志中必须包含 Agent 编写代码时的思考过程，例如"我计划设计一个 Chunked FNO 模型，参考 code-ref/model.py 中的谱卷积思想，但我会调整……"，然后实际写出代码。评审会对比 log 中记录的代码生成过程与 `code/` 中文件的内容一致性。

### 3.2 baselines-repo/ 参考基线的github原始仓库使用规范

`baselines-repo/` 目录中存放的是**供 Agent 阅读理解的外部开源方法参考仓库**。这些仓库来自公开 GitHub 项目，用于帮助 Agent 学习更先进的神经算子、物理信息学习、长时序预测、混沌系统建模和数据增强思想。

当前 `baselines-repo/` 下包含多个方法子目录，例如：

```text
baselines-repo/
├── DeepONet/
├── FNO/
├── MarkovNeuralOperator/
├── PI-DeepONet/
├── PINO/
└── U-Net/
```

每个方法子目录应遵循统一结构：

```text
方法名/
├── README.md          # 面向 Agent 的方法说明与启发式阅读指南
└── source/            # 原始 GitHub 仓库源码
```

其中，`README.md` 是 Agent 的**优先阅读入口**，用于说明该方法的核心思想、适用场景、对当前 PDE 任务可能有帮助的建模启发，以及明确的合规边界；`source/` 目录保存原始 GitHub 仓库内容，包括代码和部分数据，仅作为只读参考材料。

**Agent 对 `baselines-repo/` 的使用必须遵循以下原则**：

1. **允许阅读与理解**：Agent 可以阅读各方法目录下的 `README.md` 和 `source/` 中的原始仓库代码，理解其模型架构、训练目标、损失函数、rollout 策略、物理约束、数据增强和评估方式。
2. **必须先读方法 README**：Agent 在阅读 `source/` 之前，应优先阅读该方法目录下的 `README.md`，明确该仓库“可以提供什么思想”和“哪些内容禁止使用”。不得绕过 README 直接从 `source/` 中复制代码。
3. **禁止直接复制源码**：Agent **不得**将 `baselines-repo/*/source/` 中的任何源码文件、函数、类、配置文件或训练脚本直接复制、改名、删注释后放入最终 `code/` 或 `workspace/submission/code/` 目录。最终提交代码必须由 Agent 在正式科研迭代过程中重新设计并自主生成。
4. **禁止使用外部权重和外部数据**：除竞赛规则明确允许的官方 Task 1 PDEBench checkpoint 外，Agent 不得使用 `baselines-repo/` 中任何仓库提供的预训练权重、下载链接、外部数据集、生成数据脚本或 benchmark 数据。尤其对于 Task 2 和 Task 3，必须严格使用官方给定训练数据从头训练。
5. **禁止调用外部数值求解器生成训练数据或伪标签**：若某个原始仓库包含 PDE solver、data generation、simulation、trajectory generation 等代码，Agent 可以阅读其数学实现以理解问题，但不得调用这些代码为比赛任务生成额外轨迹、伪标签、补充训练集或测试集预测。
6. **鼓励提炼思想后重构实现**：Agent 应将外部仓库中的思想抽象为适合当前任务的自主实现方案。例如，可以学习 Markov Neural Operator 的长时稳定性思想、PINO 的物理残差正则化思想、U-Net 的多尺度结构、FNO 的谱卷积结构、DeepONet 的 Branch-Trunk 表示方式，但必须重新组织代码、重新定义接口、重新实现核心模块。
7. **日志必须体现“学习—提炼—重构”的过程**：科研日志中应清楚记录 Agent 如何阅读参考仓库、提炼哪些方法思想、为什么这些思想适用于当前 Task，以及如何在不复制源码的前提下重新实现。例如：“我阅读了 `baselines-repo/MarkovNeuralOperator/README.md`，决定借鉴其 Markovian rollout 和 dissipativity regularization 思想，但我将重新实现一个轻量的 conditional Chunked FNO，用于 Task 3 的 KS 长时预测。”
8. **不得让外部仓库污染提交代码来源**：`baselines-repo/` 是只读参考区，不是提交代码来源。评审关注的是 `code/` 或 `workspace/submission/code/` 是否能从 Agent 的 LLM 日志和写文件工具调用中追溯生成。任何无法在日志中追溯为 Agent 自主生成的代码，都不得进入最终提交。
9. **遵守原仓库许可证与竞赛规则**：Agent 阅读开源仓库时应尊重其许可证和引用要求。但即使开源许可证允许复制，比赛规则仍然优先：最终提交代码必须由 Agent 自主生成，不能直接搬运外部仓库代码。
10. **若参考方法与比赛约束冲突，以比赛约束为准**：例如，某些仓库可能默认使用外部训练数据、公开 checkpoint、长时间采样、数值 solver 或 teacher-forcing 评估。这些做法不能直接用于正式比赛。Agent 必须根据 Task 1/2/3 的数据、时间、日志和推理限制重新设计可合规执行的实现。

**合规示例**：

- ✅ Agent 阅读 `baselines-repo/PINO/README.md` 后，理解了 KS 方程残差 `u_t + u u_x + λ₂ u_xx + u_xxxx` 可以作为轻量正则项，然后在日志中说明其计划，并自主编写新的 `ks_residual_loss()`。
- ✅ Agent 阅读 `baselines-repo/MarkovNeuralOperator/source/` 后，借鉴“短步 Markov operator + 长时 rollout 验证”的思想，重新实现一个适合 Task 3 的 `rollout()` 函数。
- ✅ Agent 阅读 `baselines-repo/U-Net/README.md` 后，借鉴多尺度 encoder-decoder 思想，但重新设计一个 1D temporal-spatial U-Net 模块。
- ❌ Agent 直接将 `baselines-repo/FNO/source/` 中的 `SpectralConv1d` 复制到 `code/model.py`。
- ❌ Agent 使用 `baselines-repo/PINO/source/` 中的数据生成脚本生成额外 KS 轨迹。
- ❌ Agent 下载并加载外部仓库提供的预训练 checkpoint。
- ❌ Agent 将外部 repo 的训练脚本整体改名为 `train.py` 后放入提交代码目录。

---

## 4. 数据规范

### 4.1 Task 1：固定物理环境（$\nu = 0.001$）

- **测试输入** (`task1_test.hdf5`)：
  - `tensor`: shape `(1000, 10, 256)` — 1000 个样本，前 10 个时间步的初始条件
  - `x-coordinate`: shape `(256,)`
  - `t-coordinate`: shape `(10,)`
- **验证集** (`task1_val.hdf5`)：
  - `tensor`: shape `(100, 200, 256)` — 100 个样本，200 个时间步完整解场
- **预测要求**：基于前 10 步预测未来 190 步，输出 shape `(1000, 200, 256)`；前 10 步必须与测试输入一致（容差 `1e-3`）。

### 4.2 Task 2：多物理环境泛化（变 $\nu$）

- **训练数据**：`task2_part{0,1,2}_train.h5`，每个约 `(1000, 320, 256)`，包含 `nu` 字段。
- **验证集**：`task2_val.h5`，约 `(100, 210, 256)`，包含 `nu`。
- **测试集**：`task2_test.h5`，`(1000, 10, 256)`，**不提供 `nu`**。
- **预测要求**：输出 shape `(1000, 200, 256)`；前 10 步必须与测试输入一致。推理时间必须 ≤ 120 秒。

### 4.3 Task 3：Kuramoto-Sivashinsky 多参数长时预测

Task 3 使用一维 Kuramoto-Sivashinsky 方程：

$$
u_t + u\,u_x + \lambda_2 u_{xx} + u_{xxxx}=0
$$

其中 $\lambda_2 \in [1.0,1.5]$ 是训练集/验证集中给出的扩散系数或能量注入参数；测试集不提供该参数。$u_{xxxx}$ 项提供高波数耗散，非线性项 $u u_x$ 负责尺度间能量转移。模型需要仅根据前 20 个观测步推断隐含动力学参数并预测完整 400 步轨迹。

- **训练集** (`KS_train.hdf5`)：
  - `tensor`: shape `(2000, 400, 256)`，完整轨迹
  - `t-coordinate`: shape `(400,)`
  - `x-coordinate`: shape `(256,)`
  - `lambda2`: shape `(2000,)`
- **验证集** (`KS_val.hdf5`)：
  - `tensor`: shape `(100, 400, 256)`，完整轨迹
  - `lambda2`: shape `(100,)`
- **测试集** (`KS_test.hdf5`)：
  - `tensor`: expected shape `(100, 20, 256)`，仅前 20 个观测步，不含 `lambda2`
  - `t-coordinate`: shape `(20,)`
  - `x-coordinate`: shape `(256,)`

**Task 3 预测要求**：输出 `task3_pred.hdf5`，dataset 名为 `tensor`，shape `(100, 400, 256)`。前 20 步必须与 `KS_test.hdf5` 输入完全一致（容差 `1e-3`），仅步 20–399 为预测结果。

---

## 5. 提交规范

### 5.1 提交文件清单（`submission.zip`）

```text
submission/
├── submission.json
├── task1_pred.hdf5
├── task1_time.csv
├── task1_logs.log
├── task2_pred.hdf5
├── task2_time.csv
├── task2_logs.log
├── task3_pred.hdf5
├── task3_time.csv
├── task3_logs.log
├── methodology.pdf
└── code/
    ├── train.py
    ├── model.py
    └── ...
```

至少提交一个任务即可参与评测。但每个被提交的任务必须同时包含三个文件：`task{N}_pred.hdf5`、`task{N}_time.csv`、`task{N}_logs.log`。

### 5.2 各文件详细规范

**`submission.json`**

```json
{
  "submission_id": "队伍名称",
  "problem_id": "PDE_Burgers",
  "code_path": "code",
  "methodology": "methodology.pdf",
  "submission": "submission.zip"
}
```

**`task{N}_pred.hdf5`**

- Task 1 / Task 2：shape `(N, 200, 256)`，前 10 步必须与测试输入一致。
- Task 3：shape `(100, 400, 256)`，前 20 步必须与测试输入一致。
- dataset 名必须为 `tensor`，dtype 建议 `float32`。

**`task{N}_time.csv`**

```csv
train_time,inference_time
3600,45
```

`train_time` 是该任务的模型训练总耗时，包含 Agent 思考推理时间；`inference_time` 是该任务在测试集上的推理总耗时，单位秒。

**`task{N}_logs.log`**

- 每一行必须是一条合法 JSON。
- 必须包含 `timestamp`、`elapsed_seconds`，并至少包含 `response` 或 `tool_calls`。
- 单个任务 log 首尾记录时间差不得超过 12 小时。
- Task 3 日志还应明确记录：KS 方程动力学特性分析、未知 $\lambda_2$ 的处理策略、模型选型依据、失败实验与迭代结论。

---

## 6. 评分规则

### 6.1 总分结构

不提交 Task 3 时，总分为 Task 1 + Task 2，满分 300 分。提交 Task 3 时，总分上限为 350 分，并按以下两种方案取较高者：

- **方案 A（Task 1 + Task 2 + Task 3）**：总分 = Task 1 得分（最高 150）+ Task 2 得分（最高 150）+ Task 3 分段预测得分 × 0.5（最高 50），上限 350。
- **方案 B（Task 1 + Task 3）**：总分 = Task 1 得分（最高 150）+ Task 3 分段预测得分 × 2（最高 200），上限 350。

### 6.2 Task 1（最高 150 分）

Task 1 总分 = 预测精度得分（最高 75）+ 训练耗时得分（最高 35）+ 推理耗时得分（最高 40）。预测精度得分 = Task 1/2 通用分段预测得分 × 0.75。

### 6.3 Task 2（最高 150 分）

Task 2 总分 = Task 1/2 通用分段预测得分 × 1.5。训练时间不计入精度得分，但总时长需控制在 12 小时内；推理超过 120 秒则该任务得 0 分。

### 6.4 Task 1 / Task 2 通用分段预测得分

评分仅针对 190 个预测时间步（去掉前 10 个初始条件），分为 3 段：

| 段 | 预测步范围 | 权重 | 评分公式 |
|---|---:|---:|---|
| 第 1 段 | 0–47 | 25% | `100 × exp(-20 × Rel-MSE)` |
| 第 2 段 | 47–95 | 25% | `100 × exp(-10 × Rel-MSE)` |
| 第 3 段 | 95–190 | 50% | `max(Lorentzian, Frechet)` |

其中 `Lorentzian = 100 / (1 + 10 × RMSE)`，`Frechet = 50 × exp(-FD²)`。

### 6.5 Task 3 专用分段预测得分

评分仅针对 380 个预测时间步（去掉前 20 个观测步），分为 3 段：

| 段 | 完整轨迹步范围 | 物理时间 | 权重 | 评分公式 |
|---|---:|---:|---:|---|
| 第 1 段 | 20–49 | $t \in [10,24.5]$ | 25% | `100 × exp(-20 × Rel-MSE1)` |
| 第 2 段 | 50–199 | $t \in [25,99.5]$ | 25% | `100 × exp(-10 × Rel-MSE2)` |
| 第 3 段 | 200–399 | $t \in [100,199.5]$ | 50% | `max(Lorentzian, Frechet)` |

第 3 段中 `Lorentzian = 100 / (1 + 10 × RMSE3)`，`Frechet = 50 × exp(-FD²)`。分段预测总分 = `0.25 × s1 + 0.25 × s2 + 0.50 × s3`。

---

## 7. 科研方向与基线模型

### 7.1 基线模型

- **FNO**：频域卷积，适合规则网格，推断极快。
- **DeepONet**：Branch-Trunk 架构，无网格，适合不规则查询点。
- **PI-DeepONet / PINO**：在神经算子训练中引入 PDE 残差或物理约束。

### 7.2 Task 1 / Task 2 典型优化策略

- 物理残差约束：Burgers 残差 $u_t + u u_x - \nu u_{xx}$。
- Pushforward / Curriculum Rollout：训练时逐步增加自回归步长。
- FiLM 条件化与 $\nu$ 估计器：服务 Task 2 的未知粘性系数测试设定。
- 谱正则化与时序捆绑：抑制虚假高频并提升长时稳定性。

### 7.3 Task 3 典型优化策略

- **短窗参数识别**：训练时使用 `lambda2` 监督一个轻量参数编码器；推理时仅从前 20 步估计隐含参数或潜变量。
- **条件化长时预测**：将 $\lambda_2$ 或其估计嵌入通过 FiLM、AdaLN、条件 bias 或 latent token 注入 FNO / Transformer / U-Net 主干。
- **多步 rollout 训练**：验证和推理必须完整预测 380 个未来步，训练中应避免只做 teacher forcing 单步预测。
- **统计段对齐**：Task 3 后半段更重视统计结构，可加入谱能量、空间均值/方差、时间差分和分布匹配损失，但不能使用测试真值。
- **稳定性优先**：KS 具有混沌特性，逐点误差长时会快速放大。模型应优先保证长时轨迹的幅值、谱能量和统计分布合理。

---

## 8. 开发惯例与代码规范

### 8.1 代码组织

`code/` 目录为提交入口，应包含完整可运行的训练与推理代码。建议至少包含：

- `train.py`
- `infer.py`
- `model.py`
- `dataset.py`
- `utils.py`

代码必须完全由 Agent 自主生成，不得在 Agent session 结束后人工修改。

### 8.2 三任务统一接口

`train.py` 和 `infer.py` 应至少支持：

```bash
python code/train.py --task task1 --output_dir output/task1/iter_N --data_dir ./data/task1
python code/train.py --task task2 --output_dir output/task2/iter_N --data_dir ./data/task2
python code/train.py --task task3 --output_dir output/task3/iter_N --data_dir ./data/task3

python code/infer.py --task task1 --checkpoint output/task1/iter_N/best_checkpoint.pt --output output/task1/iter_N/pred.hdf5 --data_dir ./data/task1
python code/infer.py --task task2 --checkpoint output/task2/iter_N/best_checkpoint.pt --output output/task2/iter_N/pred.hdf5 --data_dir ./data/task2
python code/infer.py --task task3 --checkpoint output/task3/iter_N/best_checkpoint.pt --output output/task3/iter_N/pred.hdf5 --data_dir ./data/task3
```

### 8.3 实验记录

`task{N}_logs.log` 是评审核心材料，必须完整记录 Agent 的思考链路和实验轨迹。Task 3 日志尤其需要说明：

- 为什么 KS 长时预测不能只看逐点 MSE；
- 如何利用训练时可见的 `lambda2`，以及为什么推理时不能读取该字段；
- 如何保证前 20 步复制一致；
- 如何控制推理时间在 120 秒以内；
- 失败实验如何改变下一轮架构或损失设计。

---

## 9. 安全与合规

- 严禁人工干预最终提交代码、预测数组和 Agent 日志。
- 严禁调用数值求解器生成额外数据或伪标签。
- 严禁下载外部训练数据。
- Task 2 和 Task 3 严禁加载公开预训练权重或其他任务 checkpoint。
- Task 3 必须从官方 KS 训练集 `KS_train.hdf5` 从头训练。
- 单任务 log 首尾时间差超过 12 小时视为违规风险。
- 推理时间超过 120 秒的任务会得到 0 分。

---

## 10. 关键外部资源

- FNO 官方实现：https://github.com/neuraloperator/neuraloperator
- DeepONet 官方实现：https://github.com/lululxvi/deeponet
- PI-DeepONet 官方实现：https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets
- PDEBench 数据集：https://doi.org/10.18419/darus-2986
- KS 方程背景：Kuramoto 与 Sivashinsky 关于相位湍流、火焰前沿不稳定性和薄膜流动的经典研究；详见 `NEURAL_OPERATOR_PRINCIPLES.md`。

---

## 11. 给 Agent 的快速行动清单

1. [ ] 仔细阅读 `Background.md`、`NEURAL_OPERATOR_PRINCIPLES.md`、`AGENT_CODE_GUIDE.md`。
2. [ ] 判断当前正式任务是 task1、task2 还是 task3；不要跨任务混用数据、checkpoint 或日志。
3. [ ] 探查对应任务的 HDF5 文件结构、shape、key 和 dtype。
4. [ ] 阅读 `code-ref/` 参考实现，只读学习，不直接复制。
5. [ ] 自主生成 `code/model.py`、`code/dataset.py`、`code/train.py`、`code/infer.py`、`code/utils.py`。
6. [ ] Task 1/2 验证输出 shape `(N,200,256)`，前 10 步一致。
7. [ ] Task 3 验证输出 shape `(N,400,256)`，前 20 步一致。
8. [ ] 记录完整代码生成、训练、失败分析和迭代过程。
9. [ ] 生成 `task{N}_time.csv` 和 `task{N}_logs.log`。
10. [ ] 打包前运行 final validation，确认 code/log/pred/time 对应一致。
