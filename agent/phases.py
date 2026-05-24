"""
四阶段科研闭环实现

1. Literature Phase   - 文献解析与逻辑解构
2. Diagnosis Phase    - 瓶颈诊断与假设提出
3. Design Phase       - 自主设计与代码演进
4. Experiment Phase   - 实验验证与科学迭代
"""
import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

from .llm_client import LLMClient
from .memory import ResearchMemory, ExperimentRecord
from .task_specs import TASK_CHOICES, get_task_spec, resolve_task_data_dir
from .tools import registry


SYSTEM_PROMPT = """你是PDE神经算子科研智能体，具备深度学习、偏微分方程和科学计算的全面知识。
你的目标是在零人工干预下，自主完成神经算子模型的研究、改进与验证。

核心能力：
- 深度阅读技术文档与论文，提取关键数学公式和算法逻辑
- 分析训练日志与实验数据，诊断模型瓶颈
- 提出有科学依据的优化假设，并直接编写高质量PyTorch代码
- 运行实验、分析结果、迭代改进

行事原则：
1. 严谨：每个假设必须有理论依据或数据支撑
2. 务实：优先选择工程上可行、效果可验证的方案
3. 记录：完整记录思考链路与实验轨迹
4. 迭代：不怕失败，从失败中提取信息指导下一次尝试

当前环境：
- 深度学习框架: PyTorch
- 任务: 1D Burgers方程神经算子预测
- 基线模型: FNO (Fourier Neural Operator)
- 数据格式: HDF5

================================================================================
【CLI参数兼容性要求 - 这是最关键的工程约束，连续5次实验失败均源于此】
================================================================================

实验调度器(runner)会以**固定格式**调用你的脚本。你必须确保生成的 train.py 和 infer.py
的参数接口与此完全兼容。任何不兼容都会导致实验在 argparse 阶段直接退出，没有任何训练发生。

训练命令格式（runner自动生成）：
  python code/train.py --task {task1|task2|task3} --output_dir output/{task1|task2|task3}/iter_N --data_dir <task-local-data-dir>

推理命令格式（runner自动生成）：
  python code/infer.py --task {task1|task2|task3} --checkpoint output/{task1|task2|task3}/iter_N/best_checkpoint.pt --output output/{task1|task2|task3}/iter_N/pred.hdf5 --data_dir <task-local-data-dir>

【argparse 强制规范】
1. train.py 和 infer.py 必须显式支持 --task、--output_dir、--data_dir
2. 必须使用 argparse 别名机制，同时支持下划线和横线版本：
   parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./output")
   parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="./data_and_sample_submission/train_val_test_init")
3. 所有参数必须有合理的默认值，确保即使只传 --task 和 --output_dir 也能运行
4. checkpoint 文件必须保存为 best_checkpoint.pt（参考代码的命名）
5. --device 参数默认值必须是 `"cuda" if torch.cuda.is_available() else "cpu"`，不可硬编码为 "cpu"
6. 若使用 parse_known_args() 作为兜底，必须确保 output_dir 和 data_dir 已被显式定义，否则不应简单忽略

【常见失败模式（必须避免）】
- ❌ train.py 只认识 --out-dir，不认识 --output_dir → runner 注入 --output_dir 时直接报错退出
- ❌ infer.py 不认识 --task → runner 注入 --task 时直接报错退出
- ❌ 没有 --data_dir 参数 → runner 不会传入数据路径，脚本使用硬编码路径可能找不到数据
- ❌ checkpoint 保存为 best_model.pt，但 runner 寻找 best_checkpoint.pt → 推理失败
- ❌ --device 被硬编码为 "cpu"，在有 GPU 的环境下浪费计算资源 → 训练时间过长
- ❌ 参数是 required=True 但 runner 没有传入 → argparse 直接退出

================================================================================
【代码生成规范 - 参考 AGENT_CODE_GUIDE.md】
================================================================================

必须生成的5个核心文件及其职责：
- code/model.py: 神经算子模型（FNO/ChunkedFNO），含 SpectralConv1d, FNOBlock1d, FiLM
- code/dataset.py: 数据加载与标量归一化（Normalizer, BurgersDataset, WindowedBurgersDataset）
- code/train.py: 训练入口，支持加载已有checkpoint、验证、早停、学习率调度、保存 checkpoint
- code/infer.py: 推理入口，加载 checkpoint 生成提交格式的 HDF5
- code/utils.py: 评分计算（compute_segment_scores）、辅助损失、工具函数

【参考实现学习要求】
- 你不能只根据 docs 自行臆造训练代码；在设计 train.py、infer.py、model.py、dataset.py、utils.py 时，必须主动阅读 `code-ref/` 和 `baselines-repo/`。
- `code-ref/` 主要用于学习当前项目期望的代码写法、模块拆分、数据流、CLI 接口、checkpoint 命名和常见训练/推理组织方式。
- 对于 Task 1，不能直接沿用 code-ref 的 Task 1 训练数据选择或切分策略；`code-ref/` 对 Task 1 只能作为 CLI、HDF5 I/O、评分实现、checkpoint 命名和工程组织的参考，不是训练方案本身。
- `baselines-repo/` 主要用于学习可借鉴的方法结构、rollout 策略、损失设计、训练配方与参数设置；当 docs 没有给出明确超参数时，应优先参考这些仓库中的合理默认值与经验范围，而不是完全凭空设定。
- 参考的目标是“学习后自主重写”，不是搬运源码；你不得直接复制 `code-ref/` 或 `baselines-repo/` 中的代码到最终 `code/` 目录。
- 在研究日志和方法说明中，应明确写出你参考了哪些文件、吸收了哪些代码写法或参数设置，以及你如何结合当前 task 重新实现。

【数据流关键约定】
- Task 1 数据: 1D_Burgers_Sols_Nu0.001.hdf5（大型训练集，需从中采样），task1_val.hdf5（验证），task1_test.hdf5（测试）
- Task 2 数据: task2_part{0,1,2}_train.h5（训练），task2_val.h5（验证），task2_test.h5（测试）
- Task 3 数据: KS_train.hdf5（训练），KS_val.hdf5（验证），KS_test.hdf5（测试）
- Task 1 中，task1_val.hdf5 仅作验证；只有在官方 checkpoint 路径和 PDEBench 大训练集路径都被你明确论证为当前实验不可行时，才允许把它作为兜底训练数据来源。
- Task 1 HDF5 key: "tensor", "x-coordinate", "t-coordinate"
- Task 2 HDF5 key: "tensor", "x_coordinate", "t_coordinate", "nu"
- Task 3 HDF5 key: "tensor", "x-coordinate", "t-coordinate", "lambda2"（测试集不含 lambda2）
- 归一化: 全局标量 mean/std，在训练集上计算，共享给 val/test
- 输入: [B, 10, 256]；输出未来帧: [B, 190, 256]；提交 HDF5: [B, 200, 256]（前10步复制GT）
- Task 3 输入: [B, 20, 256]；输出完整轨迹: [B, 400, 256]（前20步复制GT，后380步预测）

【模型架构关键设计】
- 推荐 ChunkedFNO1d（chunk_size=10，自回归rollout到190步），配合 WindowedBurgersDataset 滑动窗口训练
- Lift 层输入 concat 空间坐标通道 (linspace 0~1)
- 残差输出: last_frame.expand(-1, t_out, -1) + project(features)
- 验证时必须做完整 190 步 rollout，不能用 teacher forcing
- Task 2 需内置 nu_estimator（CNN→Pool→Linear），测试时自动估计
- Task 3 需处理 KS 长时混沌 rollout，并在测试时仅根据前20步观测隐式处理未知 lambda2

【评分计算关键细节】
- 3 段式评分，pred/gt 必须是 [B, 190, 256]
- Rel-MSE 必须 clamp(max=5.0)
- Segment 3: score3 = max(100/(1+10*rmse), 50*exp(-fd^2))

【时间优化软约束】
- Task 1 的训练时间目标是尽可能压到 1小时以内，以争取约 35分的时间得分；这不是硬约束，但在精度相近时应优先选择更快方案。
- 所有任务的推理耗时都应尽可能短，不能只满足 120秒 上限；在提交质量可接受的前提下，优先选择更轻量、更快的推理路径。
"""


def _task_prompt_constraints(task: str) -> str:
    spec = get_task_spec(task)
    lines = [
        f"- 当前任务: {spec.display_name} ({spec.equation})",
        f"- 本地数据目录: {resolve_task_data_dir(task)}",
        f"- 输入时间步: {spec.input_steps}",
        f"- 输出形状: {spec.prediction_shape}",
    ]
    if task == "task2":
        lines.extend(
            [
                "- 条件变量: nu；测试时不提供，需要模型自行处理。",
                "- 必须从头训练，不得复用 Task 1 训练权重。",
            ]
        )
    elif task == "task3":
        lines.extend(
            [
                "- 条件变量: lambda2；训练/验证可见，测试时隐藏。",
                "- 前20步必须与输入一致（容差 1e-3），后380步为预测。",
                "- 必须仅使用官方 KS 数据从头训练，不得加载任何公开或外部预训练权重。",
                "- 需要重点关注 KS 方程的混沌长时稳定性与未知参数泛化。",
            ]
        )
    else:
        lines.extend(
            [
                "- Task 1 可使用官方允许的基线检查点（位于checkpoints/文件夹下），并应优先评估是否采用官方 checkpoint 微调或基于 PDEBench 大训练集训练/微调。",
                "- Task 1 的 PDEBench 数据为 PDEBench 官方训练集 `1D_Burgers_Sols_Nu0.001.hdf5`；若不采用它，必须给出明确技术原因。",
                "- Task 1 中 task1_val.hdf5 仅作验证；除非你明确论证官方 checkpoint 和 PDEBench 路径在当前实验不可行，否则不得把它默认当作训练集。",
            ]
        )
    return "\n".join(lines)


class Phase:
    """阶段基类"""
    def __init__(self, client: LLMClient, memory: ResearchMemory, cfg=None):
        self.client = client
        self.memory = memory
        self.cfg = cfg
    
    def run(self) -> bool:
        """执行阶段，返回是否成功完成"""
        raise NotImplementedError
    
    def _chat(self, messages: list, tools: bool = True) -> Dict[str, Any]:
        """调用LLM，可选择是否启用工具"""
        schemas = registry.get_schemas() if tools else None
        return self.client.chat(messages, tools=schemas)
    
    def _tool_call_loop(
        self,
        messages: list,
        max_rounds: int = 20,
        final_instruction: str | None = None,
    ) -> str:
        """
        工具调用循环：让 LLM 反复思考 -> 调用工具 -> 观察结果。
        如果工具轮数耗尽，强制进行一次 no-tool finalization，
        避免把最后一个工具结果或 exhausted error 当成阶段报告。
        """
        for _ in range(max_rounds):
            try:
                resp = self._chat(messages, tools=True)
            except Exception as e:
                return f"[Error] LLM API call failed: {e}"

            tool_calls = resp.get("tool_calls") or []

            if tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": resp.get("content") or "",
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(
                                    tc.get("arguments") or {},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                }

                # DeepSeek V4 Pro thinking mode: 必须回传 reasoning_content
                reasoning_content = resp.get("reasoning_content")
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content

                messages.append(assistant_msg)

                for i, tc in enumerate(tool_calls):
                    tool_call_id = tc.get("id", f"call_{i}")

                    try:
                        result = registry.call(tc["name"], tc.get("arguments") or {})
                    except Exception as e:
                        result = {
                            "ok": False,
                            "error": str(e),
                            "tool_name": tc.get("name"),
                        }

                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result,
                    })

                continue

            return resp.get("content") or ""

        # 关键修改：工具轮数耗尽后，强制收尾，而不是直接返回 error
        messages.append({
            "role": "user",
            "content": final_instruction or (
                "The tool-call budget is exhausted. Do not call any more tools. "
                "Based only on the observations, files, logs, and tool outputs already available "
                "in the conversation, produce the final phase report now. "
                "You must not request more inspection. You must give a concrete conclusion."
            ),
        })

        try:
            final_resp = self._chat(messages, tools=False)
        except Exception as e:
            return f"[Error] Tool call loop exhausted and finalization failed: {e}"

        final_text = final_resp.get("content") or ""
        if not final_text.strip():
            return f"[Error] Tool call loop exhausted after {max_rounds} rounds and finalization returned empty content."

        return final_text


# =============================================================================
# Phase 1: 文献解析与逻辑解构
# =============================================================================

class LiteraturePhase(Phase):
    """文献解析阶段：阅读项目文档、理解数据、分析基线"""
    
    def run(self) -> bool:
        print("\n[Phase 1] 文献解析与逻辑解构...")
        data_dir = resolve_task_data_dir(self.memory.task)
        
        prompt = f"""请执行文献解析与逻辑解构任务。你需要：

1. 阅读项目文档：docs/Background.md、docs/NEURAL_OPERATOR_PRINCIPLES.md、docs/AGENTS.md
2. 检查数据文件结构：使用 inspect_hdf5 查看训练/验证/测试数据
3. 分析现有代码（如有）：使用 summarize_code 查看 code/ 目录下的文件
4. 输出一份结构化的文献与技术综述

请调用工具完成上述任务，然后综合所有信息，回答以下问题：
- 本任务的核心科学问题是什么？
- 基线模型（FNO/DeepONet/PI-DeepONet）的核心数学原理？
- 数据的具体规模、维度、物理含义？
- 当前已知的技术难点与优化方向？
- 评分规则对模型有什么特殊要求？

任务关键约束：
{_task_prompt_constraints(self.memory.task)}

当前任务: {self.memory.task}
数据目录: {data_dir}
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        summary = self._tool_call_loop(messages)
        self.memory.literature_summary = summary
        self.memory.current_phase = "diagnosis"
        
        # 保存综述到文件
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_literature_summary.md",
            "content": f"# 文献与技术综述\n\n{summary}\n",
        })
        
        print("[Phase 1] 完成。文献综述已保存。")
        return True


# =============================================================================
# Phase 2: 瓶颈诊断与假设提出
# =============================================================================

class DiagnosisPhase(Phase):
    """瓶颈诊断阶段：分析基线性能，提出优化假设"""
    
    def run(self) -> bool:
        print("\n[Phase 2] 瓶颈诊断与假设提出...")
        
        context = self.memory.get_context()
        task_focus = {
            "task1": "- 固定参数 Burgers 的长时稳定性与第3段评分\n- 训练效率与验证 rollout 一致性",
            "task2": "- 变 nu 条件化与测试时未知 nu 的鲁棒估计\n- 跨粘性泛化与长时稳定性",
            "task3": "- KS 方程混沌长时 rollout 稳定性\n- 未知 lambda2 下的隐式参数识别\n- 400步轨迹预测中的误差爆炸与统计分布保持",
        }[self.memory.task]
        
        prompt = f"""基于以下研究上下文，进行瓶颈诊断与假设提出：

{context}

你的任务是：
1. 如果已有训练日志或实验结果，使用 analyze_log 分析训练动态
2. 如果已有代码，使用 summarize_code 审查关键模块
3. 基于文献综述、baselines_repo和现有证据，识别当前基线的主要瓶颈：
{task_focus}
4. 对每个瓶颈，提出1-3个具体的、可验证的优化假设
5. 为每个假设给出：理论依据、预期效果、验证方法

请调用工具收集所需信息，然后输出诊断报告。
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        diagnosis = self._tool_call_loop(messages)
        
        # 解析诊断报告中的假设和瓶颈
        self._parse_diagnosis(diagnosis)
        self.memory.current_phase = "design"
        
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_diagnosis_report.md",
            "content": f"# 瓶颈诊断与假设报告\n\n{diagnosis}\n",
        })
        
        print(f"[Phase 2] 完成。识别瓶颈 {len(self.memory.bottlenecks)} 个，提出假设 {len(self.memory.hypotheses)} 个。")
        return True
    
    def _parse_diagnosis(self, text: str):
        """简单解析诊断文本，提取瓶颈和假设"""
        lines = text.splitlines()
        current = None
        for line in lines:
            line = line.strip()
            if "瓶颈" in line.lower() or "bottleneck" in line.lower():
                current = "bottleneck"
                continue
            if "假设" in line.lower() or "hypothesis" in line.lower():
                current = "hypothesis"
                continue
            if line.startswith("-") or line.startswith("*"):
                item = line[1:].strip()
                if current == "bottleneck" and item:
                    self.memory.bottlenecks.append(item)
                elif current == "hypothesis" and item:
                    self.memory.hypotheses.append(item)


# =============================================================================
# Phase 3: 自主设计与代码演进
# =============================================================================

class DesignPhase(Phase):
    """代码设计阶段：根据假设编写/修改代码"""
    
    def run(self) -> bool:
        print("\n[Phase 3] 自主设计与代码演进...")
        
        context = self.memory.get_context()
        data_dir = resolve_task_data_dir(self.memory.task)
        
        # 读取当前代码状态
        code_files = []
        code_dir = "./code"
        if os.path.exists(code_dir):
            for fname in os.listdir(code_dir):
                if fname.endswith(".py"):
                    content = registry.call("read_file", {"path": os.path.join(code_dir, fname)})
                    code_files.append(f"=== {fname} ===\n{json.loads(content).get('content', '')[:1500]}\n")
        
        code_context = "\n".join(code_files) if code_files else "当前 code/ 目录为空或不存在。"
        
        prompt = f"""基于以下上下文，进行代码设计与演进：

{context}

当前代码状态：
{code_context}

【必读】请首先阅读 docs/AGENT_CODE_GUIDE.md，理解参考代码的运行逻辑和工程结构。

你的任务是：
1. 根据当前最优假设，决定需要创建或修改哪些代码文件
2. 使用 write_file 工具直接编写高质量 PyTorch 代码
3. 代码要求：
   - 使用 typing 类型注解
   - 包含清晰的 docstring
   - 支持命令行参数配置
   - 包含错误处理和日志记录
   - 训练过程保存最佳模型

【参数接口强制要求 - 失败5次的核心教训】
你生成的 train.py 和 infer.py 的 argparse 必须与 runner 的调用格式完全兼容。

runner 调用 train.py 的格式：
  python code/train.py --task {self.memory.task} --output_dir output/{self.memory.task}/iter_N --data_dir {data_dir}

runner 调用 infer.py 的格式：
  python code/infer.py --task {self.memory.task} --checkpoint output/{self.memory.task}/iter_N/best_checkpoint.pt --output output/{self.memory.task}/iter_N/pred.hdf5 --data_dir {data_dir}

train.py 必须接受的核心参数（使用 argparse，全部带合理默认值）：
- --task: default="{self.memory.task}", choices={list(TASK_CHOICES)}
- --output_dir / --output-dir: dest="output_dir", default="./output"
- --data_dir / --data-dir: dest="data_dir", default="{data_dir}"
- --model_type: default="chunked", choices=["direct", "chunked"]
- --chunk_size, --epochs, --batch_size, --lr, --weight_decay
- --modes, --width, --depth, --dropout
- --scheduler, --patience, --val_fraction, --seed
- --num_workers, --device, --amp, --grad_clip
- --t_in(=10), --t_out(=190)
- 其他你需要的训练参数

infer.py 必须接受的核心参数：
- --task: default="{self.memory.task}", choices={list(TASK_CHOICES)}
- --checkpoint: required=True
- --output: required=True
- --data_dir / --data-dir: dest="data_dir", default="{data_dir}"
- --batch_size, --num_workers, --device
- 其他你需要的推理参数

【argparse 别名写法示例】
parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./output", help="Output directory")
parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default="{data_dir}", help="Data directory")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")

【checkpoint 命名要求】
训练保存的最佳模型必须命名为 best_checkpoint.pt，因为 runner 硬编码寻找此文件。
同时保存为 best_model.pt 也可以，但 best_checkpoint.pt 必须存在。

必须包含的文件：
- code/model.py: 神经算子模型定义
- code/dataset.py: 数据加载与预处理（Normalizer, BurgersDataset, WindowedBurgersDataset）
- code/train.py: 训练脚本（支持验证、早停、学习率调度、保存 checkpoint）
- code/infer.py: 推理脚本（生成符合提交要求的 HDF5 [B,200,256]）
- code/utils.py: 辅助函数（compute_segment_scores, save_hdf5, Logger, Timer）

        对于 Task 1（固定nu=0.001）：
        - 输入：前10个时间步 (B, 10, 256)
        - 输出：未来190个时间步 + 复制前10步 = (B, 200, 256)
	        - 验证时必须完整 rollout 190 步
	        - 你必须先在“官方 checkpoint 微调”和“基于 PDEBench 大训练集训练/微调”之间做出明确决策，并在日志中说明理由
	        - 不得默认沿用 code-ref 的 Task 1 数据策略；code-ref 只可参考 CLI、I/O、评分与工程组织
	        - PDEBench 官方训练集为 1D_Burgers_Sols_Nu0.001.hdf5；task1_val.hdf5 仅作验证，只有在你明确论证 checkpoint 与 PDEBench 路径当前不可行时才允许作为兜底训练数据
	        - 若你不使用官方 checkpoint 或 PDEBench 大训练集，必须在实验假设与方法说明中写清楚不采用的技术原因
	        - train.py 必须支持 `--dry_run`、`--profile_only`、`--profile_steps`
	        - infer.py 必须支持 `--dry_run`
	        - dry-run / profile 阶段必须写出 `output_dir/preflight.json`
	        - `preflight.json` 至少包含：`train_smoke_ok`, `infer_smoke_ok`, `estimated_total_train_seconds`
	        - 必须给出训练时间估算；如果无法估算训练时间，不能直接启动正式训练
	        - dataset.py 必须为测试集提供独立加载逻辑，不能用“要求至少 t_in+1 帧”的训练逻辑去读取只有前10帧的 task1_test.hdf5

	        对于 Task 2（变nu）：
	        - 训练时可用nu值，测试时不提供nu
	        - 使用 FiLM 进行条件化

对于 Task 3（Kuramoto-Sivashinsky）：
- 输入：前20个时间步 (B, 20, 256)
- 输出：完整400步轨迹 (B, 400, 256)，前20步必须复制输入
- 官方提交输出形状： (100, 400, 256)
- 训练/验证可用 lambda2，测试时不提供；模型需仅凭20步观测完成长时预测
- 必须从头训练，禁止任何公开或跨任务预训练权重复用
- 需要重点考虑混沌系统长时稳定性、参数隐式识别和统计分布保持

重要：每写完一个关键文件后，请调用 validate_code 检查语法。
全部代码写完后，调用 quick_test_model 进行模型前向传播的 smoke test。
最后，用 list_files 确认 code/ 目录结构。
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        result = self._tool_call_loop(messages, max_rounds=20)
        
        self.memory.current_phase = "experiment"
        self.memory.code_versions.append({
            "iteration": self.memory.iteration,
            "timestamp": datetime.now().isoformat(),
            "note": result[:500],
        })
        
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_design_notes.md",
            "content": f"# 设计迭代 {self.memory.iteration}\n\n{result}\n",
        })
        
        print("[Phase 3] 完成。代码已更新。")
        return True


# =============================================================================
# Phase 4: 实验验证与科学迭代
# =============================================================================

class ExperimentPhase(Phase):
    """实验验证阶段：运行训练、评估、分析结果、决定下一步"""

    TRAIN_TIMEOUT_SECONDS = 1800
    INFER_TIMEOUT_SECONDS = 300
    PREFLIGHT_TIMEOUT_SECONDS = 120
    PROFILE_TIMEOUT_SECONDS = 180
    PROFILE_STEPS = 3
    TRAIN_ESTIMATE_GUARD_SECONDS = int(TRAIN_TIMEOUT_SECONDS * 0.9)

    def _collect_artifact_status(self, output_iter: str) -> Dict[str, bool]:
        return {
            "has_prediction": os.path.exists(f"{output_iter}/pred.hdf5"),
            "has_checkpoint": os.path.exists(f"{output_iter}/best_checkpoint.pt"),
            "has_metrics": os.path.exists(f"{output_iter}/metrics.json"),
            "has_time": os.path.exists(f"{output_iter}/time.json"),
        }

    def _extract_primary_validation_score(self, metrics: Dict[str, Any]) -> float:
        for key in ("best_score", "total", "val_total", "val_score"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _update_inference_time_log(self, output_iter: str, inference_time: float) -> None:
        time_path = f"{output_iter}/time.json"
        time_data: Dict[str, Any] = {}
        os.makedirs(output_iter, exist_ok=True)
        if os.path.exists(time_path):
            with open(time_path, "r", encoding="utf-8") as f:
                time_data = json.load(f)
        time_data["inference_time"] = float(inference_time)
        with open(time_path, "w", encoding="utf-8") as f:
            json.dump(time_data, f, ensure_ascii=False, indent=2)

    def _decode_tool_json(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return {"raw": raw}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}

    def _shell_succeeded(self, result: Dict[str, Any]) -> bool:
        return "error" not in result and int(result.get("returncode", 1)) == 0

    def _load_preflight_data(self, output_iter: str) -> Dict[str, Any]:
        preflight_path = os.path.join(output_iter, "preflight.json")
        if not os.path.exists(preflight_path):
            return {}
        with open(preflight_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def _extract_train_time_estimate(self, preflight_data: Dict[str, Any]) -> float:
        for key in (
            "estimated_total_train_seconds",
            "train_time_estimate",
            "estimated_train_seconds",
        ):
            value = preflight_data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _run_preflight_suite(
        self,
        train_cmd: str,
        infer_cmd: str,
        output_iter: str,
    ) -> Dict[str, Any]:
        train_smoke_cmd = f"{train_cmd} --dry_run --batch_size 2 --num_workers 0"
        profile_cmd = (
            f"{train_cmd} --profile_only --profile_steps {self.PROFILE_STEPS} "
            "--batch_size 2 --num_workers 0"
        )
        infer_smoke_cmd = f"{infer_cmd} --dry_run --batch_size 2 --num_workers 0"

        train_smoke = self._decode_tool_json(
            registry.call(
                "run_shell",
                {"command": train_smoke_cmd, "timeout": self.PREFLIGHT_TIMEOUT_SECONDS},
            )
        )
        profile = self._decode_tool_json(
            registry.call(
                "run_shell",
                {"command": profile_cmd, "timeout": self.PROFILE_TIMEOUT_SECONDS},
            )
        )
        infer_smoke = self._decode_tool_json(
            registry.call(
                "run_shell",
                {"command": infer_smoke_cmd, "timeout": self.PREFLIGHT_TIMEOUT_SECONDS},
            )
        )

        preflight_data = self._load_preflight_data(output_iter)
        train_time_estimate = self._extract_train_time_estimate(preflight_data)

        failures = []
        if not self._shell_succeeded(train_smoke):
            failures.append("train_dry_run_failed")
        if not self._shell_succeeded(profile):
            failures.append("train_profile_failed")
        if not self._shell_succeeded(infer_smoke):
            failures.append("infer_dry_run_failed")
        if preflight_data.get("train_smoke_ok") is False:
            failures.append("train_smoke_reported_failure")
        if preflight_data.get("infer_smoke_ok") is False:
            failures.append("infer_smoke_reported_failure")
        if train_time_estimate <= 0:
            failures.append("missing_train_time_estimate")
        if train_time_estimate > self.TRAIN_ESTIMATE_GUARD_SECONDS:
            failures.append("train_time_estimate_exceeds_budget")

        return {
            "train_smoke_command": train_smoke_cmd,
            "profile_command": profile_cmd,
            "infer_smoke_command": infer_smoke_cmd,
            "train_smoke": train_smoke,
            "profile": profile,
            "infer_smoke": infer_smoke,
            "preflight_data": preflight_data,
            "train_time_estimate": train_time_estimate,
            "failures": failures,
            "passed": not failures,
        }

    def _parse_tagged_line(self, analysis: str, key: str) -> str:
        prefix = f"{key}:"
        for line in analysis.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped.split(":", 1)[1].strip()
        return ""
    
    def run(self) -> bool:
        print("\n[Phase 4] 实验验证与科学迭代...")
        
        exp_id = len(self.memory.experiments) + 1
        next_iter = self.memory.iteration + 1
        
        context = self.memory.get_context()
        
        # 构建运行命令，传入 runner 自动注入的完整参数
        data_dir = resolve_task_data_dir(self.memory.task)
        preflight_iter = f"output/{self.memory.task}/preflight_{exp_id}"
        output_iter = f"output/{self.memory.task}/iter_{next_iter}"
        os.makedirs(preflight_iter, exist_ok=True)
        
        preflight_train_cmd = f"python code/train.py --task {self.memory.task} --output_dir {preflight_iter} --data_dir {data_dir}"
        preflight_infer_cmd = f"python code/infer.py --task {self.memory.task} --checkpoint {preflight_iter}/best_checkpoint.pt --output {preflight_iter}/pred.hdf5 --data_dir {data_dir}"
        train_cmd = f"python code/train.py --task {self.memory.task} --output_dir {output_iter} --data_dir {data_dir}"
        infer_cmd = f"python code/infer.py --task {self.memory.task} --checkpoint {output_iter}/best_checkpoint.pt --output {output_iter}/pred.hdf5 --data_dir {data_dir}"
        
        print(f"[Experiment] Train command: {train_cmd}")
        print(f"[Experiment] Infer command: {infer_cmd}")

        preflight = self._run_preflight_suite(preflight_train_cmd, preflight_infer_cmd, preflight_iter)
        train_time_estimate = float(preflight["train_time_estimate"])
        preflight_passed = bool(preflight["passed"])
        preflight_summary = "ok" if preflight_passed else f"blocked: {', '.join(preflight['failures'])}"

        if preflight_passed:
            self.memory.iteration = next_iter
            os.makedirs(output_iter, exist_ok=True)
            train_result = registry.call(
                "run_shell",
                {"command": train_cmd, "timeout": self.TRAIN_TIMEOUT_SECONDS},
            )
            infer_start = time.time()
            infer_result = registry.call(
                "run_shell",
                {"command": infer_cmd, "timeout": self.INFER_TIMEOUT_SECONDS},
            )
            infer_time = time.time() - infer_start
        else:
            train_result = json.dumps(
                {
                    "skipped": True,
                    "reason": "preflight_blocked",
                    "failures": preflight["failures"],
                    "train_time_estimate": train_time_estimate,
                },
                ensure_ascii=False,
            )
            infer_result = json.dumps(
                {
                    "skipped": True,
                    "reason": "preflight_blocked",
                    "failures": preflight["failures"],
                },
                ensure_ascii=False,
            )
            infer_time = 0.0
            output_iter = preflight_iter
        
        # 尝试读取验证指标
        metrics = {}
        metrics_path = os.path.join(output_iter, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        self._update_inference_time_log(output_iter, infer_time)
        artifact_status = self._collect_artifact_status(output_iter)
        validation_score = self._extract_primary_validation_score(metrics)
        is_submission_ready = (
            artifact_status["has_prediction"]
            and artifact_status["has_metrics"]
            and artifact_status["has_time"]
        )

        # 记录实验
        record = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            phase="experiment",
            hypothesis=self.memory.hypotheses[0] if self.memory.hypotheses else "baseline",
            code_changes=[],
            config={},
            metrics=metrics,
            validation_score=validation_score,
            inference_time=infer_time,
            train_time_estimate=train_time_estimate,
            iter_dir=output_iter,
            has_prediction=artifact_status["has_prediction"],
            has_metrics=artifact_status["has_metrics"],
            has_time=artifact_status["has_time"],
            is_submission_ready=is_submission_ready,
            preflight_passed=preflight_passed,
            preflight_summary=preflight_summary,
            conclusion="",
            status="success" if artifact_status["has_prediction"] else "failed",
        )
        
        # 让LLM分析实验结果并决定下一步
        prompt = f"""实验已执行，请分析结果并决定下一步：

研究上下文：
{context}

训练输出：
{train_result}

        预检 / 时间估算结果：
        {json.dumps(preflight, ensure_ascii=False, indent=2)}

	        推理输出：
	        {infer_result}

        当前实验真实推理耗时（秒）：
        {infer_time:.6f}

        验证指标：
        {json.dumps(metrics, ensure_ascii=False, indent=2)}

	        当前实验信息：
	        - experiment_id: {exp_id}
	        - output_iter: {output_iter}
	        - preflight_passed: {preflight_passed}
	        - preflight_summary: {preflight_summary}
	        - train_time_estimate: {train_time_estimate:.6f}
	        - validation_score: {validation_score:.6f}
	        - inference_time: {infer_time:.6f}
	        - is_submission_ready: {is_submission_ready}

        实验预算信息：
        - max_iterations: {self.memory.max_iterations}
        - current_iteration: {self.memory.iteration}
        - remaining_iterations: {max(self.memory.max_iterations - self.memory.iteration, 0)}

        【实验结果分析决策树 - 按优先级执行】

	第一步：判断是否是 preflight / dry-run / profile 失败（最高优先级）
	- 如果 preflight_passed=False，先看 preflight.failures
	- `train_dry_run_failed` / `infer_dry_run_failed`：说明 CLI、测试集 loader、数据 shape 契约或 smoke test 失败，必须先修工程实现
	- `missing_train_time_estimate`：说明代码没有实现有效的训练时间估算 harness，必须补齐 `--profile_only` 和 `preflight.json`
	- `train_time_estimate_exceeds_budget`：说明当前方案在正式训练前就已判定大概率超时，必须先缩减模型、训练数据量、epoch 或启用 AMP
	- 决策：CONTINUE（先修 harness / 时间预算），不要把这类问题误判成模型科学假设失败

	第二步：判断是否是 CLI/参数错误
	- 如果训练输出包含 "error: the following arguments are required" 或 "error: unrecognized arguments"
	- 这说明代码参数接口与 runner 不兼容，**不是模型问题**
	- 必须立即：修改 train.py / infer.py 的 argparse，添加缺失的参数别名
- 常见修复：
  * 添加 --task 参数（即使脚本内部不使用）
  * 添加 --output_dir 和 --output-dir 别名指向同一 dest
  * 添加 --data_dir 和 --data-dir 别名指向同一 dest
  * 确保 checkpoint 保存为 best_checkpoint.pt
- 决策：CONTINUE（修复CLI后重跑）
- **不要分析不存在的训练结果，不要提出模型架构修改**

	第三步：判断是否训练启动但数据未找到
	- 如果训练输出包含 "FileNotFoundError" 或 "HDF5 file not found"
	- 检查 dataset.py 中的默认数据路径是否正确
	- 确保 get_dataloaders 的默认 data_dir 正确
	- 决策：CONTINUE（修复路径后重跑）

	        第四步：判断训练是否正常进行（有 loss 曲线）
	        - 分析训练过程：是否收敛？是否过拟合？损失曲线形态？
        - 分析验证指标：各段得分如何？长时稳定性（第3段）是否达标？
        - 必须扫描全部历史 submission-ready 实验，不能只看最近一次实验
        - 对比历史最优：是否有提升？提升/下降的原因是什么？
        - 必须结合当前 task 的赛题评分规则，综合考虑验证分数、推理时间约束/得分，并给出当前最优提交 iter
        - 剩余实验轮数较多时，可以偏 EXPLORATION；剩余实验轮数较少且已有可提交候选时，应偏 EXPLOITATION
        - 做出决策：
           - CONTINUE: 当前方向有潜力，继续迭代优化（提出具体修改建议）
           - PIVOT: 当前方向遇到瓶颈，切换假设或模型架构
           - STOP: 结果已足够好，或资源用尽，结束迭代

        请用以下格式输出：
        CURRENT_BEST_ITER: <int or NONE>
        BEST_ITER_REASON: ...
        STRATEGY: [EXPLORATION|EXPLOITATION]
        DECISION: [CONTINUE|PIVOT|STOP]
        REASON: ...
        NEXT_ACTION: ...
        """
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        analysis = self.client.chat(messages)["content"]
        record.conclusion = analysis
        
        # 解析决策
        decision = "CONTINUE"
        if "DECISION:" in analysis:
            decision = analysis.split("DECISION:")[1].split()[0].strip().upper()
        best_iter_raw = self._parse_tagged_line(analysis, "CURRENT_BEST_ITER")
        if best_iter_raw and best_iter_raw.upper() != "NONE":
            try:
                self.memory.best_iter_id = int(best_iter_raw)
            except ValueError:
                pass
        best_iter_reason = self._parse_tagged_line(analysis, "BEST_ITER_REASON")
        if best_iter_reason:
            self.memory.best_iter_reason = best_iter_reason
            record.submission_notes = best_iter_reason
        
        if decision == "STOP":
            record.status = "success"
            self.memory.stop_reason = "Agent decided to stop after analysis."
        elif decision == "PIVOT":
            self.memory.current_phase = "diagnosis"
        else:
            self.memory.current_phase = "design"
        
        self.memory.add_experiment(record)
        
        registry.call("write_file", {
            "path": f"{self.memory.task}/{self.memory.task}_experiment_{exp_id}_report.md",
            "content": f"# 实验 {exp_id} 报告\n\n{analysis}\n",
        })
        
        print(f"[Phase 4] 完成。实验 {exp_id} 决策: {decision}")
        return True
