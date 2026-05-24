"""
科研记忆模块

维护Agent的研究状态、实验历史、假设与结论。
支持持久化到JSON，确保科研迭代的连续性。
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ExperimentRecord:
    """单次实验记录"""
    id: int
    timestamp: str
    phase: str  # 所属阶段
    hypothesis: str  # 实验假设
    code_changes: List[str] = field(default_factory=list)  # 修改的文件列表
    config: Dict[str, Any] = field(default_factory=dict)  # 实验配置
    metrics: Dict[str, float] = field(default_factory=dict)  # 评估指标
    validation_score: float = 0.0  # 验证集主分
    inference_time: float = 0.0  # 该 iter 的真实推理耗时
    train_time_estimate: float = 0.0  # 该 iter 开跑前的训练耗时估算
    iter_dir: str = ""  # 该 iter 对应的输出目录
    has_prediction: bool = False
    has_metrics: bool = False
    has_time: bool = False
    is_submission_ready: bool = False
    preflight_passed: bool = False
    preflight_summary: str = ""
    submission_notes: str = ""
    conclusion: str = ""  # 实验结论
    status: str = "running"  # running, success, failed


@dataclass
class ResearchMemory:
    """科研记忆总状态"""
    task: str = "task1"
    current_phase: str = "literature"  # literature, diagnosis, design, experiment
    iteration: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 阶段产出
    literature_summary: str = ""
    bottlenecks: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    
    # 实验历史
    experiments: List[ExperimentRecord] = field(default_factory=list)
    best_iter_id: Optional[int] = None
    best_iter_reason: str = ""
    max_iterations: int = 0
    
    # 代码版本追踪
    code_versions: List[Dict] = field(default_factory=list)
    
    # 终止条件
    stop_reason: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str = "research_memory.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str = "research_memory.json") -> Optional["ResearchMemory"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容旧字段，避免历史 memory 文件加载失败或污染新语义
            data.pop("best_experiment_id", None)
            data.pop("best_metrics", None)
            # 将 experiments 的 dict 列表转换回 ExperimentRecord 对象
            if "experiments" in data and isinstance(data["experiments"], list):
                data["experiments"] = [
                    ExperimentRecord(**e) if isinstance(e, dict) else e
                    for e in data["experiments"]
                ]
            return cls(**data)
        return None
    
    def add_experiment(self, record: ExperimentRecord):
        self.experiments.append(record)
    
    def get_experiment(self, exp_id: int) -> Optional[ExperimentRecord]:
        for e in self.experiments:
            if e.id == exp_id:
                return e
        return None

    def get_best_validation_experiment(self) -> Optional[ExperimentRecord]:
        candidates = [e for e in self.experiments if e.status == "success"]
        if not candidates:
            return None
        return max(candidates, key=lambda e: (float(e.validation_score), -int(e.id)))
    
    def get_context(self, max_experiments: int = 3) -> str:
        """生成给LLM的上下文摘要"""
        lines = []
        lines.append(f"=== 当前任务: {self.task} | 阶段: {self.current_phase} | 迭代: {self.iteration} ===")
        if self.max_iterations > 0:
            remaining = max(self.max_iterations - self.iteration, 0)
            lines.append(f"=== 总实验预算: {self.max_iterations} | 剩余实验轮数: {remaining} ===")
        lines.append("")
        
        if self.literature_summary:
            lines.append("【文献综述摘要】")
            lines.append(self.literature_summary[:800])
            lines.append("")
        
        if self.bottlenecks:
            lines.append("【已识别瓶颈】")
            for b in self.bottlenecks:
                lines.append(f"- {b}")
            lines.append("")
        
        if self.hypotheses:
            lines.append("【当前假设】")
            for h in self.hypotheses:
                lines.append(f"- {h}")
            lines.append("")
        
        if self.experiments:
            lines.append("【近期实验】")
            for e in self.experiments[-max_experiments:]:
                lines.append(f"- Exp {e.id} ({e.status}): {e.hypothesis}")
                if e.metrics:
                    lines.append(f"  Metrics: {json.dumps(e.metrics, ensure_ascii=False)}")
                lines.append(
                    f"  Validation Score: {e.validation_score:.6f} | "
                    f"Train Estimate: {e.train_time_estimate:.2f}s | "
                    f"Inference Time: {e.inference_time:.6f}s | "
                    f"Preflight Passed: {e.preflight_passed} | "
                    f"Submission Ready: {e.is_submission_ready}"
                )
                if e.preflight_summary:
                    lines.append(f"  Preflight: {e.preflight_summary}")
                if e.conclusion:
                    lines.append(f"  Conclusion: {e.conclusion}")
            lines.append("")
        
        ready = [e for e in self.experiments if e.is_submission_ready]
        if ready:
            lines.append("【历史候选摘要】")
            for e in ready:
                marker = " [CURRENT_BEST_ITER]" if self.best_iter_id == e.id else ""
                lines.append(
                    f"- Iter {e.id}{marker}: val_score={e.validation_score:.6f}, "
                    f"train_est={e.train_time_estimate:.2f}s, "
                    f"infer_time={e.inference_time:.6f}s, iter_dir={e.iter_dir}"
                )
            lines.append("")

        if self.best_iter_id:
            lines.append("【当前最优提交候选】")
            lines.append(f"best_iter_id={self.best_iter_id}")
            if self.best_iter_reason:
                lines.append(self.best_iter_reason)
            lines.append("")

        best_validation = self.get_best_validation_experiment()
        if best_validation is not None:
            lines.append("【当前验证集历史最优】")
            lines.append(
                json.dumps(
                    {
                        "id": best_validation.id,
                        "validation_score": best_validation.validation_score,
                        "metrics": best_validation.metrics,
                    },
                    ensure_ascii=False,
                )
            )
            lines.append("")
        
        return "\n".join(lines)
