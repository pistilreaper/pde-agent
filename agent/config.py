"""
配置管理模块
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """LLM API 配置"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0
    reasoning_effort: str | None = None
    verbosity: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMProfile:
    """Phase-aware LLM profile"""
    name: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    reasoning_effort: str | None = None
    verbosity: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchConfig:
    """科研流程配置"""
    max_iterations: int = 15
    max_time_hours: float = 10.5  # 留点余量，不超过12小时
    early_stop_patience: int = 3
    task: str = "task1"  # task1, task2, task3
    output_dir: str = "./output"
    code_dir: str = "./code"


@dataclass
class AgentConfig:
    """Agent 总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_profiles: dict[str, LLMProfile] = field(default_factory=dict)
    phase_profile_map: dict[str, str] = field(default_factory=dict)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    def _default_profile(self) -> LLMProfile:
        model = str(self.llm.model).strip()
        if not model:
            raise ValueError("default llm.model must not be empty")
        return LLMProfile(
            name="default",
            model=model,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            timeout=self.llm.timeout,
            reasoning_effort=self.llm.reasoning_effort,
            verbosity=self.llm.verbosity,
            extra_body=dict(self.llm.extra_body),
        )

    def get_llm_profile(self, phase: str | None = None) -> LLMProfile:
        default_profile = self._default_profile()
        if phase is None or not self.llm_profiles:
            return default_profile

        profile_name = self.phase_profile_map.get(phase, phase)
        profile = self.llm_profiles.get(profile_name)
        if profile is None:
            return default_profile
        if not str(profile.model).strip():
            raise ValueError(f"llm profile {profile.name!r} must define a non-empty model")
        return profile


def _coerce_extra_body(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("llm.extra_body must be a mapping")
    return dict(value)


def _build_profile(name: str, raw: Any, default_llm: LLMConfig) -> LLMProfile:
    if isinstance(raw, LLMProfile):
        profile = raw
    else:
        if not isinstance(raw, dict):
            raise ValueError(f"llm_profiles.{name} must be a mapping")
        if "model" in raw:
            model = str(raw.get("model", "")).strip()
            if not model:
                raise ValueError(f"llm_profiles.{name}.model must not be empty")
        else:
            model = default_llm.model
        profile_extra = _coerce_extra_body(raw.get("extra_body"))
        merged_extra = dict(default_llm.extra_body)
        merged_extra.update(profile_extra)
        profile = LLMProfile(
            name=name,
            model=model,
            temperature=raw.get("temperature", default_llm.temperature),
            max_tokens=raw.get("max_tokens", default_llm.max_tokens),
            timeout=raw.get("timeout", default_llm.timeout),
            reasoning_effort=raw.get("reasoning_effort", default_llm.reasoning_effort),
            verbosity=raw.get("verbosity", default_llm.verbosity),
            extra_body=merged_extra,
        )
    if not str(profile.model).strip():
        raise ValueError(f"llm profile {name!r} must define a non-empty model")
    return profile


def load_config(path: str = "config.yaml") -> AgentConfig:
    """从YAML加载配置，环境变量可覆盖"""
    cfg = AgentConfig()
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            if "llm" in data:
                for k, v in data["llm"].items():
                    if hasattr(cfg.llm, k):
                        if k == "extra_body":
                            setattr(cfg.llm, k, _coerce_extra_body(v))
                        else:
                            setattr(cfg.llm, k, v)
            if "research" in data:
                for k, v in data["research"].items():
                    if hasattr(cfg.research, k):
                        setattr(cfg.research, k, v)
            raw_profiles = data.get("llm_profiles", {})
            raw_phase_profile_map = data.get("phase_profile_map", {})
        else:
            raw_profiles = {}
            raw_phase_profile_map = {}
    else:
        raw_profiles = {}
        raw_phase_profile_map = {}
    
    # 环境变量覆盖
    if os.environ.get("OPENAI_API_KEY"):
        cfg.llm.api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL"):
        cfg.llm.base_url = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("LLM_MODEL"):
        cfg.llm.model = os.environ["LLM_MODEL"]

    cfg.llm_profiles = {
        str(name): _build_profile(str(name), profile_data, cfg.llm)
        for name, profile_data in raw_profiles.items()
    }
    if not isinstance(raw_phase_profile_map, dict):
        raise ValueError("phase_profile_map must be a mapping")
    cfg.phase_profile_map = {
        str(phase): str(profile_name)
        for phase, profile_name in raw_phase_profile_map.items()
    }
    
    return cfg


def save_config(cfg: AgentConfig, path: str = "config.yaml"):
    """保存配置到YAML"""
    data = {
        "llm": {
            "api_key": cfg.llm.api_key if cfg.llm.api_key else "<YOUR_API_KEY>",
            "base_url": cfg.llm.base_url,
            "model": cfg.llm.model,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens,
            "timeout": cfg.llm.timeout,
            "reasoning_effort": cfg.llm.reasoning_effort,
            "verbosity": cfg.llm.verbosity,
            "extra_body": cfg.llm.extra_body,
        },
        "llm_profiles": {
            name: {
                "model": profile.model,
                "temperature": profile.temperature,
                "max_tokens": profile.max_tokens,
                "timeout": profile.timeout,
                "reasoning_effort": profile.reasoning_effort,
                "verbosity": profile.verbosity,
                "extra_body": profile.extra_body,
            }
            for name, profile in cfg.llm_profiles.items()
        },
        "phase_profile_map": dict(cfg.phase_profile_map),
        "research": {
            "max_iterations": cfg.research.max_iterations,
            "max_time_hours": cfg.research.max_time_hours,
            "early_stop_patience": cfg.research.early_stop_patience,
            "task": cfg.research.task,
            "output_dir": cfg.research.output_dir,
            "code_dir": cfg.research.code_dir,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
