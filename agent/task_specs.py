"""
PDEAgent 内建任务规格。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    display_name: str
    equation: str
    input_steps: int
    total_steps: int
    prediction_shape: tuple[int, int, int]
    local_data_dir: str
    conditioning_field: str | None
    train_from_scratch: bool
    hidden_test_parameter: str | None


_PDEAGENT_ROOT = Path(__file__).resolve().parents[1]


TASK_SPECS = {
    "task1": TaskSpec(
        task_id="task1",
        display_name="Task 1",
        equation="Burgers",
        input_steps=10,
        total_steps=200,
        prediction_shape=(1000, 200, 256),
        local_data_dir=str(_PDEAGENT_ROOT / "data" / "task1"),
        conditioning_field=None,
        train_from_scratch=False,
        hidden_test_parameter=None,
    ),
    "task2": TaskSpec(
        task_id="task2",
        display_name="Task 2",
        equation="Burgers",
        input_steps=10,
        total_steps=200,
        prediction_shape=(1000, 200, 256),
        local_data_dir=str(_PDEAGENT_ROOT / "data" / "task2"),
        conditioning_field="nu",
        train_from_scratch=True,
        hidden_test_parameter="nu",
    ),
    "task3": TaskSpec(
        task_id="task3",
        display_name="Task 3",
        equation="Kuramoto-Sivashinsky",
        input_steps=20,
        total_steps=400,
        prediction_shape=(100, 400, 256),
        local_data_dir=str(_PDEAGENT_ROOT / "data" / "task3"),
        conditioning_field="lambda2",
        train_from_scratch=True,
        hidden_test_parameter="lambda2",
    ),
}

TASK_CHOICES = tuple(TASK_SPECS)


def get_task_spec(task: str) -> TaskSpec:
    return TASK_SPECS[task]


def resolve_task_data_dir(task: str) -> str:
    return get_task_spec(task).local_data_dir
