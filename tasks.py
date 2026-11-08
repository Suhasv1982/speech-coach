"""Load and validate the assessment tasks from assessment_tasks.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

import config

TASK_TYPES = {"scripted", "spontaneous"}


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    type: str
    time_limit_s: int
    instructions: str
    reference_text: str | None = None
    prompt: str | None = None
    reveal_on_start: bool = False
    min_coverage: float | None = None  # warn below this share of reference words recognised


def load_tasks(path: Path | None = None) -> list[Task]:
    """Read the YAML file and return the tasks in order, raising ValueError if any are invalid."""
    path = path or config.TASKS_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = [Task(**item) for item in data["tasks"]]

    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"Duplicate task id: {task.id}")
        seen.add(task.id)
        if task.type not in TASK_TYPES:
            raise ValueError(f"Task {task.id}: type must be one of {sorted(TASK_TYPES)}")
        if task.time_limit_s <= 0:
            raise ValueError(f"Task {task.id}: time_limit_s must be positive")
        if task.type == "scripted" and not task.reference_text:
            raise ValueError(f"Task {task.id}: scripted tasks need reference_text")
        if task.type == "spontaneous" and task.reference_text:
            raise ValueError(f"Task {task.id}: spontaneous tasks must not have reference_text")
        if task.reveal_on_start and not task.prompt:
            raise ValueError(f"Task {task.id}: reveal_on_start needs a prompt")
        if task.min_coverage is not None:
            if task.type != "scripted":
                raise ValueError(f"Task {task.id}: min_coverage needs a scripted task")
            if not 0 < task.min_coverage <= 1:
                raise ValueError(f"Task {task.id}: min_coverage must be between 0 and 1")
    return tasks
