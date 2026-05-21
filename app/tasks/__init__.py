"""Task entry points — Task A lives in app.pipeline; Task B here."""

from app.tasks.task_b import recommend_items

__all__ = ["recommend_items"]
