"""Benchmark VLMs and custom methods on AgML base-data.

    from agbench.eval import Benchmark, Method, EvalItem

    class MyMethod(Method):
        async def predict(self, item: EvalItem) -> str: ...

    Benchmark(data_dir, tasks=["classification"]).run(MyMethod(), out_dir="runs/mine")
"""

from agbench.eval.client import Prediction
from agbench.eval.items import FORMATS, TASKS, EvalItem, build_items
from agbench.eval.methods import Method, VLMMethod
from agbench.eval.runner import Benchmark, Result

__all__ = ["Benchmark", "Method", "VLMMethod", "EvalItem", "Prediction", "Result", "build_items", "TASKS", "FORMATS"]
