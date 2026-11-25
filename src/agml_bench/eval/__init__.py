"""Benchmark VLMs and custom methods on AgML base-data.

    from agml_bench.eval import Benchmark, Method, EvalItem

    class MyMethod(Method):
        async def predict(self, item: EvalItem) -> str: ...

    Benchmark(data_dir, tasks=["classification"]).run(MyMethod(), out_dir="runs/mine")
"""

from agml_bench.eval.client import Prediction
from agml_bench.eval.items import FORMATS, TASKS, EvalItem, build_items
from agml_bench.eval.methods import Method, VLMMethod
from agml_bench.eval.resources import register_models
from agml_bench.eval.runner import Benchmark, Result

__all__ = ["Benchmark", "Method", "VLMMethod", "EvalItem", "Prediction", "Result", "build_items", "register_models", "TASKS", "FORMATS"]
