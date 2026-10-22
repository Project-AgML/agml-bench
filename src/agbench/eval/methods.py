"""Methods produce a prediction for an EvalItem. Subclass Method to benchmark your own approach."""

from agbench.eval.client import Client, Prediction, merge
from agbench.eval.items import EvalItem
from agbench.eval.resources import models, prompts

EFFORTS = ("low", "medium", "high")


class Method:
    """Override `predict`. Return a string (the answer) or a Prediction (to report reasoning/tokens)."""

    name = "method"

    async def setup(self) -> None:
        pass

    async def predict(self, item: EvalItem) -> str | Prediction:
        raise NotImplementedError

    async def close(self) -> None:
        pass

    def config(self) -> dict:
        return {"name": self.name}


def model_spec(model_id: str) -> dict:
    reg = models()
    if model_id not in reg["models"]:
        return {"backend": "vllm", **reg["defaults"]}  # unregistered: plain `vllm serve <id>`, no sampling overrides
    return {**reg["defaults"], **reg["models"][model_id]}


def key_formats(spec: dict) -> dict:
    """How to read box keys in this model's answers: box_2d follows the prompted `box_order`."""
    return {**(spec.get("bbox_key_format") or {}), "box_2d": f"{spec.get('box_order', 'yxyx')}_1000"}


def request_body(spec: dict, thinking: bool, effort: str | None) -> dict:
    body = dict(spec.get("sampling") or {})
    if thinking:
        body = merge(body, spec.get("thinking") or {})
        if effort and spec.get("effort"):
            body = merge(body, spec["effort"].get(effort) or {})
    else:
        body = merge(body, spec.get("no_thinking") or {})
    return body


class VLMMethod(Method):
    """Prompt a VLM behind an OpenAI-compatible endpoint, using the registry settings for `model_id`."""

    def __init__(self, endpoint: str, model_id: str, thinking: bool = True, effort: str | None = "medium",
                 max_tokens: int | None = None, body: dict | None = None, served_name: str | None = None):
        self.spec = model_spec(model_id)
        self.name, self.model_id, self.thinking, self.effort = model_id, model_id, thinking, effort
        self.max_tokens = max_tokens or self.spec["max_tokens"]
        self.body = merge(request_body(self.spec, thinking, effort), body or {})
        self.system = prompts()["cot_system"] if (thinking and self.spec.get("cot")) else None
        self.box_order = self.spec.get("box_order", "yxyx")
        self.client = Client(endpoint, served_name)

    def prompt(self, item: EvalItem) -> str:
        """The item prompt, with the grounding coordinate order switched to this model's `box_order`."""
        orders = prompts()["box_orders"]
        if item.format == "grounding" and self.box_order != "yxyx":
            return item.prompt.replace(orders["yxyx"], orders[self.box_order])
        return item.prompt

    async def predict(self, item: EvalItem) -> Prediction:
        return await self.client.chat(self.prompt(item), item.images, system=self.system, max_tokens=self.max_tokens, body=self.body)

    def config(self) -> dict:
        return {"name": self.name, "model_id": self.model_id, "thinking": self.thinking, "effort": self.effort,
                "max_tokens": self.max_tokens, "body": self.body, "cot": bool(self.system), "box_order": self.box_order}
