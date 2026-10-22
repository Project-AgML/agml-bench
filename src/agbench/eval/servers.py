"""Launch and stop vLLM / llama-server for a registry model."""

import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

from agbench.eval.methods import model_spec
from agbench.progress import log


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bin(env: str, default: str) -> str:
    b = os.environ.get(env) or shutil.which(default)
    if not b:
        raise RuntimeError(f"{default} not found; install it or set {env}")
    return b


def _overlay(model_id: str, overlay: dict) -> str:
    """Symlinked copy of the model snapshot with `overlay.files` taken from `overlay.repo` (e.g. fixed processor configs)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    src = Path(snapshot_download(model_id))
    dst = Path(os.environ.get("AGBENCH_CACHE", Path.home() / ".cache" / "agbench")) / "models" / model_id.replace("/", "__")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        t = dst / f.name
        if f.name not in overlay["files"] and not t.exists():
            t.symlink_to(f.resolve())
    for name in overlay["files"]:
        (dst / name).unlink(missing_ok=True)
        shutil.copy(hf_hub_download(overlay["repo"], name), dst / name)
    return str(dst)


def command(model_id: str, port: int, max_model_len: int | None = None, extra_args: list[str] | None = None) -> list[str]:
    spec = model_spec(model_id)
    ctx = str(max_model_len or spec["max_model_len"])
    if spec["backend"] == "vllm":
        model = _overlay(model_id, spec["config_overlay"]) if spec.get("config_overlay") else model_id
        return [_bin("VLLM_BIN", "vllm"), "serve", model, "--served-model-name", model_id, "--port", str(port), "--host", "127.0.0.1",
                "--max-model-len", ctx, "--gpu-memory-utilization", str(spec["gpu_memory_utilization"]),
                "--limit-mm-per-prompt", json.dumps({"image": spec["max_images"]}), "--enable-prefix-caching",
                "--max-num-seqs", str(spec["max_num_seqs"]),
                *map(str, spec.get("serve_args") or []), *(extra_args or [])]
    if spec["backend"] == "llamacpp":
        from huggingface_hub import hf_hub_download

        model = hf_hub_download(model_id, spec["gguf"])
        mmproj = hf_hub_download(model_id, spec["mmproj"])
        slots = int(spec.get("parallel", 2))  # each slot gets the full max_model_len
        cmd = [os.environ.get(spec.get("bin_env", ""), "") or _bin("LLAMA_SERVER_BIN", "llama-server"),
               "-m", model, "--mmproj", mmproj, "--port", str(port), "--host", "127.0.0.1",
               "-c", str(int(ctx) * slots), "-np", str(slots), "-ngl", "999", "--flash-attn", "on", "--jinja"]
        if spec.get("draft"):
            draft = hf_hub_download(spec["draft"].get("repo", model_id), spec["draft"]["file"])
            cmd += [spec.get("draft_flag", "-md"), draft]
        return cmd + [*map(str, spec.get("serve_args") or []), *(extra_args or [])]
    raise ValueError(f"{model_id}: backend {spec['backend']!r} cannot be launched; pass --endpoint")


class Server:
    """Context manager: start the server, wait for /health, stop it on exit."""

    def __init__(self, model_id: str, log_path: Path, port: int | None = None, max_model_len: int | None = None,
                 extra_args: list[str] | None = None, startup_timeout: int = 3600):
        self.model_id, self.log_path, self.port = model_id, Path(log_path), port or free_port()
        self.max_model_len, self.extra_args, self.startup_timeout = max_model_len, extra_args, startup_timeout
        self.proc, self.load_seconds = None, None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        cmd = command(self.model_id, self.port, self.max_model_len, self.extra_args)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"starting {self.model_id} on :{self.port} (log: {self.log_path})")
        self._log = open(self.log_path, "w")
        self._log.write(" ".join(cmd) + "\n\n"); self._log.flush()
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        while time.time() - t0 < self.startup_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                if requests.get(self.endpoint + "/health", timeout=5).status_code == 200 and \
                        requests.get(self.endpoint + "/v1/models", timeout=5).status_code == 200:
                    self.load_seconds = time.time() - t0
                    log(f"{self.model_id} ready in {self.load_seconds:.0f}s")
                    return self
            except requests.RequestException:
                pass
            time.sleep(5)
        self.__exit__(None, None, None)
        raise TimeoutError(f"{self.model_id} not ready after {self.startup_timeout}s; see {self.log_path}")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        if getattr(self, "_log", None):
            self._log.close()


def delete_weights(model_id: str) -> None:
    """Remove a model's (and its drafter's) HF cache entries."""
    from huggingface_hub import scan_cache_dir

    spec = model_spec(model_id)
    repos = {model_id, (spec.get("draft") or {}).get("repo")}
    cache = scan_cache_dir()
    revs = [rev.commit_hash for repo in cache.repos if repo.repo_id in repos for rev in repo.revisions]
    if revs:
        strategy = cache.delete_revisions(*revs)
        log(f"deleting {strategy.expected_freed_size_str} of weights for {model_id}")
        strategy.execute()
