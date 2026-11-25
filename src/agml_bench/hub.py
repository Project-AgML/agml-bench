"""Hugging Face repo access: file discovery, config selection, remote reads."""

import fnmatch
import io
import json
import re
from dataclasses import dataclass
from functools import cached_property

import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, HfFileSystem

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
_SPLIT_RE = re.compile(r"^(?P<split>[A-Za-z_]+?)[-_]\d+(-of-\d+)?\.parquet$")


@dataclass
class FileRef:
    path: str
    config: str
    split: str


def split_from_path(path: str) -> str:
    m = _SPLIT_RE.match(path.rsplit("/", 1)[-1])
    return m.group("split") if m else "train"


def parse_card(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    try:
        return yaml.safe_load(text.split("---", 2)[1]) or {}
    except yaml.YAMLError:
        return {}


def classlabel_names(feat) -> list[str] | None:
    """Find the first ClassLabel's names anywhere inside an HF features dict."""
    if isinstance(feat, dict):
        if feat.get("_type") == "ClassLabel":
            return feat.get("names")
        for v in feat.values():
            n = classlabel_names(v)
            if n is not None:
                return n
    elif isinstance(feat, list):
        for v in feat:
            n = classlabel_names(v)
            if n is not None:
                return n
    return None


def feature_type(feat) -> str | None:
    if isinstance(feat, dict):
        return feat.get("_type") or "struct"
    if isinstance(feat, list):
        return "list"
    return None


class Repo:
    def __init__(self, repo_id: str, revision: str | None = None):
        self.repo_id = repo_id
        self.api = HfApi()
        self.fs = HfFileSystem()
        self.revision = revision or self.api.dataset_info(repo_id).sha

    def url(self, path: str) -> str:
        return f"datasets/{self.repo_id}@{self.revision}/{path}"

    @cached_property
    def files(self) -> list[str]:
        return self.api.list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision)

    @cached_property
    def card(self) -> dict:
        if "README.md" not in self.files:
            return {}
        return parse_card(self.fs.read_text(self.url("README.md")))

    def read_bytes(self, path: str) -> bytes:
        return self.fs.cat_file(self.url(path))

    def open(self, path: str, block_size: int = 2**22):
        return self.fs.open(self.url(path), block_size=block_size)

    def parquet(self, path: str) -> pq.ParquetFile:
        return pq.ParquetFile(self.open(path))

    def layout(self) -> str:
        data_pq = [f for f in self.files if f.endswith(".parquet") and "/images/" not in f"/{f}"]
        if data_pq:
            return "parquet"
        if any(f.endswith("metadata.jsonl") for f in self.files):
            return "metadata_jsonl"
        return "unsupported"

    def _configs(self) -> list[dict]:
        cfgs = self.card.get("configs") or []
        return [c for c in cfgs if isinstance(c, dict) and c.get("config_name")]

    def _config_patterns(self, c: dict) -> list[tuple[str, str | None]]:
        pats = []
        df = c.get("data_files")
        if isinstance(df, str):
            pats.append((df, None))
        elif isinstance(df, list):
            for d in df:
                if isinstance(d, str):
                    pats.append((d, None))
                elif isinstance(d, dict):
                    paths = d.get("path")
                    for p in paths if isinstance(paths, list) else [paths]:
                        pats.append((p, d.get("split")))
        if not pats and c.get("data_dir"):
            pats.append((c["data_dir"].rstrip("/") + "/*", None))
        return pats

    def data_files(self, ext: str, all_configs: bool, skip_keywords: list[str], prefer: str | None = None) -> list[FileRef]:
        """Data files grouped by HF config.

        all_configs=False picks one config (override > default > 'raw' > first non-skipped).
        """
        candidates = [f for f in self.files if f.endswith(ext) and "/images/" not in f"/{f}"
                      and not f.startswith("images/")]
        configs = self._configs()
        refs: list[FileRef] = []
        if configs:
            for c in configs:
                for pat, split in self._config_patterns(c):
                    for f in candidates:
                        if fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(f, pat.rstrip("/") + "/*"):
                            refs.append(FileRef(f, c["config_name"], split or split_from_path(f)))
        if not refs:
            for f in candidates:
                cfg_name = f.split("/")[0] if "/" in f and f.split("/")[0] not in ("data",) else "default"
                refs.append(FileRef(f, cfg_name, split_from_path(f)))
        seen, uniq = set(), []
        for r in refs:
            if r.path not in seen:
                seen.add(r.path)
                uniq.append(r)
        refs = uniq
        if all_configs:
            return refs

        names = list(dict.fromkeys(r.config for r in refs))
        usable = [n for n in names if not any(k in n.lower() for k in skip_keywords)] or names
        chosen = None
        if prefer and prefer in names:
            chosen = prefer
        else:
            defaults = [c["config_name"] for c in configs if c.get("default")]
            for cand in defaults + ["default", "raw"]:
                if cand in usable:
                    chosen = cand
                    break
            chosen = chosen or (usable[0] if usable else None)
        return [r for r in refs if r.config == chosen]

    def hf_features(self, path: str) -> dict:
        md = self.parquet(path).schema_arrow.metadata or {}
        info = json.loads(md.get(b"huggingface", b"{}"))
        return info.get("info", {}).get("features", {})

    def shard_index(self, base: str) -> dict[str, str]:
        """{image path: full shard path} from <base>/images/path_to_shard.parquet."""
        idx_path = f"{base}/images/path_to_shard.parquet" if base else "images/path_to_shard.parquet"
        if idx_path not in self.files:
            return {}
        t = pq.read_table(self.open(idx_path), columns=["path", "shard_file"])
        img_dir = idx_path.rsplit("/", 1)[0]
        return {p: f"{img_dir}/{s}" for p, s in zip(t.column("path").to_pylist(), t.column("shard_file").to_pylist())}


def image_format(data: bytes) -> tuple[str, tuple[int, int] | None]:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "bin").lower()
            size = im.size
    except Exception:
        return "bin", None
    return {"jpeg": "jpg", "tiff": "tif", "mpo": "jpg"}.get(fmt, fmt), size
