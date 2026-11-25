import time


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, m = s // 3600, s % 3600 // 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


class Progress:
    """Prints `label done/total` at most every `every` seconds, and once when complete."""

    def __init__(self, label: str, total: int, every: float = 10.0):
        self.label, self.total, self.every = label, total, every
        self.done = 0
        self.start = self.last = time.time()

    def add(self, n: int = 1) -> None:
        self.done += n
        now = time.time()
        if self.done >= self.total or now - self.last >= self.every:
            self.last = now
            log(f"{self.label} {min(self.done, self.total)}/{self.total} ({fmt_duration(now - self.start)})")
