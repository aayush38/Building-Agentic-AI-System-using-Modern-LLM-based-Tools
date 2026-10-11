"""JSONL trace logger: one line per Thought / Action / Observation / Final answer."""
import json
import time
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class TraceLogger:
    def __init__(self, run_name: str):
        LOG_DIR.mkdir(exist_ok=True)
        self.path = LOG_DIR / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.step = 0

    def log(self, kind: str, content, **extra):
        self.step += 1
        rec = {"step": self.step, "t": round(time.time(), 2), "kind": kind,
               "content": content, **extra}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec
