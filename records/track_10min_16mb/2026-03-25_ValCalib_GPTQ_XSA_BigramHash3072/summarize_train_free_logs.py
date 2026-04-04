#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path


def parse_log(path: Path) -> dict[str, str]:
    data: dict[str, str] = {"run_id": path.stem}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        data[key] = value
    return data


def float_or_nan(value: str | None) -> float:
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/train_free_proxy")
    rows = [parse_log(path) for path in sorted(root.glob("*.log"))]
    if not rows:
        raise SystemExit(f"No logs found under {root}")

    baseline_bpb = math.nan
    for row in rows:
        if row.get("run_id") == "baseline_steps0":
            baseline_bpb = float_or_nan(row.get("val_bpb"))
            break
    if math.isnan(baseline_bpb):
        for row in rows:
            if row.get("ttt_slot_steps") == "0":
                baseline_bpb = float_or_nan(row.get("val_bpb"))
                break

    rows.sort(key=lambda row: (float_or_nan(row.get("val_bpb")), row["run_id"]))

    header = [
        "run_id",
        "checkpoint_kind",
        "proxy_docs",
        "ttt_eval_seq_len",
        "ttt_batch_size",
        "ttt_slot_steps",
        "ttt_slot_adapt_block",
        "ttt_slot_lr",
        "ttt_slot_update",
        "ttt_slot_update_every",
        "ttt_slot_update_first_k_chunks",
        "ttt_slot_update_loss_threshold",
        "val_loss",
        "val_bpb",
        "gain_vs_baseline_bpb",
    ]
    print("\t".join(header))
    for row in rows:
        val_bpb = float_or_nan(row.get("val_bpb"))
        gain = val_bpb - baseline_bpb if not math.isnan(val_bpb) and not math.isnan(baseline_bpb) else math.nan
        out = [
            row.get("run_id", ""),
            row.get("checkpoint_kind", ""),
            row.get("proxy_docs", ""),
            row.get("ttt_eval_seq_len", ""),
            row.get("ttt_batch_size", ""),
            row.get("ttt_slot_steps", ""),
            row.get("ttt_slot_adapt_block", ""),
            row.get("ttt_slot_lr", ""),
            row.get("ttt_slot_update", ""),
            row.get("ttt_slot_update_every", ""),
            row.get("ttt_slot_update_first_k_chunks", ""),
            row.get("ttt_slot_update_loss_threshold", ""),
            row.get("val_loss", ""),
            row.get("val_bpb", ""),
            f"{gain:.8f}" if not math.isnan(gain) else "",
        ]
        print("\t".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
