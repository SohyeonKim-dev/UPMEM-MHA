import re
import csv
import os

import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

LOGFILE = "log.txt"
OUTDIR = "results"
CSVFILE = os.path.join(OUTDIR, "results.csv")
os.makedirs(OUTDIR, exist_ok=True)

exp_seq_re   = re.compile(r"\[EXP_SEQ\]\s*BATCH=(\d+),\s*SEQ_LEN=(\d+)")
exp_batch_re = re.compile(r"\[EXP_BATCH\]\s*BATCH=(\d+),\s*SEQ_LEN=(\d+)")
exp_hd_re    = re.compile(r"\[EXP_HD\]\s*BATCH=(\d+),\s*SEQ_LEN=(\d+),\s*HEAD_DIM=(\d+)")
exp_nh_re    = re.compile(r"\[EXP_NH\]\s*BATCH=(\d+),\s*SEQ_LEN=(\d+),\s*NUM_HEADS=(\d+)")
exp_tl_re    = re.compile(r"\[EXP_TL\].*NR_TASKLETS=(\d+)")

host_re      = re.compile(r"Host total computation time:\s*([0-9.]+)\s*ms")
dpu_re       = re.compile(r"Average cycles per slot:\s*([0-9.]+)\s*\(\s*([0-9.]+)\s*ms\s*\)")
alloc_re     = re.compile(r"DPUs allocated:\s*(\d+)")

rows: List[Dict[str, Any]] = []

current: Dict[str, Optional[Any]] = {
    "batch": None,
    "seq_len": None,
    "head_dim": None,
    "num_heads": None,
    "tasklets": None,
    "host_ms": None,
    "allocated": None,
    "exp_type": None,
}

with open(LOGFILE, "r") as f:
    for raw_line in f:
        line = raw_line.strip()
        m = exp_seq_re.search(line)
        if m:
            current = {
                "batch": int(m.group(1)),
                "seq_len": int(m.group(2)),
                "head_dim": None,
                "num_heads": None,
                "tasklets": None,
                "host_ms": None,
                "allocated": None,
                "exp_type": "EXP_SEQ",
            }
            continue

        m = exp_batch_re.search(line)
        if m:
            current = {
                "batch": int(m.group(1)),
                "seq_len": int(m.group(2)),
                "head_dim": None,
                "num_heads": None,
                "tasklets": None,
                "host_ms": None,
                "allocated": None,
                "exp_type": "EXP_BATCH",
            }
            continue

        m = exp_hd_re.search(line)
        if m:
            current = {
                "batch": int(m.group(1)),
                "seq_len": int(m.group(2)),
                "head_dim": int(m.group(3)),
                "num_heads": None,
                "tasklets": None,
                "host_ms": None,
                "allocated": None,
                "exp_type": "EXP_HD",
            }
            continue

        m = exp_nh_re.search(line)
        if m:
            current = {
                "batch": int(m.group(1)),
                "seq_len": int(m.group(2)),
                "head_dim": None,
                "num_heads": int(m.group(3)),
                "tasklets": None,
                "host_ms": None,
                "allocated": None,
                "exp_type": "EXP_NH",
            }
            continue

        m = exp_tl_re.search(line)
        if m:
            current = {
                "batch": 128,
                "seq_len": 64,
                "head_dim": 32,
                "num_heads": 16,
                "tasklets": int(m.group(1)),
                "host_ms": None,
                "allocated": None,
                "exp_type": "EXP_TL",
            }
            continue

        m = alloc_re.search(line)
        if m and current.get("exp_type") is not None:
            try:
                current["allocated"] = int(m.group(1))
            except Exception:
                current["allocated"] = None
            continue

        m = host_re.search(line)
        if m and current.get("exp_type") is not None:
            try:
                current["host_ms"] = float(m.group(1))
            except Exception:
                current["host_ms"] = None
            continue

        m = dpu_re.search(line)
        if m and current.get("exp_type") is not None:
            try:
                dpu_ms = float(m.group(2))  
            except Exception:
                dpu_ms = None

            snapshot = {
                "batch": current.get("batch"),
                "seq_len": current.get("seq_len"),
                "head_dim": current.get("head_dim"),
                "num_heads": current.get("num_heads"),
                "tasklets": current.get("tasklets"),
                "host_ms": current.get("host_ms"),
                "dpu_ms": dpu_ms,
                "allocated": current.get("allocated"),
                "exp_type": current.get("exp_type"),
            }
            rows.append(snapshot)
            continue

with open(CSVFILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["batch", "seq_len", "head_dim", "num_heads", "tasklets",
                     "host_ms", "dpu_ms", "allocated", "exp_type"])
    for r in rows:
        writer.writerow([
            r.get("batch"), r.get("seq_len"), r.get("head_dim"), r.get("num_heads"),
            r.get("tasklets"), r.get("host_ms"), r.get("dpu_ms"), r.get("allocated"),
            r.get("exp_type")
        ])
print("CSV saved →", CSVFILE)

def filter_rows(batch=None, seq=None, head_dim=None, num_heads=None, tasklets=None, exp_type=None):
    out = []
    for r in rows:
        if batch is not None and r.get("batch") != batch:
            continue
        if seq is not None and r.get("seq_len") != seq:
            continue
        if head_dim is not None and r.get("head_dim") != head_dim:
            continue
        if num_heads is not None and r.get("num_heads") != num_heads:
            continue
        if tasklets is not None and r.get("tasklets") != tasklets:
            continue
        if exp_type is not None and r.get("exp_type") != exp_type:
            continue
        if r.get("dpu_ms") is None:
            continue
        out.append(r)
    return out

bar_width = 0.35
gap = 0.06
bar_offset = (bar_width + gap) / 2

COLOR_CPU = "#FF7A6E"
COLOR_DPU = "#6DC96F"

def plot_graph_from_rows(rows_list: List[Dict[str, Any]],
                         x_key: str,
                         xlabel: str,
                         title: str,
                         filename: str,
                         use_log: bool = False,
                         show_alloc: bool = False):
    if not rows_list:
        print(f"Skipping {title}: no data")
        return
    rows_sorted = sorted(rows_list, key=lambda r: (r.get(x_key) if r.get(x_key) is not None else -1))

    xvals = [r.get(x_key) for r in rows_sorted]
    host_y = [r.get("host_ms") if r.get("host_ms") is not None else np.nan for r in rows_sorted]
    dpu_y  = [r.get("dpu_ms") if r.get("dpu_ms") is not None else np.nan for r in rows_sorted]
    allocated = [r.get("allocated") for r in rows_sorted]
    x_pos = np.arange(len(xvals))
    plt.figure(figsize=(9,5))
    host_bars = plt.bar(x_pos - bar_offset, host_y, bar_width, label="CPU Host (ms)", color=COLOR_CPU)
    dpu_bars  = plt.bar(x_pos + bar_offset, dpu_y, bar_width, label="UPMEM DPU (ms)", color=COLOR_DPU)

    if use_log:
        plt.yscale("log")

    plt.xticks(x_pos, [str(x) for x in xvals])
    plt.xlabel(xlabel)
    plt.ylabel("Time (ms)" + (" (log)" if use_log else ""))
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.35, which='both' if use_log else 'major')

    if show_alloc:
        for idx, bar in enumerate(dpu_bars):
            alloc = allocated[idx]
            if alloc is None:
                continue
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, h,
                     f"{int(alloc)}", ha='center', va='bottom', fontsize=9)

    outpath = os.path.join(OUTDIR, filename)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved", outpath)

# 1) SEQ 
seq_rows = filter_rows(batch=128, exp_type="EXP_SEQ")
plot_graph_from_rows(seq_rows,
                     x_key="seq_len",
                     xlabel="SEQ_LEN",
                     title="CPU vs UPMEM-PIM",
                     filename="seq_bar.png",
                     use_log=False,
                     show_alloc=False)

# 2) BATCH 
batch_rows = filter_rows(seq=128, exp_type="EXP_BATCH")
plot_graph_from_rows(batch_rows,
                     x_key="batch",
                     xlabel="BATCH_SIZE",
                     title="CPU vs UPMEM-PIM",
                     filename="batch_bar.png",
                     use_log=False,
                     show_alloc=True)

# 3) HEAD_DIM 
hd_rows = filter_rows(batch=128, seq=32, exp_type="EXP_HD")
plot_graph_from_rows(hd_rows,
                     x_key="head_dim",
                     xlabel="HEAD_DIM",
                     title="CPU vs UPMEM-PIM",
                     filename="headdim_bar.png",
                     use_log=False,
                     show_alloc=False)

# 4) NUM_HEADS 
nh_rows = filter_rows(batch=64, seq=32, exp_type="EXP_NH")
plot_graph_from_rows(nh_rows,
                     x_key="num_heads",
                     xlabel="NUM_HEADS",
                     title="CPU vs UPMEM-PIM",
                     filename="numheads_bar.png",
                     use_log=False,
                     show_alloc=True)

# 5) NR_TASKLETS 
tl_rows = filter_rows(exp_type="EXP_TL")
plot_graph_from_rows(tl_rows,
                     x_key="tasklets",
                     xlabel="NR_TASKLETS",
                     title="CPU vs UPMEM-PIM",
                     filename="tasklets_bar.png",
                     use_log=False,
                     show_alloc=False)

print("Done.")