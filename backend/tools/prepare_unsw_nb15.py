"""
Prepare UNSW-NB15 CSVs (train/test) into the engine's expected schema.

Usage:
  python backend/tools/prepare_unsw_nb15.py --input-dir data/UNSW-NB15 --output data/flows_unsw.csv

Input: any folder containing UNSW-NB15 CSV files (train/test). The script discovers all .csv/.csvx.
Output columns:
  packet_count,byte_count,duration,src_port,dst_port,protocol,flags,src_ip,dst_ip,label
"""

import argparse
import os
import glob
import pandas as pd


ENGINE_COLUMNS = [
    "packet_count",
    "byte_count",
    "duration",
    "src_port",
    "dst_port",
    "protocol",
    "flags",
    "src_ip",
    "dst_ip",
    "label",
]


def map_protocol(proto: str) -> str:
    if not isinstance(proto, str):
        return "TCP"
    p = proto.strip().lower()
    if p in {"tcp", "t"}:
        return "TCP"
    if p in {"udp", "u"}:
        return "UDP"
    if p in {"icmp"}:
        return "ICMP"
    if p in {"http"}:
        return "HTTP"
    if p in {"https"}:
        return "HTTPS"
    if p in {"dns"}:
        return "DNS"
    if p in {"smtp"}:
        return "SMTP"
    if p in {"ssh"}:
        return "SSH"
    return p.upper()


def coalesce(df: pd.DataFrame, *cols, default=None) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c]
    # Return a Series of defaults if column not found
    if default is None:
        default = 0
    return pd.Series([default] * len(df))


def load_and_map(path: str) -> pd.DataFrame:
    # Read CSV (csv or csvx)
    if path.lower().endswith(".csvx"):
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Column candidates across UNSW variants
    src_ip = coalesce(df, "srcip", "source_ip", default="")
    dst_ip = coalesce(df, "dstip", "dest_ip", default="")
    sport = coalesce(df, "sport", "src_port", default=0).fillna(0)
    dport = coalesce(df, "dport", "dsport", "dst_port", default=0).fillna(0)
    proto = coalesce(df, "proto", "protocol", default="TCP").fillna("TCP")
    dur = coalesce(df, "dur", "duration", default=0.0).fillna(0.0)
    spkts = coalesce(df, "spkts", "src_pkts", default=0).fillna(0)
    dpkts = coalesce(df, "dpkts", "dst_pkts", default=0).fillna(0)
    sbytes = coalesce(df, "sbytes", "src_bytes", default=0).fillna(0)
    dbytes = coalesce(df, "dbytes", "dst_bytes", default=0).fillna(0)
    state = coalesce(df, "state", default="").fillna("")

    # Labels: prefer numeric 'label' (0 normal, 1 attack); otherwise derive from attack_cat
    if "label" in df.columns:
        label_series = df["label"].fillna(0)
    elif "attack_cat" in df.columns:
        label_series = (df["attack_cat"].fillna("Benign").str.lower() != "benign").astype(int)
    else:
        label_series = pd.Series([0] * len(df))

    out = pd.DataFrame({
        "packet_count": (pd.to_numeric(spkts, errors="coerce").fillna(0) + pd.to_numeric(dpkts, errors="coerce").fillna(0)).astype(int),
        "byte_count": (pd.to_numeric(sbytes, errors="coerce").fillna(0) + pd.to_numeric(dbytes, errors="coerce").fillna(0)).astype(int),
        "duration": pd.to_numeric(dur, errors="coerce").fillna(0.0).astype(float),
        "src_port": pd.to_numeric(sport, errors="coerce").fillna(0).astype(int),
        "dst_port": pd.to_numeric(dport, errors="coerce").fillna(0).astype(int),
        "protocol": proto.astype(str).map(map_protocol),
        "flags": state.astype(str),
        "src_ip": src_ip.astype(str),
        "dst_ip": dst_ip.astype(str),
        "label": pd.to_numeric(label_series, errors="coerce").fillna(0).astype(int),
    })

    # Ensure exactly required columns in order
    return out[ENGINE_COLUMNS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing UNSW-NB15 CSVs (train/test)")
    parser.add_argument("--output", default="data/flows_unsw.csv", help="Output CSV path")
    args = parser.parse_args()

    in_dir = args.input_dir
    paths = sorted(glob.glob(os.path.join(in_dir, "**", "*.csv"), recursive=True) +
                   glob.glob(os.path.join(in_dir, "**", "*.csvx"), recursive=True))
    if not paths:
        raise SystemExit(f"No CSV/CSVX files found under {in_dir}")

    frames = []
    for p in paths:
        try:
            print(f"Mapping {p}")
            frames.append(load_and_map(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not frames:
        raise SystemExit("No valid UNSW-NB15 CSVs were mapped. Please check the input directory and file format.")
    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined):,} rows to {args.output}")


if __name__ == "__main__":
    main()


