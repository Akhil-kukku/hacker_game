"""
Prepare KDD Cup 99 CSV to the engine's expected schema.

Usage:
  python backend/tools/prepare_kddcup99.py --input data/kddcup99.csv --output data/flows_kdd.csv

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


SERVICE_TO_PORT = {
    "http": 80,
    "https": 443,
    "smtp": 25,
    "ftp": 21,
    "ftp_data": 20,
    "ssh": 22,
    "telnet": 23,
    "domain": 53,
    "pop_3": 110,
    "imap4": 143,
    "finger": 79,
    "time": 37,
    "auth": 113,
    "eco_i": 7,
    "ecr_i": 7,
    "ntp_u": 123,
    "urp_i": 540,
    "klogin": 543,
    "kshell": 544,
    "other": 80,
}


def map_service_to_port(service: str) -> int:
    if not isinstance(service, str):
        return 80
    s = service.strip().lower()
    return SERVICE_TO_PORT.get(s, 80)


def map_protocol(p: str) -> str:
    if not isinstance(p, str):
        return "TCP"
    return p.strip().upper()


def load_and_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Normalize column names if different variants exist
    cols = {c.lower(): c for c in df.columns}

    def get(col_name: str, default=0):
        c = cols.get(col_name, None)
        if c is None:
            return pd.Series([default] * len(df))
        return df[c]

    duration = pd.to_numeric(get("duration", 0.0), errors="coerce").fillna(0.0).astype(float)
    protocol_type = get("protocol_type", "tcp").astype(str).map(map_protocol)
    service = get("service", "http").astype(str)
    flag = get("flag", "").astype(str)
    src_bytes = pd.to_numeric(get("src_bytes", 0), errors="coerce").fillna(0).astype(int)
    dst_bytes = pd.to_numeric(get("dst_bytes", 0), errors="coerce").fillna(0).astype(int)
    label_raw = get("label", "normal.").astype(str)

    # Build features
    total_bytes = (src_bytes + dst_bytes).astype(int)
    # Approximate packet_count assuming ~512 bytes per packet (min 1)
    packet_count = (total_bytes // 512).clip(lower=1).astype(int)
    byte_count = total_bytes
    src_port = pd.Series([1024] * len(df)).astype(int)
    dst_port = service.map(map_service_to_port).astype(int)
    protocol = protocol_type
    flags = flag
    # Synthesize IPs (KDD lacks IPs)
    idx = pd.Series(range(len(df)))
    src_ip = ("192.168." + (idx % 254 + 1).astype(str) + "." + ((idx // 254) % 254 + 1).astype(str))
    dst_ip = ("10.0." + (idx % 254 + 1).astype(str) + "." + ((idx // 254) % 254 + 1).astype(str))
    label = (~label_raw.str.lower().str.startswith("normal")).astype(int)

    out = pd.DataFrame({
        "packet_count": packet_count,
        "byte_count": byte_count,
        "duration": duration,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "flags": flags,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "label": label,
    })
    return out[ENGINE_COLUMNS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="KDD Cup 99 CSV file or directory")
    parser.add_argument("--output", default="data/flows_kdd.csv", help="Output CSV path")
    args = parser.parse_args()

    inp = args.input
    paths = []
    if os.path.isdir(inp):
        paths = sorted(glob.glob(os.path.join(inp, "**", "*.csv"), recursive=True))
    else:
        paths = [inp]

    frames = []
    for p in paths:
        try:
            print(f"Mapping {p}")
            frames.append(load_and_map(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not frames:
        raise SystemExit("No valid KDD Cup 99 CSVs were mapped.")

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined):,} rows to {args.output}")


if __name__ == "__main__":
    main()


