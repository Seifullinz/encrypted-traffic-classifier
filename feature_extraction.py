#!/usr/bin/env python3
"""
feature_extraction.py
=====================================================================
Extracts flow-level statistical features from .pcap files for the
classification of encrypted multimedia traffic (YouTube vs Apple Music).
The payload is NEVER inspected (traffic is encrypted with TLS 1.3 / QUIC).
Only metadata is used: packet sizes, timings and direction.

Expected dataset layout:
    <data_dir>/
        YouTube/  *.pcap | *.pcapng
        Music/    *.pcap | *.pcapng

Two modes:
  * Session mode (default): one pcap -> one feature row.
        python3 feature_extraction.py --data-dir ./dataset --out features.csv
  * Window mode (stricter, more samples): each pcap is split into N-second windows.
        python3 feature_extraction.py --data-dir ./dataset --out features_win.csv --window 15

The CSV always contains a 'source_file' column (the originating pcap). It is
used by train_model.py for group-aware splitting, so that windows from the same
capture never appear in both the training and the test set.
=====================================================================
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

try:
    from scapy.all import PcapReader, IP, IPv6, TCP, UDP
except ImportError:
    print("ERROR: scapy is not installed. Install it with: pip install scapy", file=sys.stderr)
    raise

MEDIA_PORTS = {443}            # TLS (TCP/443) and QUIC (UDP/443)
BURST_GAP_SECONDS = 0.5        # a gap larger than this starts a new burst
MIN_PACKETS_PER_WINDOW = 20    # windows smaller than this are skipped
PRIVATE_PREFIXES = ("10.", "192.168.") + tuple(f"172.{i}." for i in range(16, 32))


def _is_private(ip):
    return isinstance(ip, str) and ip.startswith(PRIVATE_PREFIXES)


def _safe_stats(values):
    """Mean / std / median / min / max, guarded against an empty list."""
    if len(values) == 0:
        return dict(mean=0.0, std=0.0, median=0.0, min=0.0, max=0.0)
    arr = np.asarray(values, dtype=float)
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)),
                median=float(np.median(arr)), min=float(np.min(arr)), max=float(np.max(arr)))


def _read_packets(pcap_path):
    """Stream a pcap into a list of (time, length, src, dst, is_udp) records.
    Keep media ports (443); if none are present, fall back to all IP packets."""
    media, all_ip = [], []
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if IP in pkt:
                ip_layer = pkt[IP]
            elif IPv6 in pkt:
                ip_layer = pkt[IPv6]
            else:
                continue
            src, dst = ip_layer.src, ip_layer.dst
            length, t = len(pkt), float(pkt.time)
            sport = dport = None
            is_udp = False
            if TCP in pkt:
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
            elif UDP in pkt:
                sport, dport = pkt[UDP].sport, pkt[UDP].dport
                is_udp = True
            rec = (t, length, src, dst, is_udp)
            all_ip.append(rec)
            if sport in MEDIA_PORTS or dport in MEDIA_PORTS:
                media.append(rec)
    return media if media else all_ip


def _guess_local_ip(records):
    """The local (client) host is the most frequent private IP address.
    If no private address is present, fall back to the most frequent IP."""
    counter = Counter()
    for _, _, src, dst, _ in records:
        counter[src] += 1
        counter[dst] += 1
    if not counter:
        return None
    private = [(ip, c) for ip, c in counter.items() if _is_private(ip)]
    if private:
        return max(private, key=lambda x: x[1])[0]
    return counter.most_common(1)[0][0]


def _compute_bursts(down_times, down_sizes):
    """Group downstream packets into bursts (a gap > BURST_GAP_SECONDS starts a
    new burst). Return lists of: bytes per burst, burst duration, packets per burst."""
    if len(down_times) < 2:
        return [], [], []
    order = np.argsort(down_times)
    t = np.asarray(down_times)[order]
    s = np.asarray(down_sizes)[order]
    burst_bytes, burst_durations, burst_pkts = [], [], []
    cur_bytes, cur_start, cur_last, cur_count = float(s[0]), t[0], t[0], 1
    for i in range(1, len(t)):
        if t[i] - cur_last > BURST_GAP_SECONDS:
            burst_bytes.append(cur_bytes)
            burst_durations.append(cur_last - cur_start)
            burst_pkts.append(cur_count)
            cur_bytes, cur_start, cur_count = 0.0, t[i], 0
        cur_bytes += float(s[i])
        cur_last = t[i]
        cur_count += 1
    burst_bytes.append(cur_bytes)
    burst_durations.append(cur_last - cur_start)
    burst_pkts.append(cur_count)
    return burst_bytes, burst_durations, burst_pkts


FEATURE_KEYS = [
    "total_packets", "total_bytes", "duration_s", "packets_per_s", "bytes_per_s",
    "pkt_size_mean", "pkt_size_std", "pkt_size_median", "pkt_size_max",
    "down_pkt_size_mean", "down_pkt_size_std", "down_pkt_size_median", "down_pkt_size_max",
    "up_pkt_size_mean", "up_pkt_size_std", "up_pkt_size_median", "up_pkt_size_max",
    "iat_mean", "iat_std", "iat_max",
    "down_up_byte_ratio", "down_up_pkt_ratio", "downstream_byte_fraction",
    "num_bursts", "burst_bytes_mean", "burst_bytes_max",
    "burst_duration_mean", "burst_pkts_mean", "udp_fraction",
]


def _empty_feature_row():
    return {k: 0.0 for k in FEATURE_KEYS}


def extract_features_from_records(records):
    """Turn a list of packets into one feature dictionary.
    This core function works both on a whole capture and on a single window."""
    feats = {}
    if len(records) < 2:
        return _empty_feature_row()
    records = sorted(records, key=lambda r: r[0])
    local_ip = _guess_local_ip(records)
    times = [r[0] for r in records]
    sizes = [r[1] for r in records]
    duration = max(max(times) - min(times), 1e-6)

    # Split into downstream (server -> client) and upstream (client -> server)
    down_sizes, up_sizes, down_times = [], [], []
    udp_count = 0
    for t, length, src, dst, is_udp in records:
        if is_udp:
            udp_count += 1
        if dst == local_ip:
            down_sizes.append(length)
            down_times.append(t)
        elif src == local_ip:
            up_sizes.append(length)

    # 1. Volume and rate
    feats["total_packets"] = len(records)
    feats["total_bytes"] = float(sum(sizes))
    feats["duration_s"] = float(duration)
    feats["packets_per_s"] = len(records) / duration
    feats["bytes_per_s"] = float(sum(sizes)) / duration

    # 2. Packet-size statistics (all / downstream / upstream)
    for prefix, vals in (("pkt", sizes), ("down_pkt", down_sizes), ("up_pkt", up_sizes)):
        s = _safe_stats(vals)
        feats[f"{prefix}_size_mean"] = s["mean"]
        feats[f"{prefix}_size_std"] = s["std"]
        feats[f"{prefix}_size_median"] = s["median"]
        feats[f"{prefix}_size_max"] = s["max"]

    # 3. Inter-arrival time (IAT) and jitter
    iat = _safe_stats(np.diff(np.asarray(times, dtype=float)))
    feats["iat_mean"] = iat["mean"]
    feats["iat_std"] = iat["std"]  # jitter
    feats["iat_max"] = iat["max"]

    # 4. Flow directionality
    down_bytes, up_bytes = float(sum(down_sizes)), float(sum(up_sizes))
    feats["down_up_byte_ratio"] = down_bytes / (up_bytes + 1.0)
    feats["down_up_pkt_ratio"] = len(down_sizes) / (len(up_sizes) + 1.0)
    feats["downstream_byte_fraction"] = down_bytes / (down_bytes + up_bytes + 1.0)

    # 5. Burst features on the downstream stream (mimics video buffer refill)
    bursts = _compute_bursts(down_times, down_sizes)
    bb, bd, bp = (_safe_stats(x) for x in bursts)
    feats["num_bursts"] = float(len(bursts[0]))
    feats["burst_bytes_mean"] = bb["mean"]
    feats["burst_bytes_max"] = bb["max"]
    feats["burst_duration_mean"] = bd["mean"]
    feats["burst_pkts_mean"] = bp["mean"]

    # 6. Fraction of QUIC (UDP/443) against TLS (TCP/443)
    feats["udp_fraction"] = udp_count / len(records)
    return feats


def extract_features_from_pcap(pcap_path):
    """Whole capture -> one feature vector (used by the dashboard)."""
    return extract_features_from_records(_read_packets(pcap_path))


def _split_windows(records, window):
    """Split time-sorted packets into windows of `window` seconds."""
    records = sorted(records, key=lambda r: r[0])
    t0 = records[0][0]
    bins = {}
    for r in records:
        bins.setdefault(int((r[0] - t0) // window), []).append(r)
    return [bins[k] for k in sorted(bins)]


def build_dataset(data_dir, out_csv, window=0):
    """Walk the per-class sub-folders, extract features from every pcap, write CSV."""
    rows = []
    classes = [d for d in sorted(os.listdir(data_dir))
               if os.path.isdir(os.path.join(data_dir, d))]
    if not classes:
        print(f"No class sub-folders found in {data_dir} (expected YouTube/ and Music/).", file=sys.stderr)
        sys.exit(1)
    windowed = bool(window and window > 0)
    mode = f"windowed ({window}s)" if windowed else "session"
    print(f"Mode: {mode} | Classes: {classes}")

    for label in classes:
        class_dir = os.path.join(data_dir, label)
        pcaps = [f for f in sorted(os.listdir(class_dir))
                 if f.lower().endswith((".pcap", ".pcapng"))]
        print(f"  [{label}] files: {len(pcaps)}")
        for i, fname in enumerate(pcaps, 1):
            path = os.path.join(class_dir, fname)
            try:
                records = _read_packets(path)
            except Exception as e:  # noqa: BLE001
                print(f"    ! skipped {fname}: {e}", file=sys.stderr)
                continue
            if len(records) < 2:
                print(f"    ! empty/corrupt: {fname}", file=sys.stderr)
                continue
            chunks = _split_windows(records, window) if windowed else [records]
            for k, chunk in enumerate(chunks):
                if windowed and len(chunk) < MIN_PACKETS_PER_WINDOW:
                    continue
                feats = extract_features_from_records(chunk)
                feats["source_file"] = fname
                feats["filename"] = f"{fname}#w{k}" if windowed else fname
                feats["label"] = label
                rows.append(feats)
            if i % 10 == 0:
                print(f"    processed {i}/{len(pcaps)}")

    if not rows:
        print("No samples could be extracted.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    tail = ["source_file", "filename", "label"]
    cols = [c for c in df.columns if c not in tail] + tail
    df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"\nDone: {out_csv} ({df.shape[0]} rows, {df.shape[1] - 3} features)")
    print(df["label"].value_counts().to_string())
    if windowed:
        print(f"Distinct source captures: {df['source_file'].nunique()}")


def main():
    ap = argparse.ArgumentParser(description="Extract flow features from pcap files.")
    ap.add_argument("--data-dir", default="./dataset")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--window", type=int, default=0,
                    help="window length in seconds (0 = whole capture as one row)")
    args = ap.parse_args()
    build_dataset(args.data_dir, args.out, args.window)


if __name__ == "__main__":
    main()
