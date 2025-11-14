"python AiNDS.py to run"

import time
from pathlib import Path

import pandas as pd
import pyshark

# Path to tshark (Homebrew default on Apple Silicon). Adjust if needed.
TSHARK_PATH = "/opt/homebrew/bin/tshark"


# ------------------ LIVE CAPTURE → LIGHT PREPROCESS ------------------ #
# This file is now focused ONLY on capturing packets and doing
# lightweight preprocessing / saving to CSV.
# Heavy AI / transformer preprocessing will be done in a separate
# script (e.g., transformer_model.py).


def capture_packets(
    interface: str = "en0",
    packet_limit: int = 50,
    max_time: int = 30,
    out_csv: str | None = "live_packets.csv",
    bpf_filter: str | None = None,
) -> pd.DataFrame:
    """Capture packets live and return a lightly processed DataFrame.

    Parameters
    ----------
    interface : str
        Network interface to capture from (e.g., "en0").
    packet_limit : int
        Maximum number of packets to capture.
    max_time : int
        Maximum time in seconds to wait for capture.
    out_csv : str | None
        If not None, save the captured packets to this CSV path.
    bpf_filter : str | None
        Optional BPF filter string to filter captured packets.

    Returns
    -------
    pandas.DataFrame
        DataFrame with raw fields suitable for both:
        - classic models (after further encoding), and
        - text-based transformer preprocessing in a separate script.
    """

    print(
        f"\n[CAPTURE] Starting capture on {interface} for up to "
        f"{packet_limit} packets (max {max_time}s)..."
    )

    # Set up live capture. only_summaries=True keeps it lighter.
    capture = pyshark.LiveCapture(
        interface=interface,
        tshark_path=TSHARK_PATH,
        only_summaries=True,
        bpf_filter=bpf_filter,
    )

    start_time = time.time()

    # Use sniff with timeout + packet_count so we don't block forever.
    # This call returns after either max_time seconds OR packet_limit packets.
    capture.sniff(timeout=max_time, packet_count=packet_limit)

    packets_data: list[dict] = []

    for index, packet in enumerate(capture, start=1):
        try:
            ip_src = getattr(packet.ip, "src", "0") if hasattr(packet, "ip") else "0"
            ip_dst = getattr(packet.ip, "dst", "0") if hasattr(packet, "ip") else "0"
            mac_src = getattr(packet.eth, "src", "0") if hasattr(packet, "eth") else "0"
            mac_dst = getattr(packet.eth, "dst", "0") if hasattr(packet, "eth") else "0"
            protocol = (
                packet.highest_layer if hasattr(packet, "highest_layer") else "Unknown"
            )

            # length and sniff_time are attributes on the summary packet.
            length = int(getattr(packet, "length", 0) or 0)
            timestamp = getattr(packet, "sniff_time", "0")

            packets_data.append(
                {
                    "Timestamp": timestamp,
                    "IP Source": ip_src,
                    "IP Destination": ip_dst,
                    "MAC Source": mac_src,
                    "MAC Destination": mac_dst,
                    "Protocol": protocol,
                    "Length": length,
                }
            )

        except Exception as e:  # noqa: BLE001
            print(f"[CAPTURE] Error reading packet {index}: {e}")

    capture.close()

    elapsed = time.time() - start_time
    print(
        f"[CAPTURE] Done. Collected {len(packets_data)} packets in "
        f"{elapsed:.2f}s."
    )

    if not packets_data:
        print("[CAPTURE] No packets captured. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(packets_data)

    # Ensure numeric type for Length; keep IP/MAC/Protocol as strings.
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(0).astype(int)

    # Fill missing string fields with simple placeholders.
    df.fillna(
        {
            "IP Source": "0",
            "IP Destination": "0",
            "MAC Source": "0",
            "MAC Destination": "0",
            "Protocol": "Unknown",
        },
        inplace=True,
    )

    # Add a label column as a placeholder.
    # For live traffic, everything is assumed benign (0) by default.
    # When you build labeled datasets from NSL-KDD / CICIDS2017,
    # you will overwrite this column with real labels.
    df["Label"] = 0

    # Optionally save to CSV for offline model training / transformer prep.
    if out_csv is not None:
        out_path = Path(out_csv)
        df.to_csv(out_path, index=False)
        print(f"[CAPTURE] Saved {len(df)} packets to {out_path.resolve()}")

    print("[CAPTURE] Preprocessing complete.")
    return df


if __name__ == "__main__":
    # Example: capture ICMP (ping) packets to test live capture
    capture_packets(
        interface="en0",
        packet_limit=20,
        max_time=30,
        bpf_filter="icmp",
    )