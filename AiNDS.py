import pyshark
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time

# Path to tshark
TSHARK_PATH = "/opt/homebrew/bin/tshark"  # Adjust if needed

# ------------------ LIVE CAPTURE + PREPROCESS ------------------ #
def capture_and_preprocess(interface="en0", packet_limit=10, max_time=30):
    """
    Capture packets live and preprocess them into a DataFrame.
    - interface: network interface to capture from
    - packet_limit: max number of packets to capture
    - max_time: maximum time (seconds) to wait for capture
    """
    print(f"=Starting capture on {interface} for up to {packet_limit} packets (max {max_time}s)...")

    capture = pyshark.LiveCapture(
        interface=interface,
        tshark_path=TSHARK_PATH,
        only_summaries=True,
    )

    packets_data = []
    start_time = time.time()

    try:
        for index, packet in enumerate(capture.sniff_continuously(), start=1):
            # Stop if reached packet limit
            if index > packet_limit:
                break

            # Stop if max time exceeded
            if time.time() - start_time > max_time:
                print("Max capture time reached.")
                break

            try:
                ip_src = getattr(packet.ip, "src", "0") if hasattr(packet, "ip") else "0"
                ip_dst = getattr(packet.ip, "dst", "0") if hasattr(packet, "ip") else "0"
                mac_src = getattr(packet.eth, "src", "0") if hasattr(packet, "eth") else "0"
                mac_dst = getattr(packet.eth, "dst", "0") if hasattr(packet, "eth") else "0"
                protocol = packet.highest_layer if hasattr(packet, "highest_layer") else "Unknown"
                length = int(getattr(packet, "length", 0))
                timestamp = getattr(packet, "sniff_time", "0")

                packet_dict = {
                    "Timestamp": timestamp,
                    "IP Source": ip_src,
                    "IP Destination": ip_dst,
                    "MAC Source": mac_src,
                    "MAC Destination": mac_dst,
                    "Protocol": protocol,
                    "Length": length
                }

                packets_data.append(packet_dict)

            except Exception as e:
                print(f"Error reading packet {index}: {e}")

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    finally:
        capture.close()
        print(f"Capture complete. {len(packets_data)} packets collected.")

    # ------------------ PREPROCESSING ------------------ #
    if not packets_data:
        print("No packets captured. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(packets_data)

    # Convert numeric columns
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce").fillna(0).astype(int)
    df.fillna({
        "IP Source": "0",
        "IP Destination": "0",
        "MAC Source": "0",
        "MAC Destination": "0",
        "Protocol": "Unknown"
    }, inplace=True)

    # Encode categorical columns
    for col in ["IP Source", "IP Destination", "MAC Source", "MAC Destination", "Protocol"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Add label placeholder
    df["Label"] = 0  # Benign = 0, Malicious = 1

    print("Preprocessing complete.")
    return df

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    df = capture_and_preprocess(interface="en0", packet_limit=10, max_time=30)
    