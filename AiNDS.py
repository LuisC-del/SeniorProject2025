import json
import pyshark
import csv
import os

# Path for tshark (adjust if needed)
TSHARK_PATH = "/opt/homebrew/bin/tshark"
# Output dataset file
DATASET_FILE = "packets_dataset.csv"

def initialize_csv():
    #Creates CSV file with headers if it doesn't exist."""
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, mode='w', newline='') as f:
            writer =csv.writer(f)
            writer.writerow([
                "timestamp", "ip_src", "ip_dst",
                "mac_src", "mac_dst",
                "protocols", "length"
            ])

def save_to_csv(packet_info):
    with open(DATASET_FILE, mode='a', newline='') as f:
        writer =csv.writer(f)
        writer.writerow([
            packet_info["timestamp"],
            packet_info["IP Source"],
            packet_info["IP Destination"],
            packet_info["MAC Source"],
            packet_info["MAC Destination"],
            ",".join(packet_info["layers"]),
            packet_info["length"]
        ])


def live_capture(interface,packet_limit=10):
    print(f"Capturing on {interface}... Press CTRL+C to stop.")
    initialize_csv()

    capture = pyshark.LiveCapture(
        interface=interface,
        tshark_path=TSHARK_PATH  # Explicit path for macOS Homebrew
    )

    try:
        for index, packet in enumerate(capture.sniff_continuously()):
            if index >= packet_limit:
                break
            try:
                ip_source = packet.ip.src if hasattr(packet, "ip") else "Unknown"
                ip_destination = packet.ip.dst if hasattr(packet, "ip") else "Unknown"
                mac_source = packet.eth.src if hasattr(packet, "eth") else "Unknown"
                mac_destination = packet.eth.dst if hasattr(packet, "eth") else "Unknown"

                packet_dict = {
                    "timestamp": packet.sniff_time.isoformat(),
                    "length": packet.length,
                    "layers": [layer.layer_name for layer in packet.layers],
                    "IP Source": ip_source,
                    "IP Destination": ip_destination,
                    "MAC Source": mac_source,
                    "MAC Destination": mac_destination,
                }
                save_to_csv(packet_dict)
                print(f"[{index+1}] Packet captured and saved.")
                print(json.dumps(packet_dict, indent=4))
                print("-" * 100)

            except AttributeError as e:
                print(f"Error processing packet: {e}")

    except KeyboardInterrupt:
        print("\nStopped live capture. Goodbye!")
    finally:
        print(f"Data saved to {DATASET_FILE}")

def read_file(pcap_path):

    try:
        cap = pyshark.FileCapture(
            pcap_path,
            tshark_path=TSHARK_PATH
        )

        packet = cap[0]  # read first packet
        print(f"Packet Length: {packet.length}")
        print(f"Available Layers: {[layer.layer_name for layer in packet.layers]}")
        print(packet)

    except Exception as e:
        print(f"Error reading capture file: {e}")

if __name__ == "__main__":
    # Switch between live and file capture here
    live_capture("en0",packet_limit=10)  
    # read_file("/tmp/mycapture.cap")
