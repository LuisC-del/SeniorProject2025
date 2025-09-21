import json
import pyshark

def live_capture(interface):
    print(f"Capturing on {interface}... Press CTRL+C to stop.")

    capture = pyshark.LiveCapture(
        interface=interface,
        tshark_path="/opt/homebrew/bin/tshark"  # Explicit path for macOS Homebrew
    )

    try:
        for index, packet in enumerate(capture.sniff_continuously()):
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

                print(json.dumps(packet_dict, indent=4))
                print("-" * 100)

            except AttributeError as e:
                print(f"Error processing packet: {e}")

    except KeyboardInterrupt:
        print("\nStopped live capture. Goodbye!")

def read_file(pcap_path):
    try:
        cap = pyshark.FileCapture(
            pcap_path,
            tshark_path="/opt/homebrew/bin/tshark"
        )

        packet = cap[0]  # read first packet
        print(f"Packet Length: {packet.length}")
        print(f"Available Layers: {[layer.layer_name for layer in packet.layers]}")
        print(packet)

    except Exception as e:
        print(f"Error reading capture file: {e}")

if __name__ == "__main__":
    # Switch between live and file capture here
    live_capture("en0")  
    # read_file("/tmp/mycapture.cap")
