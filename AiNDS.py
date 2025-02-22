import json
import pyshark

capture = pyshark.LiveCapture(interface='en0')

# Capture 10 packets
capture.sniff(packet_count=10)

# Extract and convert captured packets to JSON

for index,packet in enumerate(capture):
   #print(packet)
    try:
        #Extract source address from IP or Ethernet layer if available
        source = packet.ip.src if hasattr(packet, "ip") else (packet.eth.src if hasattr(packet, "eth") else "Unknown")

        packet_dict = {
            "timestamp": packet.sniff_time.isoformat(),
            "length": packet.length,
            "layers": [layer.layer_name for layer in packet.layers],  # List of layer names
            "Source": source,
            "Destination": packet.ip.dst,
       }

        #Convert dictionary to JSON string
        JSONData = json.dumps(packet_dict, indent=4)

        #Convert JSON string back to dictionary to access keys
        parsed_data = json.loads(JSONData)

        #Print the extracted data
        print("Packet: ",index+1 ," | Timestamp: ", parsed_data["timestamp"],"| Length: ",parsed_data["length"]," | Layers:", parsed_data["layers"]," | Source:",parsed_data["Source"]," | Destination: ",parsed_data["Destination"])
        print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    except AttributeError as e:
        print(f"Error processing packet: {e}")

#for packet in capture:
  # print(packet)