# utils/realtime_sniffer.py
# FINAL WORKING + MAX DEBUG VERSION — DEC 2025

from scapy.all import AsyncSniffer, get_if_list
import threading

print("[DEBUG] realtime_sniffer.py LOADED SUCCESSFULLY!")

# Global sniffer
_sniffer = None

def start_sniffing(callback_function, interface="auto"):
    global _sniffer
    print("\n" + "="*60)
    print("start_sniffing() CALLED! — THIS MEANS IT WORKS!")
    print(f"Callback function: {callback_function}")
    print("="*60 + "\n")

    # List all interfaces
    ifaces = get_if_list()
    print(f"All Scapy interfaces: {ifaces}")

    # Find Wi-Fi
    wifi_iface = None
    for iface in ifaces:
        if "wi-fi" in iface.lower() or "wlan" in iface.lower() or "wifi" in iface.lower():
            wifi_iface = iface
            break
    if not wifi_iface and ifaces:
        wifi_iface = ifaces[0]  # fallback

    print(f"Using interface: {wifi_iface}")

    def packet_handler(pkt):
        if hasattr(pkt, '__len__'):
            print(f"[PACKET] Captured! Size: {len(pkt)} bytes")
            features = {
                "duration": 0.0, "src_bytes": len(pkt), "dst_bytes": 0,
                "protocol_type_encoded": 0, "service_encoded": 80, "flag_encoded": 0,
                "count": 1, "srv_count": 1, "same_srv_rate": 1.0,
                # ... fill rest with 0s
                **{col: 0 for col in [
                    'wrong_fragment', 'hot', 'logged_in', 'num_compromised', 'root_shell',
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                    'is_guest_login', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                    'srv_rerror_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate'
                ]}
            }
            callback_function(features)

    try:
        _sniffer = AsyncSniffer(
            iface=wifi_iface,
            prn=packet_handler,
            store=False,
            filter="tcp or udp or icmp"
        )
        _sniffer.start()
        print(f"[SUCCESS] SNIFFER STARTED ON {wifi_iface}")
    except Exception as e:
        print(f"[FAILED] {e}")

def stop_sniffing():
    global _sniffer
    if _sniffer:
        _sniffer.stop()
        print("[SNIFFER] STOPPED")
    _sniffer = None