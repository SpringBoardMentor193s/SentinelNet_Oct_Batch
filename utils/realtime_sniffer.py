from scapy.all import AsyncSniffer
import threading
import time

# Global control variables
_capture_active = False
_sniffer_instance = None
_stop_signal = threading.Event()

def start_packet_capture(callback, interface=None):
    """
    Starts the packet capture process in a separate thread.
    """
    global _sniffer_instance, _capture_active
    
    _stop_signal.clear()
    _capture_active = True
    
    print("Initializing Network Sniffer...")

    def process_packet(packet):
        if _stop_signal.is_set():
            return
            
        if packet.haslayer('IP'):
            # Extract basic features (simplified for demo purposes)
            # In a real scenario, you would extract all KDD features here
            packet_data = {
                "duration": 0.0,
                "src_bytes": len(packet),
                "dst_bytes": 0,
                "wrong_fragment": 0,
                "hot": 0,
                "logged_in": 0,
                "num_compromised": 0,
                "root_shell": 0,
                "num_root": 0,
                "num_file_creations": 0,
                "num_shells": 0,
                "num_access_files": 0,
                "is_guest_login": 0,
                "count": 1,
                "srv_count": 1,
                "serror_rate": 0.0,
                "srv_serror_rate": 0.0,
                "rerror_rate": 0.0,
                "srv_rerror_rate": 0.0,
                "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0,
                "srv_diff_host_rate": 0.0,
                "dst_host_count": 1,
                "dst_host_srv_count": 1,
                "dst_host_same_srv_rate": 1.0,
                "dst_host_diff_srv_rate": 0.0,
                "dst_host_same_src_port_rate": 0.5,
                "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": 0.0,
                "dst_host_srv_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0,
                "dst_host_srv_rerror_rate": 0.0,
                "protocol_type_encoded": 0,
                "service_encoded": 80,
                "flag_encoded": 0
            }
            callback(packet_data)

    _sniffer_instance = AsyncSniffer(
        prn=process_packet,
        store=False,
        filter="ip",
        iface=interface
    )
    _sniffer_instance.start()

def stop_packet_capture():
    """
    Stops the active packet capture.
    """
    global _sniffer_instance, _capture_active
    
    _stop_signal.set()
    if _sniffer_instance:
        try:
            _sniffer_instance.stop()
            print("Network Sniffer Halted.")
        except Exception as e:
            print(f"Warning: Could not stop sniffer gracefully (likely due to missing Npcap): {e}")
    
    _capture_active = False
    _sniffer_instance = None
