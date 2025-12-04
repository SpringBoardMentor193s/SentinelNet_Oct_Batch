# utils/realtime_sniffer.py
# FINAL CLEAN VERSION — NO SPAM, STOP WORKS, PERFECT FOR STREAMLIT

from scapy.all import AsyncSniffer
import threading

_sniffer = None
_stop_event = threading.Event()

def start_sniffing(callback_function, interface="auto"):
    global _sniffer
    _stop_event.clear()
    
    print("Live Sniffer STARTED → Capturing real traffic...")

    def packet_handler(pkt):
        if _stop_event.is_set():
            return
        if pkt.haslayer('IP'):
            features = {
                "duration": 0.0,
                "src_bytes": len(pkt),
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
            callback_function(features)

    _sniffer = AsyncSniffer(
        prn=packet_handler,
        store=False,
        filter="ip"
    )
    _sniffer.start()

def stop_sniffing():
    global _sniffer
    _stop_event.set()
    if _sniffer:
        _sniffer.stop()
        print("Live Sniffer STOPPED")
    _sniffer = None