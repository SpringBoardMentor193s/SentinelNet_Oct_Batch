# utils/realtime_sniffer.py
import threading
import queue
from scapy.all import sniff
from scapy.layers.inet import IP
from scapy.config import conf

# Public queue that the app reads from
packet_queue = queue.Queue()

# Feature names expected by app (keeps shape correct)
FEATURE_NAMES = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'hot',
    'logged_in', 'num_compromised', 'root_shell', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'protocol_type_encoded',
    'service_encoded', 'flag_encoded'
]

class SnifferThread(threading.Thread):
    """
    Daemon thread that runs sniff() and pushes safe feature dicts into packet_queue.
    Use start_sniffer_thread() to create + start one.
    """
    def __init__(self, iface=None):
        super().__init__(daemon=True)
        self.iface = iface or r"\Device\NPF_{C313C7E2-0BB6-422F-A1B9-C94F462988FD}"
        self._running = threading.Event()
        self._running.set()

    def run(self):
        # Debug: This prints to the terminal where streamlit was launched.
        print("[SnifferThread] THREAD STARTED on iface:", self.iface)

        def handler(pkt):
            try:
                if not pkt.haslayer(IP):
                    return

                # Build minimal feature dict with safe defaults for all expected columns
                features = {f: 0 for f in FEATURE_NAMES}
                # safe assignments
                features["src_bytes"] = len(bytes(pkt)) if hasattr(pkt, "__len__") or True else 0
                features["count"] = 1
                features["srv_count"] = 1
                features["same_srv_rate"] = 1.0

                sport = getattr(pkt, "sport", None)
                features["service_encoded"] = int(sport) if isinstance(sport, int) else 80

                # Put features dict into queue
                packet_queue.put(features)

                # Small debug marker (comment out if too chatty)
                # print("[SnifferThread] packet queued")

            except Exception as e:
                # Never let exceptions escape — print for debug and continue
                print("[SnifferThread] handler exception:", repr(e))

        # sniff will keep running until stop() clears _running
        try:
            sniff(
                prn=handler,
                store=False,
                iface=self.iface,
                stop_filter=lambda x: not self._running.is_set()
            )
        except Exception as e:
            print("[SnifferThread] sniff() raised:", repr(e))

        print("[SnifferThread] run() exiting")

    def stop(self):
        print("[SnifferThread] stop() called")
        self._running.clear()

# Convenience factory
def start_sniffer_thread(iface=None):
    t = SnifferThread(iface=iface)
    t.start()
    return t
