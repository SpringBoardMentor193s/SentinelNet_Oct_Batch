# utils/feature_extractor.py

from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.packet import Packet
import time

# Columns expected by the model (same order as training)
FEATURE_COLUMNS = [
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

# Encoding maps (ensure consistent with your preprocessing)
PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}
SERVICE_MAP = {"http": 0, "ftp": 1, "ssh": 2, "dns": 3, "smtp": 4}
FLAG_MAP = {"SF": 0, "S0": 1, "REJ": 2, "RSTO": 3, "RSTR": 4, "SH": 5}

# Session storage for session-based features (simplified)
flow_stats = {}

def extract_features(packet: Packet):
    """
    Extracts features from a Scapy packet and returns a dict
    matching the exact model training columns.
    Missing features default to zero.
    """
    try:
        # ---- Duration (approx via timestamp) ----
        timestamp = time.time()
        duration = 0.0

        # ---- Byte Counts ----
        src_bytes = len(packet.original) if hasattr(packet, 'original') else 0
        dst_bytes = 0  # we cannot know without flow reconstruction

        # ---- Wrong Fragment ----
        wrong_fragment = packet[IP].frag if IP in packet else 0

        # ---- Logged In / Hot / Compromised (not extractable live) ----
        hot = 0
        logged_in = 0
        num_compromised = 0
        root_shell = 0
        num_root = 0
        num_file_creations = 0
        num_shells = 0
        num_access_files = 0
        is_guest_login = 0

        # ---- Session-Based Counts (VERY simplified) ----
        src = packet[IP].src if IP in packet else "0.0.0.0"
        dst = packet[IP].dst if IP in packet else "0.0.0.0"
        key = (src, dst)

        if key not in flow_stats:
            flow_stats[key] = {
                "count": 0,
                "srv_count": 0,
                "last_service": None
            }

        flow_stats[key]["count"] += 1

        count = flow_stats[key]["count"]
        srv_count = flow_stats[key]["srv_count"]

        # ---- Protocol Type ----
        if TCP in packet:
            protocol_type_encoded = PROTOCOL_MAP["tcp"]
            service_encoded = SERVICE_MAP.get("http", 0)
            flag_encoded = FLAG_MAP.get("SF", 0)
        elif UDP in packet:
            protocol_type_encoded = PROTOCOL_MAP["udp"]
            service_encoded = SERVICE_MAP.get("dns", 3)
            flag_encoded = FLAG_MAP.get("SF", 0)
        elif ICMP in packet:
            protocol_type_encoded = PROTOCOL_MAP["icmp"]
            service_encoded = 0
            flag_encoded = 0
        else:
            protocol_type_encoded = 0
            service_encoded = 0
            flag_encoded = 0

        # ---- Rates and Host-based features (not extractable) ----
        serror_rate = 0
        srv_serror_rate = 0
        rerror_rate = 0
        srv_rerror_rate = 0
        same_srv_rate = 0
        diff_srv_rate = 0
        srv_diff_host_rate = 0
        dst_host_count = 0
        dst_host_srv_count = 0
        dst_host_same_srv_rate = 0
        dst_host_diff_srv_rate = 0
        dst_host_same_src_port_rate = 0
        dst_host_srv_diff_host_rate = 0
        dst_host_serror_rate = 0
        dst_host_srv_serror_rate = 0
        dst_host_rerror_rate = 0
        dst_host_srv_rerror_rate = 0

        # ---- Build feature dict ----
        feature_dict = {
            'duration': duration,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'wrong_fragment': wrong_fragment,
            'hot': hot,
            'logged_in': logged_in,
            'num_compromised': num_compromised,
            'root_shell': root_shell,
            'num_root': num_root,
            'num_file_creations': num_file_creations,
            'num_shells': num_shells,
            'num_access_files': num_access_files,
            'is_guest_login': is_guest_login,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'srv_serror_rate': srv_serror_rate,
            'rerror_rate': rerror_rate,
            'srv_rerror_rate': srv_rerror_rate,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'srv_diff_host_rate': srv_diff_host_rate,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': dst_host_same_srv_rate,
            'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
            'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
            'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
            'dst_host_serror_rate': dst_host_serror_rate,
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
            'dst_host_rerror_rate': dst_host_rerror_rate,
            'dst_host_srv_rerror_rate': dst_host_srv_rerror_rate,
            'protocol_type_encoded': protocol_type_encoded,
            'service_encoded': service_encoded,
            'flag_encoded': flag_encoded
        }

        return feature_dict

    except Exception as e:
        print(f"[FEATURE ERROR] {e}")
        return None
