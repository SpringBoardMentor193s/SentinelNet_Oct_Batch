from scapy.all import sniff
from utils.feature_extractor import extract_features

def start_sniffing(callback_function):
    def process(pkt):
        try:
            features = extract_features(pkt)
            if features:
                callback_function(features)
        except:
            pass

    sniff(prn=process, store=False)
