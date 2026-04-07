"""
feature_extractor.py — Flow Builder & Feature Computer
Device  : Raspberry Pi 4

Groups raw packets into bidirectional IP flows and computes
20 CICIDS2017-compatible features per flow for inference.

Flow lifecycle:
  1. Packets arrive from Scapy sniffer (IP-level) or ESP32 (MAC-level).
  2. IP packets are grouped by 5-tuple key into FlowRecord objects.
  3. A background thread expires flows inactive > FLOW_TIMEOUT seconds.
  4. Each expired flow is scored: features → InferenceEngine.

Common mistake: NOT using bidirectional flow keys leads to double-counting
every flow.  _make_key() normalises direction before creating the key.
"""

import time
import logging
import threading
import numpy as np
from collections import defaultdict, deque

import config

logger = logging.getLogger(__name__)


# ============================================================
#  FLOW RECORD
# ============================================================
class FlowRecord:
    """
    Accumulates packets for one bidirectional flow and computes
    the 20-dimensional feature vector used by the models.
    """

    __slots__ = (
        "flow_key", "start_time", "last_time",
        "fwd_packets", "bwd_packets", "_fwd_src",
    )

    def __init__(self, flow_key, first_pkt):
        self.flow_key   = flow_key
        self.start_time = first_pkt["timestamp"]
        self.last_time  = first_pkt["timestamp"]
        self.fwd_packets = []   # list of (timestamp, pkt_len, flags)
        self.bwd_packets = []
        # Canonical forward source = first seen src_ip
        self._fwd_src   = first_pkt.get("src_ip", "")
        self._add(first_pkt)

    # ---- add -------------------------------------------------------
    def add_packet(self, pkt):
        self.last_time = pkt["timestamp"]
        self._add(pkt)

    def _add(self, pkt):
        rec = (pkt["timestamp"], pkt["pkt_len"], pkt.get("flags", 0))
        if pkt.get("src_ip", "") == self._fwd_src:
            self.fwd_packets.append(rec)
        else:
            self.bwd_packets.append(rec)

    # ---- properties ------------------------------------------------
    @property
    def total_packets(self):
        return len(self.fwd_packets) + len(self.bwd_packets)

    @property
    def duration(self):
        return max(self.last_time - self.start_time, 1e-9)

    # ---- feature computation ---------------------------------------
    def compute_features(self):
        """
        Returns numpy float32 array of shape (20,).

        Index → Feature name (matching CICIDS2017 columns):
          0  dst_port            8  bwd_pkt_len_mean
          1  flow_duration (µs)  9  flow_bytes_per_sec
          2  fwd_pkt_count      10  flow_pkts_per_sec
          3  bwd_pkt_count      11  flow_iat_mean (µs)
          4  fwd_bytes_total    12  flow_iat_std  (µs)
          5  bwd_bytes_total    13  fwd_iat_mean  (µs)
          6  fwd_pkt_len_max    14  bwd_iat_mean  (µs)
          7  fwd_pkt_len_mean   15  min_pkt_len
                                16  max_pkt_len
                                17  pkt_len_mean
                                18  pkt_len_std
                                19  avg_pkt_size
        """
        try:
            _, _, _, dst_port, _ = self.flow_key

            fwd_lens = [p[1] for p in self.fwd_packets] or [0]
            bwd_lens = [p[1] for p in self.bwd_packets] or [0]
            all_lens = fwd_lens + bwd_lens

            fwd_ts   = sorted(p[0] for p in self.fwd_packets)
            bwd_ts   = sorted(p[0] for p in self.bwd_packets)
            all_ts   = sorted(fwd_ts + bwd_ts)

            def iats(ts):
                if len(ts) < 2:
                    return [0.0]
                return [ts[i+1] - ts[i] for i in range(len(ts)-1)]

            all_iats = iats(all_ts)
            fwd_iats = iats(fwd_ts)
            bwd_iats = iats(bwd_ts)

            dur_us = self.duration * 1e6

            features = [
                float(dst_port),                             # 0
                float(dur_us),                               # 1
                float(len(self.fwd_packets)),                # 2
                float(len(self.bwd_packets)),                # 3
                float(sum(fwd_lens)),                        # 4
                float(sum(bwd_lens)),                        # 5
                float(max(fwd_lens)),                        # 6
                float(np.mean(fwd_lens)),                    # 7
                float(np.mean(bwd_lens)),                    # 8
                float(sum(all_lens) / self.duration),        # 9  bytes/s
                float(self.total_packets / self.duration),   # 10 pkts/s
                float(np.mean(all_iats) * 1e6),             # 11 IAT mean µs
                float(np.std(all_iats)  * 1e6),             # 12 IAT std  µs
                float(np.mean(fwd_iats) * 1e6),             # 13
                float(np.mean(bwd_iats) * 1e6),             # 14
                float(min(all_lens)),                        # 15
                float(max(all_lens)),                        # 16
                float(np.mean(all_lens)),                    # 17
                float(np.std(all_lens)),                     # 18
                float(np.mean(all_lens)),                    # 19 avg pkt size
            ]

            # Replace NaN / Inf with 0
            features = [0.0 if (not np.isfinite(f)) else f for f in features]
            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error("Feature computation error: %s", e)
            return np.zeros(config.NUM_FEATURES, dtype=np.float32)

    def get_flag_counts(self):
        """Return TCP flag counts across all packets."""
        syn = ack = fin = rst = psh = 0
        for _, _, flags in self.fwd_packets + self.bwd_packets:
            if flags & 0x02: syn += 1
            if flags & 0x10: ack += 1
            if flags & 0x01: fin += 1
            if flags & 0x04: rst += 1
            if flags & 0x08: psh += 1
        return {"SYN": syn, "ACK": ack, "FIN": fin, "RST": rst, "PSH": psh}


# ============================================================
#  ESP32 Wi-Fi LEVEL TRACKER  (MAC-based supplementary data)
# ============================================================
class ESP32Tracker:
    """Lightweight tracker for Wi-Fi metadata from the ESP32."""

    def __init__(self):
        self._data   = defaultdict(list)   # src_mac → [{ts, len, rssi}]
        self._lock   = threading.Lock()
        self._last_gc = time.time()

    def add(self, info):
        key = info.get("src_mac", "unknown")
        with self._lock:
            self._data[key].append({
                "ts":   info["timestamp"],
                "len":  info["pkt_len"],
                "rssi": info.get("rssi", -70),
            })
        self._maybe_gc()

    def _maybe_gc(self):
        now = time.time()
        if now - self._last_gc < 60:
            return
        cutoff = now - 120
        with self._lock:
            for k in list(self._data):
                self._data[k] = [p for p in self._data[k] if p["ts"] > cutoff]
                if not self._data[k]:
                    del self._data[k]
        self._last_gc = now


# ============================================================
#  MAIN FEATURE EXTRACTOR
# ============================================================
class FeatureExtractor:
    """
    Manages active flow records and exports completed flows
    to the InferenceEngine.
    """

    def __init__(self, state, inference_engine):
        self.state    = state
        self.ie       = inference_engine
        self._flows   = {}              # flow_key → FlowRecord
        self._lock    = threading.Lock()
        self._seq_buf = deque(maxlen=config.SEQUENCE_LENGTH)
        self._esp32   = ESP32Tracker()
        self._processed = 0

        # Background thread: expire idle flows
        threading.Thread(
            target=self._expiry_loop, daemon=True, name="FlowExpiry"
        ).start()

        logger.info("FeatureExtractor ready (timeout=%.0fs)", config.FLOW_TIMEOUT)

    # ----------------------------------------------------------------
    def add_packet(self, info):
        """Add an IP-layer packet from Scapy."""
        src = info.get("src_ip", "0.0.0.0")
        dst = info.get("dst_ip", "0.0.0.0")
        sp  = info.get("src_port", 0)
        dp  = info.get("dst_port", 0)
        pr  = info.get("protocol", 0)

        # Ignore broadcast / multicast
        if dst.endswith(".255") or dst.startswith("224.") or dst.startswith("239."):
            return

        key = self._make_key(src, dst, sp, dp, pr)

        with self._lock:
            if key not in self._flows:
                self._flows[key] = FlowRecord(key, info)
            else:
                self._flows[key].add_packet(info)

    def add_packet_esp32(self, info):
        """Add a MAC-layer packet from the ESP32 (no IP info)."""
        self._esp32.add(info)

    # ----------------------------------------------------------------
    @staticmethod
    def _make_key(src, dst, sp, dp, proto):
        """
        Create a canonical bidirectional key so (A→B) and (B→A)
        map to the same flow.
        """
        if (src, sp) > (dst, dp):
            return (dst, src, dp, sp, proto)
        return (src, dst, sp, dp, proto)

    # ----------------------------------------------------------------
    def _expiry_loop(self):
        """
        Every second, check for flows inactive longer than FLOW_TIMEOUT.
        Export qualifying flows to the InferenceEngine.
        """
        while True:
            try:
                time.sleep(1.0)
                now     = time.time()
                to_exp  = []

                with self._lock:
                    for key, flow in list(self._flows.items()):
                        age = now - flow.last_time
                        if (age > config.FLOW_TIMEOUT and
                                flow.total_packets >= config.MIN_PACKETS_PER_FLOW):
                            to_exp.append(flow)
                            del self._flows[key]

                for flow in to_exp:
                    self._export(flow)

            except Exception as e:
                logger.error("Flow expiry error: %s", e)

    def _export(self, flow):
        """Compute features and hand flow off to the InferenceEngine."""
        try:
            features = flow.compute_features()
            self._seq_buf.append(features)
            self._processed += 1

            with self.state.lock:
                self.state.total_flows += 1

            flow_data = {
                "features":  features,
                "sequence":  list(self._seq_buf),
                "flow_key":  str(flow.flow_key),
                "duration":  flow.duration,
                "pkt_count": flow.total_packets,
                "flags":     flow.get_flag_counts(),
                "timestamp": flow.last_time,
            }

            self.ie.process_flow(flow_data)

        except Exception as e:
            logger.error("Flow export error: %s", e)

    # ----------------------------------------------------------------
    def get_stats(self):
        with self._lock:
            return {
                "active_flows":    len(self._flows),
                "flows_processed": self._processed,
                "seq_buffer_len":  len(self._seq_buf),
            }
