"""
packet_sniffer.py — Live Packet Capture using Scapy
Device  : Raspberry Pi 4

Sniffs packets from the configured network interface,
extracts IP/TCP/UDP metadata, and feeds them to FeatureExtractor.

IMPORTANT: Must run as root (sudo) for raw-socket access.
Common mistake: Running without sudo gives a PermissionError
that Scapy silently swallows — you'll capture nothing!
"""

import time
import logging
import threading

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

import config

logger = logging.getLogger(__name__)


class PacketSniffer:
    """
    Captures live network packets using Scapy on the Pi's own
    network interface (wlan0 / eth0).  Provides richer IP-layer
    data than the ESP32 MAC-layer view.
    """

    def __init__(self, state, feature_extractor):
        self.state = state
        self.fe    = feature_extractor
        self._stop = threading.Event()

    # ----------------------------------------------------------------
    def start(self):
        if not SCAPY_OK:
            logger.error("Scapy not installed.  pip3 install scapy")
            return

        logger.info("Scapy sniffer starting on interface: %s", config.SNIFF_INTERFACE)
        try:
            sniff(
                iface=config.SNIFF_INTERFACE,
                prn=self._process,
                store=False,
                stop_filter=lambda _: not self.state.running,
            )
        except PermissionError:
            logger.error(
                "Scapy: PermissionError — run with sudo!  "
                "sudo python3 main.py"
            )
        except OSError as e:
            logger.error("Scapy OSError (%s): check SNIFF_INTERFACE in config.py", e)
        except Exception as e:
            logger.error("Sniffer error: %s", e)

    def stop(self):
        self._stop.set()

    # ----------------------------------------------------------------
    def _process(self, pkt):
        """Extract metadata from each captured packet."""
        try:
            if not pkt.haslayer(IP):
                return

            ip  = pkt[IP]
            src = ip.src
            dst = ip.dst

            # Skip loopback
            if src.startswith("127.") or dst.startswith("127."):
                return

            info = {
                "src_ip":    src,
                "dst_ip":    dst,
                "protocol":  ip.proto,
                "pkt_len":   len(pkt),
                "timestamp": time.time(),
                "src_port":  0,
                "dst_port":  0,
                "flags":     0,
                "ttl":       ip.ttl,
                "source":    "scapy",
            }

            if pkt.haslayer(TCP):
                t = pkt[TCP]
                info["src_port"] = t.sport
                info["dst_port"] = t.dport
                info["flags"]    = int(t.flags)
                info["protocol"] = 6
            elif pkt.haslayer(UDP):
                u = pkt[UDP]
                info["src_port"] = u.sport
                info["dst_port"] = u.dport
                info["protocol"] = 17
            elif pkt.haslayer(ICMP):
                info["protocol"] = 1

            with self.state.lock:
                self.state.total_packets += 1

            self.fe.add_packet(info)

        except Exception as e:
            logger.debug("Packet processing error: %s", e)
