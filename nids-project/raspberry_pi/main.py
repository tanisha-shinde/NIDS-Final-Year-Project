"""
main.py — NIDS Entry Point
Device  : Raspberry Pi 4

Starts all system components in background threads:
  • Mosquitto MQTT subscriber  (receives ESP32 packet data)
  • Scapy packet sniffer       (captures local IP traffic)
  • Feature extractor          (builds flows, computes 20 features)
  • TFLite inference engine    (LSTM + CNN-1D + Autoencoder ensemble)
  • Alert system               (GPIO LED + Telegram)
  • Flask-SocketIO dashboard   (runs on main thread, port 5000)

Run with:  sudo python3 main.py
(sudo required for Scapy raw-socket capture)
"""

import sys
import os
import time
import signal
import logging
import threading
import requests
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.feature_extractor import FeatureExtractor
from core.inference_engine   import InferenceEngine
from core.alert_system       import AlertSystem
from core.mqtt_client        import MQTTClient
from core.packet_sniffer     import PacketSniffer
from dashboard.app           import create_app, socketio


# ============================================================
#  LOGGING SETUP
# ============================================================
def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return root


def send_telegram_alert(attack_type=None):
    """
    Send a Telegram alert for non-normal predictions.
    """
    logger = logging.getLogger("main")
    bot_token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    chat_id = getattr(config, "TELEGRAM_CHAT_ID", "")

    if (
        not bot_token
        or not chat_id
        or "YOUR_BOT_TOKEN" in str(bot_token)
        or "YOUR_CHAT_ID" in str(chat_id)
    ):
        logger.warning("Telegram alert skipped: bot token/chat id not configured")
        return False

    message = "🚨 ALERT: Network Attack Detected"
    if attack_type:
        message += f"\nAttack Type: {attack_type}"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Telegram alert sent")
            return True
        logger.error("Telegram alert failed (%d): %s", resp.status_code, resp.text[:120])
    except requests.RequestException as e:
        logger.error("Telegram alert error: %s", e)
    return False


# ============================================================
#  SHARED STATE
# ============================================================
class NIDSState:
    """
    Single shared-state object passed to every component.
    All multi-thread access must acquire self.lock first.
    """
    def __init__(self):
        self.running       = True
        self.total_packets = 0
        self.total_flows   = 0
        self.esp32_online  = False
        self.model_loaded  = False
        self.alerts        = []          # newest-first list of alert dicts
        self.attack_counts = {name: 0 for name in config.ATTACK_CLASSES.values()}
        self.attack_counts.update({"ZERO-DAY": 0, "ANOMALY": 0})
        self.lock = threading.Lock()


# ============================================================
#  GRACEFUL SHUTDOWN
# ============================================================
def shutdown(state, components):
    logging.getLogger("main").info("Shutting down NIDS…")
    state.running = False
    for c in components:
        try:
            c.stop()
        except Exception:
            pass
    sys.exit(0)


# ============================================================
#  MAIN
# ============================================================
def main():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("  NIDS — Network Intrusion Detection System")
    logger.info("  Device: Raspberry Pi 4 | Python %s", sys.version.split()[0])
    logger.info("=" * 60)

    state = NIDSState()

    # Build component chain (order matters for dependency injection)
    alert_sys  = AlertSystem(state)
    inference  = InferenceEngine(state, alert_sys)
    feature_ex = FeatureExtractor(state, inference)
    mqtt_cli   = MQTTClient(state, feature_ex)
    pkt_sniff  = PacketSniffer(state, feature_ex)

    original_handle = inference._handle

    def handle_with_telegram(result, fd):
        original_handle(result, fd)
        prediction = result.get("label") if isinstance(result, dict) else None
        if prediction and prediction != "NORMAL":
            send_telegram_alert(attack_type=prediction)

    inference._handle = handle_with_telegram

    app = create_app(state, inference, alert_sys)

    components = [alert_sys, inference, feature_ex, mqtt_cli, pkt_sniff]

    # Signal handlers for graceful stop
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda s, f: shutdown(state, components))

    # ---- Start background threads ----
    threads = [
        threading.Thread(target=mqtt_cli.start,       name="MQTT",      daemon=True),
        threading.Thread(target=inference.run_loop,   name="Inference", daemon=True),
        threading.Thread(target=alert_sys.start,      name="Alerts",    daemon=True),
    ]

    if config.SNIFF_ENABLED:
        threads.append(
            threading.Thread(target=pkt_sniff.start, name="Sniffer", daemon=True)
        )

    for t in threads:
        t.start()
        logger.info("Started thread: %s", t.name)

    logger.info("Dashboard → http://0.0.0.0:%d", config.DASHBOARD_PORT)
    logger.info("All systems go. Monitoring network traffic…")

    # Flask-SocketIO runs on the main thread (blocking)
    try:
        socketio.run(
            app,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            debug=False,
            use_reloader=False,
            log_output=False,
        )
    except KeyboardInterrupt:
        shutdown(state, components)


if __name__ == "__main__":
    main()
