"""
config.py — Central configuration for the NIDS system
Device : Raspberry Pi 4
Edit this file to match your setup before running main.py
"""

import os

# ============================================================
#  PATHS
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

LSTM_MODEL_PATH        = os.path.join(MODELS_DIR, "lstm_model.tflite")
CNN_MODEL_PATH         = os.path.join(MODELS_DIR, "cnn_model.tflite")
AUTOENCODER_MODEL_PATH = os.path.join(MODELS_DIR, "autoencoder_model.tflite")
SCALER_PATH            = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH     = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ============================================================
#  MQTT  (Mosquitto broker running on this Pi)
# ============================================================
MQTT_BROKER_HOST     = "localhost"
MQTT_BROKER_PORT     = 1883
MQTT_CLIENT_ID       = "pi-nids-main"
MQTT_TOPIC_PACKETS   = "nids/packets"
MQTT_TOPIC_STATUS    = "nids/status"
MQTT_TOPIC_ALERTS    = "nids/alerts"
MQTT_TOPIC_HEARTBEAT = "nids/heartbeat"

# ============================================================
#  NETWORK SNIFFING  (Scapy on Pi)
# ============================================================
SNIFF_INTERFACE = "Wi-Fi"  # Change to eth0 if using Ethernet cable
SNIFF_ENABLED   = True      # Set False to rely only on ESP32 MQTT data

# ============================================================
#  FEATURES
# ============================================================
NUM_FEATURES    = 20
SEQUENCE_LENGTH = 10   # Number of flows fed to LSTM as a sequence

FEATURE_NAMES = [
    "dst_port",        "flow_duration",   "fwd_pkt_count",  "bwd_pkt_count",
    "fwd_bytes_total", "bwd_bytes_total", "fwd_pkt_len_max","fwd_pkt_len_mean",
    "bwd_pkt_len_mean","flow_bytes_per_sec","flow_pkts_per_sec","flow_iat_mean",
    "flow_iat_std",    "fwd_iat_mean",    "bwd_iat_mean",   "min_pkt_len",
    "max_pkt_len",     "pkt_len_mean",    "pkt_len_std",    "avg_pkt_size",
]

# ============================================================
#  INFERENCE
# ============================================================
INFERENCE_INTERVAL   = 2.0   # seconds between inference passes
FLOW_TIMEOUT         = 30.0  # seconds of inactivity → export flow
MIN_PACKETS_PER_FLOW = 3     # skip flows with fewer packets

ATTACK_THRESHOLD  = 0.50   # min confidence to call something an attack
ANOMALY_THRESHOLD = 0.73144  # autoencoder MSE threshold — set from training (95th pct normal MSE)

# Must match label encoding used during training
ATTACK_CLASSES = {
    0: "NORMAL",
    1: "DDoS",
    2: "PortScan",
    3: "BruteForce",
    4: "Bot",
    5: "Infiltration",
}

ATTACK_SEVERITY = {
    "NORMAL":      "info",
    "DDoS":        "critical",
    "PortScan":    "high",
    "BruteForce":  "high",
    "Bot":         "medium",
    "Infiltration":"critical",
    "ZERO-DAY":    "critical",
    "ANOMALY":     "high",
}

# ============================================================
#  ALERTS
# ============================================================
GPIO_LED_PIN   = 17   # BCM pin (GPIO17 = Physical pin 11)
GPIO_ENABLED   = True # Set False if no LED is connected

# Seconds before the same attack type can trigger another alert
ALERT_COOLDOWN = 30

# ---- Telegram (optional — works over internet) ----
TELEGRAM_ENABLED   = False              # Flip to True once configured
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"  # From @BotFather
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID"    # Your Telegram chat/group ID

# ============================================================
#  DASHBOARD
# ============================================================
DASHBOARD_HOST  = "0.0.0.0"
DASHBOARD_PORT  = 5000
MAX_LOG_ENTRIES = 500
SECRET_KEY      = "nids-secret-2024"

# ============================================================
#  LOGGING
# ============================================================
LOG_LEVEL        = "INFO"
LOG_FILE         = os.path.join(LOG_DIR, "nids.log")
LOG_MAX_BYTES    = 10 * 1024 * 1024   # 10 MB per file
LOG_BACKUP_COUNT = 3
