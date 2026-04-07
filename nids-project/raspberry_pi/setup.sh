#!/bin/bash
# ============================================================
#  NIDS Raspberry Pi — Automated Setup Script
#  Run once on a fresh Raspberry Pi OS installation.
#
#  Usage:
#    chmod +x setup.sh
#    sudo bash setup.sh
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════╗"
echo "║   NIDS Raspberry Pi — Setup Script      ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ---- Step 1: System update ----
echo -e "${CYAN}[1/7] Updating system packages...${NC}"
apt-get update -qq
apt-get upgrade -y -qq

# ---- Step 2: System dependencies ----
echo -e "${CYAN}[2/7] Installing system dependencies...${NC}"
apt-get install -y \
    python3-pip python3-dev python3-venv \
    mosquitto mosquitto-clients \
    libatlas-base-dev libopenjp2-7 libjpeg-dev \
    tcpdump net-tools curl git

# ---- Step 3: Mosquitto MQTT broker ----
echo -e "${CYAN}[3/7] Configuring Mosquitto MQTT broker...${NC}"
cat > /etc/mosquitto/conf.d/nids.conf << 'MQTTCONF'
listener 1883
allow_anonymous true
persistence true
persistence_location /var/lib/mosquitto/
log_dest file /var/log/mosquitto/mosquitto.log
MQTTCONF

systemctl enable mosquitto
systemctl restart mosquitto
echo -e "${GREEN}    Mosquitto running on port 1883 ✓${NC}"

# ---- Step 4: Python packages ----
echo -e "${CYAN}[4/7] Installing Python packages...${NC}"
pip3 install --upgrade pip --quiet
pip3 install flask flask-socketio flask-cors eventlet \
    paho-mqtt scapy numpy scipy scikit-learn joblib pandas \
    requests psutil --quiet

# ---- Step 5: TFLite runtime ----
echo -e "${CYAN}[5/7] Installing TFLite runtime (ARM-optimised)...${NC}"
pip3 install tflite-runtime --quiet || \
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime --quiet || \
echo "    TFLite install failed — will use tensorflow fallback"

# ---- Step 6: RPi.GPIO ----
echo -e "${CYAN}[6/7] Installing RPi.GPIO...${NC}"
pip3 install RPi.GPIO --quiet

# ---- Step 7: systemd service ----
echo -e "${CYAN}[7/7] Creating systemd auto-start service...${NC}"
cat > /etc/systemd/system/nids.service << SVCEOF
[Unit]
Description=NIDS — Real-Time Network Intrusion Detection System
After=network.target mosquitto.service
Requires=mosquitto.service

[Service]
Type=simple
User=pi
WorkingDirectory=${SCRIPT_DIR}
ExecStart=/usr/bin/python3 ${SCRIPT_DIR}/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nids

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable nids.service
echo -e "${GREEN}    Service registered ✓${NC}"

# ---- Done ----
PI_IP=$(hostname -I | awk '{print $1}')
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗"
echo "║          Setup Complete! ✓               ║"
echo "╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Next steps:"
echo "  1. Copy .tflite models  → ${SCRIPT_DIR}/models/"
echo "  2. Edit config.py       → set SNIFF_INTERFACE, GPIO pin, etc."
echo "  3. Start NIDS           → sudo systemctl start nids"
echo "  4. Dashboard URL        → http://${PI_IP}:5000"
echo "  5. View live logs       → sudo journalctl -u nids -f"
echo ""
echo "  Common mistake: ALWAYS run main.py with sudo (Scapy needs root)"
echo "  → sudo python3 main.py"
