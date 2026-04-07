# Real-Time Network Intrusion Detection System (NIDS)
### Deep Learning on Edge Hardware — BE ENTC Final Year Project

> **One-line summary:** An ESP32 sniffs Wi-Fi packets → sends data to a Raspberry Pi 4 via MQTT → Pi runs a 3-model TFLite ensemble (LSTM + CNN-1D + Autoencoder) → classifies traffic as Normal / Attack / Zero-Day → triggers GPIO LED + Telegram alert + live web dashboard.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Hardware Required](#2-hardware-required)
3. [Software Stack](#3-software-stack)
4. [System Architecture](#4-system-architecture)
5. [Project Structure](#5-project-structure)
6. [Step-by-Step Setup](#6-step-by-step-setup)
   - [A. Train Models (Google Colab)](#a-train-models-google-colab)
   - [B. Convert to TFLite (Google Colab)](#b-convert-to-tflite-google-colab)
   - [C. Set Up Raspberry Pi](#c-set-up-raspberry-pi)
   - [D. Flash ESP32](#d-flash-esp32)
   - [E. Run the System](#e-run-the-system)
7. [Dataset](#7-dataset-cicids2017)
8. [Deep Learning Models](#8-deep-learning-models)
9. [Dashboard](#9-dashboard)
10. [Alerts](#10-alerts)
11. [Common Mistakes & Fixes](#11-common-mistakes--fixes)
12. [Results Summary](#12-results-summary)

---

## 1. Project Overview

This project builds a **fully offline, real-time NIDS** that runs entirely on a Raspberry Pi 4 at the network edge — no cloud, no internet dependency.

| Feature | Details |
|---|---|
| Dataset | CICIDS2017 (Canadian Institute for Cybersecurity) |
| Attack classes | Normal, DDoS, PortScan, BruteForce, Bot, Infiltration |
| Models | LSTM + CNN-1D + Autoencoder (ensemble) |
| Inference device | Raspberry Pi 4 (4 GB RAM) |
| Sensor | ESP32 DevKit v1 (Wi-Fi packet capture) |
| Alert output | GPIO LED + Telegram Bot + Web Dashboard |
| Prior work | CatBoost IDS on static CICIDS2017 (Semester 1) |

**Key novelty over prior work:**
- Real-time edge inference (vs. offline batch classification)
- Hardware sensor (ESP32) for dedicated packet capture
- Zero-day detection via unsupervised Autoencoder
- Three-model ensemble reduces false positives

---

## 2. Hardware Required

| Component | Purpose | Qty |
|---|---|---|
| Raspberry Pi 4 (4 GB RAM) | Edge inference + MQTT broker + dashboard | 1 |
| MicroSD card (32 GB, Class 10) | Raspberry Pi OS + models | 1 |
| ESP32 DevKit v1 | Wi-Fi packet sniffer / hardware sensor | 1 |
| LED (any colour) + 330 Ω resistor | Physical attack alert | 1 each |
| USB-C power supply (5V 3A) for Pi | Power | 1 |
| Micro-USB cable for ESP32 | Power + flashing firmware | 1 |
| Wi-Fi router / mobile hotspot | Local network | 1 |

> All devices communicate **over local Wi-Fi only** — no internet required once trained.

---

## 3. Software Stack

| Layer | Technology |
|---|---|
| OS | Raspberry Pi OS Lite (64-bit) |
| Language | Python 3.9+ (Pi), C++ / Arduino (ESP32) |
| ML framework | TensorFlow → TFLite (Colab training), tflite-runtime (Pi inference) |
| Feature extraction | Scapy (packet parsing), custom flow aggregator |
| Messaging | MQTT via Mosquitto broker (on Pi) + paho-mqtt |
| Dashboard | Flask + Flask-SocketIO + Chart.js |
| Alerts | RPi.GPIO (LED), python-telegram-bot (Telegram) |
| Training | Google Colab (GPU), CICIDS2017 CSV files |
| IDE (ESP32) | Arduino IDE 2.x + ESP32 board support package |

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    LOCAL Wi-Fi NETWORK                       │
│                                                              │
│  ┌────────────┐  MQTT/JSON   ┌───────────────────────────┐  │
│  │   ESP32    │ ──────────── │     Raspberry Pi 4         │  │
│  │ (sniffer)  │  topic:      │                            │  │
│  │            │  nids/packets│  ┌─────────────────────┐  │  │
│  │ Promiscuous│              │  │  Feature Extractor   │  │  │
│  │ mode Wi-Fi │              │  │  (flow statistics)   │  │  │
│  │ sniffing   │              │  └────────┬────────────┘  │  │
│  └────────────┘              │           │                │  │
│                              │  ┌────────▼────────────┐  │  │
│  ┌────────────┐              │  │  Inference Engine    │  │  │
│  │ Pi (Scapy) │              │  │  ┌──────┐ ┌──────┐  │  │  │
│  │ local sniff│──────────────┤  │  │ LSTM │ │CNN-1D│  │  │  │
│  └────────────┘              │  │  └──┬───┘ └──┬───┘  │  │  │
│                              │  │     └────┬────┘      │  │  │
│                              │  │  ┌───────▼────────┐  │  │  │
│                              │  │  │  Autoencoder   │  │  │  │
│                              │  │  │  (zero-day)    │  │  │  │
│                              │  │  └───────┬────────┘  │  │  │
│                              │  └──────────┼────────────┘  │  │
│                              │             │                │  │
│                              │  ┌──────────▼──────────┐   │  │
│                              │  │    Alert System      │   │  │
│                              │  │  GPIO LED + Telegram │   │  │
│                              │  └──────────────────────┘   │  │
│                              │                              │  │
│                              │  ┌──────────────────────┐   │  │
│                              │  │  Flask Dashboard      │   │  │
│                              │  │  http://<PI_IP>:5000  │   │  │
│                              │  └──────────────────────┘   │  │
│                              └───────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. ESP32 captures Wi-Fi packet metadata → publishes to MQTT topic `nids/packets`
2. Pi MQTT client receives packet data; Pi also runs Scapy for local sniffing
3. Feature Extractor aggregates packets into flows and extracts 20 features
4. Inference Engine runs LSTM + CNN-1D (classification) + Autoencoder (anomaly)
5. Ensemble vote → label + confidence
6. Alert System triggers LED / Telegram if attack detected
7. Flask dashboard streams live results via SocketIO

---

## 5. Project Structure

```
nids-project/
├── esp32_firmware/
│   └── esp32_sniffer.ino        # Arduino sketch — flash to ESP32
│
├── raspberry_pi/
│   ├── main.py                  # Entry point — run this to start NIDS
│   ├── config.py                # All settings (IPs, thresholds, paths)
│   ├── requirements.txt         # Python dependencies
│   ├── setup.sh                 # Automated Pi setup script
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── packet_sniffer.py    # Scapy-based local packet capture
│   │   ├── mqtt_client.py       # Receives ESP32 data via MQTT
│   │   ├── feature_extractor.py # Flow aggregation + 20-feature extraction
│   │   ├── inference_engine.py  # LSTM + CNN + Autoencoder ensemble
│   │   └── alert_system.py      # GPIO LED + Telegram alerts
│   │
│   ├── dashboard/
│   │   ├── app.py               # Flask + SocketIO backend
│   │   ├── templates/
│   │   │   └── index.html       # Dark cybersecurity dashboard UI
│   │   └── static/
│   │       ├── css/style.css    # Dark theme CSS
│   │       └── js/dashboard.js  # Chart.js + SocketIO live updates
│   │
│   ├── models/                  # Place .tflite + scaler.pkl here
│   │   ├── lstm_model.tflite
│   │   ├── cnn_model.tflite
│   │   ├── autoencoder_model.tflite
│   │   └── scaler.pkl
│   │
│   └── logs/                    # Runtime logs (auto-created)
│       └── nids.log
│
└── training/                    # Run on Google Colab (not on Pi)
    ├── colab_train.py           # Train all 3 models on CICIDS2017
    └── convert_tflite.py        # Convert .h5 → .tflite for Pi
```

---

## 6. Step-by-Step Setup

### A. Train Models (Google Colab)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `training/colab_train.py` → open as notebook or paste into a cell
3. Set Runtime → Change runtime type → **GPU**
4. Download CICIDS2017 dataset from https://www.unb.ca/cic/datasets/ids-2017.html
   - Upload CSV files to your Google Drive at: `MyDrive/NIDS/cicids2017/`
   - *(If no CSVs found, script auto-generates synthetic demo data)*
5. Run all cells — training takes ~20-40 minutes on GPU
6. Output saved to `MyDrive/NIDS/models/`:
   - `lstm_model.h5`
   - `cnn_model.h5`
   - `autoencoder_model.h5`
   - `scaler.pkl`
7. **Note the `ANOMALY_THRESHOLD` value** printed at the end — update `config.py` line 64

### B. Convert to TFLite (Google Colab)

1. Upload `training/convert_tflite.py` to Colab (same session or new)
2. Run all cells
3. Output `.tflite` files saved to `MyDrive/NIDS/models/`
4. Download to your PC:
   - `lstm_model.tflite`
   - `cnn_model.tflite`
   - `autoencoder_model.tflite`
   - `scaler.pkl` ← **do not forget this one**

### C. Set Up Raspberry Pi

**1. Install OS**
```bash
# Flash Raspberry Pi OS Lite (64-bit) using Raspberry Pi Imager
# Enable SSH + set hostname/password in Imager advanced settings
```

**2. Copy project files to Pi**
```bash
# From your PC (replace PI_IP with your Pi's IP address):
scp -r nids-project/raspberry_pi pi@PI_IP:~/nids-project/
```

**3. Copy models to Pi**
```bash
scp lstm_model.tflite          pi@PI_IP:~/nids-project/raspberry_pi/models/
scp cnn_model.tflite           pi@PI_IP:~/nids-project/raspberry_pi/models/
scp autoencoder_model.tflite   pi@PI_IP:~/nids-project/raspberry_pi/models/
scp scaler.pkl                 pi@PI_IP:~/nids-project/raspberry_pi/models/
```

**4. Run setup script on Pi (SSH in first)**
```bash
ssh pi@PI_IP
cd ~/nids-project/raspberry_pi
chmod +x setup.sh
sudo ./setup.sh
```
This installs: Mosquitto, Python packages, tflite-runtime, configures GPIO.

**5. Edit config.py if needed**
```python
# Key settings to check:
SNIFF_INTERFACE  = "wlan0"     # or "eth0" if using Ethernet
ANOMALY_THRESHOLD = 0.0XXX     # paste value from Colab training output
TELEGRAM_ENABLED  = True       # set True and fill in token/chat_id
TELEGRAM_BOT_TOKEN = "..."
TELEGRAM_CHAT_ID   = "..."
```

**6. Wire up LED**
```
Pi GPIO17 (pin 11) ──── 330Ω resistor ──── LED anode (+)
Pi GND    (pin 9)  ──────────────────────── LED cathode (-)
```

### D. Flash ESP32

1. Install [Arduino IDE 2.x](https://www.arduino.cc/en/software)
2. Add ESP32 board support: File → Preferences → Additional boards URL:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Tools → Board → ESP32 Arduino → **ESP32 Dev Module**
4. Open `esp32_firmware/esp32_sniffer.ino`
5. Edit at the top of the file:
   ```cpp
   const char* ssid     = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* mqtt_server = "PI_IP_ADDRESS";  // your Pi's IP
   ```
6. Tools → Port → select your ESP32 COM port
7. Upload (Ctrl+U)
8. Open Serial Monitor (115200 baud) to verify connection

### E. Run the System

```bash
# SSH into Pi
ssh pi@PI_IP
cd ~/nids-project/raspberry_pi

# Start NIDS (requires sudo for raw packet capture)
sudo python3 main.py
```

Open dashboard in browser: **http://PI_IP:5000**

To run as a background service (auto-start on boot):
```bash
sudo systemctl enable nids
sudo systemctl start nids
```
*(Service file is created by setup.sh)*

---

## 7. Dataset: CICIDS2017

- **Source:** Canadian Institute for Cybersecurity
- **URL:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Size:** ~2.8 GB (8 CSV files, one per day of capture)
- **Features used:** 20 out of 80 available features
- **Class mapping used in this project:**

| Class ID | Name | Attack Examples |
|---|---|---|
| 0 | NORMAL | Benign traffic |
| 1 | DDoS | DDoS, DoS Hulk, DoS GoldenEye, Heartbleed |
| 2 | PortScan | Port scanning |
| 3 | BruteForce | FTP-Patator, SSH-Patator, Web XSS, SQLi |
| 4 | Bot | Botnet C&C |
| 5 | Infiltration | Infiltration attacks |

---

## 8. Deep Learning Models

### LSTM (Long Short-Term Memory)
- **Input:** Sequence of 10 consecutive flow vectors `(1, 10, 20)`
- **Purpose:** Captures temporal patterns — e.g., repeated connection attempts over time
- **Architecture:** LSTM(128) → Dropout → LSTM(64) → Dense(64) → Softmax(6)
- **Good at:** BruteForce (repeated login attempts), Bot (periodic C&C)

### CNN-1D (1D Convolutional Neural Network)
- **Input:** Single flow feature vector reshaped as `(1, 20, 1)`
- **Purpose:** Detects spatial patterns in feature space — e.g., packet flood signatures
- **Architecture:** Conv1D(64) → Conv1D(128) → GlobalMaxPool → Dense(128) → Softmax(6)
- **Good at:** DDoS (high packet rate pattern), PortScan (many short flows)

### Autoencoder (Anomaly Detector)
- **Input:** Single flow vector `(1, 20)`
- **Purpose:** Trained only on Normal traffic; high reconstruction error = anomaly
- **Architecture:** Dense(16) → Dense(8) → Dense(4) [bottleneck] → Dense(8) → Dense(16) → Dense(20)
- **Good at:** Zero-day attacks (never-seen-before traffic patterns)
- **Threshold:** Set from 95th percentile of MSE on Normal test data (printed during training)

### Ensemble Decision Logic
```
LSTM probability + CNN probability → average → supervised label
                                                      ↓
                          Autoencoder MSE > threshold?
                          YES + supervised=Normal → ZERO-DAY alert
                          YES + supervised=Attack → confirmed attack
                          NO                     → use supervised label
```

---

## 9. Dashboard

Access at **http://PI_IP:5000** from any device on the same Wi-Fi.

Features:
- Live packet/flow rate graph (Chart.js, WebSocket updates)
- Attack type distribution donut chart
- Real-time alert log table (colour-coded by severity)
- Model confidence scores per prediction
- System status: CPU, RAM, models loaded, uptime
- Anomaly score timeline

---

## 10. Alerts

### GPIO LED (Physical)
- LED on GPIO17 blinks when an attack is detected
- Blink pattern: 3 fast blinks for CRITICAL, 1 long for HIGH, 1 short for MEDIUM
- Edit `config.py`: `GPIO_LED_PIN`, `GPIO_ENABLED`

### Telegram Bot (Remote)
1. Message [@BotFather](https://t.me/botfather) on Telegram → `/newbot` → copy token
2. Get your chat ID: message [@userinfobot](https://t.me/userinfobot)
3. Edit `config.py`:
   ```python
   TELEGRAM_ENABLED   = True
   TELEGRAM_BOT_TOKEN = "1234567890:ABCDEF..."
   TELEGRAM_CHAT_ID   = "987654321"
   ```
4. Alert message example: `🚨 [CRITICAL] DDoS detected | conf=94.2% | 1,203 pkts | 192.168.1.5:80`

Alert cooldown (default 30s) prevents spam — edit `ALERT_COOLDOWN` in `config.py`.

---

## 11. Common Mistakes & Fixes

| Mistake | Symptom | Fix |
|---|---|---|
| Forgot to copy `scaler.pkl` to Pi | Near-random predictions, ~16% accuracy | Copy scaler.pkl to `raspberry_pi/models/` |
| Wrong `ANOMALY_THRESHOLD` in config.py | Too many false positives OR misses zero-days | Use value printed by `colab_train.py` Step 5 |
| SNIFF_INTERFACE wrong | "No packets captured" | Run `ip a` on Pi to find your interface name |
| ESP32 MQTT broker IP wrong | ESP32 can't connect | Set `mqtt_server` to Pi's static IP, not `localhost` |
| Running main.py without sudo | Scapy can't capture packets (permission error) | Use `sudo python3 main.py` |
| TFLite model compiled for wrong arch | ImportError on Pi | Use `tflite-runtime` wheel built for ARM64 (setup.sh handles this) |
| CICIDS2017 column names have spaces | KeyError in training | Script strips whitespace: `df.columns.str.strip()` ✓ |
| class imbalance ignored | Model always predicts NORMAL | Script applies SMOTE — check SMOTE output in Colab |

---

## 12. Results Summary

*(Fill in after running your experiments)*

| Model | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| LSTM | __ % | | | |
| CNN-1D | __ % | | | |
| Ensemble (LSTM + CNN) | __ % | | | |
| Autoencoder AUC-ROC | __ | — | — | — |
| **CatBoost baseline** (Sem 1) | __ % | | | |

**Pi inference speed** (measured on device):

| Model | Avg inference time |
|---|---|
| LSTM | __ ms |
| CNN-1D | __ ms |
| Autoencoder | __ ms |
| Total ensemble | __ ms |

---

## Authors

BE Electronics & Telecommunication Engineering  
Final Year Project — 2024-25

> Co-developed with Oz AI agent (Warp)

---

## License

This project is for academic purposes only.  
CICIDS2017 dataset © Canadian Institute for Cybersecurity — for research use.
