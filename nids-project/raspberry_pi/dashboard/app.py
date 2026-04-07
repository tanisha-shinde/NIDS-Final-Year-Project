"""
dashboard/app.py — Flask + Socket.IO Real-Time Dashboard
Device  : Raspberry Pi 4

REST endpoints:
  GET  /               → main dashboard page
  GET  /api/stats      → current stats JSON
  GET  /api/alerts     → recent alerts list
  POST /api/alerts/clear
  GET  /api/system     → CPU, RAM, disk, temperature
  GET  /api/models     → model load status

Socket.IO events:
  server → client:  stats_update, new_alert, initial_data
  client → server:  request_stats
"""

import time
import threading
import logging
from datetime import datetime
from collections import deque

import psutil
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import config

logger = logging.getLogger(__name__)

# ============================================================
#  Shared globals (set by create_app)
# ============================================================
_state     = None
_inference = None
_alerts    = None

# Socket.IO instance exported to main.py
socketio = SocketIO(
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=False,
    engineio_logger=False,
    ping_timeout=20,
    ping_interval=10,
)

# Rolling traffic history for the line chart (60 data points = 2 min)
_traffic_history   = deque(maxlen=60)
_last_pkt_count    = 0
_last_history_time = time.time()


# ============================================================
#  CREATE APP
# ============================================================
def create_app(state, inference_engine, alert_system):
    global _state, _inference, _alerts
    _state     = state
    _inference = inference_engine
    _alerts    = alert_system

    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY

    CORS(app)
    socketio.init_app(app)
    _register_routes(app)

    # Start background stats broadcaster
    threading.Thread(
        target=_broadcast_loop, name="StatsBroadcast", daemon=True
    ).start()

    logger.info("Flask dashboard app created")
    return app


# ============================================================
#  ROUTES
# ============================================================
def _register_routes(app):

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/stats")
    def api_stats():
        return jsonify(_build_stats())

    @app.route("/api/alerts")
    def api_alerts():
        with _state.lock:
            return jsonify(_state.alerts[:100])

    @app.route("/api/alerts/clear", methods=["POST"])
    def api_clear_alerts():
        with _state.lock:
            _state.alerts = []
            for k in _state.attack_counts:
                _state.attack_counts[k] = 0
        return jsonify({"status": "cleared"})

    @app.route("/api/system")
    def api_system():
        return jsonify(_system_health())

    @app.route("/api/models")
    def api_models():
        return jsonify(_inference.get_stats() if _inference else {})


# ============================================================
#  SOCKET.IO EVENTS
# ============================================================
@socketio.on("connect")
def on_connect():
    logger.debug("Dashboard client connected")
    emit("initial_data", {
        "stats":   _build_stats(),
        "alerts":  _state.alerts[:50] if _state else [],
        "history": list(_traffic_history)[-30:],
    })


@socketio.on("disconnect")
def on_disconnect():
    logger.debug("Dashboard client disconnected")


@socketio.on("request_stats")
def on_request_stats():
    emit("stats_update", _build_stats())


# ============================================================
#  BROADCAST HELPERS
# ============================================================
def broadcast_alert(alert_data):
    """Called by InferenceEngine to push a new alert to all clients."""
    try:
        socketio.emit("new_alert", alert_data)
    except Exception as e:
        logger.debug("broadcast_alert error: %s", e)


def _broadcast_loop():
    """Background thread — pushes stats to all clients every 2 seconds."""
    global _last_pkt_count, _last_history_time

    while True:
        try:
            now = time.time()
            dt  = now - _last_history_time

            if dt >= 1.0 and _state is not None:
                curr = _state.total_packets
                pps  = (curr - _last_pkt_count) / dt
                _last_pkt_count    = curr
                _last_history_time = now

                total_attacks = sum(
                    v for k, v in _state.attack_counts.items()
                    if k != "NORMAL"
                )
                _traffic_history.append({
                    "time":    datetime.now().strftime("%H:%M:%S"),
                    "pps":     round(pps, 1),
                    "attacks": total_attacks,
                })

            socketio.emit("stats_update", _build_stats())
            time.sleep(2)

        except Exception as e:
            logger.debug("Broadcast loop error: %s", e)
            time.sleep(2)


# ============================================================
#  DATA BUILDERS
# ============================================================
def _build_stats():
    if _state is None:
        return {}

    with _state.lock:
        attack_counts = dict(_state.attack_counts)
        total_attacks = sum(v for k, v in attack_counts.items() if k != "NORMAL")

    inf = _inference.get_stats() if _inference else {}

    return {
        "total_packets":   _state.total_packets,
        "total_flows":     _state.total_flows,
        "total_attacks":   total_attacks,
        "attack_counts":   attack_counts,
        "esp32_online":    _state.esp32_online,
        "model_loaded":    _state.model_loaded,
        "inference_count": inf.get("total_inferences", 0),
        "models_loaded":   inf.get("models_loaded", {}),
        "traffic_history": list(_traffic_history)[-30:],
        "timestamp":       datetime.now().isoformat(),
    }


def _system_health():
    """Return CPU/RAM/disk/temperature metrics."""
    try:
        cpu  = psutil.cpu_percent(interval=0.1)
        mem  = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        temp = None

        # Raspberry Pi thermal zone
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = int(f.read().strip()) / 1000.0
        except Exception:
            pass

        return {
            "cpu_percent":  cpu,
            "ram_percent":  mem.percent,
            "ram_used_mb":  round(mem.used  / (1024 ** 2), 1),
            "ram_total_mb": round(mem.total / (1024 ** 2), 1),
            "disk_percent": disk.percent,
            "cpu_temp":     temp,
            "timestamp":    datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}
