"""
alert_system.py — Physical & Digital Alert Handler
Device  : Raspberry Pi 4

Handles:
  • GPIO LED blink patterns  (physical on-device alert)
  • Telegram bot messages    (remote mobile notification)

LED wiring:
  GPIO17 (BCM) → 330Ω resistor → LED(+) → LED(-) → GND
  (GPIO17 = Physical pin 11 on the 40-pin header)

Common mistake: confusing BCM pin numbers with physical pin numbers.
This code uses BCM mode.  Always check the Pi pinout diagram.
"""

import time
import logging
import threading
import requests
from collections import defaultdict

import config

logger = logging.getLogger(__name__)

# ---- Graceful GPIO import (fails on non-Pi machines) ----
GPIO_OK = False
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(config.GPIO_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO_OK = True
    logger.info("GPIO ready — LED on BCM pin %d", config.GPIO_LED_PIN)
except Exception as e:
    logger.warning("GPIO not available (%s) — LED alerts disabled", e)


# ============================================================
class AlertSystem:
    """
    Non-blocking alert dispatcher.
    Incoming alerts are queued; a dedicated thread processes them
    so we never block the inference pipeline.

    LED blink patterns per severity:
      info     → 1 slow blink
      medium   → 2 medium blinks
      high     → 3 fast blinks
      critical → solid ON for 3 seconds
    """

    def __init__(self, state):
        self.state       = state
        self._queue      = []
        self._lock       = threading.Lock()
        self._cooldowns  = defaultdict(float)   # label → last-alert timestamp
        self._running    = True

    # ----------------------------------------------------------------
    def start(self):
        logger.info("Alert system running")
        while self._running:
            alert = None
            with self._lock:
                if self._queue:
                    alert = self._queue.pop(0)
            if alert:
                self._dispatch(alert)
            time.sleep(0.05)

    def stop(self):
        self._running = False
        self._led_off()
        if GPIO_OK:
            try:
                GPIO.cleanup()
            except Exception:
                pass

    # ----------------------------------------------------------------
    def trigger_alert(self, alert_data):
        """Enqueue an alert (non-blocking, called from inference thread)."""
        label = alert_data.get("label", "UNKNOWN")
        now   = time.time()

        # Cooldown check — don't spam the same alert type
        if now - self._cooldowns[label] < config.ALERT_COOLDOWN:
            logger.debug("Alert cooldown active for %s", label)
            return

        self._cooldowns[label] = now
        with self._lock:
            self._queue.append(alert_data)

    # ----------------------------------------------------------------
    def _dispatch(self, alert):
        severity = alert.get("severity", "medium")
        label    = alert.get("label",    "UNKNOWN")
        logger.warning("[ALERT] %s — severity: %s", label, severity.upper())

        if GPIO_OK and config.GPIO_ENABLED:
            self._led_blink(severity)

        if config.TELEGRAM_ENABLED:
            self._send_telegram(alert)

    # ----------------------------------------------------------------
    def _led_blink(self, severity):
        """
        Execute LED blink pattern.
        Patterns are (on_seconds, off_seconds) pairs.
        """
        patterns = {
            "info":     [(0.5, 0.4)],
            "medium":   [(0.3, 0.15), (0.3, 0.15)],
            "high":     [(0.1, 0.08)] * 3,
            "critical": [(3.0, 0.0)],
        }
        sequence = patterns.get(severity, patterns["medium"])
        try:
            for on_t, off_t in sequence:
                GPIO.output(config.GPIO_LED_PIN, GPIO.HIGH)
                time.sleep(on_t)
                GPIO.output(config.GPIO_LED_PIN, GPIO.LOW)
                if off_t:
                    time.sleep(off_t)
        except Exception as e:
            logger.error("LED blink error: %s", e)

    def _led_off(self):
        if GPIO_OK:
            try:
                GPIO.output(config.GPIO_LED_PIN, GPIO.LOW)
            except Exception:
                pass

    # ----------------------------------------------------------------
    def _send_telegram(self, alert):
        """Send a Telegram message notification."""
        try:
            label    = alert.get("label",        "UNKNOWN")
            conf     = alert.get("confidence",   0.0)
            severity = alert.get("severity",     "medium")
            flow_key = alert.get("flow_key",     "")
            pkts     = alert.get("pkt_count",    0)
            ts       = alert.get("timestamp",    "")
            anomaly  = alert.get("anomaly_score",0.0)

            emoji_map = {
                "info": "ℹ️", "medium": "⚠️", "high": "🔴", "critical": "🚨"
            }
            emoji = emoji_map.get(severity, "⚠️")

            text = (
                f"{emoji} *NIDS ALERT — {severity.upper()}*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 Type       : `{label}`\n"
                f"📊 Confidence : `{conf:.1%}`\n"
                f"🔬 Anomaly    : `{anomaly:.4f}`\n"
                f"📦 Packets    : `{pkts}`\n"
                f"🔑 Flow       : `{str(flow_key)[:40]}`\n"
                f"⏰ Time       : `{ts}`\n"
                f"📍 Sensor     : Raspberry Pi 4 NIDS\n"
                f"━━━━━━━━━━━━━━━━━━━━"
            )

            url = (
                f"https://api.telegram.org/bot"
                f"{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            )
            resp = requests.post(
                url,
                json={
                    "chat_id":    config.TELEGRAM_CHAT_ID,
                    "text":       text,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Telegram alert sent for %s", label)
            else:
                logger.error(
                    "Telegram error %d: %s", resp.status_code, resp.text[:100]
                )

        except requests.exceptions.ConnectionError:
            logger.warning("No internet — Telegram alert skipped (offline mode)")
        except Exception as e:
            logger.error("Telegram send error: %s", e)
