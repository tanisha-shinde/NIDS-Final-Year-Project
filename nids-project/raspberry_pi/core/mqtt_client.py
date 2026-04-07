"""
mqtt_client.py — MQTT Subscriber for ESP32 Packet Data
Device  : Raspberry Pi 4

Subscribes to the Mosquitto broker (running on this Pi) and
receives batched packet records from the ESP32 sniffer.
"""

import json
import time
import logging
import threading

import paho.mqtt.client as mqtt

import config

logger = logging.getLogger(__name__)


class MQTTClient:

    def __init__(self, state, feature_extractor):
        self.state = state
        self.fe    = feature_extractor

        self._client = mqtt.Client(client_id=config.MQTT_CLIENT_ID)
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        self.connected  = False
        self.msg_count  = 0

    # ----------------------------------------------------------------
    def start(self):
        logger.info(
            "MQTT client connecting to %s:%d",
            config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT,
        )
        try:
            self._client.connect(
                config.MQTT_BROKER_HOST,
                config.MQTT_BROKER_PORT,
                keepalive=60,
            )
            self._client.loop_forever()
        except ConnectionRefusedError:
            logger.error(
                "MQTT connection refused — is Mosquitto running?\n"
                "  sudo systemctl start mosquitto"
            )
        except Exception as e:
            logger.error("MQTT error: %s", e)

    def stop(self):
        try:
            self._client.disconnect()
        except Exception:
            pass

    # ----------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info("MQTT connected to broker")
            client.subscribe(config.MQTT_TOPIC_PACKETS)
            client.subscribe(config.MQTT_TOPIC_STATUS)
            client.subscribe(config.MQTT_TOPIC_HEARTBEAT)
        else:
            logger.error("MQTT connect failed (rc=%d)", rc)

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            logger.warning("MQTT unexpectedly disconnected (rc=%d)", rc)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.msg_count += 1

            if msg.topic == config.MQTT_TOPIC_PACKETS:
                self._handle_packets(payload)
            elif msg.topic == config.MQTT_TOPIC_STATUS:
                self._handle_status(payload)
            elif msg.topic == config.MQTT_TOPIC_HEARTBEAT:
                with self.state.lock:
                    self.state.esp32_online = True

        except json.JSONDecodeError:
            logger.debug("Invalid JSON on MQTT topic %s", msg.topic)
        except Exception as e:
            logger.error("MQTT message handler error: %s", e)

    # ----------------------------------------------------------------
    def _handle_packets(self, payload):
        """Process a batch of packets from the ESP32."""
        packets = payload.get("packets", [])
        now     = time.time()

        for i, p in enumerate(packets):
            info = {
                "src_mac":   p.get("s", ""),
                "dst_mac":   p.get("d", ""),
                "pkt_len":   p.get("l", 0),
                "rssi":      p.get("r", -70),
                "channel":   p.get("c", 1),
                "timestamp": now + i * 0.001,
                "source":    "esp32",
                # ESP32 sees 802.11 frames — no IP info available
                "src_ip":    "0.0.0.0",
                "dst_ip":    "0.0.0.0",
                "src_port":  0,
                "dst_port":  0,
                "protocol":  0,
                "flags":     0,
            }

            with self.state.lock:
                self.state.total_packets += 1

            # Store Wi-Fi-level stats separately in FeatureExtractor
            self.fe.add_packet_esp32(info)

    def _handle_status(self, payload):
        status = payload.get("status", "")
        device = payload.get("device", "?")
        logger.info("ESP32 [%s] status: %s", device, status)
        with self.state.lock:
            self.state.esp32_online = (status == "online")

    # ----------------------------------------------------------------
    def publish_alert(self, alert_data):
        """Optional: publish detected alert back to MQTT bus."""
        if self.connected:
            try:
                self._client.publish(
                    config.MQTT_TOPIC_ALERTS,
                    json.dumps(alert_data),
                )
            except Exception:
                pass
