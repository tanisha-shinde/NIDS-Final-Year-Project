/*
 * ============================================================
 *  NIDS — ESP32 Wi-Fi Packet Sniffer v1.0
 *  Device  : ESP32 DevKit v1
 *  Purpose : Capture Wi-Fi packets in promiscuous mode and
 *            publish metadata to Raspberry Pi via MQTT.
 *
 *  Arduino Library Manager — install these before compiling:
 *    • PubSubClient  (Nick O'Leary)     v2.8+
 *    • ArduinoJson   (Benoit Blanchon)  v6.x
 *
 *  Wiring: No extra wiring needed — just USB to PC for flashing.
 * ============================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ============================================================
//  USER CONFIGURATION  ← EDIT THESE
// ============================================================
#define WIFI_SSID        "YOUR_WIFI_SSID"
#define WIFI_PASSWORD    "YOUR_WIFI_PASSWORD"
#define MQTT_BROKER_IP   "192.168.1.100"   // ← Raspberry Pi IP
#define MQTT_PORT        1883
#define DEVICE_ID        "ESP32-NIDS-01"
#define SNIFF_CHANNEL    1                  // Wi-Fi ch to sniff (1–13)
#define PUBLISH_MS       200                // Publish batch every N ms
#define HEARTBEAT_MS     10000             // Heartbeat every N ms

// MQTT topics (must match config.py on Pi)
#define TOPIC_PACKETS    "nids/packets"
#define TOPIC_STATUS     "nids/status"
#define TOPIC_HEARTBEAT  "nids/heartbeat"

// ============================================================
//  IEEE 802.11 FRAME HEADER
// ============================================================
typedef struct {
  unsigned frame_ctrl : 16;
  unsigned duration   : 16;
  uint8_t  addr1[6];   // Destination MAC
  uint8_t  addr2[6];   // Source MAC
  uint8_t  addr3[6];   // BSSID
  unsigned seq_ctrl   : 16;
} __attribute__((packed)) ieee80211_hdr_t;

typedef struct {
  ieee80211_hdr_t hdr;
  uint8_t payload[0];
} __attribute__((packed)) ieee80211_pkt_t;

// ============================================================
//  PACKET BATCH BUFFER
// ============================================================
#define BATCH_SIZE 10

struct PktRecord {
  char    src[18];
  char    dst[18];
  uint16_t len;
  int8_t   rssi;
  uint8_t  ch;
  uint8_t  type;   // 0=DATA 1=MGMT 2=CTRL
  uint32_t ts;
};

volatile PktRecord batch[BATCH_SIZE];
volatile int       batchIdx  = 0;
portMUX_TYPE       batchMux  = portMUX_INITIALIZER_UNLOCKED;

// Global counters
volatile uint32_t totalPkts   = 0;
uint32_t          lastPublish = 0;
uint32_t          lastHB      = 0;

WiFiClient   espClient;
PubSubClient mqtt(espClient);

// ============================================================
//  PROMISCUOUS CALLBACK  (runs in IRAM — keep it fast)
// ============================================================
void IRAM_ATTR snifferCB(void *buf, wifi_promiscuous_pkt_type_t type) {
  if (type == WIFI_PKT_MISC) return;

  const wifi_promiscuous_pkt_t *raw = (wifi_promiscuous_pkt_t *)buf;
  const ieee80211_pkt_t        *frm = (ieee80211_pkt_t *)raw->payload;
  const ieee80211_hdr_t        *hdr = &frm->hdr;

  portENTER_CRITICAL_ISR(&batchMux);
  if (batchIdx < BATCH_SIZE) {
    PktRecord &r = (PktRecord &)batch[batchIdx];
    snprintf(r.src, 18, "%02X:%02X:%02X:%02X:%02X:%02X",
      hdr->addr2[0], hdr->addr2[1], hdr->addr2[2],
      hdr->addr2[3], hdr->addr2[4], hdr->addr2[5]);
    snprintf(r.dst, 18, "%02X:%02X:%02X:%02X:%02X:%02X",
      hdr->addr1[0], hdr->addr1[1], hdr->addr1[2],
      hdr->addr1[3], hdr->addr1[4], hdr->addr1[5]);
    r.len  = raw->rx_ctrl.sig_len;
    r.rssi = raw->rx_ctrl.rssi;
    r.ch   = raw->rx_ctrl.channel;
    r.type = (uint8_t)type;
    r.ts   = (uint32_t)(esp_timer_get_time() / 1000ULL);
    batchIdx++;
    totalPkts++;
  }
  portEXIT_CRITICAL_ISR(&batchMux);
}

// ============================================================
//  HELPERS
// ============================================================
void setupWiFi() {
  Serial.printf("\n[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500); Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected! IP: %s\n",
      WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] FAILED — restarting");
    ESP.restart();
  }
}

void mqttConnect() {
  for (int i = 0; i < 5 && !mqtt.connected(); i++) {
    Serial.printf("[MQTT] Connecting to %s:%d ...\n", MQTT_BROKER_IP, MQTT_PORT);
    String cid = String(DEVICE_ID) + "-" + String(random(0xFFFF), HEX);
    if (mqtt.connect(cid.c_str())) {
      Serial.println("[MQTT] Connected!");
      // Announce online
      StaticJsonDocument<128> s;
      s["device"] = DEVICE_ID;
      s["status"] = "online";
      s["ip"]     = WiFi.localIP().toString();
      char buf[128]; serializeJson(s, buf);
      mqtt.publish(TOPIC_STATUS, buf, true);
    } else {
      Serial.printf("[MQTT] rc=%d  retrying...\n", mqtt.state());
      delay(3000);
    }
  }
}

void startSniffer() {
  wifi_promiscuous_filter_t f;
  f.filter_mask = WIFI_PROMIS_FILTER_MASK_DATA | WIFI_PROMIS_FILTER_MASK_MGMT;
  esp_wifi_set_promiscuous(false);
  esp_wifi_set_promiscuous_filter(&f);
  esp_wifi_set_promiscuous_rx_cb(&snifferCB);
  esp_wifi_set_channel(SNIFF_CHANNEL, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(true);
  Serial.printf("[SNIFF] Promiscuous ON — channel %d\n", SNIFF_CHANNEL);
}

// ============================================================
//  PUBLISH BATCH
// ============================================================
void publishBatch(int count) {
  // Snapshot batch under lock
  PktRecord snap[BATCH_SIZE];
  portENTER_CRITICAL(&batchMux);
  for (int i = 0; i < count; i++) snap[i] = (PktRecord &)batch[i];
  portEXIT_CRITICAL(&batchMux);

  StaticJsonDocument<2048> doc;
  doc["device"]  = DEVICE_ID;
  doc["ts"]      = millis();
  doc["total"]   = totalPkts;
  doc["bsz"]     = count;

  JsonArray pkts = doc.createNestedArray("packets");
  for (int i = 0; i < count; i++) {
    JsonObject p = pkts.createNestedObject();
    p["s"] = snap[i].src;
    p["d"] = snap[i].dst;
    p["l"] = snap[i].len;
    p["r"] = snap[i].rssi;
    p["c"] = snap[i].ch;
    p["t"] = snap[i].ts;
  }

  char buf[2048];
  size_t n = serializeJson(doc, buf, sizeof(buf));
  if (!mqtt.publish(TOPIC_PACKETS, (uint8_t *)buf, n, false)) {
    Serial.println("[MQTT] Publish failed — increase buffer?");
  }
}

// ============================================================
//  HEARTBEAT
// ============================================================
void sendHeartbeat() {
  StaticJsonDocument<256> doc;
  doc["device"]    = DEVICE_ID;
  doc["uptime_ms"] = millis();
  doc["pkt_total"] = totalPkts;
  doc["wifi_rssi"] = WiFi.RSSI();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["sniffing"]  = true;
  char buf[256]; serializeJson(doc, buf);
  mqtt.publish(TOPIC_HEARTBEAT, buf);
  Serial.printf("[♥] up=%lus pkts=%u heap=%u\n",
    millis() / 1000, totalPkts, ESP.getFreeHeap());
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("╔══════════════════════════════════════╗");
  Serial.println("║   NIDS  ESP32 Packet Sniffer v1.0   ║");
  Serial.println("╚══════════════════════════════════════╝");

  setupWiFi();

  mqtt.setServer(MQTT_BROKER_IP, MQTT_PORT);
  mqtt.setBufferSize(4096);
  mqttConnect();

  startSniffer();

  Serial.println("[READY] Sniffing — publishing to Pi via MQTT");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  if (!mqtt.connected()) mqttConnect();
  mqtt.loop();

  uint32_t now = millis();

  // Publish batch
  if (now - lastPublish >= PUBLISH_MS) {
    int count = 0;
    portENTER_CRITICAL(&batchMux);
    count    = batchIdx;
    batchIdx = 0;
    portEXIT_CRITICAL(&batchMux);
    if (count > 0) publishBatch(count);
    lastPublish = now;
  }

  // Heartbeat
  if (now - lastHB >= HEARTBEAT_MS) {
    sendHeartbeat();
    lastHB = now;
  }

  delay(5);
}
