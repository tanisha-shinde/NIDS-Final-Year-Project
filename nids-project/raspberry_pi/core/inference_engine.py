"""
inference_engine.py — Three-Model TFLite Ensemble
Device  : Raspberry Pi 4

Ensemble strategy:
  1. LSTM     — detects temporal / sequential attack patterns
  2. CNN-1D   — detects spatial packet-flood patterns (DDoS)
  3. Autoencoder — reconstruction error flags zero-day anomalies

Decision logic:
  • Average LSTM + CNN probabilities  → supervised label + confidence
  • If autoencoder MSE > threshold AND supervised says Normal → ZERO-DAY
  • If autoencoder MSE > threshold AND supervised says Attack → confirmed

Common mistake: forgetting to normalise features with the SAME scaler
used during training.  Models trained on scaled data give nonsense
on raw features.
"""

import os
import time
import logging
import threading
import numpy as np
from collections import deque
from datetime import datetime

import config

logger = logging.getLogger(__name__)


# ============================================================
#  TFLite MODEL WRAPPER
# ============================================================
class TFLiteModel:
    """
    Loads a .tflite file and provides a predict() method.
    Falls back to *demo mode* (random predictions) if the file
    is missing — useful for development on a non-Pi machine.
    """

    def __init__(self, path, name):
        self.name      = name
        self.loaded    = False
        self._interp   = None
        self._in_idx   = None
        self._out_idx  = None
        self._load(path)

    def _load(self, path):
        if not os.path.exists(path):
            logger.warning(
                "%s: model file not found (%s) → DEMO MODE", self.name, path
            )
            return

        try:
            try:
                import tflite_runtime.interpreter as tflite
                self._interp = tflite.Interpreter(model_path=path)
            except ImportError:
                import tensorflow as tf
                self._interp = tf.lite.Interpreter(model_path=path)

            self._interp.allocate_tensors()
            in_det  = self._interp.get_input_details()
            out_det = self._interp.get_output_details()
            self._in_idx  = in_det[0]["index"]
            self._out_idx = out_det[0]["index"]
            self.loaded   = True

            logger.info(
                "%s loaded | input=%s output=%s",
                self.name,
                in_det[0]["shape"],
                out_det[0]["shape"],
            )
        except Exception as e:
            logger.error("%s: failed to load — %s", self.name, e)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x    : float32 numpy array already shaped for this model
        Returns float32 1-D array (model output).
        """
        if not self.loaded:
            # Demo mode: uniform noise so the UI still shows something
            n = len(config.ATTACK_CLASSES)
            return np.random.dirichlet(np.ones(n)).astype(np.float32)

        try:
            self._interp.set_tensor(self._in_idx, x.astype(np.float32))
            self._interp.invoke()
            return self._interp.get_tensor(self._out_idx)[0]
        except Exception as e:
            logger.error("%s inference error: %s", self.name, e)
            return np.zeros(len(config.ATTACK_CLASSES), dtype=np.float32)


# ============================================================
#  INFERENCE ENGINE
# ============================================================
class InferenceEngine:

    def __init__(self, state, alert_system):
        self.state  = state
        self.alerts = alert_system
        self._queue = deque(maxlen=1000)
        self._lock  = threading.Lock()

        # Load scaler
        self._scaler = self._load_scaler()

        # Load models
        self.lstm_model = TFLiteModel(config.LSTM_MODEL_PATH,        "LSTM")
        self.cnn_model  = TFLiteModel(config.CNN_MODEL_PATH,         "CNN-1D")
        self.ae_model   = TFLiteModel(config.AUTOENCODER_MODEL_PATH, "Autoencoder")

        # Counters
        self.total_inferences = 0
        self.total_attacks    = 0
        self.total_anomalies  = 0

        with self.state.lock:
            self.state.model_loaded = True

        logger.info("InferenceEngine ready — 3-model ensemble")

    # ----------------------------------------------------------------
    def _load_scaler(self):
        try:
            import joblib
            if os.path.exists(config.SCALER_PATH):
                s = joblib.load(config.SCALER_PATH)
                logger.info("Feature scaler loaded from %s", config.SCALER_PATH)
                return s
            logger.warning("Scaler not found — features used raw (reduced accuracy)")
        except Exception as e:
            logger.error("Scaler load error: %s", e)
        return None

    def _scale(self, features: np.ndarray) -> np.ndarray:
        """Apply scaler; return raw if unavailable."""
        if self._scaler is None:
            return features
        try:
            return self._scaler.transform(features.reshape(1, -1))[0].astype(np.float32)
        except Exception:
            return features

    def _scale_seq(self, seq: np.ndarray) -> np.ndarray:
        """Scale a 2-D sequence array (seq_len × features)."""
        if self._scaler is None:
            return seq
        try:
            return self._scaler.transform(seq).astype(np.float32)
        except Exception:
            return seq

    # ----------------------------------------------------------------
    def process_flow(self, flow_data):
        """Called by FeatureExtractor — enqueues a flow for inference."""
        self._queue.append(flow_data)

    # ----------------------------------------------------------------
    def run_loop(self):
        """Main inference loop — runs every INFERENCE_INTERVAL seconds."""
        logger.info("Inference loop started (interval=%.1fs)", config.INFERENCE_INTERVAL)
        while self.state.running:
            try:
                processed = 0
                while self._queue:
                    fd     = self._queue.popleft()
                    result = self._run_ensemble(fd)
                    if result:
                        self._handle(result, fd)
                        processed += 1

                if processed:
                    logger.debug("Processed %d flows", processed)

                time.sleep(config.INFERENCE_INTERVAL)
            except Exception as e:
                logger.error("Inference loop error: %s", e)
                time.sleep(1)

    # ----------------------------------------------------------------
    def _run_ensemble(self, fd) -> dict:
        """
        Run all 3 models on a single flow and combine their outputs.

        Returns dict with label, confidence, anomaly info, and raw scores.
        """
        try:
            raw_feat = fd["features"]                  # shape (20,)
            sequence = fd["sequence"]                  # list of arrays

            # --- Normalise ---
            norm_feat = self._scale(raw_feat)

            # ---- CNN-1D: input shape (1, 20, 1) ----
            cnn_in   = norm_feat.reshape(1, config.NUM_FEATURES, 1)
            cnn_prob = self.cnn_model.predict(cnn_in)

            # ---- LSTM: input shape (1, SEQ_LEN, 20) ----
            seq_arr = np.array(sequence, dtype=np.float32)
            # Pad if sequence shorter than needed
            if len(seq_arr) < config.SEQUENCE_LENGTH:
                pad = np.zeros(
                    (config.SEQUENCE_LENGTH - len(seq_arr), config.NUM_FEATURES),
                    dtype=np.float32,
                )
                seq_arr = np.vstack([pad, seq_arr])
            seq_arr    = self._scale_seq(seq_arr)
            lstm_in    = seq_arr.reshape(1, config.SEQUENCE_LENGTH, config.NUM_FEATURES)
            lstm_prob  = self.lstm_model.predict(lstm_in)

            # ---- Autoencoder: input shape (1, 20) ----
            ae_in  = norm_feat.reshape(1, config.NUM_FEATURES)
            ae_out = self.ae_model.predict(ae_in)
            # Reconstruction error (MSE)
            if ae_out.shape == norm_feat.shape:
                ae_mse = float(np.mean((norm_feat - ae_out) ** 2))
            else:
                ae_mse = 0.0
            is_anomaly = ae_mse > config.ANOMALY_THRESHOLD

            # ---- Ensemble: average LSTM + CNN ----
            n = len(config.ATTACK_CLASSES)
            if len(lstm_prob) == n and len(cnn_prob) == n:
                ens = 0.5 * lstm_prob + 0.5 * cnn_prob
            elif len(lstm_prob) == n:
                ens = lstm_prob
            elif len(cnn_prob) == n:
                ens = cnn_prob
            else:
                ens = np.ones(n, dtype=np.float32) / n

            pred_idx   = int(np.argmax(ens))
            confidence = float(ens[pred_idx])
            label      = config.ATTACK_CLASSES.get(pred_idx, "UNKNOWN")
            is_attack  = label != "NORMAL" and confidence >= config.ATTACK_THRESHOLD

            # Override: autoencoder anomaly on otherwise-Normal traffic → ZERO-DAY
            if is_anomaly and not is_attack:
                label      = "ZERO-DAY"
                is_attack  = True
                confidence = min(ae_mse * 10.0, 1.0)

            self.total_inferences += 1
            return {
                "label":         label,
                "confidence":    round(confidence, 4),
                "is_attack":     is_attack,
                "is_anomaly":    is_anomaly,
                "anomaly_score": round(ae_mse, 6),
                "lstm_probs":    lstm_prob.tolist(),
                "cnn_probs":     cnn_prob.tolist(),
                "ensemble":      ens.tolist(),
                "pred_idx":      pred_idx,
                "timestamp":     datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Ensemble error: %s", e)
            return None

    # ----------------------------------------------------------------
    def _handle(self, result, fd):
        """Update state, build alert record, notify alert system."""
        label      = result["label"]
        is_attack  = result["is_attack"]
        confidence = result["confidence"]

        # Update attack counter
        with self.state.lock:
            key = label if label in self.state.attack_counts else "ZERO-DAY"
            self.state.attack_counts[key] = self.state.attack_counts.get(key, 0) + 1

        alert = {
            "id":            self.total_inferences,
            "timestamp":     result["timestamp"],
            "label":         label,
            "confidence":    confidence,
            "is_attack":     is_attack,
            "is_anomaly":    result["is_anomaly"],
            "anomaly_score": result["anomaly_score"],
            "flow_key":      fd.get("flow_key", ""),
            "pkt_count":     fd.get("pkt_count", 0),
            "duration":      round(fd.get("duration", 0.0), 3),
            "flags":         fd.get("flags", {}),
            "severity":      config.ATTACK_SEVERITY.get(label, "medium"),
            "model_scores": {
                "lstm":     result["lstm_probs"],
                "cnn":      result["cnn_probs"],
                "ensemble": result["ensemble"],
            },
        }

        # Store in shared state
        with self.state.lock:
            self.state.alerts.insert(0, alert)
            if len(self.state.alerts) > config.MAX_LOG_ENTRIES:
                self.state.alerts = self.state.alerts[:config.MAX_LOG_ENTRIES]

        # Trigger physical + notification alerts
        if is_attack:
            self.total_attacks += 1
            self.alerts.trigger_alert(alert)
            logger.warning(
                "ATTACK: %-12s  conf=%.1f%%  pkts=%-4d  flow=%s",
                label,
                confidence * 100,
                fd.get("pkt_count", 0),
                fd.get("flow_key", "?")[:40],
            )
        else:
            logger.info(
                "Normal  %-12s  conf=%.1f%%  pkts=%-4d",
                label,
                confidence * 100,
                fd.get("pkt_count", 0),
            )

        # Push to dashboard via SocketIO
        try:
            from dashboard.app import broadcast_alert
            broadcast_alert(alert)
        except Exception:
            pass

    # ----------------------------------------------------------------
    def get_stats(self):
        return {
            "total_inferences": self.total_inferences,
            "total_attacks":    self.total_attacks,
            "total_anomalies":  self.total_anomalies,
            "models_loaded": {
                "lstm":        self.lstm_model.loaded,
                "cnn":         self.cnn_model.loaded,
                "autoencoder": self.ae_model.loaded,
            },
        }
