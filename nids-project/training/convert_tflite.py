"""
convert_tflite.py — Convert Trained Keras Models → TFLite for Raspberry Pi
============================================================================
Run this in Google Colab AFTER colab_train.py has finished.

Runtime → Run all  (or run each cell manually)

What this script does:
  1. Loads the 3 trained Keras models (.h5) from Google Drive
  2. Converts each to TFLite format in TWO flavours:
       a) Float32  — full precision, most accurate
       b) INT8     — quantized, ~4× smaller, ~2× faster on Pi
  3. Verifies each converted model by running a dummy inference
  4. Prints file sizes and estimated inference times
  5. Saves everything to /content/drive/MyDrive/NIDS/models/

Files produced:
  lstm_model.tflite          ← copy to Pi: raspberry_pi/models/lstm_model.tflite
  cnn_model.tflite           ← copy to Pi: raspberry_pi/models/cnn_model.tflite
  autoencoder_model.tflite   ← copy to Pi: raspberry_pi/models/autoencoder_model.tflite
  (+ optional *_int8.tflite variants)

Common mistake: students forget to copy scaler.pkl to the Pi models/ folder.
  The .tflite files alone are NOT enough — the scaler is needed to normalise
  live traffic features to the same scale used during training.
"""

import os, time, struct, warnings
warnings.filterwarnings('ignore')

# ── Mount Drive ───────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/NIDS'
    print("Google Drive mounted ✓")
except ImportError:
    DRIVE = os.path.expanduser('~/NIDS')
    print(f"Local mode — using {DRIVE}")

MODELS_DIR = f'{DRIVE}/models'
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────
import numpy as np
import tensorflow as tf
print(f"TensorFlow {tf.__version__}")

# ── Model input shapes (MUST match colab_train.py) ────────────
NUM_FEATURES    = 20
SEQUENCE_LENGTH = 10
NUM_CLASSES     = 6

MODELS = {
    "lstm": {
        "h5_path":     f"{MODELS_DIR}/lstm_model.h5",
        "tflite_path": f"{MODELS_DIR}/lstm_model.tflite",
        "dummy_input": np.random.rand(1, SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32),
        "description": f"LSTM  — input (1, {SEQUENCE_LENGTH}, {NUM_FEATURES})",
    },
    "cnn": {
        "h5_path":     f"{MODELS_DIR}/cnn_model.h5",
        "tflite_path": f"{MODELS_DIR}/cnn_model.tflite",
        "dummy_input": np.random.rand(1, NUM_FEATURES, 1).astype(np.float32),
        "description": f"CNN-1D — input (1, {NUM_FEATURES}, 1)",
    },
    "autoencoder": {
        "h5_path":     f"{MODELS_DIR}/autoencoder_model.h5",
        "tflite_path": f"{MODELS_DIR}/autoencoder_model.tflite",
        "dummy_input": np.random.rand(1, NUM_FEATURES).astype(np.float32),
        "description": f"Autoencoder — input (1, {NUM_FEATURES})",
    },
}

# ============================================================
#  HELPER — verify a .tflite file with a dummy inference
# ============================================================
def verify_tflite(tflite_path: str, dummy_input: np.ndarray, label: str) -> bool:
    """
    Load the .tflite file, run one inference, and check the output shape.
    Returns True if everything looks correct.
    """
    if not os.path.exists(tflite_path):
        print(f"  [FAIL] {label}: file not found")
        return False

    try:
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()

        in_det  = interp.get_input_details()
        out_det = interp.get_output_details()

        # Resize if needed (handles dynamic-batch models)
        interp.resize_input_tensor(in_det[0]["index"], dummy_input.shape)
        interp.allocate_tensors()

        interp.set_tensor(in_det[0]["index"], dummy_input)

        t0 = time.perf_counter()
        interp.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        output = interp.get_tensor(out_det[0]["index"])
        size_kb = os.path.getsize(tflite_path) / 1024

        print(f"  [OK]  {label}")
        print(f"        input  : {in_det[0]['shape']}")
        print(f"        output : {out_det[0]['shape']}  →  {output.shape}")
        print(f"        inference time  : {elapsed_ms:.2f} ms (single sample, Colab CPU)")
        print(f"        file size       : {size_kb:.1f} KB")
        return True

    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


# ============================================================
#  STEP 1 — FLOAT32 CONVERSION
# ============================================================
print("\n" + "="*60)
print("STEP 1 — Float32 TFLite Conversion")
print("="*60)
print("Float32 gives the highest accuracy.")
print("Use these .tflite files on the Raspberry Pi.\n")

float32_results = {}

for name, cfg in MODELS.items():
    print(f"Converting {name.upper()}  ({cfg['description']}) ...")

    if not os.path.exists(cfg["h5_path"]):
        print(f"  WARNING: {cfg['h5_path']} not found — skipping.\n"
              f"  Run colab_train.py first!\n")
        float32_results[name] = False
        continue

    try:
        model    = tf.keras.models.load_model(cfg["h5_path"], compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Keep float32 precision
        converter.optimizations              = []
        converter.target_spec.supported_types = [tf.float32]

        tflite_model = converter.convert()

        with open(cfg["tflite_path"], "wb") as f:
            f.write(tflite_model)

        print(f"  Saved → {cfg['tflite_path']}")
        ok = verify_tflite(cfg["tflite_path"], cfg["dummy_input"], name.upper())
        float32_results[name] = ok
        print()

    except Exception as e:
        print(f"  ERROR during conversion: {e}\n")
        float32_results[name] = False


# ============================================================
#  STEP 2 — INT8 QUANTIZED CONVERSION (optional)
# ============================================================
print("\n" + "="*60)
print("STEP 2 — INT8 Quantized TFLite Conversion (optional)")
print("="*60)
print("INT8 quantization makes models ~4× smaller and ~2× faster.")
print("Slight accuracy drop (~1-2%) — acceptable for edge deployment.\n")
print("Requires a representative dataset for calibration.\n")

# We use the scaler to generate calibration data that looks like real features
def make_representative_dataset(dummy_shape, n_samples=200):
    """
    Generator that yields calibration samples.
    Ideally use real validation data; we use random normal here as a fallback.
    """
    for _ in range(n_samples):
        sample = np.random.randn(*dummy_shape).astype(np.float32)
        yield [sample]

int8_results = {}

for name, cfg in MODELS.items():
    if not os.path.exists(cfg["h5_path"]):
        int8_results[name] = False
        continue

    int8_path = cfg["tflite_path"].replace(".tflite", "_int8.tflite")
    print(f"INT8 converting {name.upper()} ...")

    try:
        model     = tf.keras.models.load_model(cfg["h5_path"], compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Calibration dataset
        dummy_shape = cfg["dummy_input"].shape   # e.g. (1, 10, 20)
        converter.representative_dataset = lambda s=dummy_shape: make_representative_dataset(s)

        # Request int8 ops — falls back to float for unsupported ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback
        ]
        converter.inference_input_type  = tf.float32  # keep float I/O for easy use
        converter.inference_output_type = tf.float32

        tflite_int8 = converter.convert()

        with open(int8_path, "wb") as f:
            f.write(tflite_int8)

        print(f"  Saved → {int8_path}")
        ok = verify_tflite(int8_path, cfg["dummy_input"], f"{name.upper()} INT8")
        int8_results[name] = ok
        print()

    except Exception as e:
        print(f"  WARNING: INT8 conversion failed for {name}: {e}")
        print("  → Float32 model will be used instead.\n")
        int8_results[name] = False


# ============================================================
#  STEP 3 — SIZE COMPARISON TABLE
# ============================================================
print("\n" + "="*60)
print("STEP 3 — Model Size Comparison")
print("="*60)
print(f"{'Model':<20} {'Float32 (KB)':>14} {'INT8 (KB)':>12} {'Reduction':>12}")
print("-" * 60)

for name, cfg in MODELS.items():
    fp32_kb = (os.path.getsize(cfg["tflite_path"]) / 1024
               if os.path.exists(cfg["tflite_path"]) else 0)
    int8_path = cfg["tflite_path"].replace(".tflite", "_int8.tflite")
    int8_kb   = (os.path.getsize(int8_path) / 1024
                 if os.path.exists(int8_path) else 0)

    reduction = f"{100*(1 - int8_kb/fp32_kb):.0f}%" if fp32_kb > 0 and int8_kb > 0 else "n/a"
    print(f"{name.upper():<20} {fp32_kb:>12.1f} {int8_kb:>12.1f} {reduction:>12}")

print("-" * 60)


# ============================================================
#  STEP 4 — BENCHMARK Pi inference time estimate
# ============================================================
print("\n" + "="*60)
print("STEP 4 — Benchmark (Colab CPU, ~0.5× Pi speed)")
print("="*60)
print("Raspberry Pi 4 is roughly 0.5× the speed of Colab CPU.")
print("Multiply times below by ~2 for real Pi estimate.\n")

N_RUNS = 50

for name, cfg in MODELS.items():
    if not os.path.exists(cfg["tflite_path"]):
        continue

    try:
        interp = tf.lite.Interpreter(model_path=cfg["tflite_path"])
        interp.allocate_tensors()
        in_idx  = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]

        # Warm up
        interp.resize_input_tensor(in_idx, cfg["dummy_input"].shape)
        interp.allocate_tensors()

        times = []
        for _ in range(N_RUNS):
            interp.set_tensor(in_idx, cfg["dummy_input"])
            t0 = time.perf_counter()
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000)

        avg    = np.mean(times)
        p95    = np.percentile(times, 95)
        pi_est = avg * 2.0

        print(f"{name.upper():<15}  avg={avg:.2f}ms  p95={p95:.2f}ms  "
              f"Pi estimate≈{pi_est:.1f}ms")

    except Exception as e:
        print(f"{name.upper()}: benchmark error — {e}")


# ============================================================
#  STEP 5 — COPY INSTRUCTIONS
# ============================================================
print("\n" + "="*60)
print("STEP 5 — Copy Models to Raspberry Pi")
print("="*60)
print("""
Files to copy from Google Drive to your Raspberry Pi:

  SOURCE (Google Drive)                   DESTINATION (Raspberry Pi)
  /MyDrive/NIDS/models/lstm_model.tflite  → raspberry_pi/models/lstm_model.tflite
  /MyDrive/NIDS/models/cnn_model.tflite   → raspberry_pi/models/cnn_model.tflite
  /MyDrive/NIDS/models/autoencoder_model.tflite
                                          → raspberry_pi/models/autoencoder_model.tflite
  /MyDrive/NIDS/models/scaler.pkl         → raspberry_pi/models/scaler.pkl

Transfer options:
  Option A — USB drive:
    1. Download files from Google Drive to your PC
    2. Copy to USB drive
    3. Plug USB into Pi and copy to ~/nids-project/raspberry_pi/models/

  Option B — SCP over Wi-Fi (Pi and laptop on same network):
    scp lstm_model.tflite pi@<PI_IP>:~/nids-project/raspberry_pi/models/
    scp cnn_model.tflite  pi@<PI_IP>:~/nids-project/raspberry_pi/models/
    scp autoencoder_model.tflite pi@<PI_IP>:~/nids-project/raspberry_pi/models/
    scp scaler.pkl        pi@<PI_IP>:~/nids-project/raspberry_pi/models/

  Option C — rclone (Colab → Drive → Pi):
    Install rclone on Pi, configure Google Drive remote, then:
    rclone copy "gdrive:NIDS/models" ~/nids-project/raspberry_pi/models/

IMPORTANT: Do NOT forget scaler.pkl!
  Without it, the Pi will run raw (unscaled) features through models trained
  on scaled data → near-random predictions.

After copying, verify on the Pi:
  ls -lh ~/nids-project/raspberry_pi/models/
  # Should see: lstm_model.tflite  cnn_model.tflite  autoencoder_model.tflite  scaler.pkl
""")

# Also update config.py ANOMALY_THRESHOLD note
try:
    import joblib
    scaler_path = f"{MODELS_DIR}/scaler.pkl"
    if os.path.exists(scaler_path):
        print("scaler.pkl found ✓")
    else:
        print("WARNING: scaler.pkl not found — make sure colab_train.py ran successfully.")
except Exception:
    pass

print("\n✅ TFLite conversion complete!")
print(f"   Float32 models saved to: {MODELS_DIR}/")
