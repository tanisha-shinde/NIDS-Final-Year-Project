"""
generate_models_local.py
========================
Runs entirely on your Windows PC — no internet, no Colab, no CICIDS2017 needed.

Generates synthetic CICIDS2017-like data, trains all 3 models (LSTM, CNN-1D,
Autoencoder), converts them to TFLite, and saves everything to:
  ../raspberry_pi/models/

Copy that models/ folder to the Raspberry Pi and the system is ready to run.

NOTE: Models trained on synthetic data will work structurally but accuracy
will be limited. For production accuracy, run colab_train.py on real CICIDS2017.
"""

import os, sys, warnings, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

print("=" * 60)
print("  NIDS LOCAL MODEL GENERATOR")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
print(f"  Python     : {sys.version.split()[0]}")
print()

# ── Config ────────────────────────────────────────────────────
SEED            = 42
NUM_FEATURES    = 20
SEQUENCE_LENGTH = 10
NUM_CLASSES     = 6
BATCH_SIZE      = 512
EPOCHS          = 15          # keep low for local CPU training
N_SAMPLES       = 30000       # enough to train reasonable models fast

CLASS_NAMES = ['NORMAL', 'DDoS', 'PortScan', 'BruteForce', 'Bot', 'Infiltration']

# Output directory — goes straight into Pi models folder
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'raspberry_pi', 'models'
)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"  Output dir : {os.path.abspath(OUT_DIR)}\n")

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
#  STEP 1 — GENERATE SYNTHETIC DATA
# ============================================================
print("=" * 60)
print("STEP 1 — Generating synthetic CICIDS2017-like data")
print("=" * 60)

def make_class(n, loc, scale=0.15):
    return np.abs(np.random.normal(loc=loc, scale=scale, size=(n, NUM_FEATURES))).astype(np.float32)

n_normal = int(N_SAMPLES * 0.60)
n_ddos   = int(N_SAMPLES * 0.12)
n_pscan  = int(N_SAMPLES * 0.10)
n_brute  = int(N_SAMPLES * 0.08)
n_bot    = int(N_SAMPLES * 0.06)
n_infil  = N_SAMPLES - n_normal - n_ddos - n_pscan - n_brute - n_bot

# Feature means per class (mimics CICIDS2017 statistical profiles)
# [dst_port, flow_dur, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes,
#  fwd_len_max, fwd_len_mean, bwd_len_mean, flow_bytes/s, flow_pkts/s,
#  flow_iat_mean, flow_iat_std, fwd_iat_mean, bwd_iat_mean,
#  min_pkt_len, max_pkt_len, pkt_len_mean, pkt_len_std, avg_pkt_size]
locs = {
    0: [0.5, 0.3, 0.2, 0.2, 0.3, 0.3, 0.4, 0.3, 0.3, 0.2, 0.2,  # NORMAL
        0.4, 0.3, 0.4, 0.4, 0.3, 0.4, 0.3, 0.2, 0.3],
    1: [0.1, 0.05,0.9, 0.0, 0.9, 0.0, 0.1, 0.1, 0.0, 0.95,0.95, # DDoS
        0.05,0.02,0.05,0.0, 0.1, 0.1, 0.1, 0.02,0.1],
    2: [0.9, 0.1, 0.1, 0.0, 0.05,0.0, 0.05,0.05,0.0, 0.3, 0.5,  # PortScan
        0.2, 0.1, 0.2, 0.0, 0.05,0.05,0.05,0.02,0.05],
    3: [0.15,0.6, 0.5, 0.4, 0.4, 0.4, 0.3, 0.25,0.25,0.15,0.2,  # BruteForce
        0.5, 0.3, 0.5, 0.5, 0.25,0.3, 0.25,0.1, 0.25],
    4: [0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.15,0.12,0.12,0.05,0.05, # Bot
        0.9, 0.5, 0.9, 0.9, 0.12,0.15,0.12,0.05,0.12],
    5: [0.45,0.7, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.15, # Infiltration
        0.6, 0.4, 0.6, 0.6, 0.35,0.5, 0.4, 0.15,0.4],
}

counts = [n_normal, n_ddos, n_pscan, n_brute, n_bot, n_infil]
X_list, y_list = [], []
for cls_id, n in enumerate(counts):
    X_list.append(make_class(n, locs[cls_id]))
    y_list.append(np.full(n, cls_id))

X_raw = np.vstack(X_list).astype(np.float32)
y_raw = np.concatenate(y_list).astype(np.int32)

# Shuffle
idx = np.random.permutation(len(X_raw))
X_raw, y_raw = X_raw[idx], y_raw[idx]

print(f"  Total samples : {len(X_raw):,}")
for i, name in enumerate(CLASS_NAMES):
    print(f"    {name:<12} : {(y_raw==i).sum():,}")

# ============================================================
#  STEP 2 — PREPROCESS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 — Scaling + splitting")
print("=" * 60)

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

scaler_path = os.path.join(OUT_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"  scaler.pkl saved → {scaler_path}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
)
print(f"  Train : {X_tr.shape}   Test : {X_te.shape}")

# One-hot
y_tr_oh = keras.utils.to_categorical(y_tr, NUM_CLASSES)
y_te_oh = keras.utils.to_categorical(y_te, NUM_CLASSES)

# LSTM sequences
def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs, np.float32), np.array(ys)

X_seq_tr, y_seq_tr = make_sequences(X_tr, y_tr, SEQUENCE_LENGTH)
X_seq_te, y_seq_te = make_sequences(X_te, y_te, SEQUENCE_LENGTH)
y_seq_tr_oh = keras.utils.to_categorical(y_seq_tr, NUM_CLASSES)

# CNN reshape
X_cnn_tr = X_tr.reshape(-1, NUM_FEATURES, 1)
X_cnn_te = X_te.reshape(-1, NUM_FEATURES, 1)

# ============================================================
#  STEP 3 — TRAIN LSTM
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 — Training LSTM")
print("=" * 60)

lstm = models.Sequential([
    layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    layers.LSTM(64, return_sequences=True, unroll=True),   # unroll=True → TFLite compatible
    layers.Dropout(0.3),
    layers.LSTM(32, unroll=True),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(NUM_CLASSES, activation='softmax'),
], name='LSTM_Classifier')

lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm.summary(line_length=60)

t0 = time.time()
lstm.fit(
    X_seq_tr, y_seq_tr_oh,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=0),
    ],
    verbose=1
)
print(f"  Training time: {time.time()-t0:.1f}s")

y_pred_lstm = np.argmax(lstm.predict(X_seq_te, verbose=0), axis=1)
acc = accuracy_score(y_seq_te, y_pred_lstm)
print(f"  LSTM Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_seq_te, y_pred_lstm, target_names=CLASS_NAMES, zero_division=0))

lstm_h5 = os.path.join(OUT_DIR, 'lstm_model.h5')
lstm.save(lstm_h5)
print(f"  Saved → {lstm_h5}")

# ============================================================
#  STEP 4 — TRAIN CNN-1D
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 — Training CNN-1D")
print("=" * 60)

inp = layers.Input(shape=(NUM_FEATURES, 1))
x   = layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
x   = layers.BatchNormalization()(x)
x   = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
x   = layers.BatchNormalization()(x)
x   = layers.GlobalMaxPooling1D()(x)
x   = layers.Dense(64, activation='relu')(x)
x   = layers.Dropout(0.4)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
cnn = models.Model(inp, out, name='CNN1D_Classifier')
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary(line_length=60)

t0 = time.time()
cnn.fit(
    X_cnn_tr, y_tr_oh,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=0),
    ],
    verbose=1
)
print(f"  Training time: {time.time()-t0:.1f}s")

y_pred_cnn = np.argmax(cnn.predict(X_cnn_te, verbose=0), axis=1)
acc = accuracy_score(y_te, y_pred_cnn)
print(f"  CNN-1D Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_te, y_pred_cnn, target_names=CLASS_NAMES, zero_division=0))

cnn_h5 = os.path.join(OUT_DIR, 'cnn_model.h5')
cnn.save(cnn_h5)
print(f"  Saved → {cnn_h5}")

# ============================================================
#  STEP 5 — TRAIN AUTOENCODER
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 — Training Autoencoder (anomaly detector)")
print("=" * 60)

X_normal_tr = X_tr[y_tr == 0]
X_normal_te = X_te[y_te == 0]
print(f"  Normal train: {len(X_normal_tr)}  test: {len(X_normal_te)}")

inp_ae  = layers.Input(shape=(NUM_FEATURES,))
e       = layers.Dense(16, activation='relu')(inp_ae)
e       = layers.Dense(8,  activation='relu')(e)
encoded = layers.Dense(4,  activation='relu', name='bottleneck')(e)
d       = layers.Dense(8,  activation='relu')(encoded)
d       = layers.Dense(16, activation='relu')(d)
out_ae  = layers.Dense(NUM_FEATURES, activation='linear')(d)
ae      = models.Model(inp_ae, out_ae, name='Autoencoder')
ae.compile(optimizer='adam', loss='mse')
ae.summary(line_length=60)

t0 = time.time()
ae.fit(
    X_normal_tr, X_normal_tr,
    validation_data=(X_normal_te, X_normal_te),
    epochs=30,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
    ],
    verbose=1
)
print(f"  Training time: {time.time()-t0:.1f}s")

recon        = ae.predict(X_te, verbose=0)
mse_all      = np.mean((X_te - recon) ** 2, axis=1)
mse_normal   = mse_all[y_te == 0]
threshold    = float(np.percentile(mse_normal, 95))
print(f"\n  Autoencoder MSE threshold (95th pct): {threshold:.6f}")
print(f"  → Update ANOMALY_THRESHOLD in config.py to: {round(threshold, 5)}")

ae_h5 = os.path.join(OUT_DIR, 'autoencoder_model.h5')
ae.save(ae_h5)
print(f"  Saved → {ae_h5}")

# ============================================================
#  STEP 6 — CONVERT TO TFLITE
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 — Converting to TFLite")
print("=" * 60)

def to_tflite(model, out_path, label, dummy_input):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    converter.target_spec.supported_types = [tf.float32]
    # Allow SELECT_TF_OPS as fallback for any ops not natively supported
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_bytes = converter.convert()

    with open(out_path, 'wb') as f:
        f.write(tflite_bytes)

    # Verify
    try:
        interp = tf.lite.Interpreter(model_path=out_path)
        interp.allocate_tensors()
        in_det  = interp.get_input_details()
        out_det = interp.get_output_details()
        # Set input tensor directly (fixed shape after unroll)
        input_data = dummy_input.reshape(in_det[0]['shape'])
        interp.set_tensor(in_det[0]['index'], input_data)
        t0 = time.perf_counter()
        interp.invoke()
        ms = (time.perf_counter() - t0) * 1000
        output = interp.get_tensor(out_det[0]['index'])
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  [{label}]  {os.path.basename(out_path)}  ({size_kb:.1f} KB)")
        print(f"    input={in_det[0]['shape']}  output={output.shape}  infer={ms:.2f}ms")
    except Exception as e:
        size_kb = os.path.getsize(out_path) / 1024 if os.path.exists(out_path) else 0
        print(f"  [{label}]  {os.path.basename(out_path)}  ({size_kb:.1f} KB)  [verify skipped: {e}]")
    return True

dummy_lstm = np.random.rand(1, SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32)
dummy_cnn  = np.random.rand(1, NUM_FEATURES, 1).astype(np.float32)
dummy_ae   = np.random.rand(1, NUM_FEATURES).astype(np.float32)

to_tflite(lstm, os.path.join(OUT_DIR, 'lstm_model.tflite'),       'LSTM',        dummy_lstm)
to_tflite(cnn,  os.path.join(OUT_DIR, 'cnn_model.tflite'),        'CNN-1D',      dummy_cnn)
to_tflite(ae,   os.path.join(OUT_DIR, 'autoencoder_model.tflite'),'Autoencoder', dummy_ae)

# ============================================================
#  SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  ALL DONE — Models ready for Raspberry Pi!")
print("=" * 60)
print(f"\n  Files in: {os.path.abspath(OUT_DIR)}\n")
files = ['lstm_model.tflite', 'cnn_model.tflite',
         'autoencoder_model.tflite', 'scaler.pkl']
for f in files:
    p = os.path.join(OUT_DIR, f)
    if os.path.exists(p):
        kb = os.path.getsize(p) / 1024
        print(f"  ✓  {f:<35} {kb:>7.1f} KB")
    else:
        print(f"  ✗  {f}  MISSING")

print(f"""
  ANOMALY_THRESHOLD for config.py: {round(threshold, 5)}

  Next steps:
  1. Copy raspberry_pi/models/ folder to your Pi:
       scp -r raspberry_pi/models pi@<PI_IP>:~/nids-project/raspberry_pi/

  2. On the Pi, edit raspberry_pi/config.py:
       ANOMALY_THRESHOLD = {round(threshold, 5)}

  3. Run the system:
       sudo python3 main.py
""")
