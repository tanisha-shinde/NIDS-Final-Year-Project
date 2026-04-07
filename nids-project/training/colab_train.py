"""
colab_train.py — NIDS Model Training on Google Colab
======================================================
Run this ENTIRE file in Google Colab (Runtime → Run all).

What this script does:
  1. Mounts Google Drive
  2. Downloads / loads CICIDS2017 CSV files
  3. Preprocesses: cleans, maps labels, scales features
  4. Trains 3 models:
       a) LSTM        — (10, 20) sequence → 6-class classifier
       b) CNN-1D      — (20, 1)  single flow → 6-class classifier
       c) Autoencoder — (20,)    normal only  → reconstruction
  5. Evaluates each model (confusion matrix, classification report)
  6. Saves models + scaler to Drive as:
       lstm_model.h5  cnn_model.h5  autoencoder_model.h5  scaler.pkl

After training, run convert_tflite.py to get the .tflite files.

CICIDS2017 download:
  https://www.unb.ca/cic/datasets/ids-2017.html
  Put the CSV files in: /content/drive/MyDrive/NIDS/cicids2017/
"""

# ── Colab setup ──────────────────────────────────────────────
import os, sys, warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/NIDS'
    print("Google Drive mounted ✓")
except ImportError:
    # Running locally
    DRIVE = os.path.expanduser('~/NIDS')
    print(f"Local mode. Using: {DRIVE}")

os.makedirs(f'{DRIVE}/models', exist_ok=True)
os.makedirs(f'{DRIVE}/cicids2017', exist_ok=True)

# Install packages (Colab already has TF)
os.system('pip install -q imbalanced-learn scikit-learn pandas numpy matplotlib seaborn')

# ── Imports ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix, accuracy_score)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# ── Configuration ────────────────────────────────────────────
DATA_DIR   = f'{DRIVE}/cicids2017'
MODELS_DIR = f'{DRIVE}/models'

NUM_FEATURES    = 20
SEQUENCE_LENGTH = 10
NUM_CLASSES     = 6
BATCH_SIZE      = 1024
EPOCHS          = 30
SEED            = 42

# The 20 features we extract (must match feature_extractor.py on Pi)
SELECTED_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Average Packet Size',
]

# Label mapping — consolidate similar attacks
LABEL_MAP = {
    'BENIGN':                   0,
    # DDoS group
    'DDoS':                     1,
    'DoS Hulk':                 1,
    'DoS GoldenEye':            1,
    'DoS slowloris':            1,
    'DoS Slowhttptest':         1,
    'Heartbleed':               1,
    # Port Scan
    'PortScan':                 2,
    # Brute Force group
    'FTP-Patator':              3,
    'SSH-Patator':              3,
    'Web Attack \x96 Brute Force': 3,
    'Web Attack – Brute Force': 3,
    'Web Attack \x96 XSS':      3,
    'Web Attack – XSS':         3,
    'Web Attack \x96 Sql Injection': 3,
    'Web Attack – Sql Injection': 3,
    # Bot
    'Bot':                      4,
    # Infiltration
    'Infiltration':             5,
}

CLASS_NAMES = ['NORMAL','DDoS','PortScan','BruteForce','Bot','Infiltration']

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
#  STEP 1 — LOAD DATA
# ============================================================
print("\n" + "="*60)
print("STEP 1 — Loading CICIDS2017 CSV files")
print("="*60)

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files: {csv_files}")

if not csv_files:
    print("\n⚠  No CSV files found in", DATA_DIR)
    print("   Download from: https://www.unb.ca/cic/datasets/ids-2017.html")
    print("   Generating SYNTHETIC data for demo purposes...\n")

    # ── Synthetic demo data ──────────────────────────────────
    N = 50000
    np.random.seed(SEED)
    # Normal traffic: low, stable values
    normal = np.random.normal(loc=[80,1e6,10,8,5000,4000,1500,500,450,
                                    5000,20,50000,10000,45000,55000,40,1500,
                                    500,100,500], scale=0.1, size=(int(N*0.7), 20))
    normal = np.abs(normal)
    # DDoS: very high packet rates, small packets
    ddos   = np.random.normal(loc=[80,100,1000,0,60000,0,60,60,0,
                                    600000,10000,100,50,80,0,60,60,60,5,60],
                               scale=0.05, size=(int(N*0.1), 20))
    ddos   = np.abs(ddos)
    # PortScan: many short flows, various ports
    pscan  = np.random.normal(loc=[random_port:=np.random.randint(1,65535,int(N*0.05)),
                                   *(np.random.uniform(500,200000,19).tolist())],
                               scale=0.1, size=(int(N*0.05), 20)) if False else \
             np.hstack([np.random.randint(1,65535,(int(N*0.05),1)),
                        np.random.uniform(500,200000,(int(N*0.05),19))])
    # BruteForce: repeated auth attempts
    brute  = np.random.normal(loc=[22,2e6,50,50,10000,10000,200,200,200,
                                    5000,50,20000,5000,18000,22000,200,200,
                                    200,10,200], scale=0.1, size=(int(N*0.06), 20))
    brute  = np.abs(brute)
    # Bot: low-rate, periodic
    bot    = np.random.normal(loc=[4444,3e7,5,4,500,400,100,100,100,
                                    50,0.3,1e6,200000,900000,1.1e6,100,100,
                                    100,5,100], scale=0.1, size=(int(N*0.05), 20))
    bot    = np.abs(bot)
    # Infiltration
    infil  = np.random.normal(loc=[443,5e6,20,18,8000,7000,400,400,390,
                                    3200,8,625000,50000,600000,650000,350,
                                    400,390,30,395], scale=0.1, size=(int(N*0.04), 20))
    infil  = np.abs(infil)

    X_raw = np.vstack([normal, ddos, pscan, brute, bot, infil]).astype(np.float32)
    y_raw = np.array(
        [0]*len(normal) + [1]*len(ddos) + [2]*len(pscan) +
        [3]*len(brute)  + [4]*len(bot)  + [5]*len(infil)
    )
    print(f"Synthetic data shape: {X_raw.shape}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y_raw==i).sum()}")

else:
    # ── Load real CSV files ──────────────────────────────────
    dfs = []
    for f in csv_files:
        path = os.path.join(DATA_DIR, f)
        print(f"Loading {f}...")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all):,}")
    print("Label distribution:\n", df_all['Label'].value_counts())

    # ── Select features + label ──────────────────────────────
    available = [c for c in SELECTED_FEATURES if c in df_all.columns]
    missing   = [c for c in SELECTED_FEATURES if c not in df_all.columns]
    if missing:
        print(f"\n⚠  Missing columns (will use zeros): {missing}")
        for c in missing:
            df_all[c] = 0.0

    df_all[SELECTED_FEATURES] = df_all[SELECTED_FEATURES].apply(
        pd.to_numeric, errors='coerce'
    )

    # Map labels
    df_all['Label_Clean'] = df_all['Label'].str.strip()
    df_all['Label_Int'] = df_all['Label_Clean'].map(LABEL_MAP)
    df_all = df_all.dropna(subset=['Label_Int'])
    df_all['Label_Int'] = df_all['Label_Int'].astype(int)

    # Remove NaN/Inf
    df_all[SELECTED_FEATURES] = df_all[SELECTED_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    df_all = df_all.dropna(subset=SELECTED_FEATURES)

    # Clip extreme values (99th percentile per feature)
    for col in SELECTED_FEATURES:
        p99 = df_all[col].quantile(0.99)
        df_all[col] = df_all[col].clip(upper=p99)

    X_raw = df_all[SELECTED_FEATURES].values.astype(np.float32)
    y_raw = df_all['Label_Int'].values

    print(f"\nClean data shape: {X_raw.shape}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(y_raw==i).sum():,}")

# ============================================================
#  STEP 2 — PREPROCESS
# ============================================================
print("\n" + "="*60)
print("STEP 2 — Preprocessing")
print("="*60)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
print("Scaler saved ✓")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# SMOTE for class imbalance (only if minority class has enough samples)
try:
    min_samples = min(np.bincount(y_train))
    k_neighbors = min(5, min_samples - 1)
    if k_neighbors >= 1:
        sm = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_bal.shape}")
    else:
        X_train_bal, y_train_bal = X_train, y_train
        print("Skipped SMOTE (too few minority samples)")
except Exception as e:
    X_train_bal, y_train_bal = X_train, y_train
    print(f"SMOTE skipped: {e}")

# One-hot labels
y_train_oh = keras.utils.to_categorical(y_train_bal, NUM_CLASSES)
y_test_oh  = keras.utils.to_categorical(y_test,      NUM_CLASSES)

# ── Sequence data for LSTM ────────────────────────────────────
def make_sequences(X, y, seq_len):
    """Sliding window to create (N, seq_len, features) sequences."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])   # label = last element in window
    return np.array(Xs, dtype=np.float32), np.array(ys)

print("Building LSTM sequences...")
X_seq_tr, y_seq_tr = make_sequences(X_train_bal, y_train_bal, SEQUENCE_LENGTH)
X_seq_te, y_seq_te = make_sequences(X_test,       y_test,      SEQUENCE_LENGTH)
y_seq_tr_oh = keras.utils.to_categorical(y_seq_tr, NUM_CLASSES)
y_seq_te_oh = keras.utils.to_categorical(y_seq_te, NUM_CLASSES)
print(f"LSTM train: {X_seq_tr.shape}   test: {X_seq_te.shape}")

# Reshape for CNN-1D: (N, 20, 1)
X_cnn_tr = X_train_bal.reshape(-1, NUM_FEATURES, 1)
X_cnn_te = X_test.reshape(-1, NUM_FEATURES, 1)

# ============================================================
#  STEP 3 — TRAIN LSTM
# ============================================================
print("\n" + "="*60)
print("STEP 3 — Training LSTM")
print("="*60)

def build_lstm():
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name='LSTM_Classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

lstm_model = build_lstm()
lstm_model.summary()

lstm_hist = lstm_model.fit(
    X_seq_tr, y_seq_tr_oh,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        callbacks.ModelCheckpoint(f'{MODELS_DIR}/lstm_model.h5', save_best_only=True),
    ],
    verbose=1
)

# Evaluate
y_pred_lstm = np.argmax(lstm_model.predict(X_seq_te, verbose=0), axis=1)
acc_lstm    = accuracy_score(y_seq_te, y_pred_lstm)
print(f"\nLSTM Test Accuracy: {acc_lstm:.4f}")
print(classification_report(y_seq_te, y_pred_lstm, target_names=CLASS_NAMES, zero_division=0))

# ============================================================
#  STEP 4 — TRAIN CNN-1D
# ============================================================
print("\n" + "="*60)
print("STEP 4 — Training CNN-1D")
print("="*60)

def build_cnn():
    inp = layers.Input(shape=(NUM_FEATURES, 1))
    x   = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalMaxPooling1D()(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inp, out, name='CNN1D_Classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

cnn_model = build_cnn()
cnn_model.summary()

cnn_hist = cnn_model.fit(
    X_cnn_tr, y_train_oh,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        callbacks.ModelCheckpoint(f'{MODELS_DIR}/cnn_model.h5', save_best_only=True),
    ],
    verbose=1
)

y_pred_cnn = np.argmax(cnn_model.predict(X_cnn_te, verbose=0), axis=1)
acc_cnn    = accuracy_score(y_test, y_pred_cnn)
print(f"\nCNN-1D Test Accuracy: {acc_cnn:.4f}")
print(classification_report(y_test, y_pred_cnn, target_names=CLASS_NAMES, zero_division=0))

# ============================================================
#  STEP 5 — TRAIN AUTOENCODER (normal traffic only)
# ============================================================
print("\n" + "="*60)
print("STEP 5 — Training Autoencoder (anomaly detector)")
print("="*60)

# Train ONLY on normal traffic
X_normal_tr = X_train_bal[y_train_bal == 0]
X_normal_te = X_test[y_test == 0]
print(f"Normal train samples: {len(X_normal_tr)}")
print(f"Normal test  samples: {len(X_normal_te)}")

def build_autoencoder():
    inp = layers.Input(shape=(NUM_FEATURES,))
    # Encoder
    e   = layers.Dense(16, activation='relu')(inp)
    e   = layers.Dense(8,  activation='relu')(e)
    encoded = layers.Dense(4, activation='relu', name='bottleneck')(e)
    # Decoder
    d   = layers.Dense(8,  activation='relu')(encoded)
    d   = layers.Dense(16, activation='relu')(d)
    out = layers.Dense(NUM_FEATURES, activation='linear')(d)

    ae  = models.Model(inp, out, name='Autoencoder')
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return ae

ae_model = build_autoencoder()
ae_model.summary()

ae_hist = ae_model.fit(
    X_normal_tr, X_normal_tr,
    validation_data=(X_normal_te, X_normal_te),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint(f'{MODELS_DIR}/autoencoder_model.h5', save_best_only=True),
    ],
    verbose=1
)

# Find MSE threshold on normal test set (95th percentile = anomaly cutoff)
X_ae_all = X_test
y_ae_all  = y_test

recon      = ae_model.predict(X_ae_all, verbose=0)
mse_all    = np.mean((X_ae_all - recon)**2, axis=1)
mse_normal = mse_all[y_ae_all == 0]
threshold  = float(np.percentile(mse_normal, 95))
print(f"\nAutoencoder MSE threshold (95th pct): {threshold:.6f}")
print("Update ANOMALY_THRESHOLD in config.py to:", round(threshold, 5))

# Evaluate anomaly detection
y_anomaly_pred = (mse_all > threshold).astype(int)  # 1 = anomaly
y_anomaly_true = (y_ae_all != 0).astype(int)         # 1 = attack
from sklearn.metrics import roc_auc_score
try:
    auc = roc_auc_score(y_anomaly_true, mse_all)
    print(f"Autoencoder AUC-ROC: {auc:.4f}")
except:
    pass

# ============================================================
#  STEP 6 — PLOT TRAINING CURVES
# ============================================================
print("\n" + "="*60)
print("STEP 6 — Training Curves")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#0d1420')

for ax, hist, title in zip(
    axes,
    [lstm_hist, cnn_hist, ae_hist],
    ['LSTM', 'CNN-1D', 'Autoencoder']
):
    ax.set_facecolor('#0f1928')
    metric = 'accuracy' if 'accuracy' in hist.history else 'loss'
    ax.plot(hist.history[metric], color='#00ffaa', linewidth=2, label=f'Train {metric}')
    val_key = 'val_' + metric
    if val_key in hist.history:
        ax.plot(hist.history[val_key], color='#ff3b5c', linewidth=2, linestyle='--', label=f'Val {metric}')
    ax.set_title(title, color='#dff0ff', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch', color='#7a99b8')
    ax.set_ylabel(metric.capitalize(), color='#7a99b8')
    ax.tick_params(colors='#7a99b8')
    ax.legend(facecolor='#0d1420', labelcolor='#dff0ff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a546c')

plt.tight_layout()
plt.savefig(f'{MODELS_DIR}/training_curves.png', dpi=120, bbox_inches='tight',
            facecolor='#0d1420')
plt.show()
print("Training curves saved.")

# ============================================================
#  STEP 7 — CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1420')

for ax, y_true, y_pred, title in [
    (axes[0], y_seq_te, y_pred_lstm, 'LSTM'),
    (axes[1], y_test,  y_pred_cnn,  'CNN-1D'),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cmap='YlOrRd', cbar=False,
        linewidths=0.5, linecolor='#0d1420'
    )
    ax.set_facecolor('#0f1928')
    ax.set_title(f'{title} Confusion Matrix', color='#dff0ff', fontsize=12)
    ax.tick_params(colors='#7a99b8', labelsize=8)
    ax.set_xlabel('Predicted', color='#7a99b8')
    ax.set_ylabel('True', color='#7a99b8')

plt.tight_layout()
plt.savefig(f'{MODELS_DIR}/confusion_matrices.png', dpi=120, bbox_inches='tight',
            facecolor='#0d1420')
plt.show()
print("Confusion matrices saved.")

# ============================================================
#  SUMMARY
# ============================================================
print("\n" + "="*60)
print("  TRAINING COMPLETE")
print("="*60)
print(f"  LSTM   accuracy : {acc_lstm:.4f} ({acc_lstm*100:.2f}%)")
print(f"  CNN-1D accuracy : {acc_cnn:.4f}  ({acc_cnn*100:.2f}%)")
print(f"  AE MSE threshold: {threshold:.6f}")
print(f"\n  Files saved to: {MODELS_DIR}/")
print("  lstm_model.h5")
print("  cnn_model.h5")
print("  autoencoder_model.h5")
print("  scaler.pkl")
print("\n  NEXT: Run convert_tflite.py to generate .tflite files")
print("="*60)
