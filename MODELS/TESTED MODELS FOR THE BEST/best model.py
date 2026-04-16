"""
=============================================================================
  CSI Fall Detection - Best Model Search & Training Pipeline
  Author: Generated for Joseph M Benny
  Description:
    - Parses raw ESP32 CSI CSV files (fall / no fall)
    - Applies EMA denoising + Z-score normalisation
    - Builds sliding windows of 256 packets x 52 subcarriers
    - Trains 6 model architectures and picks the best one
    - Saves the best model + scaler for live inference
=============================================================================
"""

import os
import csv
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─── Suppress TF noise ───────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_THREAD_MODE"]   = "gpu_private"   # reduces GPU stall
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ─── GPU Setup ───────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # no OOM crashes
        print(f"  ✓ GPU detected: {[g.name for g in gpus]}")
        print(f"  ✓ Memory growth enabled — GPU will be used for all training")
    except RuntimeError as e:
        print(f"  [WARN] GPU config error: {e}")
else:
    print("  ⚠ No GPU detected — running on CPU")
    print("    Install CUDA + cuDNN and tensorflow-gpu if you want GPU training")

# ─── Mixed precision (float16 on GPU = 2-3x faster, same accuracy) ───────────
try:
    from tensorflow.keras import mixed_precision
    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
        print("  ✓ Mixed precision (float16) enabled — ~2x speedup on GPU")
    else:
        mixed_precision.set_global_policy("float32")
except Exception as e:
    print(f"  [WARN] Mixed precision not available: {e}")

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
#  USER CONFIG  ← change these paths if needed
# =============================================================================
DATA_ROOT   = r"C:\Users\josep\OneDrive\Desktop\Project data\RAW"
OUTPUT_DIR  = r"C:\Users\josep\OneDrive\Desktop\Project data\MODELS"

FALL_DIR    = os.path.join(DATA_ROOT, "fall")
NOFALL_DIR  = os.path.join(DATA_ROOT, "no fall")

# Signal parameters
# Real data format (confirmed from sample file):
#   Row: timestamp,CSI_DATA,STATION,MAC,...,[header(4 nums) imag0 real0 imag1 real1 ...]
#   62 numbers in brackets; skip first 4 header bytes
#   Active subcarrier indices (after header skip): 0-25 and 27-52 → 52 active
#   Zeros at positions 26, 57-61 (pilot tones / guard bands)
N_SUBCARRIERS = 52          # 52 active subcarriers after removing nulls/pilots
ACTIVE_IDX    = list(range(0, 26)) + list(range(27, 53))  # within payload pairs
WINDOW_SIZE   = 460         # ~2.5 s at 184 Hz
HOP_SIZE      = 92          # 80 % overlap → many windows per file
EMA_ALPHA     = 0.05        # background subtraction strength

# Training
BATCH_SIZE    = 32
MAX_EPOCHS    = 120
PATIENCE      = 18          # early stopping

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 ─ DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def parse_csi_line(line_str: str):
    """
    Extract 52-element magnitude vector from one raw ESP32 CSI row.

    Real data format (confirmed):
      Row inside CSV: timestamp,CSI_DATA,STATION,MAC,...,[header imag0 real0 ...]
      Numbers in brackets are SPACE-separated.
      First 4 numbers are a hardware header (84 -64 4 0) — skip them.
      Then pairs (imag, real) for each subcarrier.
      62 total numbers → 4 header + 58 payload → 29 pairs, but some are pilot zeros.
      Active subcarrier payload indices: 0-25 and 27-52 → 52 values.
    """
    if "[" not in line_str or "]" not in line_str:
        return None
    try:
        inner   = line_str.split("[")[1].split("]")[0]
        nums    = [int(x) for x in inner.split()
                   if x.lstrip("-").lstrip("+").isdigit()]
        if len(nums) < 14:
            return None
        # Skip 4-byte hardware header
        payload = nums[4:]
        imag    = np.array(payload[0::2], dtype=np.float32)
        real    = np.array(payload[1::2], dtype=np.float32)
        m       = min(len(imag), len(real))
        mag_all = np.sqrt(imag[:m]**2 + real[:m]**2)
        # Select only the 52 active subcarriers
        if m >= max(ACTIVE_IDX) + 1:
            return mag_all[ACTIVE_IDX]
        # Fallback: pad if slightly short
        if m >= N_SUBCARRIERS:
            return mag_all[:N_SUBCARRIERS]
        return None
    except Exception:
        return None


def ema_background_subtract(matrix: np.ndarray, alpha: float = EMA_ALPHA):
    """
    Subtract a running EMA background from each subcarrier column.
    matrix shape: (T, N_SUBCARRIERS)
    """
    bg  = matrix[0].copy()
    out = np.zeros_like(matrix)
    for t in range(len(matrix)):
        bg      = alpha * matrix[t] + (1 - alpha) * bg
        out[t]  = matrix[t] - bg
    return out


def load_file(path: str):
    """
    Load one raw CSI CSV file → 2-D magnitude matrix (T, 52).
    Handles the ESP32 format: each row is a quoted string containing
    'timestamp,CSI_DATA,STATION,...,[space-separated numbers]'
    Returns None if file has too few packets.
    """
    mags = []
    with open(path, newline="", errors="ignore") as f:
        for row in csv.reader(f):
            if not row:
                continue
            # Row may be a single quoted field or multiple fields
            line = row[0].strip().strip('"').strip("'")
            # The bracket data is always in the last part of the line
            m = parse_csi_line(line)
            if m is not None:
                mags.append(m)

    if len(mags) < WINDOW_SIZE:
        return None
    return np.stack(mags, axis=0)   # (T, 52)


def make_windows(matrix: np.ndarray, label: int):
    """Sliding-window segmentation with overlap."""
    T = len(matrix)
    X, y = [], []
    for start in range(0, T - WINDOW_SIZE + 1, HOP_SIZE):
        win = matrix[start:start + WINDOW_SIZE]   # (256, 52)
        X.append(win)
        y.append(label)
    return X, y


def load_dataset():
    print("\n" + "="*60)
    print("  LOADING & PREPROCESSING DATASET")
    print("="*60)

    X_all, y_all = [], []

    for label, folder in [(1, FALL_DIR), (0, NOFALL_DIR)]:
        tag = "FALL" if label == 1 else "NO FALL"
        files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        print(f"\n  [{tag}] Found {len(files)} files in {folder}")
        kept = 0
        for fname in files:
            path   = os.path.join(folder, fname)
            matrix = load_file(path)
            if matrix is None:
                continue
            # EMA background subtraction
            matrix = ema_background_subtract(matrix)
            X, y   = make_windows(matrix, label)
            X_all.extend(X)
            y_all.extend(y)
            kept += 1
        print(f"  [{tag}] Parsed {kept}/{len(files)} files → "
              f"{sum(1 for yy in y_all if yy == label)} windows")

    X_all = np.array(X_all, dtype=np.float32)   # (N, 256, 52)
    y_all = np.array(y_all, dtype=np.int32)

    print(f"\n  TOTAL Windows : {len(X_all)}")
    print(f"  Fall    (1)   : {np.sum(y_all == 1)}")
    print(f"  No-Fall (0)   : {np.sum(y_all == 0)}")
    print(f"  Shape         : {X_all.shape}")
    return X_all, y_all


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 ─ PREPROCESSING  (Z-score per subcarrier channel)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X):
    """
    Z-score normalise across the time axis for each window/subcarrier.
    Shape stays (N, 256, 52).
    """
    N, T, C = X.shape
    X_flat  = X.reshape(N * T, C)
    scaler  = StandardScaler()
    X_norm  = scaler.fit_transform(X_flat).reshape(N, T, C)
    return X_norm.astype(np.float32), scaler


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 ─ MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

def build_1dcnn_bilstm(input_shape):
    """
    MODEL A – 1D-CNN + Bidirectional LSTM
    Best for time-series with local feature + global sequence.
    """
    inp = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                          dropout=0.3, recurrent_dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, dropout=0.3))(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    # Cast to float32 — required when mixed_float16 is active
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name="1DCNN_BiLSTM")


def build_hybrid_cnn_lstm(input_shape):
    """
    MODEL B – 2D-CNN + LSTM (treats window as a 2D image)
    Captures spatial subcarrier patterns + temporal sequence.
    """
    inp = keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inp)   # (256,52,1)

    x = layers.Conv2D(32, (5, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (5, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.RepeatVector(32)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(64, dropout=0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    # Cast to float32 — required when mixed_float16 is active
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name="Hybrid_2DCNN_LSTM")


def build_tcn(input_shape):
    """
    MODEL C – Temporal Convolutional Network (TCN)
    Dilated causal convolutions; excellent at long-range temporal patterns.
    """
    def tcn_block(x, filters, kernel_size, dilation):
        skip = x
        x = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=dilation, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        if skip.shape[-1] != filters:
            skip = layers.Conv1D(filters, 1)(skip)
        return layers.Add()([x, skip])

    inp = keras.Input(shape=input_shape)
    x   = inp
    for d in [1, 2, 4, 8, 16]:
        x = tcn_block(x, 128, 3, d)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    # Cast to float32 — required when mixed_float16 is active
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name="TCN")


def build_transformer(input_shape):
    """
    MODEL D – Patch-Transformer (FAST version)
    CNN downsamples 460→32 "patches" FIRST so attention is O(32²) not O(460²).
    This makes it ~200x faster while keeping global temporal reasoning.
    """
    def transformer_block(x, embed_dim, num_heads, ff_dim, dropout=0.15):
        attn = layers.MultiHeadAttention(num_heads=num_heads,
                                          key_dim=embed_dim // num_heads,
                                          dropout=dropout)(x, x)
        x    = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff   = layers.Dense(ff_dim, activation="relu")(x)
        ff   = layers.Dropout(dropout)(ff)
        ff   = layers.Dense(embed_dim)(ff)
        x    = layers.LayerNormalization(epsilon=1e-6)(x + ff)
        return x

    inp = keras.Input(shape=input_shape)       # (460, 52)

    # ── CNN patch encoder: 460 → ~29 patches ──────────────────────────────
    x = layers.Conv1D(64,  7, strides=2, padding="same", activation="relu")(inp)  # 230
    x = layers.Conv1D(128, 5, strides=2, padding="same", activation="relu")(x)    # 115
    x = layers.Conv1D(128, 3, strides=4, padding="same", activation="relu")(x)    # ~29
    x = layers.BatchNormalization()(x)
    # Now x shape ≈ (29, 128)  — self-attention is 29×29 = cheap!

    # ── 2 Transformer blocks ───────────────────────────────────────────────
    for _ in range(2):
        x = transformer_block(x, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.15)

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.4)(x)
    # Cast to float32 — required when mixed_float16 is active
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name="PatchTransformer")


def build_deep_1dcnn(input_shape):
    """
    MODEL E – Deep Residual 1D-CNN (ResNet-style)
    Very effective at detecting sharp transient events like falls.
    """
    def res_block(x, filters, kernel_size=3, stride=1):
        skip = x
        x = layers.Conv1D(filters, kernel_size, strides=stride,
                          padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if skip.shape[-1] != filters or stride != 1:
            skip = layers.Conv1D(filters, 1, strides=stride, padding="same")(skip)
        x = layers.Add()([x, skip])
        x = layers.Activation("relu")(x)
        return x

    inp = keras.Input(shape=input_shape)
    x   = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x   = layers.BatchNormalization()(x)

    x   = res_block(x, 64)
    x   = layers.MaxPooling1D(2)(x)
    x   = res_block(x, 128)
    x   = layers.MaxPooling1D(2)(x)
    x   = res_block(x, 256)
    x   = layers.MaxPooling1D(2)(x)
    x   = res_block(x, 256)

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(256, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    # Cast to float32 — required when mixed_float16 is active
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name="Deep_ResNet1D")


def build_cnn_attention_lstm(input_shape):
    """
    MODEL F – CNN + Self-Attention + LSTM  (our custom champion)
    Combines spatial feature extraction, attention gating, and temporal modelling.
    """
    inp = keras.Input(shape=input_shape)

    # CNN tower
    x = layers.Conv1D(64,  7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Self-attention gate
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attn = layers.Dropout(0.2)(attn)
    x    = layers.LayerNormalization()(x + attn)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                          dropout=0.3, recurrent_dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64, dropout=0.3))(x)

    x   = layers.Dense(256, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.5)(x)
    # Cast to float32 — required when mixed_float16 is active
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inp, out, name="CNN_Attention_BiLSTM")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 ─ TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal loss — down-weights easy negatives, focuses on hard fall samples.
    alpha=0.75 gives extra weight to the minority fall class.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt     = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        at     = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        fl     = -at * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(fl)
    return loss_fn


def get_class_weights(y):
    classes  = np.unique(y)
    weights  = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def get_callbacks(model_name, ckpt_path):
    """
    Version-safe callbacks. Uses save_weights_only=True to avoid
    the native Keras 'options' argument bug on some TF/Keras versions.
    EarlyStopping with restore_best_weights=True keeps best weights in RAM.
    """
    cbs = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=0
        ),
    ]
    try:
        cbs.append(
            ModelCheckpoint(
                ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
        )
    except Exception as e:
        print(f"  [WARN] ModelCheckpoint skipped: {e}")
    return cbs


def train_model(model, X, y, class_weights, model_name, epochs_override=None):
    # .weights.h5 avoids the native Keras 'options' arg bug
    ckpt = os.path.join(OUTPUT_DIR, f"ckpt_{model_name}.weights.h5")
    cbs  = get_callbacks(model_name, ckpt)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )

    print(f"\n  ► Training {model_name}  (params: {model.count_params():,})")

    # Simple model.fit — fastest on CPU
    hist = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=epochs_override if epochs_override else MAX_EPOCHS,
        validation_split=0.15,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
        shuffle=True,
    )
    # EarlyStopping restore_best_weights=True already loaded best weights
    return hist



# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 ─ EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X, y, model_name, threshold=0.5):
    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)

    report = classification_report(y, preds,
                                   target_names=["No Fall", "Fall"],
                                   output_dict=True)
    auc    = roc_auc_score(y, probs)
    f1     = f1_score(y, preds, average="macro")
    cm     = confusion_matrix(y, preds)

    print(f"\n  ─── {model_name} ───")
    print(classification_report(y, preds, target_names=["No Fall", "Fall"]))
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Macro-F1: {f1:.4f}")
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    print(f"  Sensitivity (Fall Recall) : {sensitivity:.4f}")
    print(f"  Specificity (NoFall Recall): {specificity:.4f}")

    return {
        "name"       : model_name,
        "auc"        : auc,
        "macro_f1"   : f1,
        "fall_f1"    : report["Fall"]["f1-score"],
        "fall_recall": report["Fall"]["recall"],
        "accuracy"   : report["accuracy"],
        "sensitivity": sensitivity,
        "specificity": specificity,
        "cm"         : cm,
        "probs"      : probs
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 ─ VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_results(results, histories):
    plt.style.use("dark_background")
    n = len(results)
    fig = plt.figure(figsize=(20, 18))
    gs  = gridspec.GridSpec(3, n, figure=fig, hspace=0.5, wspace=0.4)

    # Row 0 – Confusion matrices
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Fall", "Fall"],
                    yticklabels=["No Fall", "Fall"], ax=ax,
                    cbar=False, linewidths=0.5)
        ax.set_title(r["name"], fontsize=9, color="white", pad=6)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

    # Row 1 – Training loss curves
    for i, (r, h) in enumerate(zip(results, histories)):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(h.history["loss"],    label="Train", color="#ff5252")
        ax.plot(h.history["val_loss"], label="Val",  color="#69f0ae")
        ax.set_title(f"{r['name']}\nLoss", fontsize=9, color="white")
        ax.legend(fontsize=7)
        ax.set_xlabel("Epoch", fontsize=8)

    # Row 2 – Bar chart comparison
    ax = fig.add_subplot(gs[2, :])
    names    = [r["name"] for r in results]
    metrics  = {
        "AUC"          : [r["auc"]         for r in results],
        "Macro-F1"     : [r["macro_f1"]    for r in results],
        "Fall-F1"      : [r["fall_f1"]     for r in results],
        "Sensitivity"  : [r["sensitivity"] for r in results],
    }
    x        = np.arange(len(names))
    width    = 0.2
    colors   = ["#ff5252", "#448aff", "#69f0ae", "#ffea00"]
    for j, (mname, mvals) in enumerate(metrics.items()):
        ax.bar(x + j * width, mvals, width, label=mname, color=colors[j], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.set_title("Model Comparison", fontsize=14, color="white")
    ax.set_ylabel("Score", fontsize=10)

    plt.suptitle("CSI Fall Detection — Model Tournament Results",
                 fontsize=16, color="white", y=1.01)
    out_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 ─ MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*60)
    print("  CSI FALL DETECTION — BEST MODEL SEARCH")
    print("█"*60)

    # ── Load data ──────────────────────────────────────────────
    X, y = load_dataset()

    # ── Normalise ──────────────────────────────────────────────
    print("\n  Z-score normalising …")
    X, scaler = preprocess(X)
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    # ── Class weights (handle 207:591 imbalance) ───────────────
    cw = get_class_weights(y)
    print(f"\n  Class weights: {cw}")

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    input_shape = (WINDOW_SIZE, N_SUBCARRIERS)   # (256, 52)

    # ── Model factory ──────────────────────────────────────────
    # max_epochs override per model (None = use global MAX_EPOCHS)
    # PatchTransformer is fast now but give it a cap as safety net
    builders = [
        ("1DCNN_BiLSTM",    build_1dcnn_bilstm,        None),
        ("2DCNN_LSTM",      build_hybrid_cnn_lstm,     None),
        ("TCN",             build_tcn,                  None),
        ("PatchTransformer",build_transformer,          60),   # cap at 60
        ("ResNet1D",        build_deep_1dcnn,           None),
        ("CNN_Attn_BiLSTM", build_cnn_attention_lstm,  None),
    ]

    all_results   = []
    all_histories = []
    all_models    = []

    for mname, builder, ep_override in builders:
        print("\n" + "─"*60)
        t0    = time.time()
        model = builder(input_shape)
        hist  = train_model(model, X, y, cw, mname,
                            epochs_override=ep_override)
        elapsed = (time.time() - t0) / 60
        print(f"  [{mname}] total training time: {elapsed:.1f} min")
        res   = evaluate_model(model, X, y, mname)
        all_results.append(res)
        all_histories.append(hist)
        all_models.append(model)

    # ── Pick best model ────────────────────────────────────────
    # Score = 0.4 * fall_f1 + 0.3 * sensitivity + 0.2 * auc + 0.1 * macro_f1
    # (prioritise not missing a real fall above all else)
    def composite_score(r):
        return (0.40 * r["fall_f1"]
              + 0.30 * r["sensitivity"]
              + 0.20 * r["auc"]
              + 0.10 * r["macro_f1"])

    for r in all_results:
        r["score"] = composite_score(r)

    ranked = sorted(range(len(all_results)),
                    key=lambda i: all_results[i]["score"],
                    reverse=True)

    print("\n" + "="*60)
    print("  MODEL RANKING  (composite score)")
    print("="*60)
    for rank, idx in enumerate(ranked, 1):
        r = all_results[idx]
        print(f"  #{rank:02d}  {r['name']:<22}  "
              f"Score={r['score']:.4f}  "
              f"Fall-F1={r['fall_f1']:.4f}  "
              f"Sensitivity={r['sensitivity']:.4f}  "
              f"AUC={r['auc']:.4f}")

    best_idx   = ranked[0]
    best_res   = all_results[best_idx]
    best_model = all_models[best_idx]
    print(f"\n  ★ BEST MODEL → {best_res['name']}")

    # ── Save best model ────────────────────────────────────────
    # Try .keras first; fall back to legacy .h5 if the version has the options bug
    best_path = os.path.join(OUTPUT_DIR, "best_fall_model.keras")
    try:
        best_model.save(best_path)
        print(f"  Best model saved (native Keras) → {best_path}")
    except Exception as e:
        print(f"  [WARN] .keras save failed ({e}), falling back to .h5")
        best_path = os.path.join(OUTPUT_DIR, "best_fall_model.h5")
        best_model.save(best_path)
        print(f"  Best model saved (legacy HDF5) → {best_path}")

    # ── Save TFLite for embedded/edge deployment ───────────────
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path  = os.path.join(OUTPUT_DIR, "best_fall_model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"  TFLite model saved → {tflite_path}")
    except Exception as e:
        print(f"  TFLite conversion skipped: {e}")

    # ── Save summary CSV ───────────────────────────────────────
    df = pd.DataFrame([{
        "model"       : r["name"],
        "score"       : r["score"],
        "fall_f1"     : r["fall_f1"],
        "sensitivity" : r["sensitivity"],
        "specificity" : r["specificity"],
        "auc"         : r["auc"],
        "macro_f1"    : r["macro_f1"],
        "accuracy"    : r["accuracy"]
    } for r in all_results]).sort_values("score", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "model_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Scores CSV saved → {csv_path}")

    # ── Visualise ──────────────────────────────────────────────
    ordered_results   = [all_results[i]   for i in ranked]
    ordered_histories = [all_histories[i] for i in ranked]
    plot_all_results(ordered_results, ordered_histories)

    print("\n  ALL DONE  ✓")
    print("="*60)


if __name__ == "__main__":
    main()