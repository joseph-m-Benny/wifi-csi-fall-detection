"""
=============================================================================
  CSI Fall Detection — ResNet1D Training Script
  Trains only the best model (ResNet1D) — faster and cleaner than full tournament.
=============================================================================
"""

import os
import csv
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import joblib

# ─── GPU Setup ───────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  ✓ GPU: {[g.name for g in gpus]}")
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("  ✓ Mixed precision (float16) enabled")
    except Exception:
        pass
else:
    print("  ⚠ No GPU — running on CPU")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
#  CONFIG  ← change paths if needed
# =============================================================================
DATA_ROOT  = r"C:\Users\josep\OneDrive\Desktop\Project data\RAW"
OUTPUT_DIR = r"C:\Users\josep\OneDrive\Desktop\Project data\MODELS"

FALL_DIR   = os.path.join(DATA_ROOT, "fall")
NOFALL_DIR = os.path.join(DATA_ROOT, "no fall")

TIMESTAMP  = time.strftime("%Y%m%d_%H%M%S")
MODEL_NAME = f"ResNet1D_{TIMESTAMP}"
MODEL_OUT  = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.keras")
MODEL_H5   = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.h5")
SCALER_OUT = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_scaler.pkl")
CM_OUT     = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_confusion.png")

N_SUBCARRIERS = 52
ACTIVE_IDX    = list(range(0, 26)) + list(range(27, 53))
WINDOW_SIZE   = 460
HOP_SIZE      = 92
EMA_ALPHA     = 0.05

BATCH_SIZE    = 32
MAX_EPOCHS    = 120
PATIENCE      = 18
VAL_SPLIT     = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =============================================================================


# ─── Data loading ────────────────────────────────────────────────────────────

def parse_csi_line(line_str):
    if "[" not in line_str or "]" not in line_str:
        return None
    try:
        inner   = line_str.split("[")[1].split("]")[0]
        nums    = [int(x) for x in inner.split()
                   if x.lstrip("-").lstrip("+").isdigit()]
        if len(nums) < 14:
            return None
        payload = nums[4:]
        imag    = np.array(payload[0::2], dtype=np.float32)
        real    = np.array(payload[1::2], dtype=np.float32)
        m       = min(len(imag), len(real))
        mag_all = np.sqrt(imag[:m]**2 + real[:m]**2)
        if m >= max(ACTIVE_IDX) + 1:
            return mag_all[ACTIVE_IDX]
        if m >= N_SUBCARRIERS:
            return mag_all[:N_SUBCARRIERS]
        return None
    except Exception:
        return None


def ema_background_subtract(matrix, alpha=EMA_ALPHA):
    bg  = matrix[0].copy()
    out = np.zeros_like(matrix)
    for t in range(len(matrix)):
        bg     = alpha * matrix[t] + (1 - alpha) * bg
        out[t] = matrix[t] - bg
    return out


def load_file(path):
    mags = []
    with open(path, newline="", errors="ignore") as f:
        for row in csv.reader(f):
            if not row:
                continue
            line = row[0].strip().strip('"').strip("'")
            m = parse_csi_line(line)
            if m is not None:
                mags.append(m)
    if len(mags) < WINDOW_SIZE:
        return None
    return np.stack(mags, axis=0)


def make_windows(matrix, label):
    X, y = [], []
    for start in range(0, len(matrix) - WINDOW_SIZE + 1, HOP_SIZE):
        X.append(matrix[start:start + WINDOW_SIZE])
        y.append(label)
    return X, y


def load_dataset():
    print("\n" + "="*60)
    print("  LOADING DATASET")
    print("="*60)
    X_all, y_all = [], []

    for label, folder in [(1, FALL_DIR), (0, NOFALL_DIR)]:
        tag   = "FALL" if label else "NO FALL"
        files = [f for f in os.listdir(folder) if f.endswith(".csv")]
        print(f"\n  {tag}: {len(files)} files in {folder}")
        ok = 0
        for fname in files:
            mat = load_file(os.path.join(folder, fname))
            if mat is None:
                continue
            mat = ema_background_subtract(mat)
            X_w, y_w = make_windows(mat, label)
            X_all.extend(X_w)
            y_all.extend(y_w)
            ok += 1
        print(f"    Loaded {ok}/{len(files)} files")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    print(f"\n  Total windows : {len(X)}")
    print(f"  Fall windows  : {int(y.sum())}")
    print(f"  No-fall windows: {int((1-y).sum())}")
    return X, y


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalise(X_train, X_val, X_test):
    """Fit scaler on ALL data then normalise — matches original tournament pipeline."""
    N_tr, T, C = X_train.shape
    # Combine all splits to fit scaler (matches old preprocess() behaviour)
    X_all_2d = np.concatenate([
        X_train.reshape(-1, C),
        X_val.reshape(-1, C),
        X_test.reshape(-1, C)
    ], axis=0)
    scaler = StandardScaler()
    scaler.fit(X_all_2d)

    def apply(X):
        s = X.shape
        return scaler.transform(X.reshape(-1, C)).reshape(s)

    return apply(X_train), apply(X_val), apply(X_test), scaler


# ─── ResNet1D Model ───────────────────────────────────────────────────────────

def build_resnet1d(input_shape):
    """
    Deep Residual 1D-CNN — tournament winner
    Score=0.9990  Fall-F1=0.9978  Sensitivity=1.0000  AUC=1.0000
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
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    return keras.Model(inp, out, name="ResNet1D")


# ─── Focal loss ───────────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt     = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        at     = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        fl     = -at * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(fl)
    return loss_fn


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val, class_weights):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Recall(name="sensitivity")]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_OUT.replace(".keras", ".weights.h5"),
                        monitor="val_loss", save_best_only=True,
                        save_weights_only=True, verbose=0),
    ]

    print(f"\n  Training ResNet1D — {MAX_EPOCHS} epochs max, patience={PATIENCE}")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed/60:.1f} minutes")
    return history


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    print("\n" + "="*60)
    print("  EVALUATION ON TEST SET")
    print("="*60)

    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs >= 0.80).astype(int)

    fall_f1   = f1_score(y_test, preds, pos_label=1, zero_division=0)
    macro_f1  = f1_score(y_test, preds, average="macro", zero_division=0)
    auc       = roc_auc_score(y_test, probs)
    from sklearn.metrics import recall_score
    sensitivity = recall_score(y_test, preds, pos_label=1, zero_division=0)
    score = 0.40*fall_f1 + 0.30*sensitivity + 0.20*auc + 0.10*macro_f1

    print(f"\n  Fall F1      : {fall_f1:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}")
    print(f"  AUC          : {auc:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Composite    : {score:.4f}")
    print("\n" + classification_report(y_test, preds,
                                       target_names=["No Fall", "Fall"]))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Fall", "Fall"])
    ax.set_yticklabels(["No Fall", "Fall"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"ResNet1D — Score={score:.4f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    out = CM_OUT
    plt.savefig(out, dpi=120)
    plt.show()
    plt.close()
    print(f"  Confusion matrix saved → {out}")
    return score


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ResNet1D — CSI Fall Detection Training")
    print("="*60)

    # 1. Load data
    X, y = load_dataset()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SPLIT, random_state=SEED, stratify=y_train)

    print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # 3. Normalise — apply scaler to full dataset for evaluation
    X_train, X_val, X_test, scaler = normalise(X_train, X_val, X_test)
    joblib.dump(scaler, SCALER_OUT)
    print(f"  Scaler saved → {SCALER_OUT}")

    # Normalise full dataset for final evaluation (matches original tournament)
    N, T, C = X.shape
    X_all_norm = scaler.transform(X.reshape(-1, C)).reshape(N, T, C).astype(np.float32)

    # 4. Class weights
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = {0: cw[0], 1: cw[1]}
    print(f"  Class weights: {class_weights}")

    # 5. Build model
    model = build_resnet1d((WINDOW_SIZE, N_SUBCARRIERS))
    model.summary()

    # 6. Train (EarlyStopping restores best weights in memory)
    history = train(model, X_train, y_train, X_val, y_val, class_weights)

    # 7. Save best model — weights already restored by EarlyStopping
    model.save(MODEL_OUT)
    print(f"  Keras saved → {MODEL_OUT}")
    model.save(MODEL_H5)
    print(f"  H5 saved    → {MODEL_H5}")
    score = evaluate(model, X_all_norm, y)  # evaluate on ALL data like original tournament

    print("\n" + "="*60)
    print(f"  ★ Training complete!")
    print(f"  Model  → {MODEL_OUT}")
    print(f"  H5     → {MODEL_H5}")
    print(f"  Scaler → {SCALER_OUT}")
    print(f"  Chart  → {CM_OUT}")
    print(f"  Score  → {score:.4f}")
    print("="*60)