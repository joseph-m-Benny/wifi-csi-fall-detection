import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

# --- 1. SYSTEM CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_PATH = r"C:\Users\josep\OneDrive\Desktop\Project data\RAW"
WINDOW_SIZE = 256   
SUB_CARRIERS = 52
STEP_SIZE = 64      

# Math Alignment (Matches Live Script Exactly)
EMA_ALPHA = 0.01 

# --- 2. PARSING & PREPROCESSING (EMA-Aligned) ---
def parse_and_center(filepath):
    """Parses CSI and applies the Live EMA background subtraction point-by-point."""
    mags = []
    with open(filepath, 'r', errors='ignore') as f:
        for row in f:
            line = row.strip()
            if not line: continue
            if "[" in line and "]" in line:
                try:
                    csi = line.split("[")[1].split("]")[0]
                    nums = [int(x) for x in csi.replace(",", " ").split() if x.lstrip("-").isdigit()]
                    if len(nums) >= 104:
                        narr = np.array(nums)
                        mag = np.sqrt(narr[0::2][:52]**2 + narr[1::2][:52]**2)
                        mags.append(mag)
                except: continue
                
    mags = np.array(mags)
    if len(mags) == 0:
        return mags

    centered_mags = np.zeros_like(mags)
    ema_bg = mags[0].copy() 
    
    for i in range(len(mags)):
        ema_bg = (EMA_ALPHA * mags[i]) + ((1 - EMA_ALPHA) * ema_bg)
        centered_mags[i] = mags[i] - ema_bg

    return centered_mags

def load_dataset():
    X_train, y_train = [], []
    X_val, y_val = [], []
    categories = {'no fall': 0, 'fall': 1}
    
    for cat, label in categories.items():
        path = os.path.join(BASE_PATH, cat)
        if not os.path.exists(path): continue
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        
        if len(files) > 1:
            train_files, val_files = train_test_split(files, test_size=0.15, random_state=42)
        else:
            train_files, val_files = files, []
            
        def process_files(file_list, target_X, target_y):
            for filepath in file_list:
                data = parse_and_center(filepath)
                if len(data) >= WINDOW_SIZE:
                    for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
                        window = data[i : i + WINDOW_SIZE]
                        
                        # MIN_VARIANCE FILTER REMOVED! 
                        # The AI will now properly learn the "stillness" after a fall.
                        
                        norm_win = (window - np.mean(window)) / (np.std(window) + 1e-6)
                        target_X.append(norm_win)
                        target_y.append(label)

        process_files(train_files, X_train, y_train)
        process_files(val_files, X_val, y_val)
                    
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

# --- 3. CLINICAL DATA AUGMENTATION ---
def augment_data(X, y):
    print("🧬 Applying Domain-Shift Augmentations (Scaling & Jittering)...")
    scales = np.random.uniform(0.8, 1.2, size=(X.shape[0], 1, 1, 1))
    X_scaled = X * scales
    noise = np.random.normal(0, 0.01, size=X.shape)
    X_jittered = X + noise
    
    X_aug = np.concatenate([X, X_scaled, X_jittered], axis=0)
    y_aug = np.concatenate([y, y, y], axis=0)
    return X_aug, y_aug

# --- 4. DATA PREP ---
print("⚙️ Processing Dataset (EMA Aligned, Full Sequences)...")
X_train_raw, y_train_raw, X_val_raw, y_val = load_dataset()

X_train = X_train_raw.reshape(-1, WINDOW_SIZE, SUB_CARRIERS, 1)
X_val = X_val_raw.reshape(-1, WINDOW_SIZE, SUB_CARRIERS, 1)

X_train_aug, y_train_aug = augment_data(X_train, y_train_raw)

weights = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weights = dict(enumerate(weights))

# --- 5. THE HYBRID CNN-LSTM ARCHITECTURE ---
def build_cnn_lstm_model():
    model = models.Sequential([
        layers.Input(shape=(WINDOW_SIZE, SUB_CARRIERS, 1)),
        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), 
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), 
        
        layers.Reshape((64, 13 * 64)), 
        
        # Increased LSTM complexity slightly to better understand sequences
        layers.LSTM(64, return_sequences=False, dropout=0.4),
        
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn_lstm_model()

# --- 6. TRAINING ---
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n🧠 Training the Medical-Grade CNN-LSTM...")
model.fit(X_train_aug, y_train_aug, validation_data=(X_val, y_val), 
          epochs=50, batch_size=32, class_weight=class_weights, callbacks=[early_stop])

model.save('cnn.keras')
model.save('cnn.h5')

# --- 7. EVALUATION ---
y_probs = model.predict(X_val, verbose=0).flatten()
fpr, tpr, thresholds = roc_curve(y_val, y_probs)
best_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"\n🎯 Optimal Training Threshold calculated at: {best_threshold:.4f}")

y_pred = (y_probs > best_threshold).astype(int) 
print(f"\n--- VALIDATION EVALUATION ---")
print(classification_report(y_val, y_pred, target_names=['No Fall', 'Fall']))
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fall', 'Fall'],
            yticklabels=['No Fall', 'Fall'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("✅ Confusion matrix saved to confusion_matrix.png")