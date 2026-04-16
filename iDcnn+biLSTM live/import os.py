import os
import numpy as np
import tensorflow as tf
from collections import deque

# --- 1. SYSTEM CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🛑 UPDATE THIS TO YOUR TEST FOLDER 🛑
TEST_FOLDER_PATH = r"C:\Users\josep\OneDrive\Desktop\Project data\TEST"
MODEL_PATH = 'csi_fall_hybrid_cnnlstm.keras'

# Exact math parity with your live system
WINDOW_SIZE = 256   
SUB_CARRIERS = 52
STEP_SIZE = 64      
EMA_ALPHA = 0.01 

# 🏆 THE WINNING GRID-SEARCH PARAMETERS 🏆
SAFETY_THRESHOLD = 0.80  
VOTE_COUNT = 3           
BUFFER_SIZE = 3          

# --- 2. EXACT MATH PARITY PARSER ---
def parse_and_center(filepath):
    """Parses CSI and applies the EMA background subtraction point-by-point."""
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

# --- 3. BATCH DIAGNOSIS ENGINE ---
def test_entire_folder():
    print(f"🚀 Loading Medical-Grade CNN-LSTM Model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model. Error: {e}")
        return

    if not os.path.exists(TEST_FOLDER_PATH):
        print(f"❌ Folder not found! Please check: {TEST_FOLDER_PATH}")
        return

    files = [f for f in os.listdir(TEST_FOLDER_PATH) if f.endswith('.csv')]
    print(f"\n📂 Found {len(files)} CSV files in {TEST_FOLDER_PATH}\n")
    
    print("-" * 80)
    print(f"{'FILE NAME':<30} | {'FRAMES':<8} | {'PEAK CONFIDENCE':<16} | {'CLINICAL DIAGNOSIS'}")
    print("-" * 80)

    total_falls = 0
    total_safe = 0

    for file in files:
        filepath = os.path.join(TEST_FOLDER_PATH, file)
        data = parse_and_center(filepath)
        
        if len(data) < WINDOW_SIZE:
            print(f"{file:<30} | {len(data):<8} | {'N/A':<16} | ⚠️ SKIPPED (Too short)")
            continue

        # Real-time memory buffer simulation
        prediction_buffer = deque(maxlen=BUFFER_SIZE)
        alarm_triggered = False
        max_prob = 0.0

        # Slide the window across the file
        for i in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
            window = data[i : i + WINDOW_SIZE]
            
            # Normalize and Reshape
            norm_win = (window - np.mean(window)) / (np.std(window) + 1e-6)
            inp = norm_win.reshape(1, WINDOW_SIZE, SUB_CARRIERS, 1)
            
            # Predict
            prob = float(model.predict(inp, verbose=0)[0][0])
            if prob > max_prob: 
                max_prob = prob
                
            # 1. Threshold Check
            current_pred = 1 if prob > SAFETY_THRESHOLD else 0
            
            # 2. Push to Circular Buffer
            prediction_buffer.append(current_pred)
            
            # 3. Vote Check (Does the buffer have 3 consecutive 1s?)
            if sum(prediction_buffer) >= VOTE_COUNT:
                alarm_triggered = True
                # We don't break here so we can still find the max_prob of the whole file

        # Formatting the output row
        prob_str = f"{max_prob*100:.1f}%"
        if alarm_triggered:
            diagnosis = "🚨 FALL DETECTED"
            total_falls += 1
        else:
            diagnosis = "✅ SAFE (No 3-Vote trigger)"
            total_safe += 1

        print(f"{file:<30} | {len(data):<8} | {prob_str:<16} | {diagnosis}")

    print("-" * 80)
    print("📊 BATCH SUMMARY")
    print("-" * 80)
    print(f"Total Files Processed : {total_falls + total_safe}")
    print(f"Falls Alarms Triggered: {total_falls}")
    print(f"Safe Files Suppressed : {total_safe}")
    print("=" * 80)

if __name__ == "__main__":
    test_entire_folder()