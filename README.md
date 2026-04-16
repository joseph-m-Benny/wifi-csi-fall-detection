# WiFi CSI Fall Detection System

An AI-powered fall detection system using WiFi Channel State Information (CSI). This project utilizes Deep Learning (ResNet1D and CNN+biLSTM) to identify fall events from WiFi signal disturbances.

## 🚀 Features
- **Real-time Detection**: Live monitoring of WiFi CSI data via serial port.
- **Deep Learning Models**: Includes ResNet1D and Hybrid CNN-biLSTM architectures.
- **Automated Alerts**: Integrated WhatsApp notifications via `pywhatkit` when a fall is detected.
- **Data Collection**: Built-in tools for collecting and labeling new CSI data.
- **Visualization**: Real-time plotting of signal magnitudes and detection confidence.

## 🛠️ Technologies Used
- **Python 3.x**
- **TensorFlow / Keras**: For building and training deep learning models.
- **Scikit-learn**: Data preprocessing and evaluation metrics.
- **Pandas & NumPy**: Data manipulation and signal processing.
- **Matplotlib & Seaborn**: Data visualization and confusion matrices.
- **Tkinter**: Graphical User Interface (GUI).
- **PySerial**: For communication with WiFi CSI hardware.
- **PyWhatKit**: For automated WhatsApp alerts.

## 📂 Project Structure
- `final model.py`: Main script for training the ResNet1D model.
- `iDcnn+biLSTM live/`: Contains the live application and hybrid model logic.
  - `app.py`: The main GUI application for real-time detection.
- `MODELS/`: Pre-trained models and scalers.
- `RAW/`, `TEST/`, `NEW_ROOM_DATA/`: Datasets containing CSI samples (Fall vs. No-Fall).
- `message.py`: Script for handling alerts and notifications.

## ⚙️ Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/wifi-csi-fall-detection.git
   cd wifi-csi-fall-detection
   ```
2. **Install dependencies**:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib pyserial pywhatkit joblib seaborn
   ```
3. **Hardware Setup**:
   - Connect your WiFi CSI sensing device (e.g., ESP32 with CSI firmware).
   - Update the `_PORT` variable in `app.py` or `message.py` to match your device's COM port (e.g., `COM5`).

## 🖥️ Usage
- **To Train the Model**: Run `python "final model.py"`.
- **To Start Live Detection**: Run `python "iDcnn+biLSTM live/app.py"`.

## 📊 Results
The system achieves high accuracy in detecting falls by analyzing the amplitude variations in WiFi subcarriers. Detailed performance metrics and confusion matrices can be found in the `MODELS/` directory.

---
*Developed as part of a WiFi CSI Mini Project.*
