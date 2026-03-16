# Prognostic and Health Monitoring System (PHMS)

A full-stack real-time dashboard for monitoring jet engine health using a physical ESP32 sensor rig or a built-in simulation mode. Sensor data is scored by a trained Machine Learning model every 100ms and visualized through a 3D engine model, live charts, and a risk prediction panel.

---

## What It Does

The dashboard has two modes:

- **SIM mode** — you control 7 virtual sensor values via sliders. The frontend sends these to the backend over WebSocket, the ML model scores them, and the risk prediction updates in real time. No hardware needed.
- **HW mode** — a physical ESP32 reads real sensors and sends JSON over a USB serial connection. The backend picks up the data, runs ML inference, and broadcasts it to the frontend. The dashboard becomes a live window into a real running engine.

In both modes the ML model is always running — every frame of sensor data passes through the trained Logistic Regression pipeline and produces a risk percentage, a label (NORMAL / WARNING / CRITICAL), and a confidence score.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| 3D Visualization | React Three Fiber, Three.js, @react-three/drei |
| Charts | Recharts |
| Backend | FastAPI, Python, WebSockets |
| Serial Communication | PySerial |
| Machine Learning | Scikit-Learn, Joblib, NumPy, Pandas |
| Font | IBM Plex Mono |
| Hardware | ESP32, MPU6500, DS18B20, ACS712-5A, KY-003, TB6612FNG |

---

## Project Structure

```
jet-engine-phm/
├── app/
│   ├── layout.tsx                  IBM Plex Mono font setup, page metadata
│   ├── page.tsx                    Main dashboard — mode state, WebSocket logic, layout
│   ├── globals.css                 Tailwind, custom scrollbar, range input styling
│   └── components/
│       ├── jetEngineScene.tsx      3D engine — OBJ loader + procedural fallback, jitter, glow
│       ├── simulationCards.tsx     7 sensor sliders + Idle/Cruise/Stress/FAIL presets
│       ├── sensorCards.tsx         Read-only live hardware sensor cards with status bars
│       ├── predictionCards.tsx     Risk bar, ML label, confidence, contributing factors
│       ├── toggleButton.tsx        SIM / HW pill toggle button
│       └── Charts.tsx              Vibration history line chart with threshold lines
├── machineLearning/
│   ├── requirements.txt
│   ├── train_model.py              Trains pipeline, prints metrics, saves .pkl
│   └── data/
│       └── synthetic_data.py       Generates 10,000 rows of synthetic sensor data
├── api/
│   ├── main.py                     FastAPI server — /predict endpoint, /ws WebSocket, serial reader
│   └── requirements.txt
├── public/
│   └── engine.obj                  Your 3D engine model (place it here)
├── package.json
├── tailwind.config.ts
├── postcss.config.js
├── next.config.mjs
└── tsconfig.json
```

---

## Dashboard Features

### Header
- **WS Status indicator** — green LIVE / orange CONNECTING / red OFFLINE. Auto-reconnects every 3 seconds if the backend drops.
- **SIM / HW toggle** — switches between simulation and hardware mode.

### 3D Engine View (left column)
- Loads your `engine.obj` from the `public/` folder. If the file is missing it automatically renders a procedural fallback engine built from primitives — cylinder nacelle, inlet torus, exhaust cone, 12 fan blades, 4 struts.
- The model rotates smoothly to match pitch, roll, and yaw values. Rotation is lerped at a coefficient of 0.12 so it feels physically weighted rather than snapping instantly.
- Vibration drives sinusoidal jitter on all three axes — at 0g the model is perfectly still, at 5g it shakes visibly.
- The border glow shifts from Royal Blue at low vibration to red at high vibration.
- At very low vibration (< 0.05g) the model auto-rotates slowly. Click and drag to orbit manually at any time.
- Axis legend overlay: X = PITCH (red), Y = YAW (green), Z = ROLL (blue).

### Simulation Controls (middle column — SIM mode)
Seven sliders covering all sensor channels:

| Sensor | Range | Unit |
|---|---|---|
| Temperature | 20 – 200 | °C |
| Vibration | 0 – 5 | g |
| Current | 0 – 5 | A |
| RPM | 0 – 50,000 | rpm |
| Roll | -180 – 180 | ° |
| Pitch | -90 – 90 | ° |
| Yaw | -180 – 180 | ° |

**Quick presets:**

| Preset | Temp | Vibration | Current | RPM | Description |
|---|---|---|---|---|---|
|  Idle | 45°C | 0.10g | 0.8A | 5,000 | Engine at rest, all values well below thresholds |
|  Cruise | 72°C | 0.35g | 2.2A | 26,000 | Normal operating conditions |
|  Stress | 148°C | 3.1g | 3.8A | 43,000 | Pushing toward warning and critical thresholds |
|  FAIL | 190°C | 4.8g | 4.9A | 49,500 | All sensors deep in critical range |

These are hardcoded snapshots — clicking them snaps all 7 sliders simultaneously. The ML model then scores those values and produces the risk output.

### Live Sensor Cards (middle column — HW mode)
Read-only cards for all 7 sensors. Each card shows the current value, a progress bar, and a status badge when thresholds are exceeded.

**Per-sensor thresholds:**

| Sensor | Warning | Critical |
|---|---|---|
| Temperature | 100°C | 150°C |
| Vibration | 1.5g | 3.0g |
| Current | 3.0A | 4.0A |
| RPM | 35,000 | 45,000 |
| Roll / Pitch / Yaw | 30° | 60° |

Cards turn orange at warning and red at critical.

### Vibration History Chart (right column)
- Rolling line chart of the last 120 vibration readings (~12 seconds at 10Hz).
- Orange dashed reference line at 1.5g (warning).
- Red dashed reference line at 3.0g (critical).
- No animation for smooth real-time rendering.
- Hover tooltip shows exact value at each sample.

### Prediction Footer (fixed bottom bar)
The ML inference output, updated every 100ms:

- **Pulse dot** — blue when NORMAL, orange when WARNING, red and pulsing when CRITICAL.
- **Risk bar** — gradient bar from 0–100%. Dashed lines at 40% (WARNING boundary) and 70% (CRITICAL boundary).
- **Confidence** — how certain the model is, e.g. 97.3%.
- **Contributing factors** — mini bars for temperature and vibration showing their individual contribution, colored by threshold status.
- **Model info** — confirms the active pipeline (Logistic Regression + StandardScaler v1.0.0).

---

## Machine Learning Model

### Why vibration is the primary indicator
Vibration is the most physically meaningful fault signal in rotating machinery because it is a direct mechanical symptom rather than a downstream side effect. Temperature and current only rise *after* damage has been occurring for some time. Vibration changes the moment a bearing starts to degrade, a blade cracks, or a shaft goes out of balance — often days or weeks before temperature or current show any anomaly. This is why real-world Engine Health Monitoring (EHM) systems in commercial aviation are built primarily around vibration analysis.

### Features used for inference
Only 4 of the 7 sensor channels are fed into the model:

```
[temp, vibration, current, rpm]
```

Roll, pitch, and yaw affect the 3D visualization only — they are not used for fault prediction.

### Pipeline
```
Raw sensor values → StandardScaler → LogisticRegression → failure probability
```

### Failure modes in training data
The synthetic dataset contains three distinct failure patterns:

| Mode | Key Signal | How it manifests |
|---|---|---|
| Thermal + vibration | temp ~N(155,15), vib ~N(3.2,0.7) | Overheating combined with mechanical vibration |
| Overspeed | rpm ~N(46000,1500), current ~N(4.0,0.3) | Shaft spinning beyond safe limits |
| Electrical fault | current ~N(4.5,0.25) | Anomalous current draw with otherwise normal sensors |

Normal operation: temp ~N(65,12), vib ~N(0.3,0.15), rpm ~N(24000,4000). 30% failure, 70% normal.

### Risk thresholds
| Risk % | Label |
|---|---|
| 0 – 39% | NORMAL |
| 40 – 69% | WARNING |
| 70 – 100% | CRITICAL |

### Training results
Running `train_model.py` on the synthetic data produces near-perfect separation because the failure modes were designed with clear statistical boundaries:
```
AUC-ROC (test)  : ~1.0000
5-Fold CV AUC   : ~1.0000 ± 0.0000
```
In production with real sensor noise the accuracy would be lower and a more sophisticated model (gradient boosting, LSTM, autoencoder) would be appropriate.

---

## Hardware — ESP32 Sensor Rig

### What HW means
HW = Hardware. It refers to the physical sensor rig built around an ESP32 microcontroller. In HW mode the dashboard displays real measurements from a running motor rather than simulated values.

### Sensor wiring

| Sensor | Purpose | Interface | GPIO |
|---|---|---|---|
| MPU6500 | Gyroscope + accelerometer — roll, pitch, yaw, vibration | I²C | SDA=21, SCL=22 |
| DS18B20 | Temperature | OneWire | GPIO 4 |
| ACS712-5A | Current sensing — 185mV/A | ADC | GPIO 34 |
| KY-003 Hall Effect | RPM via pulse counting | Digital | GPIO 35 |
| TB6612FNG | Motor driver — controls the engine being monitored | PWM | GPIO 25 |

### ESP32 firmware — what it must output
The ESP32 must emit newline-terminated JSON at **115200 baud, 10Hz**:

```json
{"temp": 72.4, "vibration": 0.35, "current": 2.18, "rpm": 25800, "roll": 0.5, "pitch": 2.8, "yaw": -0.3}
```

All 7 fields must be present. The backend will reject malformed lines silently and wait for the next one.

### Connecting on Windows
On Windows the serial port will be a COM port, not `/dev/ttyUSB0`. Check Device Manager for the correct port (e.g. COM3) then start the backend with:

```bash
set SERIAL_PORT=COM3
python -m uvicorn main:app --reload --port 8000
```

Or on PowerShell:
```powershell
$env:SERIAL_PORT = "COM3"
python -m uvicorn main:app --reload --port 8000
```

---

## Setup & Run

### Prerequisites
- Node.js v18+
- Python 3.9+
- npm

### 1. Install frontend dependencies
```bash
cd jet-engine-phm
npm install
```

### 2. Generate training data
```bash
cd machineLearning
pip install -r requirements.txt
python data/synthetic_data.py
```

### 3. Train the model
```bash
python train_model.py
# Saves → machineLearning/model/logistic_regression.pkl
```

### 4. Start the API
```bash
cd api
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### 5. Start the frontend
```bash
cd ..
npm run dev
# → http://localhost:3000
```

### 6. (Optional) Place your 3D model
Copy your `engine.obj` into the `public/` folder:
```
jet-engine-phm/public/engine.obj
```
If absent, the procedural fallback renders automatically.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws` | WebSocket URL for the frontend |
| `SERIAL_PORT` | `/dev/ttyUSB0` | Serial device for ESP32 (use COM3 etc. on Windows) |
| `MODEL_PATH` | `../machineLearning/model/logistic_regression.pkl` | Path to trained model |

Create `.env.local` in the project root:
```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

---

## How Everything Connects

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│   Next.js + React Three Fiber + Recharts                    │
│                                                             │
│   SIM mode: sends slider values over WebSocket @ 10Hz       │
│   HW mode:  receives live scored telemetry from backend     │
│                                                             │
│   Displays: 3D engine, sensor cards, chart, risk footer     │
└────────────────────┬────────────────────────────────────────┘
                     │ WebSocket ws://localhost:8000/ws
┌────────────────────▼────────────────────────────────────────┐
│                         BACKEND                             │
│   FastAPI + PySerial + Scikit-Learn                         │
│                                                             │
│   /ws endpoint:                                             │
│     - Receives sim frames from frontend                     │
│     - Runs ML inference on [temp, vib, current, rpm]        │
│     - Echoes back enriched payload with risk_pct/label      │
│                                                             │
│   Serial thread:                                            │
│     - Reads JSON lines from ESP32 at 115200 baud            │
│     - Runs ML inference                                     │
│     - Broadcasts scored payload to all WS clients           │
│                                                             │
│   /predict POST:                                            │
│     - REST endpoint for one-off scoring                     │
└────────────────────┬────────────────────────────────────────┘
                     │ USB Serial 115200 baud (Physical mode only)
┌────────────────────▼────────────────────────────────────────┐
│                        ESP32 RIG                            │
│                                                             │
│   MPU6500  →  roll, pitch, yaw, vibration                   │
│   DS18B20  →  temperature                                   │
│   ACS712   →  current                                       │
│   KY-003   →  RPM                                           │
│                                                             │
│   Emits JSON @ 10Hz over USB serial                         │
└─────────────────────────────────────────────────────────────┘
```




## Media
![PHMS2](https://github.com/user-attachments/assets/17a862fe-a40c-4269-9db0-cd61a51a5349)
![phms_dashboard](https://github.com/user-attachments/assets/9dd97035-f427-4130-ace8-ed0d30224caf)

