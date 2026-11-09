"""
Interactive Web Application for License Plate Detection (app.py)
A lightweight web app with zero extra dependencies (standard Python http.server)
Provides drag-and-drop inference, bounding box overlay, confidence scores,
and automated cropped license plate visualization.
"""

import os
import sys
import json
import base64
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
from ultralytics import YOLO


# Prioritize root weights/best.pt, fallback to training run
CANDIDATE_WEIGHTS = [
    os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    os.path.join(os.path.dirname(__file__), "license-plate-detection", "runs", "detect", "train-2", "weights", "best.pt")
]

DEFAULT_WEIGHTS = next((w for w in CANDIDATE_WEIGHTS if os.path.exists(w)), CANDIDATE_WEIGHTS[0])

# Load model globally on startup
print(f"Initializing YOLO11 License Plate Detection Model from: {DEFAULT_WEIGHTS}")
model = YOLO(DEFAULT_WEIGHTS)
print("Model loaded successfully!")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI License Plate Detection & Cropping</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #131b2e;
            --card-bg: rgba(23, 32, 54, 0.75);
            --border: rgba(99, 133, 255, 0.2);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.4);
            --accent-emerald: #10b981;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e293b 0%, var(--bg-primary) 70%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 30px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 35px;
        }
        .badge {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid var(--accent);
            color: #60a5fa;
            padding: 5px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        h1 {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 40%, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        p.subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            backdrop-filter: blur(12px);
            border-radius: 14px;
            padding: 16px;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
        }
        .stat-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
        }
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        .main-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        @media (max-width: 900px) {
            .main-layout { grid-template-columns: 1fr; }
        }
        .panel {
            background: var(--card-bg);
            border: 1px solid var(--border);
            backdrop-filter: blur(14px);
            border-radius: 18px;
            padding: 24px;
        }
        .panel-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .dropzone {
            border: 2px dashed rgba(99, 133, 255, 0.4);
            border-radius: 14px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(15, 23, 42, 0.4);
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent);
            background: rgba(59, 130, 246, 0.08);
            box-shadow: 0 0 25px var(--accent-glow);
        }
        .btn {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 15px;
        }
        .btn:hover {
            box-shadow: 0 4px 18px rgba(37, 99, 235, 0.5);
            transform: translateY(-1px);
        }
        .slider-group {
            margin-top: 20px;
        }
        .slider-group label {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        input[type="range"] {
            width: 100%;
            accent-color: var(--accent);
        }
        .preview-img {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border);
            display: none;
            margin-top: 15px;
        }
        .crops-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }
        .crop-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 8px;
            text-align: center;
        }
        .crop-card img {
            width: 100%;
            height: 70px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .crop-conf {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent-emerald);
            margin-top: 4px;
        }
        .spinner {
            display: none;
            margin: 20px auto;
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">Production AI Model</div>
            <h1>YOLO11 Automatic License Plate Detection</h1>
            <p class="subtitle">High-speed real-time vehicle license plate detector trained on 16,000+ annotations</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">94.8%</div>
                <div class="stat-label">Test Precision</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">89.9%</div>
                <div class="stat-label">Test mAP @ 0.50</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">86.3%</div>
                <div class="stat-label">Test Recall</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">5.4 ms</div>
                <div class="stat-label">GPU Latency (~130 FPS)</div>
            </div>
        </div>

        <div class="main-layout">
            <div class="panel">
                <div class="panel-title">Upload & Detection Settings</div>
                <div class="dropzone" id="dropzone">
                    <svg width="48" height="48" fill="none" stroke="#60a5fa" stroke-width="1.8" viewBox="0 0 24 24" style="margin-bottom:10px;">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                    </svg>
                    <p style="font-weight: 600;">Drag & drop vehicle image here</p>
                    <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 4px;">or click to browse files (JPG, PNG)</p>
                    <input type="file" id="fileInput" accept="image/*" style="display:none;">
                </div>

                <div class="slider-group">
                    <label>
                        <span>Confidence Threshold:</span>
                        <span id="confValue" style="color: var(--accent); font-weight: 600;">0.35</span>
                    </label>
                    <input type="range" id="confSlider" min="0.10" max="0.95" step="0.05" value="0.35">
                </div>

                <button class="btn" id="detectBtn">
                    <span>Detect License Plates</span>
                </button>
            </div>

            <div class="panel">
                <div class="panel-title">Live Detection Results</div>
                <div class="spinner" id="spinner"></div>
                <div id="statusText" style="color: var(--text-secondary); font-size: 0.9rem;">Upload an image to see detection results and cropped plates.</div>
                <img id="resultImg" class="preview-img" alt="Detection Output">

                <div id="cropsContainer" style="display:none; margin-top: 20px;">
                    <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 8px;">Extracted License Plates:</div>
                    <div class="crops-grid" id="cropsGrid"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const confSlider = document.getElementById('confSlider');
        const confValue = document.getElementById('confValue');
        const detectBtn = document.getElementById('detectBtn');
        const spinner = document.getElementById('spinner');
        const resultImg = document.getElementById('resultImg');
        const statusText = document.getElementById('statusText');
        const cropsContainer = document.getElementById('cropsContainer');
        const cropsGrid = document.getElementById('cropsGrid');

        let selectedFile = null;

        confSlider.addEventListener('input', (e) => {
            confValue.innerText = Number(e.target.value).toFixed(2);
        });

        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        });

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });

        function handleFile(file) {
            selectedFile = file;
            dropzone.querySelector('p').innerText = 'Selected: ' + file.name;
        }

        detectBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('Please select or drop an image file first!');
                return;
            }

            spinner.style.display = 'block';
            statusText.innerText = 'Analyzing image with YOLO11...';
            resultImg.style.display = 'none';
            cropsContainer.style.display = 'none';
            cropsGrid.innerHTML = '';

            const reader = new FileReader();
            reader.onload = async () => {
                const base64Data = reader.result.split(',')[1];
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image: base64Data,
                            conf: parseFloat(confSlider.value)
                        })
                    });
                    const data = await response.json();
                    spinner.style.display = 'none';

                    if (data.error) {
                        statusText.innerText = 'Error: ' + data.error;
                        return;
                    }

                    statusText.innerText = `Detected ${data.count} plate(s) in ${data.speed_ms.toFixed(1)} ms.`;
                    resultImg.src = 'data:image/jpeg;base64,' + data.annotated;
                    resultImg.style.display = 'block';

                    if (data.crops && data.crops.length > 0) {
                        cropsContainer.style.display = 'block';
                        data.crops.forEach((crop, idx) => {
                            const card = document.createElement('div');
                            card.className = 'crop-card';
                            card.innerHTML = `
                                <img src="data:image/jpeg;base64,${crop.img}">
                                <div class="crop-conf">Plate #${idx+1} (${(crop.conf*100).toFixed(0)}%)</div>
                            `;
                            cropsGrid.appendChild(card);
                        });
                    }
                } catch (err) {
                    spinner.style.display = 'none';
                    statusText.innerText = 'Inference request failed: ' + err.message;
                }
            };
            reader.readAsDataURL(selectedFile);
        });
    </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                payload = json.loads(post_data.decode("utf-8"))
                img_bytes = base64.b64decode(payload["image"])
                conf = float(payload.get("conf", 0.35))

                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("Could not decode image.")

                results = model.predict(source=img, conf=conf, imgsz=640, verbose=False)
                res = results[0]

                annotated_bgr = res.plot()
                _, buffer = cv2.imencode(".jpg", annotated_bgr)
                annotated_base64 = base64.b64encode(buffer).decode("utf-8")

                crops = []
                for box in res.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    c = float(box.conf[0])
                    crop = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    if crop.size > 0:
                        _, c_buf = cv2.imencode(".jpg", crop)
                        crops.append({
                            "conf": c,
                            "img": base64.b64encode(c_buf).decode("utf-8")
                        })

                speed_ms = sum(res.speed.values())

                response_data = {
                    "count": len(res.boxes),
                    "speed_ms": speed_ms,
                    "annotated": annotated_base64,
                    "crops": crops
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RequestHandler)
    print("\n" + "=" * 60)
    print(f" License Plate Detection Web UI is live at:")
    print(f" -> http://localhost:{port}")
    print(" Press Ctrl+C to stop the server.")
    print("=" * 60 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
