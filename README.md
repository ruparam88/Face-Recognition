# 🎭 Face Recognition Web App

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?logo=flask)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

A **Flask-based web application** that performs **face recognition** using a pre-trained classification model (`.pkl`).  
Upload an image → the app detects and classifies faces → shows recognition results instantly.  

---

## ✨ Features

✅ Load and use a pre-trained face recognition model  
✅ Detect and classify faces only if they were part of the training set  
✅ Upload image via a simple web interface  
✅ Flask backend + HTML/CSS/JS frontend  
✅ Easy to deploy locally or on cloud platforms  

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask  
- **Model:** scikit-learn / OpenCV (`.pkl` format)  
- **Frontend:** HTML, CSS, JavaScript  
- **Dependencies:** See `requirements.txt`  

---

## ⚡ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ruparam88/Face-Recognition.git
cd Face-Recognition
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python3 -m venv venv
source bin/activate       # macOS/Linux
Scripts\activate          # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
python app.py
```

👉 Visit **http://127.0.0.1:5000** in your browser.

---

## 📂 Project Structure

```
Face-Recognition/
├── app.py                     # Flask backend
├── face_recognition_model.pkl # Pre-trained model
├── requirements.txt           # Dependencies
├── templates/
│   └── index.html             # Frontend UI
├── static/                    # CSS / JS / Images (optional)
└── README.md
```

---

## 🎯 Usage Example

1. Open the app in your browser
2. Upload an image containing a face
3. Get results like:

✅ **Recognized:** Alice  
❌ **Unknown face**

---

## 🚀 Deployment

- **Local Development:** `python app.py`
- **Production (Render, Railway, Heroku):**

```bash
gunicorn app:app
```

---

## 🤝 Contribution

Want to improve this project? Fork it and submit a PR 🚀 Some ideas:

- Add live webcam face recognition
- Improve UI with React/Tailwind
- Integrate a database for storing recognized users

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ **Don't forget to star this repo if you find it useful!**

