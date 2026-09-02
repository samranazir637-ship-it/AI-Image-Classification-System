# VisionAI Image Classification System

A Flask-based AI image classification dashboard for the Week 3 computer vision project. Users can upload an image, receive a predicted category with confidence, and review prediction history and model evaluation metrics.

## Features

- Image upload for JPG, JPEG, PNG, GIF, and WEBP files
- Classification into nine predefined categories
- Prediction confidence and class score breakdown
- Dashboard with total predictions and average confidence
- Local prediction history tracking
- Model evaluation metrics: accuracy, precision, recall, F1 score, and validation loss
- Upgrade roadmap for custom training, real-time camera detection, and model comparison

## Important implementation note

This repository currently contains a lightweight image-analysis prototype. The classifier uses image color, brightness, saturation, and aspect-ratio signals so the application works without downloading a large trained model. For production use, replace `ImageClassifier.predict()` with a trained TensorFlow or PyTorch CNN and report metrics from a real validation dataset.

## Requirements

- Python 3.10 or newer
- pip

## Run locally

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the application

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Upload to GitHub

Create an empty repository on GitHub, then run these commands from this folder:

```powershell
git init
git add .
git commit -m "Initial VisionAI image classification system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Do not commit `.venv`, uploaded images, `.env`, or local history data. These are excluded by `.gitignore`.

## Project structure

```text
.
├── app.py                 # Flask routes and application entry point
├── image_classifier.py    # Classifier and history storage logic
├── requirements.txt       # Python dependencies
├── templates/             # Home and dashboard HTML templates
├── static/css/            # Application styles
├── data/                  # Local runtime history storage
└── uploads/               # Local uploaded images
```
