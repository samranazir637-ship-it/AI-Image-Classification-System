from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from uuid import uuid4

from image_classifier import ImageClassifier, HistoryStore

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "local-development-key")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png", "gif", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

classifier = ImageClassifier()
history_store = HistoryStore("data/history.json")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


@app.route("/")
def index():
    summary = history_store.get_dashboard_summary()
    return render_template(
        "index.html",
        categories=classifier.CATEGORIES,
        summary=summary,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No image was uploaded.", "error")
        return redirect(url_for("index"))

    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        flash("Please choose an image before submitting.", "error")
        return redirect(url_for("index"))
    if not allowed_file(uploaded_file.filename):
        flash("Please upload a JPG, PNG, GIF, or WEBP image.", "error")
        return redirect(url_for("index"))

    original_name = secure_filename(uploaded_file.filename)
    filename = f"{uuid4().hex}_{original_name}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(file_path)

    prediction = classifier.predict(file_path)
    record = {
        "image_name": filename,
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "timestamp": prediction["timestamp"],
        "size": prediction["metadata"]["image_size"],
    }
    history_store.add_record(record)

    return render_template(
        "dashboard.html",
        prediction=prediction,
        image_name=filename,
        image_url=url_for("uploaded_file", filename=filename),
        summary=history_store.get_dashboard_summary(),
        history=history_store.get_history(),
        metrics=classifier.get_evaluation_metrics(),
    )


@app.route("/dashboard")
def dashboard():
    summary = history_store.get_dashboard_summary()
    metrics = classifier.get_evaluation_metrics()
    history = history_store.get_history()
    return render_template(
        "dashboard.html",
        prediction=None,
        image_url=None,
        summary=summary,
        history=history,
        metrics=metrics,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
