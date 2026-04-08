# Blood cell classification web application using TensorFlow/Keras CNN model
# Provides REST API endpoints for image upload and inference

import os
import uuid
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'hematovision-dev-secret-key')

# Configure upload directory for temporary image storage
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set maximum upload file size to 8 MB
MAX_UPLOAD_SIZE_MB = 8
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Supported image formats (consistent with web standards)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Input image dimensions for CNN (must match training configuration)
IMG_SIZE = 224

# Default class labels for 4-class blood cell classification
DEFAULT_CLASS_LABELS = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']


def resolve_first_existing_path(candidates):
    """Locate model file from candidate paths; return first existing path."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


MODEL_PATH = resolve_first_existing_path([
    os.path.join(BASE_DIR, 'Blood_Cell.h5'),
    os.path.join(BASE_DIR, '..', 'Blood_Cell.h5'),
])

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Model file not found. Place 'Blood_Cell.h5' inside HematoVision/ or project root."
    )

CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')

print("Loading pre-trained CNN model...")
model = load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")


def load_class_labels():
    """Load class labels from JSON; fallback to default if unavailable."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        if isinstance(labels, list) and labels:
            return labels
    return DEFAULT_CLASS_LABELS


CLASS_LABELS = load_class_labels()


def allowed_file(filename):
    """Validate file extension against whitelist."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_valid_image(file_storage):
    """Verify file contains valid image data using Pillow."""
    try:
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as img:
            img.verify()
        file_storage.stream.seek(0)
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        file_storage.stream.seek(0)
        return False


def build_unique_filename(filename):
    """Generate collision-free filename using UUID suffix."""
    safe_name = secure_filename(filename)
    name, ext = os.path.splitext(safe_name)
    return f"{name}_{uuid.uuid4().hex[:10]}{ext.lower()}"


def predict_image(img_path):
    """
    Load image, preprocess, and run inference.
    
    - Resize to 224x224 (model input dimension)
    - Normalize pixel values to [0, 1] range via rescale=1./255
    - Add batch dimension for model.predict()
    - Return class name and confidence score
    """
    # Load and resize image to model input dimensions
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    # Convert PIL image to numpy array (height, width, channels)
    img_array = image.img_to_array(img)
    # Add batch dimension for inference: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values: [0, 255] → [0, 1]
    img_array = img_array / 255.0
    # Run model inference (suppress verbose output)
    predictions = model.predict(img_array, verbose=0)
    # Extract class with highest probability
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    # Map class index to class label
    if predicted_index >= len(CLASS_LABELS):
        predicted_class = f"Class_{predicted_index}"
    else:
        predicted_class = CLASS_LABELS[predicted_index]
    # Return class name and confidence percentage
    return predicted_class, confidence


@app.route('/')
def home():
    """Render upload form page."""
    return render_template('home.html')


@app.errorhandler(RequestEntityTooLarge)
def handle_oversize_upload(_error):
    """Handle oversized file uploads with user-friendly error message."""
    flash(f"File too large. Upload image up to {MAX_UPLOAD_SIZE_MB} MB.", 'error')
    return redirect(url_for('home'))


@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded image and return classification result."""
    # Validate file presence
    if 'file' not in request.files:
        flash('No file selected. Please upload an image.', 'error')
        return redirect(url_for('home'))
    # Retrieve file from multipart form data
    file = request.files['file']
    # Reject empty/unnamed file
    if file.filename == '':
        flash('No file selected. Please upload an image.', 'error')
        return redirect(url_for('home'))
    # Check file extension against whitelist
    if not allowed_file(file.filename):
        flash('Unsupported file type. Use PNG, JPG, JPEG, BMP, or TIFF.', 'error')
        return redirect(url_for('home'))
    # Verify file contains valid image data (not spoofed)
    if not is_valid_image(file):
        flash('Invalid image file. Please upload a real image.', 'error')
        return redirect(url_for('home'))
    # Generate collision-free filename and save to upload directory
    filename = build_unique_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Run prediction on uploaded image
    try:
        predicted_class, confidence = predict_image(filepath)
    except Exception:
        flash('Prediction failed. Please try another image.', 'error')
        return redirect(url_for('home'))
    # Render result page with prediction and confidence
    return render_template(
        'result.html',
        prediction=predicted_class,
        confidence=f"{confidence:.2f}",
        image_path=url_for('static', filename=f"uploads/{filename}")
    )


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 HematoVision - Blood Cell Classifier")
    print("="*50)
    print(f"📊 Classes: {CLASS_LABELS}")
    print(f"🌐 Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    # Start Flask development server
    app.run(debug=True, port=5000)
