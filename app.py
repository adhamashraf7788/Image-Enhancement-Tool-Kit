from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Ensure the upload and output directories exist
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Power-Law (Gamma) Transformation function
def power_law_transform(image, gamma):
    normalized_img = image / 255.0
    transformed_img = np.power(normalized_img, gamma)
    transformed_img = np.uint8(transformed_img * 255)
    return transformed_img

# Gray-Level Slicing function
def gray_level_slicing(image, min_val, max_val):
    output_image = np.zeros_like(image)
    output_image[(image >= min_val) & (image <= max_val)] = 255
    return output_image

# Function to convert image to base64 for HTML display
def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    original_img = None
    processed_img = None
    error = None

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            error = 'No file uploaded'
            return render_template('index.html', error=error)
        
        file = request.files['file']
        if file.filename == '':
            error = 'No file selected'
            return render_template('index.html', error=error)

        # Read the uploaded image
        try:
            img_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if image is None:
                error = 'Invalid image file'
                return render_template('index.html', error=error)
        except Exception as e:
            error = f'Error reading image: {str(e)}'
            return render_template('index.html', error=error)

        # Get the selected enhancement
        enhancement = request.form.get('enhancement')
        output_image = None

        try:
            if enhancement == 'gamma':
                gamma = float(request.form.get('gamma', 0.5))
                output_image = power_law_transform(image, gamma)
            elif enhancement == 'gray_slicing':
                min_gray = int(request.form.get('min_gray', 100))
                max_gray = int(request.form.get('max_gray', 200))
                output_image = gray_level_slicing(image, min_gray, max_gray)
            elif enhancement == 'histogram':
                output_image = cv2.equalizeHist(image)
            else:
                error = 'Invalid enhancement selected'
                return render_template('index.html', error=error)
        except Exception as e:
            error = f'Error processing image: {str(e)}'
            return render_template('index.html', error=error)

        # Convert images to base64 for display
        original_img = image_to_base64(image)
        processed_img = image_to_base64(output_image)

    return render_template('index.html', original_img=original_img, processed_img=processed_img, error=error)

if __name__ == '__main__':
    app.run(debug=True)