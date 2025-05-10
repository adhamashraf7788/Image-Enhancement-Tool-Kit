# breakage: Image Enhancement Web Application for CSE281
# Implements Power-Law Transformation, Gray-Level Slicing, Histogram Equalization, and Contrast Stretching

from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
from io import BytesIO
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')

# Ensure the upload and output directories exist
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Default parameters
DEFAULTS = {
    'gamma': 0.5,
    'min_gray': 100,
    'max_gray': 200
}

def power_law_transform(image, gamma):
    """
    Apply Power-Law (Gamma) Transformation to an image.
    
    Args:
        image (np.ndarray): Grayscale image array.
        gamma (float): Gamma value for transformation.
    
    Returns:
        np.ndarray: Transformed image.
    """
    normalized_img = image / 255.0
    transformed_img = np.power(normalized_img, gamma)
    transformed_img = np.uint8(transformed_img * 255)
    return transformed_img

def gray_level_slicing(image, min_val, max_val):
    """
    Apply Gray-Level Slicing to highlight a specific intensity range.
    
    Args:
        image (np.ndarray): Grayscale image array.
        min_val (int): Minimum gray level to enhance.
        max_val (int): Maximum gray level to enhance.
    
    Returns:
        np.ndarray: Sliced image.
    """
    output_image = np.zeros_like(image)
    output_image[(image >= min_val) & (image <= max_val)] = 255
    return output_image

def contrast_stretching(image):
    """
    Apply Contrast Stretching to enhance image contrast.
    
    Args:
        image (np.ndarray): Grayscale image array.
    
    Returns:
        np.ndarray: Stretched image.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image.copy()  # Avoid division by zero
    stretched = np.uint8(255 * (image - min_val) / (max_val - min_val))
    return stretched

def get_histogram(image):
    """
    Generate histogram of an image.
    
    Args:
        image (np.ndarray): Grayscale image array.
    
    Returns:
        str: Base64-encoded histogram image.
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure(figsize=(5, 3))
    plt.plot(hist, color='black')
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def image_to_base64(image):
    """
    Convert an image to base64 string for HTML display.
    
    Args:
        image (np.ndarray): Image array.
    
    Returns:
        str: Base64-encoded image.
    """
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

# Cache for processed image to support download
last_processed_image = None

# Jinja filter for cache busting
app.jinja_env.filters['datetime'] = lambda x: datetime.now()
app.jinja_env.filters['timestamp'] = lambda x: int(x.timestamp())

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_processed_image
    original_img = None
    processed_img = None
    original_hist = None
    processed_hist = None
    error = None

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            error = 'No file uploaded'
            return render_template('index.html', error=error, defaults=DEFAULTS)
        
        file = request.files['file']
        if file.filename == '':
            error = 'No file selected'
            return render_template('index.html', error=error, defaults=DEFAULTS)

        # Read the uploaded image
        try:
            img_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if image is None:
                error = 'Invalid image file'
                return render_template('index.html', error=error, defaults=DEFAULTS)
        except Exception as e:
            error = f'Error reading image: {str(e)}'
            return render_template('index.html', error=error, defaults=DEFAULTS)

        # Resize large images to improve performance
        max_size = 1024
        if image.shape[0] > max_size or image.shape[1] > max_size:
            scale = min(max_size / image.shape[0], max_size / image.shape[1])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        # Get the selected enhancement
        enhancement = request.form.get('enhancement')
        output_image = None

        try:
            if enhancement == 'gamma':
                gamma = float(request.form.get('gamma', DEFAULTS['gamma']))
                if gamma <= 0:
                    error = 'Gamma must be positive'
                    return render_template('index.html', error=error, defaults=DEFAULTS)
                output_image = power_law_transform(image, gamma)
            elif enhancement == 'gray_slicing':
                min_gray = int(request.form.get('min_gray', DEFAULTS['min_gray']))
                max_gray = int(request.form.get('max_gray', DEFAULTS['max_gray']))
                if min_gray < 0 or max_gray > 255 or min_gray > max_gray:
                    error = 'Invalid gray level range (0 ≤ min ≤ max ≤ 255)'
                    return render_template('index.html', error=error, defaults=DEFAULTS)
                output_image = gray_level_slicing(image, min_gray, max_gray)
            elif enhancement == 'histogram':
                output_image = cv2.equalizeHist(image)
            elif enhancement == 'contrast_stretching':
                output_image = contrast_stretching(image)
            else:
                error = 'Invalid enhancement selected'
                return render_template('index.html', error=error, defaults=DEFAULTS)
        except Exception as e:
            error = f'Error processing image: {str(e)}'
            return render_template('index.html', error=error, defaults=DEFAULTS)

        # Store processed image for download
        last_processed_image = output_image

        # Convert images and histograms to base64
        original_img = image_to_base64(image)
        processed_img = image_to_base64(output_image)
        original_hist = get_histogram(image)
        processed_hist = get_histogram(output_image)

    return render_template('index.html', original_img=original_img, processed_img=processed_img,
                         original_hist=original_hist, processed_hist=processed_hist,
                         error=error, defaults=DEFAULTS)

@app.route('/download')
def download():
    """
    Serve the last processed image for download.
    """
    global last_processed_image
    if last_processed_image is None:
        return "No processed image available", 404
    output_path = os.path.join(OUTPUT_FOLDER, 'processed.png')
    cv2.imwrite(output_path, last_processed_image)
    return send_file(output_path, as_attachment=True, download_name='processed_image.png')

if __name__ == '__main__':
    app.run(debug=True)