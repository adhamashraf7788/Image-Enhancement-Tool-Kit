# breakage: Image Enhancement Web Application for CSE281
# Implements Power-Law Transformation, Gray-Level Slicing, Histogram Equalization, Bit-Plane Slicing, and Piecewise Linear Transformation
# Supports both grayscale and RGB image processing

from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
from io import BytesIO
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
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
    'max_gray': 200,
    'bit_plane': 7,
    'r1': 50,
    's1': 20,
    'r2': 150,
    's2': 200
}

def power_law_transform(image, gamma, is_rgb=False):
    """
    Apply Power-Law (Gamma) Transformation to an image.
    
    Args:
        image (np.ndarray): Grayscale or RGB image array.
        gamma (float): Gamma value for transformation.
        is_rgb (bool): If True, process each RGB channel separately.
    
    Returns:
        np.ndarray: Transformed image.
    """
    if is_rgb:
        channels = cv2.split(image)
        transformed_channels = []
        for channel in channels:
            normalized = channel / 255.0
            transformed = np.power(normalized, gamma)
            transformed_channels.append(np.uint8(transformed * 255))
        return cv2.merge(transformed_channels)
    else:
        normalized_img = image / 255.0
        transformed_img = np.power(normalized_img, gamma)
        return np.uint8(transformed_img * 255)

def gray_level_slicing(image, min_val, max_val, is_rgb=False):
    """
    Apply Gray-Level Slicing to highlight a specific intensity range.
    
    Args:
        image (np.ndarray): Grayscale or RGB image array.
        min_val (int): Minimum gray level to enhance.
        max_val (int): Maximum gray level to enhance.
        is_rgb (bool): If True, convert to HSV and slice V channel.
    
    Returns:
        np.ndarray: Sliced image.
    """
    if is_rgb:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        output_v = np.zeros_like(v)
        output_v[(v >= min_val) & (v <= max_val)] = 255
        return cv2.cvtColor(cv2.merge([h, s, output_v]), cv2.COLOR_HSV2BGR)
    else:
        output_image = np.zeros_like(image)
        output_image[(image >= min_val) & (image <= max_val)] = 255
        return output_image

def histogram_equalization(image, is_rgb=False):
    """
    Apply Histogram Equalization to enhance contrast.
    
    Args:
        image (np.ndarray): Grayscale or RGB image array.
        is_rgb (bool): If True, equalize V channel in HSV.
    
    Returns:
        np.ndarray: Equalized image.
    """
    if is_rgb:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(image)

def bit_plane_slicing(image, bit, is_rgb=False):
    """
    Extract a specific bit-plane from an image.
    
    Args:
        image (np.ndarray): Grayscale or RGB image array.
        bit (int): Bit-plane to extract (0-7).
        is_rgb (bool): If True, process each RGB channel separately.
    
    Returns:
        np.ndarray: Binary image of the selected bit-plane.
    """
    if is_rgb:
        channels = cv2.split(image)
        sliced_channels = []
        for channel in channels:
            sliced = np.uint8((channel & (1 << bit)) > 0) * 255
            sliced_channels.append(sliced)
        return cv2.merge(sliced_channels)
    else:
        return np.uint8((image & (1 << bit)) > 0) * 255

def piecewise_linear_transform(image, r1, s1, r2, s2, is_rgb=False):
    """
    Apply Piecewise Linear Transformation using control points.
    
    Args:
        image (np.ndarray): Grayscale or RGB image array.
        r1, s1 (int): First control point (input, output intensity).
        r2, s2 (int): Second control point (input, output intensity).
        is_rgb (bool): If True, process each RGB channel separately.
    
    Returns:
        np.ndarray: Transformed image.
    """
    def transform_channel(channel):
        output = np.zeros_like(channel, dtype=np.float32)
        mask1 = channel <= r1
        output[mask1] = (s1 / r1) * channel[mask1] if r1 > 0 else 0
        mask2 = (channel > r1) & (channel <= r2)
        output[mask2] = s1 + ((s2 - s1) / (r2 - r1)) * (channel[mask2] - r1) if r2 > r1 else s1
        mask3 = channel > r2
        output[mask3] = s2 + ((255 - s2) / (255 - r2)) * (channel[mask3] - r2) if r2 < 255 else s2
        return np.uint8(np.clip(output, 0, 255))

    if is_rgb:
        channels = cv2.split(image)
        transformed_channels = [transform_channel(channel) for channel in channels]
        return cv2.merge(transformed_channels)
    else:
        return transform_channel(image)

def get_histogram(image, is_rgb=False):
    """
    Generate histogram of an image.
    
    Args:
        image (np.ndarray): Grayscale or RGB image array.
        is_rgb (bool): If True, generate histograms for R, G, B channels.
    
    Returns:
        str: Base64-encoded histogram image.
    """
    plt.figure(figsize=(5, 3))
    if is_rgb:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f'{color.upper()} channel')
        plt.legend()
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
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
        image (np.ndarray): Image array (grayscale or RGB).
    
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
            is_rgb = request.form.get('color_mode') == 'rgb'
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_GRAYSCALE)
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
                output_image = power_law_transform(image, gamma, is_rgb=is_rgb)
            elif enhancement == 'gray_slicing':
                min_gray = int(request.form.get('min_gray', DEFAULTS['min_gray']))
                max_gray = int(request.form.get('max_gray', DEFAULTS['max_gray']))
                if min_gray < 0 or max_gray > 255 or min_gray > max_gray:
                    error = 'Invalid gray level range (0 ≤ min ≤ max ≤ 255)'
                    return render_template('index.html', error=error, defaults=DEFAULTS)
                output_image = gray_level_slicing(image, min_gray, max_gray, is_rgb=is_rgb)
            elif enhancement == 'histogram':
                output_image = histogram_equalization(image, is_rgb=is_rgb)
            elif enhancement == 'bit_slicing':
                bit = int(request.form.get('bit_plane', DEFAULTS['bit_plane']))
                if bit < 0 or bit > 7:
                    error = 'Bit plane must be between 0 and 7'
                    return render_template('index.html', error=error, defaults=DEFAULTS)
                output_image = bit_plane_slicing(image, bit, is_rgb=is_rgb)
            elif enhancement == 'piecewise_linear':
                r1 = int(request.form.get('r1', DEFAULTS['r1']))
                s1 = int(request.form.get('s1', DEFAULTS['s1']))
                r2 = int(request.form.get('r2', DEFAULTS['r2']))
                s2 = int(request.form.get('s2', DEFAULTS['s2']))
                if r1 < 0 or r1 > 255 or s1 < 0 or s1 > 255 or r2 < 0 or r2 > 255 or s2 < 0 or s2 > 255 or r1 > r2:
                    error = 'Invalid control points (0 ≤ r1 ≤ r2 ≤ 255, 0 ≤ s1, s2 ≤ 255)'
                    return render_template('index.html', error=error, defaults=DEFAULTS)
                output_image = piecewise_linear_transform(image, r1, s1, r2, s2, is_rgb=is_rgb)
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
        original_hist = get_histogram(image, is_rgb=is_rgb)
        processed_hist = get_histogram(output_image, is_rgb=is_rgb)

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