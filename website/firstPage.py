import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import numpy as np
import joblib
import dlib
import face_recognition
import base64
import uuid
import os
import cv2
import json
from skimage import feature
from flask import Flask, render_template
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from matplotlib.pyplot import bar
new_width = 400
new_height = 400
kernel_size = 5


app = Flask(__name__)
svm_model_filename = "random_forest_model.pkl"
svm_model = joblib.load(svm_model_filename)


# Global variable to store the LBP histogram
global_lbp_histogram = None
global_detected_face=None
DETECTED_FOLDER = 'detected_faces'
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/enter_parameters', methods=['GET', 'POST'])
def enter_parameters():
    if request.method == 'POST':
        return redirect(url_for('processing'))

    return render_template('enter_parameters.html')

@app.route('/processing', methods=['GET', 'POST'])
def processing():
    saved_file_url = None
    global global_detected_face
    if request.method == 'POST':
        if 'fileUpload' in request.files:
            file = request.files['fileUpload']
            if file.filename != '':
                static_folder = os.path.join(app.root_path, 'static', 'uploaded_images')
                os.makedirs(static_folder, exist_ok=True)
                file_path = os.path.join(static_folder, file.filename)
                file.save(file_path)

                saved_file_url = process_image(file_path)
    global_detected_face=saved_file_url
    return render_template('processing.html', saved_file_url=saved_file_url)


def process_image(file_path):
    try:
        # Read the uploaded image using OpenCV
        print("File path:", file_path)
        image = cv2.imread(file_path)

        # Check if the image is successfully loaded
        if image is None:
            raise Exception("Failed to load the image.")

        print("Image shape:", image.shape)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization to the grayscale image
        equalized_image = cv2.equalizeHist(gray_image)

        # Apply Gaussian blur to the equalized image
        blurred_image = cv2.GaussianBlur(equalized_image, (kernel_size, kernel_size), 0)

        # Use dlib to detect faces
        detector = dlib.get_frontal_face_detector()
        faces = detector(blurred_image)

        # Ensure the 'detected_faces' folder exists
        detected_folder = os.path.join(app.root_path, 'static', 'detected_faces')
        os.makedirs(detected_folder, exist_ok=True)

        # Extract and save only the first detected face
        if faces:
            face = faces[0]
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            detected_face = gray_image[y:y+h, x:x+w]

            # Save the detected face
            detected_filename = 'detected_face_' + os.path.basename(file_path)
            detected_file_path = os.path.join(detected_folder, detected_filename)
            cv2.imwrite(detected_file_path, detected_face)

            # Generate URL for the saved detected face image
            detected_file_url = url_for('static', filename=f'detected_faces/{detected_filename}')

            return detected_file_url

        else:
            print("No face detected in the image.")
            return None

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None






def generate_lbp_histogram(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate LBP features
    lbp = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Flatten the histogram to a 1D array
    lbp_histogram = lbp.flatten()

    # Save the histogram plot as an image (optional)
    plt.plot(lbp_histogram)
    plt.title("LBP Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    histogram_image_path = "static/lbp_histogram.png"
    plt.savefig(histogram_image_path)
    plt.close()

    return lbp_histogram, histogram_image_path

@app.route('/extraction')
def extraction():
    global global_lbp_histogram
    global global_detected_face

    # Construct the full path by joining with the root path
    full_path = os.path.join(app.root_path, global_detected_face.lstrip('/'))

    # Generate LBP histogram for the last detected face
    lbp_histogram, histogram_image_path = generate_lbp_histogram(full_path)

    # Wrap Matplotlib code inside app context
    with app.app_context():
        # Matplotlib code to plot the LBP histogram
        plt.figure(figsize=(8, 6))
        plt.bar(range(256), lbp_histogram, width=1.0, color='b')
        plt.title('LBP Histogram')
        plt.xlabel('LBP Bin')
        plt.ylabel('Frequency')

        # Convert the plot to a base64-encoded image
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.read()).decode('utf-8')

    # Store the LBP histogram in the global variable
    global_lbp_histogram = lbp_histogram

    return render_template('extraction.html', lbp_histogram=lbp_histogram, img_base64=img_base64,global_detected_face=full_path)

@app.route('/classification')
def classification():
    global global_lbp_histogram

    if global_lbp_histogram is None:
        # Handle the case where the histogram is not available
        return render_template('classification.html', predicted_class="N/A", lbp_histogram=None)

    # Reshape the histogram to match the shape expected by the model
    lbp_histogram_reshaped = global_lbp_histogram.reshape(1, -1)

    # Make a prediction using the loaded SVM model
    prediction = svm_model.predict(lbp_histogram_reshaped)

    # Use the predicted class for further processing or rendering
    predicted_class = prediction[0]

    return render_template('classification.html', predicted_class=predicted_class, lbp_histogram=global_lbp_histogram)


if __name__ == '__main__':
    app.run(debug=True)