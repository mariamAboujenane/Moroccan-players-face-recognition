# Moroccan-players-face-recognition
## Project Overview
This project focuses on facial recognition for Moroccan players. The main functionalities include preprocessing images, detecting faces, extracting Local Binary Patterns (LBP) features, and classifying the faces using a pre-trained SVM model.

## Project Structure

### Files and Directories
- `app.py`: The main application file that contains all the routes and functions for the Flask web app.
- `random_forest_model.pkl`: Pre-trained SVM model used for classification.
- `static/`: Directory containing static files like images and CSS.
  - `uploaded_images/`: Directory to store uploaded images.
  - `detected_faces/`: Directory to store detected face images.
  - `lbp_histogram.png`: Image file of the LBP histogram.
- `templates/`: Directory containing HTML templates for the web pages.
  - `index.html`: Homepage template.
  - `enter_parameters.html`: Template for the parameter entry page.
  - `processing.html`: Template for the image processing page.
  - `extraction.html`: Template for the LBP extraction page.
  - `classification.html`: Template for the classification result page.
- `face_recognition_presentation.pdf`: Presentation file with detailed information about the project.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/face-recognition.git
   cd face-recognition
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```
### Usage
1. **Home Page**: Navigate to the home page where you can start the process.
   - **Route**: `/`
   - **Description**: This is the starting point of the application.
![phot1](https://github.com/mariamAboujenane/Moroccan-players-face-recognition/assets/106840796/440f9c02-d60e-4f77-a938-b4d463361f63)

2. **Enter Parameters**: Go to the `/enter_parameters` route to upload an image.
   - **Route**: `/enter_parameters`
   - **Description**: Here, you can upload the image that will be processed.
![photo2](https://github.com/mariamAboujenane/Moroccan-players-face-recognition/assets/106840796/e0a8fa91-78b1-453a-93e7-d1f7afe6ef97)
3. **Processing**: The uploaded image is processed in the `/processing` route, where the face is detected and saved.
   - **Route**: `/processing`
   - **Description**: This route handles the image processing, including face detection and saving the detected face.
![photo3](https://github.com/mariamAboujenane/Moroccan-players-face-recognition/assets/106840796/e9d801cd-fd38-4cd7-b700-354f3d4dbc53)
![photo4](https://github.com/mariamAboujenane/Moroccan-players-face-recognition/assets/106840796/c27768a5-6279-4372-8224-0e761ebba451)
4. **Feature Extraction**: The `/extraction` route calculates the LBP histogram of the detected face and displays it.
   - **Route**: `/extraction`
   - **Description**: This route extracts features from the detected face using Local Binary Patterns (LBP) and displays the histogram.
![photo5](https://github.com/mariamAboujenane/Moroccan-players-face-recognition/assets/106840796/103d8f56-2615-4a43-b68d-bae79b2021f1)
5. **Classification**: The `/classification` route uses the SVM model to classify the face based on the LBP histogram and shows the predicted class.
   - **Route**: `/classification`
   - **Description**: This route classifies the face using a Support Vector Machine (SVM) model based on the extracted LBP features and displays the predicted class.
![photo6](https://github.com/mariamAboujenane/Moroccan-players-face-recognition/assets/106840796/1b948c11-54ec-4efc-b9b2-9e72bd9aec51)
