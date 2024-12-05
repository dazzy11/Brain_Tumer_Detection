
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from flask import redirect, url_for, session

app = Flask(__name__, template_folder='templates')
CORS(app)

# Set a secret key for session
app.secret_key = os.urandom(24)

# Dummy user database
users = {'admin': '123'}

# Path for uploading files
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
MODEL_PATH = "models/brain_tumor_model.h5"
model = load_model(MODEL_PATH)

# Dummy Sign-in Route
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username  # Store user in session
            return redirect(url_for('patient_details'))
        else:
            return "Invalid Credentials", 401
    return render_template('signin.html')

# Patient details form route
@app.route('/patient_details', methods=['GET', 'POST'])
def patient_details():
    if 'user' not in session:
        return redirect(url_for('signin'))  # Redirect if not signed in
    if request.method == 'POST':
        # Process patient details here if needed
        return redirect(url_for('index'))
    return render_template('patient_details.html')

# Detect tumor function
def detect_tumor(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        return f"Tumor Detected (Confidence: {prediction[0][0]*100:.2f}%)"
    else:
        return f"No Tumor Detected (Confidence: {(1-prediction[0][0])*100:.2f}%)"

# Upload image route
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    # Call the detection function
    img = cv2.imread(file_path)
    result = detect_tumor(img)
    return jsonify({'result': result, 'image_url': f'/uploads/{file.filename}'})

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Index route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
