from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'model_01.h5'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def get_disease_info(disease_name):
    disease_info = {
        "Bacterial_spot": {
            "treatment": "Apply copper-based fungicides, practice crop rotation, remove infected plants",
            "prevention": "Use disease-resistant varieties, avoid overhead irrigation"
        },
        "Early_blight": {
            "treatment": "Apply fungicides, remove infected leaves",
            "prevention": "Maintain good air circulation, avoid wet leaves overnight"
        },
        "Late_blight": {
            "treatment": "Apply fungicides containing copper or chlorothalonil",
            "prevention": "Plant resistant varieties, maintain good garden hygiene"
        },
        "Leaf Mold": {
            "treatment": "Improve air circulation, apply fungicides",
            "prevention": "Reduce humidity, space plants properly"
        },
        "Septoria leaf spot": {
            "treatment": "Remove infected leaves, apply fungicides",
            "prevention": "Mulch around plants, avoid overhead watering"
        },
        "Spider mites (Two spotted spider mite)": {
            "treatment": "Apply insecticidal soap or neem oil",
            "prevention": "Maintain humidity, regular inspection"
        },
        "Target_Spot": {
            "treatment": "Apply appropriate fungicides, remove infected leaves",
            "prevention": "Avoid overhead irrigation, maintain proper spacing"
        },
        "Yellow_Leaf_Curl_Virus": {
            "treatment": "Remove infected plants, control whiteflies",
            "prevention": "Use resistant varieties, control insect vectors"
        },
        "Mosaic_virus": {
            "treatment": "Remove and destroy infected plants",
            "prevention": "Control aphids, use virus-free seeds"
        },
        "Healthy": {
            "treatment": "No treatment needed",
            "prevention": "Continue good gardening practices"
        }
    }
    return disease_info.get(disease_name, {})

def model_predict(img_path, model):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        pred_class = np.argmax(preds, axis=1)[0]

        disease_mapping = {
            0: "Bacterial_spot",
            1: "Early_blight",
            2: "Late_blight",
            3: "Leaf Mold",
            4: "Septoria leaf spot",
            5: "Spider mites (Two spotted spider mite)",
            6: "Target_Spot",
            7: "Yellow_Leaf_Curl_Virus",
            8: "Mosaic_virus",
            9: "Healthy"
        }

        predicted_disease = disease_mapping.get(pred_class, "Unknown")
        confidence = float(preds[0][pred_class]) * 100

        disease_info = get_disease_info(predicted_disease)

        return {
            "disease": predicted_disease,
            "confidence": round(confidence, 2),
            "treatment": disease_info.get("treatment", "Not available"),
            "prevention": disease_info.get("prevention", "Not available")
        }

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = model_predict(filepath, model)

        if "error" in result:
            prediction = result["error"]
            image_url = None
        else:
            prediction = {
                "disease": result["disease"],
                "confidence": result["confidence"],
                "treatment": result["treatment"],
                "prevention": result["prevention"]
            }
            image_url = url_for('uploaded_file', filename=filename)

        return render_template('index.html', prediction=prediction, image_url=image_url)

    except Exception as e:
        return render_template('index.html', prediction=f"Something went wrong: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
