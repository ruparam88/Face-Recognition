from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import cv2
import numpy as np
import face_recognition
import base64
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

class FaceRecognitionPredictor:
    def __init__(self, model_path='face_recognition_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.face_encodings = None
        self.face_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and associated data."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.face_encodings = model_data.get('face_encodings', [])
            self.face_names = model_data.get('face_names', [])
            
            print("Model loaded successfully!")
            
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict_face(self, image_array, tolerance=0.6):
        """
        Predict faces in the given image.
        
        Args:
            image_array: numpy array of the image
            tolerance: face matching tolerance (lower is more strict)
        
        Returns:
            list of dictionaries with face predictions and locations
        """
        try:
            # Convert image to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            results = []
            
            for (face_encoding, face_location) in zip(face_encodings, face_locations):
                # Method 1: If using known face encodings (recommended)
                if self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.face_encodings, face_encoding, tolerance=tolerance
                    )
                    face_distances = face_recognition.face_distance(
                        self.face_encodings, face_encoding
                    )
                    
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                        else:
                            name = "Unknown"
                            confidence = 0.0
                    else:
                        name = "Unknown"
                        confidence = 0.0
                
                # Method 2: If using trained classifier (alternative approach)
                elif self.model:
                    # Reshape face encoding for model prediction
                    face_encoding_reshaped = face_encoding.reshape(1, -1)
                    prediction = self.model.predict(face_encoding_reshaped)
                    probability = self.model.predict_proba(face_encoding_reshaped)
                    
                    # Get the predicted name
                    predicted_label = prediction[0]
                    name = self.label_encoder.inverse_transform([predicted_label])[0]
                    confidence = np.max(probability)
                
                else:
                    name = "Unknown"
                    confidence = 0.0
                
                # Only add result if confidence is 50% or higher
                if confidence >= 0.65:
                    # Convert face location coordinates
                    top, right, bottom, left = face_location
                    
                    results.append({
                        'name': name,
                        'confidence': float(confidence),
                        'location': {
                            'top': int(top),
                            'right': int(right),
                            'bottom': int(bottom),
                            'left': int(left)
                        }
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in face prediction: {str(e)}")
            return []

# Initialize the predictor
try:
    predictor = FaceRecognitionPredictor()
except:
    predictor = None
    print("Warning: Could not load face recognition model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Face recognition model not loaded'
            }), 500
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file selected'
            }), 400
        
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Get tolerance from request (optional)
        tolerance = float(request.form.get('tolerance', 0.6))
        
        # Predict faces
        predictions = predictor.predict_face(image_array, tolerance=tolerance)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_faces': len(predictions)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Alternative endpoint for base64 encoded images."""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Face recognition model not loaded'
            }), 500
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No base64 image provided'
            }), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Get tolerance from request
        tolerance = float(data.get('tolerance', 0.6))
        
        # Predict faces
        predictions = predictor.predict_face(image_array, tolerance=tolerance)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_faces': len(predictions)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)