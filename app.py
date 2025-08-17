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
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

class RealTimeFaceRecognitionPredictor:
    def __init__(self, model_path='face_recognition_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.face_encodings = None
        self.face_names = None
        
        # Real-time processing parameters
        self.is_processing = False
        self.current_frame = None
        self.latest_predictions = []
        self.frame_lock = threading.Lock()
        
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
            
            print("✅ Model loaded successfully!")
            print(f"📊 Known faces: {len(self.face_names) if self.face_names else 0}")
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}")
            self.face_encodings = []
            self.face_names = []
            print("⚠️ Running without trained model - will only detect unknown faces")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.face_encodings = []
            self.face_names = []
    
    def predict_face(self, image_array, tolerance=0.6, min_face_size=80):
        """
        Predict faces in the given image.
        
        Args:
            image_array: numpy array of the image
            tolerance: face matching tolerance (lower is more strict)
            min_face_size: minimum face size in pixels
        
        Returns:
            list of dictionaries with face predictions and locations
        """
        try:
            # Convert image to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Check if it's BGR (OpenCV format)
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_array
            
            # Resize image if it's too large (for performance)
            height, width = rgb_image.shape[:2]
            max_dimension = 800
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height))
                scale_factor = 1 / scale
            else:
                scale_factor = 1
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            results = []
            
            for (face_encoding, face_location) in zip(face_encodings, face_locations):
                top, right, bottom, left = face_location
                
                # Scale coordinates back to original image size
                top = int(top * scale_factor)
                right = int(right * scale_factor)
                bottom = int(bottom * scale_factor)
                left = int(left * scale_factor)
                
                # Check minimum face size
                face_width = right - left
                face_height = bottom - top
                if face_width < min_face_size or face_height < min_face_size:
                    continue
                
                name = "Unknown"
                confidence = 0.0
                
                # Method 1: Using known face encodings (recommended)
                if self.face_encodings and len(self.face_encodings) > 0:
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
                
                # Method 2: Using trained classifier (alternative approach)
                elif self.model:
                    try:
                        # Reshape face encoding for model prediction
                        face_encoding_reshaped = face_encoding.reshape(1, -1)
                        prediction = self.model.predict(face_encoding_reshaped)
                        probability = self.model.predict_proba(face_encoding_reshaped)
                        
                        # Get the predicted name
                        predicted_label = prediction[0]
                        name = self.label_encoder.inverse_transform([predicted_label])[0]
                        confidence = np.max(probability)
                    except Exception as e:
                        print(f"Model prediction error: {e}")
                        name = "Unknown"
                        confidence = 0.0
                
                else:
                    # No model available, just detect faces
                    name = "Unknown"
                    confidence = 0.5  # Default confidence for face detection
                
                # Only add result if confidence meets threshold
                confidence_threshold = 0.4 if name == "Unknown" else 0.5
                if confidence >= confidence_threshold:
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

    def start_realtime_processing(self):
        """Start real-time face processing (if needed for webcam stream)."""
        self.is_processing = True
        
    def stop_realtime_processing(self):
        """Stop real-time face processing."""
        self.is_processing = False

# Initialize the predictor
try:
    predictor = RealTimeFaceRecognitionPredictor()
except Exception as e:
    predictor = None
    print(f"Warning: Could not initialize face recognition: {e}")

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle face prediction requests from uploaded images or camera frames."""
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Face recognition system not available'
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
        
        # Get parameters from request
        tolerance = float(request.form.get('tolerance', 0.6))
        min_size = int(request.form.get('min_size', 80))
        
        # Predict faces
        predictions = predictor.predict_face(
            image_array, 
            tolerance=tolerance,
            min_face_size=min_size
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_faces': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
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
                'error': 'Face recognition system not available'
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
        
        # Get parameters from request
        tolerance = float(data.get('tolerance', 0.6))
        min_size = int(data.get('min_size', 80))
        
        # Predict faces
        predictions = predictor.predict_face(
            image_array,
            tolerance=tolerance,
            min_face_size=min_size
        )
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_faces': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Base64 prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'known_faces': len(predictor.face_names) if predictor and predictor.face_names else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/system_info', methods=['GET'])
def system_info():
    """Get system information for debugging."""
    info = {
        'opencv_version': cv2.__version__,
        'face_recognition_available': True,
        'model_status': 'loaded' if predictor else 'not_loaded',
        'known_faces': len(predictor.face_names) if predictor and predictor.face_names else 0
    }
    
    try:
        import face_recognition
        info['face_recognition_version'] = face_recognition.__version__
    except:
        info['face_recognition_available'] = False
    
    return jsonify(info)

@app.route('/start_realtime', methods=['POST'])
def start_realtime():
    """Start real-time processing (for future webcam streaming features)."""
    if predictor:
        predictor.start_realtime_processing()
        return jsonify({'success': True, 'message': 'Real-time processing started'})
    else:
        return jsonify({'success': False, 'error': 'Predictor not available'})

@app.route('/stop_realtime', methods=['POST'])
def stop_realtime():
    """Stop real-time processing."""
    if predictor:
        predictor.stop_realtime_processing()
        return jsonify({'success': True, 'message': 'Real-time processing stopped'})
    else:
        return jsonify({'success': False, 'error': 'Predictor not available'})

def create_templates_directory():
    """Create templates directory if it doesn't exist."""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"📁 Created templates directory: {templates_dir}")
    return templates_dir

def check_model_file():
    """Check if model file exists and provide helpful information."""
    model_path = 'face_recognition_model.pkl'
    if not os.path.exists(model_path):
        print("⚠️ " + "="*60)
        print("   MODEL FILE NOT FOUND")
        print("="*60)
        print(f"   Looking for: {model_path}")
        print("   The application will work for face detection only.")
        print("   To enable face recognition, you need to:")
        print("   1. Train a model using your face recognition training script")
        print("   2. Save it as 'face_recognition_model.pkl'")
        print("   3. Restart this application")
        print("="*60)
    else:
        print(f"✅ Model file found: {model_path}")

if __name__ == '__main__':
    # Setup
    create_templates_directory()
    check_model_file()
    
    print("\n" + "="*60)
    print("🎭 REAL-TIME FACE RECOGNITION SERVER")
    print("="*60)
    print("🌐 Server will start at: http://localhost:5000")
    print("📁 Make sure index.html is in the templates/ folder")
    print("🚀 Starting server...")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True  # Enable threading for better performance
    )