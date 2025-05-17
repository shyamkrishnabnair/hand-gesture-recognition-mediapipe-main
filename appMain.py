from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from gesture_model import process_frame  # Make sure spelling is correct here!

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data received'}), 400

        # Decode base64 image
        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Process the frame with gesture recognition
        results = process_frame(frame)

        return jsonify({
            "gesture": results.get("gesture", "Unknown"),
            "hand_sign": results.get("hand_sign", "Unknown")
        })

    except Exception as e:
        import traceback
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
