from flask import Flask, jsonify, request
import subprocess
import time
from threading import Thread
# from flask_cors import CORS
# CORS(app)
app = Flask(__name__)

# Global variable for managing the process
process = None
start_time = None
running = False


def run_recognition_script():
    """Function to run the gesture recognition script."""
    global process
    process = subprocess.Popen(['python', 'main.py'])
    process.wait()


@app.route('/start', methods=['POST'])
def start_recognition():
    """Start the gesture recognition process."""
    global process, start_time, running
    if process is None or process.poll() is not None:
        # Run the recognition script in a separate thread so it doesn't block the server
        recognition_thread = Thread(target=run_recognition_script)
        recognition_thread.start()
        start_time = time.time()
        running = True
        return jsonify({"status": "success", "message": "Recognition started"})
    else:
        return jsonify({"status": "error", "message": "Recognition is already running"})


@app.route('/stop', methods=['POST'])
def stop_recognition():
    """Stop the gesture recognition process."""
    global process, running
    if process and process.poll() is None:
        process.terminate()
        process.wait()
        process = None
        running = False
        return jsonify({"status": "success", "message": "Recognition stopped"})
    else:
        return jsonify({"status": "error", "message": "No running script to stop"})


# WebSocket for real-time communication
from flask_socketio import SocketIO, emit
from flask import render_template

# Set up SocketIO for real-time communication with frontend
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')


# Example function to emit gestures to the frontend
@socketio.on('send_gesture')
def handle_gesture(data):
    """Handle gesture data sent from the backend."""
    emit('new_gesture', data, broadcast=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)