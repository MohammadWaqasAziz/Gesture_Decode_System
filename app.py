import sys
import io
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from flask import Flask, render_template, request, redirect, url_for, Response

# Ensure UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model (replace with your model path)
model = load_model('model/Final2.h5')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(['Abdomen', 'Ankle', 'Back', 'Blood', 'Body', 'Brain', 'Elbow', 'Fist',
                    'Heart', 'Jaw', 'Knuckle', 'Lips', 'Lungs', 'Mouth', 'Nerve', 'Nose',
                    'Palm', 'Skull', 'Thumb'])

def mediapipe_detection(image, holistic):
    """Performs MediaPipe pose estimation on a given image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, result):
    # Define drawing specifications
    face_landmark_style = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
    face_connection_style = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    
    hand_landmark_style = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2)
    hand_connection_style = mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    
    pose_landmark_style = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
    pose_connection_style = mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    
    # Draw face landmarks and connections
    mp_drawing.draw_landmarks(
        image, 
        result.face_landmarks, 
        mp_holistic.FACEMESH_CONTOURS, 
        landmark_drawing_spec=face_landmark_style, 
        connection_drawing_spec=face_connection_style
    )
    
    # Draw right hand landmarks and connections
    mp_drawing.draw_landmarks(
        image, 
        result.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS, 
        landmark_drawing_spec=hand_landmark_style, 
        connection_drawing_spec=hand_connection_style
    )
    
    # Draw left hand landmarks and connections
    mp_drawing.draw_landmarks(
        image, 
        result.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS, 
        landmark_drawing_spec=hand_landmark_style, 
        connection_drawing_spec=hand_connection_style
    )
    
    # Draw pose landmarks and connections
    mp_drawing.draw_landmarks(
        image, 
        result.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS, 
        landmark_drawing_spec=pose_landmark_style, 
        connection_drawing_spec=pose_connection_style
    )

def extract_keypoints(results):
    """Extracts keypoints from MediaPipe pose estimation results."""
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return np.concatenate([pose, face, left_hand, right_hand])

def test_video(video):
    sequence, sentence, predictions = [], [], []
    threshold = 0.5

    cap = cv2.VideoCapture(video)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = (frame_count / fps / 35) * 1000
    count = 0
    video_with_landmarks = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, count * duration)
            success, frame = cap.read()
            count += 1
            if not success:
                break

            frame, results = mediapipe_detection(frame, holistic)
            draw_landmarks(frame, results)
            video_with_landmarks.append(frame)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                    if not sentence or actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

    cap.release()
    return ' '.join(sentence) if sentence else "No action detected", video_with_landmarks

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)  # No video uploaded, redirect back

        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)  # Empty filename, redirect back

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process video for predictions
            predicted_action, video_with_landmarks = test_video(filepath)

            return render_template('result.html', action=predicted_action, video_path=filepath, video_with_landmarks=video_with_landmarks)

    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
        return

    sequence = []
    threshold = 30

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            try:
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-threshold:]

                if len(sequence) == threshold:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])  # Optional: Print action to console

                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    print("Error: Failed to encode image.")
                    continue

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                print(f"Error: {e}")
                continue

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/real_time', methods=['GET', 'POST'])
def real_time():
    return render_template('real_time.html')

@app.route('/process_real_time', methods=['POST'])
def process_real_time():
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'real_time_output.mp4')
    predicted_action, video_with_landmarks = test_video(video_path)
    return render_template('result.html', action=predicted_action, video_path=video_path, video_with_landmarks=video_with_landmarks)

if __name__ == '__main__':
    app.run(debug=True)