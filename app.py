import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response


app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("class.keras")

# Define emotion labels based on your model's training classes
emotion_labels = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load emojis
emoji_map = {
    "Angry": cv2.imread("emoji/angry.png", cv2.IMREAD_UNCHANGED),
    "Disgusted": cv2.imread("emoji/disgusted.png", cv2.IMREAD_UNCHANGED),
    "Fearful": cv2.imread("emoji/fearful.png", cv2.IMREAD_UNCHANGED),
    "Happy": cv2.imread("emoji/happy.png", cv2.IMREAD_UNCHANGED),
    "Neutral": cv2.imread("emoji/neutral.png", cv2.IMREAD_UNCHANGED),
    "Sad": cv2.imread("emoji/sad.png", cv2.IMREAD_UNCHANGED),
    "Surprised": cv2.imread("emoji/surprised.png", cv2.IMREAD_UNCHANGED),  
}

# Open webcam
camera = cv2.VideoCapture(0)

def overlay_emoji_top_right(frame, emoji):
    """Overlay the emoji in the top-right corner of the frame."""
    emoji_size = 100  # Set fixed size for emoji
    emoji = cv2.resize(emoji, (emoji_size, emoji_size))

    # Position emoji at the top-right corner
    x_offset = frame.shape[1] - emoji_size - 10  # 10px from the right
    y_offset = 10  # 10px from the top

    if emoji.shape[2] == 4:  # If emoji has transparency (alpha channel)
        alpha_channel = emoji[:, :, 3] / 255.0
        for c in range(3):  # Apply transparency to RGB channels
            frame[y_offset:y_offset+emoji_size, x_offset:x_offset+emoji_size, c] = (
                (1 - alpha_channel) * frame[y_offset:y_offset+emoji_size, x_offset:x_offset+emoji_size, c] +
                alpha_channel * emoji[:, :, c]
            )
    else:
        frame[y_offset:y_offset+emoji_size, x_offset:x_offset+emoji_size] = emoji

    return frame

def detect_emotion(frame):
    """Detect faces, predict emotion, and show corresponding emoji in the top-right corner."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detected_emotion = "Neutral"  # Default emotion if no face is detected

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize for model
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension
        roi_gray = roi_gray / 255.0  # Normalize pixel values

        # Predict emotion
        predictions = model.predict(roi_gray)
        emotion_index = np.argmax(predictions)
        detected_emotion = emotion_labels[emotion_index]

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Overlay emoji in the top-right corner
    emoji = emoji_map.get(detected_emotion)
    if emoji is not None:
        frame = overlay_emoji_top_right(frame, emoji)

    return frame

def generate_frames():
    """Generate frames for the video feed with emotion detection and emoji overlay."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/open_camera.html")
def open_camera():
    return render_template("open_camera.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
