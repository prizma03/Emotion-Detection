import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
# Load the model from the .keras file
model = load_model("emo_model.keras")
# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load emoji images (Ensure these files exist in the "emojis" folder)
emoji_map = { "Angry": cv2.imread(r"C:\Users\prizm\Downloads\emojii\angry.png", cv2.IMREAD_UNCHANGED),
    "Disgusted": cv2.imread(r"C:\Users\prizm\Downloads\emojii\disgusted.png", cv2.IMREAD_UNCHANGED),
    "Fearful": cv2.imread(r"C:\Users\prizm\Downloads\emojii\fearful.png", cv2.IMREAD_UNCHANGED),
    "Happy": cv2.imread(r"C:\Users\prizm\Downloads\emojii\happy.png", cv2.IMREAD_UNCHANGED),
    "Neutral": cv2.imread(r"C:\Users\prizm\Downloads\emojii\neutral.png", cv2.IMREAD_UNCHANGED),
    "Sad": cv2.imread(r"C:\Users\prizm\Downloads\emojii\sad.png", cv2.IMREAD_UNCHANGED),
    "Surprised": cv2.imread(r"C:\Users\prizm\Downloads\emojii\surpriced.png", cv2.IMREAD_UNCHANGED),
}
# Start webcam
cap = cv2.VideoCapture(0)

def overlay_emoji_fixed_position(background, emoji):
    """ Place emoji in a fixed position (top-right corner). """
    emoji_size = 200  # Increase emoji size
    emoji_resized = cv2.resize(emoji, (emoji_size, emoji_size))

    # Fixed position: Top-right corner
    emoji_x = background.shape[1] - emoji_size - 20
    emoji_y = 20  # 20 pixels from the top

    # Extract RGBA channels
    b, g, r, a = cv2.split(emoji_resized)
    mask = a / 255.0
    inv_mask = 1.0 - mask

    # Blend emoji with background
    for c in range(3):  # Loop over BGR channels
        background[emoji_y:emoji_y + emoji_size, emoji_x:emoji_x + emoji_size, c] = (
            mask * emoji_resized[:, :, c] + inv_mask * background[emoji_y:emoji_y + emoji_size, emoji_x:emoji_x + emoji_size, c]
        )

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    detected_emotion = None  # Store the detected emotion

    for (x, y, w, h) in faces:
        # Extract face region
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict emotion
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        detected_emotion = emotion_dict[maxindex]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # Display detected emotion text
        cv2.putText(frame, detected_emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Overlay fixed-position emoji based on detected emotion
    if detected_emotion and detected_emotion in emoji_map:
        overlay_emoji_fixed_position(frame, emoji_map[detected_emotion])

    # Show output
    cv2.imshow('Face Emotion Recognition with Fixed Emoji', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()