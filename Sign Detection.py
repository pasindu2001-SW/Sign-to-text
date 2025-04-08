import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# Load trained model
MODEL_PATH = "sinhala_sign_subset_full_Greetings.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constants
N_FRAMES = 20
FEATURE_DIM = 132
class_names = ["Alright", "Ayubowan", "Hello", "How are you", "Thank you"]  # Update with actual class names

# Buffer to store frames
frame_buffer = deque(maxlen=N_FRAMES)
prev_keypoints = np.zeros(FEATURE_DIM)

def extract_keypoints(results, prev_keypoints):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] 
                     if results.pose_landmarks else np.zeros((33, 3))).flatten()
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] 
                   if results.left_hand_landmarks else np.zeros((21, 3))).flatten()
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] 
                   if results.right_hand_landmarks else np.zeros((21, 3))).flatten()
    
    keypoints = np.concatenate([pose, lh, rh])[:FEATURE_DIM]
    
    # If all zeros, reuse previous keypoints
    if np.all(keypoints == 0) and prev_keypoints is not None:
        return prev_keypoints
    return keypoints

# OpenCV Capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Extract keypoints
    keypoints = extract_keypoints(results, prev_keypoints)
    prev_keypoints = keypoints  # Store for next frame
    
    # Debugging: Check if MediaPipe detects hands/pose
    print("Pose detected:", results.pose_landmarks is not None)
    print("Left hand detected:", results.left_hand_landmarks is not None)
    print("Right hand detected:", results.right_hand_landmarks is not None)
    print("Extracted Keypoints:", keypoints)
    
    frame_buffer.append(keypoints)
    
    # Predict if buffer is full
    if len(frame_buffer) == N_FRAMES:
        input_data = np.array(frame_buffer).reshape(1, N_FRAMES, FEATURE_DIM)
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        print("Predictions:", predictions)
        print("Predicted Class:", predicted_class, "Confidence:", confidence)
        
        # Display prediction
        label = f"{class_names[predicted_class]} ({confidence:.2f})"
        cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show image
    cv2.imshow('Real-Time Action Detection', image)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

 




