# Test model from new image to classify Vietnamese money

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from keras._tf_keras.keras.preprocessing.image import img_to_array

model = load_model("weights-13-1.00.keras")

class_names = ["0", "1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000", "500000"]

# Open camera
cap = cv2.VideoCapture(0)

print("Press 'q' to quit, 's' to predict.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can not open Camera.")
        break

    # Show frame on Camera
    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 's' to scan and predict", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Camera", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Resize image ~ (128, 128)
        img = cv2.resize(frame, (128, 128))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Predict
        preds = model.predict(img)
        pred_class = np.argmax(preds)
        confidence = np.max(preds)

        # Show result
        label = f"{class_names[pred_class]} ({confidence*100:.2f}%)"
        print("Predict:", label)

        result_frame = cv2.resize(frame, (400, 400))
        cv2.putText(result_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        cv2.imshow("Prediction", result_frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Prediction")

# When everything finish, release the capture
cap.release()
cv2.destroyAllWindows()