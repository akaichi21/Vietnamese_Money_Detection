# Collect images of Vietnamese money

# Libraries
import cv2
import numpy as np
import time
import os

# Label (about denominations of Vietnamese money)
label = "500000"

# Open camera for collecting
cap = cv2.VideoCapture(0)

i = 0
while(True):
    # Capture frame-by-frame
    i = i + 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4)

    # Show camera
    cv2.imshow("Capture Vietnamese Money", frame)

    # Save data
    if ((i>=60) and (i<=1060)):
        print("Number of Captured Image: ", i-60)

        # Create folder if not exists
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))

        # Write data into folder
        cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything finish, release the capture
cap.release()
cv2.destroyAllWindows()
