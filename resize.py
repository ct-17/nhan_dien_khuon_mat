import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Duong dan den video.", required=True)
args = vars(ap.parse_args())
cap = cv2.VideoCapture(args["video"])

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

while cap.isOpened():
    ref, frame = cap.read()
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()