import time
import cv2
import numpy as np
import mediapipe as mp
import hand_tracking_module as htm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img)
    if len(lmList) != 0:
        print(lmList[1])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(1)