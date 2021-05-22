import cv2
import numpy as np
import mediapipe as mp
import hand_tracking_module_advance as htm
import time
import autopy

wcam, hcam =  640, 480
frameR = 25 #frame reduction
smoothing = 5

pTime = 0
plocx, plocy = 0,0
clocx, clocy = 0,0

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)
while True:
    #1. Find Landmrks
    success, img = cap.read()
    img= detector.find_hands(img)
    lmList, bbox = detector.find_position(img)
    # 2. get the tip of the index and middle fingers
    if len(lmList) !=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
    #3. check with ingers are up
        fingers = detector.fingers_up()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR), (255, 0, 255), 2)
    # 4. only index finger : moving mode
        if fingers[1] == 1 and fingers[2] == 0:
    # 5. convert coorinates
            x3 = np.interp(x1, (frameR,wcam - frameR), (0, wScr))
            y3 = np.interp(x1, (frameR, hcam - frameR), (0, hScr))

    # 6. smoothen values
            clocx = plocx + (x3 - plocx) / smoothing
            clocy = plocy + (y3 - plocy) / smoothing
    # 7. move mouse
            autopy.mouse.move(wScr - clocx,clocy)
            cv2.circle(img,(x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocx, plocy = clocx, clocy
    # 8. both index and moddle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:

    # 9. Checking the distance between two fingers
            length,img,lineInfo = detector.find_distance(8,12, img)
            print(length)
    # 10. Mouse click
            if length < 45:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                autopy.mouse.click()
    # 11. frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

    #12. display
    cv2.imshow("image",img)
    cv2.waitKey(1)
