import cv2
import numpy as np

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap1.set(cv2.CAP_PROP_FPS, 60)
cap2.set(cv2.CAP_PROP_FPS, 60)

lower_red1 = np.array([14, 75, 119])
upper_red1 = np.array([74, 255, 255])

lower_red2 = np.array([14, 75, 119])
upper_red2 = np.array([74, 255, 255])

positionsX1, positionsY1 = [], []

positionsX2, positionsY2 = [], []

x = 0
y = 0

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv1, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv2, lower_red2, upper_red2)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    kk = 0
    for contour in contours1:
        ((xp, yp), radius) = cv2.minEnclosingCircle(contour)
        if radius > 5:
            positionsX1.append(int(xp))
            positionsY1.append(int(yp))
        if len(positionsX1) >= 2:
            if positionsX1[len(positionsX1)-1] < positionsX1[len(positionsX1)-2]:
                a, b, c = np.polyfit(positionsX1, positionsY1, 2)
                xm = 640
                y = int(a * xm ** 2 + b * xm + c)
            else:
                # kk = 1
                break

    for contour in contours2:
        ((xp1, yp1), radius) = cv2.minEnclosingCircle(contour)
        if radius > 3:
            positionsX2.append(int(xp1))
            positionsY2.append(int(yp1))
        if len(positionsX2) >=2:
            if positionsX2[len(positionsX2) - 1] < positionsX2[len(positionsX2) - 2]:
                m, c = np.polyfit(positionsX2, positionsY2, 1)
                x = -(int(c/m))
            else:
                # kk = 1
                break

    if kk:
        break
    cv2.imshow('Frame_x', frame2)

    cv2.imshow('Frame_y', frame1)

    frameCircle = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frameCircle, (x, y), 10, (255, 0, 0), 4)
    cv2.imshow("circle", frameCircle)
    print(x,y)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()