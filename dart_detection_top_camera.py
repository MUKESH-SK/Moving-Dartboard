# Importing all modules
import cv2
import numpy as np
# defining function for finding variables of linear equation
def calc_linear_eq(x1, y1, x2, y2):
  m = (y2 - y1) / (x2 - x1)
  b = y1 - m * x1
  return m, b
# defining function for finding the final coordinate
def predict_location(x, m, b):
  y = m * x + b
  return y


x_list = []
y_list = []
i=0

# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array([15, 150, 20])
upper = np.array([35, 255, 255]) # (These ranges will detect Yellow)

# lower = np.array([0,70,50])
# upper = np.array([10,255,255])

# Capturing webcam footage
webcam_video = cv2.VideoCapture(0)

while True:
    success, video = webcam_video.read() # Reading webcam footage

    img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format

    mask = cv2.inRange(img, lower, upper) # Masking the image to find our color

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    # Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                X = (x + w) / 2
                Y = (y + h) / 2
                # print(f"{(X,Y)}")
                x_list.append(X)
                y_list.append(Y)
                if i >= 1:
                    # parabola_equation = find_parabola(x_list[i-2], y_list[i-2], x_list[i-1], y_list[i-1], x_list[i], y_list[i])
                    m,b = calc_linear_eq(x_list[i-1], y_list[i-1], x_list[i], y_list[i])
                    # lets assume our camera is fitted 5m from the person who is throwing
                    x_board = 5

                    y_new_pos = predict_location(x_board, m, b)
                    print(y_new_pos)

    cv2.imshow("mask image", mask) # Displaying mask image

    cv2.imshow("window", video) # Displaying webcam image
    i += 1

    cv2.waitKey(1)