# Importing all modules
import cv2
import numpy as np

# defining a function
def find_parabola_variables(x1, y1, x2, y2, x3, y3):
  # Create a matrix of the coefficients
  A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [x3**2, x3, 1]])

  # Create a vector of the y-values
  y = np.array([y1, y2, y3])

  # Solve the system of equations using the least-squares method
  a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]

  # Return the equation of the parabola in standard form
  # return f"y = {a}x^2 + {b}x + {c}"
  return a,b,c

# creating empty list to  store the x, y coordinates detected
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
                if i >= 2:
                    # parabola_equation = find_parabola(x_list[i-2], y_list[i-2], x_list[i-1], y_list[i-1], x_list[i], y_list[i])
                    a,b,c = find_parabola_variables(x_list[i-2], y_list[i-2], x_list[i-1], y_list[i-1], x_list[i], y_list[i])
                    # lets assume our camera is fitted 5m from the person who is throwing
                    x_board = 5
                    y_new_pos =25*a + 5*b + c
                    print(y_new_pos)

    cv2.imshow("mask image", mask) # Displaying mask image

    cv2.imshow("window", video) # Displaying webcam image

    i += 1

    cv2.waitKey(1)