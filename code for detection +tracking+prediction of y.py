import cv2
import numpy as np

# Initialize video capture and create a blank image for drawing the trajectory
cap = cv2.VideoCapture(0)
trajectory = np.zeros((480, 640, 3), dtype=np.uint8)  # 480 is along y direction and 640 is along x direction

# Set up the HSV color range for the dart
lower_red = np.array([20, 100, 100])
upper_red = np.array([30, 255, 255])

# lower_red = np.array((128, 0, 128))
# upper_red = np.array((255, 128, 255))

# Initialize variables for tracking the dart's position
positions = []
positionsX,positionsY = [],[]
xList = [item for item in range(0,640)]
frame_count = 0

# Loop through each frame of the video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_count += 1

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask the frame to only select red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Iterate through the contours and draw a bounding circle around the dart
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 5:
            # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            positions.append((int(x), int(y)))
            positionsX.append(int(x))
            positionsY.append(int(y))
        if positionsX:
            # polynomial regression y = ax^2 +bx+c
            # find the coefficients
            a, b, c = np.polyfit(positionsX,positionsY, 2)

            for i,(posX,posY) in enumerate(zip(positionsX,positionsY)):
                pos = (posX,posY)
                cv2.circle(frame, pos, 6, (0, 255, 0), cv2.FILLED)
                cv2.circle(trajectory, pos, 3, (0, 255, 0), cv2.FILLED)

                if i == 0:
                    cv2.line(frame, pos, pos, (0, 255, 0), 2)
                    cv2.line(trajectory, pos, pos, (0, 255, 0), 2)

                else:
                    cv2.line(frame,pos,(positionsX[i-1],positionsY[i-1]),(0,255,0),2)
                    cv2.line(trajectory,pos,(positionsX[i-1],positionsY[i-1]),(0,255,0),2)

            for x in xList:
                y = int(a*x**2 + b*x + c)
                cv2.circle(frame, (x,y),3,(255,0,255),cv2.FILLED)
                if(x<=680):
                    print(y)

    # Display the resulting frame and trajectory
    cv2.imshow('Frame', frame)
    cv2.imshow('Trajectory', trajectory)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
