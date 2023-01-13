import cv2
import numpy as np

# Initialize video capture and create a blank image for drawing the trajectory
cap1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FPS, 60)
cap2.set(cv2.CAP_PROP_FPS, 60)

# dartboard = cv2.imread("dartboard.png")

trajectory1 = np.zeros((480, 640, 3), dtype=np.uint8)  # 480 is along y direction and 640 is along x direction
trajectory2 = np.zeros((480, 640, 3), dtype=np.uint8)  # 480 is along y direction and 640 is along x direction
dartboard_projection = np.zeros((600,600, 3), dtype=np.uint8)

# Set up the HSV color range for the dart

lower_red1 = np.array([0, 245, 0])
upper_red1 = np.array([0, 255, 0])

lower_red2 = np.array([0, 245, 0])
upper_red2 = np.array([0, 255, 0])

# Initialize variables for tracking the dart's position
positions1 = []
positionsX1,positionsY1 = [],[]
yList = [item for item in range(0,640)]
frame_count1 = 0

positions2 = []
positionsX2,positionsY2 = [],[]
xList = [item for item in range(0,640)]
frame_count2 = 0

# resized_dartboar = cv2.resize(dartboard, (200, 200))

x_board, y_board = 300,300
# cv2.imshow("Resized", resized_dartboar)

# Loop through each frame of the video
while True:
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    # ret2, frame2 = cap2.read()

    frame_count1 += 1

    # Convert the frame to HSV color space
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    # Mask the frame to only select red colors
    mask1 = cv2.inRange(hsv1, lower_red1, upper_red1)

    # Find contours in the mask
    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Iterate through the contours and draw a bounding circle around the dart
    for contour in contours1:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 5:
            # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            positions1.append((int(x), int(y)))
            positionsX1.append(int(x))
            positionsY1.append(int(y))
        if len(positionsX1) >= 5:
            # polynomial regression y = ax^2 +bx+c
            # find the coefficients
            a, b, c = np.polyfit(positionsX1,positionsY1, 2)

            for i,(posX,posY) in enumerate(zip(positionsX1,positionsY1)):
                pos1 = (posX,posY)
                cv2.circle(frame1, pos1, 6, (0, 255, 0), cv2.FILLED)
                cv2.circle(trajectory1, pos1, 3, (0, 255, 0), cv2.FILLED)

                if i == 0:
                    cv2.line(frame1, pos1, pos1, (0, 255, 0), 2)
                    cv2.line(trajectory1, pos1, pos1, (0, 255, 0), 2)

                else:
                    cv2.line(frame1,pos1,(positionsX1[i-1],positionsY1[i-1]),(0,255,0),2)
                    cv2.line(trajectory1,pos1,(positionsX1[i-1],positionsY1[i-1]),(0,255,0),2)

            for x in yList:
                y = int(a*x**2 + b*x + c)
                cv2.circle(frame1, (x,y),3,(255,0,255),cv2.FILLED)
                if(x<=680):
                    print(y)

    # Display the resulting frame and trajectory
    cv2.imshow('Frame_y', frame1)
    cv2.imshow('Trajectory1', trajectory1)



    # 2nd frame

    ret2, frame2 = cap2.read()
    frame_count2 += 1

    # Convert the frame to HSV color space
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # Mask the frame to only select red colors
    mask2 = cv2.inRange(hsv2, lower_red2, upper_red2)

    # Find contours in the mask
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Iterate through the contours and draw a bounding circle around the dart
    for contour in contours2:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 5:
            # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            positions2.append((int(x), int(y)))
            positionsX2.append(int(x))
            positionsY2.append(int(y))
        if positionsX2:
            # linear regression y = m*x + c
            # find the coefficients
            m,c = np.polyfit(positionsX2,positionsY2, 1)

            for i,(posX,posY) in enumerate(zip(positionsX2,positionsY2)):
                pos2 = (posX,posY)
                cv2.circle(frame2, pos2, 6, (0, 255, 0), cv2.FILLED)
                cv2.circle(trajectory2, pos2, 3, (0, 255, 0), cv2.FILLED)

                if i == 0:
                    cv2.line(frame2, pos2, pos2, (0, 255, 0), 2)
                    cv2.line(trajectory2, pos2, pos2, (0, 255, 0), 2)

                else:
                    cv2.line(frame2,pos2,(positionsX2[i-1],positionsY2[i-1]),(0,255,0),2)
                    cv2.line(trajectory2,pos2,(positionsX2[i-1],positionsY2[i-1]),(0,255,0),2)

            for x in xList:
                y = int(m*x + c)
                cv2.circle(frame2, (x,y),3,(255,0,255),cv2.FILLED)
                if(x<=680):
                    print(y)

    # Display the resulting frame and trajectory
    cv2.imshow('Frame_x', frame2)
    # cv2.imshow('Dartboard_projection', dartboard_projection )
    cv2.imshow('Trajectory2', trajectory2)


    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and destroy all windows
cap1.release()
cv2.destroyAllWindows()
