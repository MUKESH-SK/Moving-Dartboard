import cv2
import numpy as np

# Initialize video capture and create a blank image for drawing the trajectory
cap = cv2.VideoCapture(0)
trajectory = np.zeros((480, 640, 3), dtype=np.uint8)

# Set up the HSV color range for the dart
lower_red = np.array([20, 100, 100])
upper_red = np.array([30, 255, 255])

# Initialize variables for tracking the dart's position
positions = []
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
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            positions.append((int(x), int(y)))
            print(f"{(x,y)}")

    # Draw the trajectory on the blank image
    for i in range(1, len(positions)):
        if positions[i - 1] is None or positions[i] is None:
            continue
        # cv2.line(trajectory, positions[i - 1], positions[i], (0, 255, 0), 2)
        #cv2.circle(trajectory, positions[i - 1], positions[i], (0, 255, 0), 2)
        cv2.circle(trajectory, positions[i - 1], 1, (0, 255, 0), 2)
        # cv2.circle(trajectory, positions[i], 1, (0, 255, 0), 2)

    # Display the resulting frame and trajectory
    cv2.imshow('Frame', frame)
    cv2.imshow('Trajectory', trajectory)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
