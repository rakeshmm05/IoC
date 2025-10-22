import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

# Define the color to track (example: red)
lower_red = np.array([0, 120, 70])      # lower HSV bound
upper_red = np.array([10, 255, 255])    # upper HSV bound

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours of the color area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 15:
            # Draw circle and center point
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3)
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"({int(x)}, {int(y)})", (int(x)-40, int(y)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show results
    cv2.imshow('Color Tracker', frame)
    cv2.imshow('Mask', mask)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
