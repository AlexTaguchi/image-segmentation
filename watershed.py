# Real-time watershed segmentation with a web camera

# Import modules
import cv2
import numpy as np

# Randomize color pallete
colors = np.random.randint(256, size=(100, 3))

# Start camera capture
capture = cv2.VideoCapture(0)
while(True):

    # Capture mirror image video frame
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)

    # Create uniformly spaced grid of 100 markers
    marker = 0
    markers = np.zeros(frame.shape[:2], dtype=np.int32)
    for y in range(frame.shape[0] // 20, frame.shape[0], frame.shape[0] // 10):
        for x in range(frame.shape[1] // 20, frame.shape[1], frame.shape[1] // 10):
            markers[y, x] = marker
            marker += 1

    # Apply watershed algorithm
    markers = cv2.watershed(frame, markers)
    for i in range(marker):
        frame[markers == i] = colors[i]

    # Display frame with overlay
    cv2.imshow('frame', frame)

    # Exit with q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera capture
capture.release()
cv2.destroyAllWindows()
