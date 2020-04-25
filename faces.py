# Real-time face segmentation with a web camera

# Import modules
import cv2
import dlib
import numpy as np

# Import dlib face alignment file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define facial landmarks
landmarks = {'jawline': list(range(0, 17)),
             'right_eyebrow': list(range(17, 22)),
             'left_eyebrow': list(range(22, 27)),
             'nose': list(range(27, 36)),
             'right_eye': list(range(36, 42)),
             'left_eye': list(range(42, 48)),
             'outer_mouth': list(range(48, 60)),
             'inner_mouth': list(range(60, 68))}

# Start camera capture
videoCapture = cv2.VideoCapture(0)
while True:

    # Capture mirror image video frame
    ret, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect dlib face rectangles
    factor = 1
    gray = cv2.resize(gray, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_LINEAR)
    rectangles = detector(gray, 0)

    # Track face features if bounding box detected
    if rectangles:

        # Face shape prediction
        shape = predictor(gray, rectangles[0])
        coordinates = np.zeros((shape.num_parts, 2), dtype='int')
        for x in range(0, shape.num_parts):
            coordinates[x] = (shape.part(x).x, shape.part(x).y)
        shape = factor * coordinates

        # Forehead top and side anchors
        forehead_rt = 2 * (shape[19] - shape[36]) + shape[19]
        forehead_lt = 2 * (shape[24] - shape[45]) + shape[24]
        forehead_rs = 2 * (shape[19] - shape[36]) + shape[0]
        forehead_ls = 2 * (shape[24] - shape[45]) + shape[16]

        # Forehead anchor midpoints
        midpoint_r = [0.25 * (forehead_rt[0] - forehead_rs[0]) + forehead_rs[0],
                      0.75 * (forehead_rt[1] - forehead_rs[1]) + forehead_rs[1]]
        midpoint_l = [0.25 * (forehead_lt[0] - forehead_ls[0]) + forehead_ls[0],
                      0.75 * (forehead_lt[1] - forehead_ls[1]) + forehead_ls[1]]

        # Add forehead anchor points
        shape = np.vstack((shape, forehead_rt, forehead_lt, forehead_rs,
                           forehead_ls, midpoint_r, midpoint_l)).astype(np.int)

        # Generate face mask
        face_mask = np.zeros(frame.shape[:2])
        cv2.fillConvexPoly(face_mask, cv2.convexHull(shape), 1)
        face_mask = face_mask.astype(np.bool)
        
        # Overlay face in blue
        faces = np.array(frame, copy=True, dtype=float)
        faces[~face_mask] = 0
        faces[face_mask] *= 0.66
        faces[face_mask, 0] += 255 / 3
        frame[face_mask] = 0
        frame = frame + faces.astype('uint8')

        # Draw landmarks
        for feature, points in landmarks.items():
            if feature == 'nose':
                points += [points[3]]
            elif feature == 'jawline':
                pass
            else:
                points += [points[0]]
            for i in range(len(points) - 1):
                cv2.line(frame, tuple(shape[points[i]]), tuple(shape[points[i+1]]), (255, 0, 0), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release the capture
videoCapture.release()
cv2.destroyAllWindows()
