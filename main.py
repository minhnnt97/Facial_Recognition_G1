import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # frame is now the image capture by the webcam (one frame of the video)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
		# Break when pressing ESC
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
