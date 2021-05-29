import numpy as np
import os
import cv2
from glob import glob


################################################################################
#######################        SETTING GLOBALS        ##########################
################################################################################
PROJECT = '/Users/MAC/Desktop/Minh Nguyen/CoderSchool/week_9/Facial_Recognition_G1'
SAMPLE_DIR = os.path.join(PROJECT, 'sample')
YOLO_DIR = os.path.join(PROJECT, 'yolo')

IMG_WIDTH, IMG_HEIGHT = 416, 416

BOX_COLOR = (255,255,0) # BGR
TEXT_ORIGIN_FACES = (10,50)


################################################################################
######################        DEFINING FUNCTIONS        ########################
################################################################################
def predict_frame(net, frame):
    blob = cv2.dnn.blobFromImage(frame,
                                 1/255,
                                 (IMG_WIDTH, IMG_HEIGHT),
                                 [0,0,0],
                                 1,
                                 crop=False)

    net.setInput(blob)
    outs = net.forward(OUTPUT_LAYERS)
    return outs


def get_final_boxes(outs, frame_height, frame_width):
    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
        # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(round(detection[0] * frame_width))
                center_y = int(round(detection[1] * frame_height))
                width    = int(round(detection[2] * frame_width))
                height   = int(round(detection[3] * frame_height))

                # Find the top left point of the bounding box
                topleft_x = center_x - width//2
                topleft_y = center_y - height//2
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    final_boxes = [boxes[i[0]] for i in indices]
    final_confidences = [confidences[i[0]] for i in indices]

    return (final_boxes, final_confidences)


def draw_final_boxes(frame, final_boxes, final_confidences, draw_boxes=True):
    num_faces_detected = len(final_boxes)
    if num_faces_detected > 0:
        if draw_boxes:
            for i,box in enumerate(final_boxes):
                # Extract position data
                left   = box[0]
                top    = box[1]
                width  = box[2]
                height = box[3]


                # Draw bounding box with the above measurements
                tl = (left, top)
                br = (left+width, top+height)
                cv2.rectangle(frame,
                              tl,
                              br,
                              BOX_COLOR,
                              2)


                # Display text about confidence rate above each box
                text_confidence = f'{final_confidences[i]:.2f}'
                text_origin = (left, top-10)
                cv2.putText(frame,
                            text_confidence,
                            text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            BOX_COLOR,
                            2)


        # Display text about number of detected faces on topleft corner
        text_num_faces = f'Faces detected: {num_faces_detected}'
        cv2.putText(frame,
                    text_num_faces,
                    TEXT_ORIGIN_FACES,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    BOX_COLOR,
                    2)


################################################################################
######################        LOADING THE MODEL        #########################
################################################################################
try:
    filename = '*.cfg'
    MODEL  = glob(os.path.join(YOLO_DIR, filename))[0]
    filename = '*.weights'
    WEIGHT = glob(os.path.join(YOLO_DIR, filename))[0]
    print('MODEL:\n', MODEL)
    print('WEIGHT:\n', WEIGHT)
except IndexError as err:
    raise OSError(f'File not found: {filename}') from err

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

OUTPUT_LAYERS = net.getUnconnectedOutLayersNames()


################################################################################
####################         RUNNING THE CAMERA           ######################
################################################################################
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    ####################
    ## FACE DETECTION ##
    ####################
    predictions = predict_frame(net, frame)
    final_boxes, final_confidences = get_final_boxes(predictions, frame.shape[0], frame.shape[1])
    draw_final_boxes(frame, final_boxes, final_confidences)


    # frame is now the image capture by the webcam (one frame of the video)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    # Break when pressing ESC
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
