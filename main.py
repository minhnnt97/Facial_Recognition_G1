import numpy as np
import os
import cv2
import argparse
from glob import glob


################################################################################
#######################        SETTING GLOBALS        ##########################
################################################################################
PROJECT = os.path.dirname(os.path.realpath(__file__))

IMG_WIDTH, IMG_HEIGHT = 416, 416
BOX_COLOR = (255,255,0) # BGR
TEXT_ORIGIN_FACES = (10,50)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="name of the sub-directory for saving image")
ap.add_argument("-y", "--yolo", required=True,
                help="name of the sub-directory containing the YOLO model and weights")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.4,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


SAMPLE_DIR = os.path.join(PROJECT, args['image'])
YOLO_DIR = os.path.join(PROJECT, args['yolo'])
CURRENT_NUM_SAMPLES = len(glob(os.path.join(SAMPLE_DIR, '*.jpg')))

CONFIDENCE_CUTOFF = args['confidence']
NMS_THRESHOLD = args['threshold']




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
            if confidence > CONFIDENCE_CUTOFF:
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_CUTOFF, NMS_THRESHOLD)
    final_boxes = [boxes[i[0]] for i in indices]
    final_confidences = [confidences[i[0]] for i in indices]

    return (final_boxes, final_confidences)


def draw_final_boxes(frame, final_boxes, final_confidences):
    num_faces_detected = len(final_boxes)
    if num_faces_detected > 0:
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
draw_boxes = True
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
    if draw_boxes:
        draw_final_boxes(frame, final_boxes, final_confidences)


    # frame is now the image capture by the webcam (one frame of the video)
    cv2.imshow('Input', frame)


    c = cv2.waitKey(1)

    if c == ord('c'):       # Save part of the frame cropped by the bounding box
        box = final_boxes[0]
        left   = box[0]
        top    = box[1]
        width  = box[2]
        height = box[3]
        cv2.imwrite(os.path.join(SAMPLE_DIR,f'{CURRENT_NUM_SAMPLES}.jpg'), frame[top:top+height, left:left+width])
        CURRENT_NUM_SAMPLES += 1

    elif c == ord('d'):     # Turn on/off bounding boxes
        draw_boxes = not draw_boxes

    elif c == 27:           # Break when pressing ESC
        break

cap.release()
cv2.destroyAllWindows()
