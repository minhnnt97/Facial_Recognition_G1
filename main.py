import numpy as np
import os
import cv2
import argparse

from architecture import *
from threading import Thread
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from glob import glob


################################################################################
#######################        SETTING GLOBALS        ##########################
################################################################################
PROJECT = os.path.dirname(os.path.realpath(__file__))

# DO NOT CHANGE
IMG_WIDTH, IMG_HEIGHT = 416, 416
CROP_W, CROP_H = 160, 160
N_FEATURES = 128

# CUSTOMIZATION
BOX_COLOR = (255,255,0) # BGR
TEXT_ORIGIN_FACES = (10,50)
KNN_K = 11
KNN_CUTOFF_DIST = 10


# ARGUMENT PARSER
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
                help='path to the .h5 file of the tensorflow model for feature extraction')
ap.add_argument("-y", "--yolo", type=str, default='yolo',
                help="name of the sub-directory containing the YOLO model and weights")
ap.add_argument('-d', '--data', type=str, default='data',
                help='name of the sub-directory containing the training images')
ap.add_argument("-s", "--save", type=str, default='',
                help="name of the sub-directory for saving image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.4,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

DATA_DIR            = os.path.join(PROJECT, args['data'])
SAMPLE_DIR          = os.path.join(PROJECT, args['save'])
YOLO_DIR            = os.path.join(PROJECT, args['yolo'])

CONFIDENCE_CUTOFF   = args['confidence']
NMS_THRESHOLD       = args['threshold']
FACE_MODEL_PATH     = args['model']

CURRENT_NUM_SAMPLES = len(glob(os.path.join(SAMPLE_DIR, '*.jpg')))
FACE_MODEL = InceptionResNetV1()
FACE_MODEL.load_weights(FACE_MODEL_PATH)


################################################################################
######################        DEFINING FUNCTIONS        ########################
################################################################################
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def load_data():
    N = sum([len(files) for _,_,files in os.walk(DATA_DIR)])-1
    labels = [os.path.basename(d) for d in glob(os.path.join(DATA_DIR,'*'))]
    X = np.zeros((N, N_FEATURES))
    y = np.empty(N, dtype=object)
    i = 0
    for lbl in labels:
        images = glob(os.path.join(DATA_DIR,lbl,'*'))
        for image in images:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = normalize(img)
            img = tf.image.resize(img, [CROP_W,CROP_H])
            try:
                ext_features = FACE_MODEL(np.array([img]), training=False)
            except TypeError:
                ext_features = FACE_MODEL(np.array([img]))
            X[i] = ext_features
            y[i] = lbl
            i += 1

    return (X,y)




def predict_boxes(frame):
    blob = cv2.dnn.blobFromImage(frame,
                                 1/255,
                                 (IMG_WIDTH, IMG_HEIGHT),
                                 [0,0,0],
                                 1,
                                 crop=False)

    net.setInput(blob)
    outs = net.forward(OUTPUT_LAYERS)

    confidences = []
    boxes = []

    frame_h, frame_w = frame.shape[:2]
    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
        # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > CONFIDENCE_CUTOFF:

                center_x = int(round(detection[0] * frame_w))
                center_y = int(round(detection[1] * frame_h))
                width    = int(round(detection[2] * frame_w))
                height   = int(round(detection[3] * frame_h))

                # Find the top left point of the bounding box
                topleft_x = center_x - width//2
                topleft_y = center_y - height//2
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_CUTOFF, NMS_THRESHOLD)
    final_boxes = [boxes[i[0]] for i in indices]

    return final_boxes



def predict_faces(frame):
    global flag_thread_finished, buffer_boxes, buffer_labels, buffer_confidences
    flag_thread_finished = False

    buffer_boxes = predict_boxes(frame)
    buffer_labels, buffer_confidences = [],[]
    num_faces_detected = len(buffer_boxes)
    if num_faces_detected > 0:
        try:
            for box in buffer_boxes:
                # Extract position data
                margin = 5
                l,t,w,h = box[:4]
                l -= margin
                t -= margin
                w += margin
                h += margin

                crop = cv2.cvtColor(frame[t:t+h, l:l+w], cv2.COLOR_BGR2RGB)
                crop = normalize(crop)
                crop = tf.image.resize(crop, [CROP_W,CROP_H])
                try:
                    features = FACE_MODEL(np.array([crop]), training=False)
                except TypeError:
                    features = FACE_MODEL(np.array([crop]))
                
                k_dist, k_idx = knn.kneighbors(features, n_neighbors=KNN_K)

                flag_unknown = (k_dist[0] > KNN_CUTOFF_DIST).sum() > 0
                if flag_unknown:
                    label = '???'
                    proba = 0
                else:
                    M = stats.mode(y_train[k_idx[0]])
                    label = M[0][0]
                    proba = M[1][0]/KNN_K

                buffer_labels.append(label)
                buffer_confidences.append(proba)
        except:
            pass

    flag_thread_finished = True

    #return (final_boxes, final_labels, final_proba)




def draw_final_boxes(frame, final_boxes, final_labels, final_proba):
    num_faces_detected = len(final_boxes)
    for i,box in enumerate(final_boxes):
        # Extract position data
        l,t,w,h = box[:4]

        # Draw bounding box with the above measurements
        tl = (l,t)
        br = (l+w, t+h)
        cv2.rectangle(frame, tl, br,
                      BOX_COLOR, 2)

        # Display text about confidence rate above each box
        try:
            display_label = final_labels[i]
            display_proba = final_proba[i]
        except IndexError:
            display_label = '???'
            display_proba = 0
        text_confidence = f'{display_label} - {display_proba:.2f}'
        text_origin = (l,t-10)
        cv2.putText(frame, text_confidence, text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, BOX_COLOR, 2)

    # Display text about number of detected faces on topleft corner
    text_num_faces = f'Faces detected: {num_faces_detected}'
    cv2.putText(frame, text_num_faces, TEXT_ORIGIN_FACES,
                cv2.FONT_HERSHEY_SIMPLEX, 1, BOX_COLOR, 2)




################################################################################
######################        LOADING THE MODEL        #########################
################################################################################

### Face Detection ###
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

### Face Identification ###
print('\n>>> Loading data...')
knn = KNeighborsClassifier(n_neighbors=KNN_K)
X_train, y_train = load_data()
knn.fit(X_train, y_train)
print('...Done')

################################################################################
####################         RUNNING THE CAMERA           ######################
################################################################################
draw_boxes = True
cap = cv2.VideoCapture(0)

flag_thread_finished = True
buffer_boxes, buffer_labels, buffer_confidences = [],[],[]
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()


    ####################
    ## FACE DETECTION ##
    ####################
    if flag_thread_finished:
        cache_boxes = buffer_boxes.copy()
        cache_labels = buffer_labels.copy()
        cache_confidences = buffer_confidences.copy()
        Thread(target=predict_faces, args=[frame]).start()

    if draw_boxes:
        draw_final_boxes(frame, cache_boxes, cache_labels, cache_confidences)


    # frame is now the image capture by the webcam (one frame of the video)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)

    if c == ord('c'):       # Save part of the frame cropped by the bounding box
        box = final_boxes[0]
        l,t,w,h = box[:4]
        cv2.imwrite(os.path.join(SAMPLE_DIR,f'{CURRENT_NUM_SAMPLES}.jpg'), frame[t:t+h, l:l+w])
        CURRENT_NUM_SAMPLES += 1

    elif c == ord('d'):     # Turn on/off bounding boxes
        draw_boxes = not draw_boxes

    elif c == 27:           # Break when pressing ESC
        break


cap.release()
cv2.destroyAllWindows()

