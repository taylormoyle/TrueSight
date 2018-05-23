import numpy as np
import tensorflow as tf
import math
import cv2
import dlib
import os

from glob import glob
from Operations import intersection_over_union as IoU

RES = 300

class Model:

    def __init__(self,
                 detection_prototxt,
                 detection_model_file,
                 landmark_model_file,
                 encoder_meta,
                 encoder_ckpt,
                 conf_threshold=0.5,
                 rec_threshold=0.6):
        self._load_detection_model(detection_prototxt, detection_model_file)
        self._load_landmark_model(landmark_model_file)
        self._load_encoder(encoder_meta, encoder_ckpt)
        self.conf_threshold = conf_threshold
        self.rec_threshold = rec_threshold

    # Load Detection and Landmark Models
    def _load_detection_model(self, prototxt, detection_model_file):
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, detection_model_file)

    def _load_landmark_model(self, landmark_model_file):
        self.landmarker = dlib.shape_predictor(landmark_model_file)

    # Load Tensorflow Session
    def _load_encoder(self, encoder_meta, encoder_ckpt):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        saver = tf.train.import_meta_graph(encoder_meta)
        saver.restore(self.sess, encoder_ckpt)
        self.encoder = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        self.image_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

    def get_faces(self, frame, crosshairs):
        h, w, _ = frame.shape
        resized_frame = cv2.resize(frame, (RES, RES), interpolation=cv2.INTER_AREA)

        # Get the mean of the frame and convert to blob
        mean = np.mean(resized_frame, axis=(0, 1))
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (RES, RES), mean)

        # Set frame (blob) as input to network and get detections
        self.detector.setInput(blob)
        detections = self.detector.forward()

        predictions = []
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Ignore ones with conf below threshold
            if confidence < self.conf_threshold:
                continue

            # Get bounding box dims with respect to original frame
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
            x1, y1, x2, y2 = box

            # calculate the IoU
            iou = IoU(crosshairs, box) if crosshairs else None

            # package
            predictions.append((iou, box))

        return predictions

    def get_landmarks(self, frame, box):

        x1, y1, x2, y2 = box
        rect = dlib.rectangle(x1, y1, x2, y2)

        # get landmark shape and extract coordinates
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.landmarker(gray_frame, rect)
        coordinates = []
        for i in range(68):
            coordinates.append((shape.part(i).x, shape.part(i).y))

        # build dictionary with keypoints
        landmarks = {'nose': coordinates[27:35],
                     'right_eye': coordinates[36:41],
                     'left_eye': coordinates[42:47],
                     'right_brow': coordinates[17:21],
                     'left_brow': coordinates[22:26],
                     'mouth': coordinates[48:],
                     'jaw': coordinates[:16],
                     'nose_tip': [coordinates[33]],
                     'right_eye_corners': [coordinates[36], coordinates[39]],
                     'left_eye_corners': [coordinates[45], coordinates[42]]
                     }
        return landmarks

    def _get_encoding(self, frame):
        #mean = np.mean(frame)
        #std = np.std(frame)
        #std = np.maximum(std, 1.0/np.sqrt(frame.size))
        #norm_frame = np.multiply(np.subtract(frame, mean), 1/std)
        #norm_frame = norm_frame.reshape(-1, 160, 160, 3)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = np.zeros_like(frame)
        gray_frame[:, :, 0] = gray_frame[:, :, 1] = gray_frame[:, :, 2] = gray
        gray_frame = gray_frame.reshape(-1, 160, 160, 3)

        feed_dict = {self.image_placeholder: gray_frame, self.phase_train_placeholder: False}
        embeddings = self.sess.run(self.encoder, feed_dict=feed_dict)
        return embeddings[0]

    def align_and_encode_face(self, frame, box, get_landmarks=False):
        '''
        align face in image and encode it
        :return: currently, cropped and aligned face
                 future, encoding of face once implemented
        '''
        frame_width, frame_height, _ = frame.shape

        landmarks = self.get_landmarks(frame, box)
        nose_x, nose_y = landmarks['nose_tip'][0]
        center_x, center_y = frame_width / 2, frame_height / 2

        # draw linear regression line between outer eye corners
        left_eye = landmarks['left_eye_corners'][0]
        right_eye = landmarks['right_eye_corners'][0]
        eye_orientation = (left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
        # get angle of regression line
        angle_rad = math.atan2(float(eye_orientation[1]), float(eye_orientation[0]))

        # build composite matrix
        cosine = math.cos(angle_rad)
        sine = math.sin(angle_rad)
        shift_x = center_x - nose_x * cosine - nose_y * sine
        shift_y = center_y - nose_x * -sine - nose_y * cosine

        P = np.float32([[cosine, sine, shift_x], [-sine, cosine, shift_y]])

        # apply composite matrix to image
        aligned_frame = cv2.warpAffine(frame, P, (frame_width, frame_height))

        # Crop image
        lbrow = landmarks['left_brow'][2]
        rbrow = landmarks['right_brow'][2]

        # get coordinates of face edges
        brow_center = (int((rbrow[0] + lbrow[0]) / 2), int((rbrow[1] + lbrow[1]) / 2))
        face_bottom = (landmarks['mouth'][8][0], int((landmarks['mouth'][8][1] + landmarks['jaw'][8][1]) / 2))
        face_right = landmarks['jaw'][3]
        face_left = landmarks['jaw'][14]

        # calculate distance between each edge and nose
        dist_top = math.sqrt((brow_center[0] - nose_x) ** 2 + ((brow_center[1] - nose_y) ** 2))
        dist_bottom = math.sqrt((face_bottom[0] - nose_x) ** 2 + (face_bottom[1] - nose_y) ** 2)
        dist_right = math.sqrt((face_right[0] - nose_x) ** 2 + (face_right[1] - nose_y) ** 2)
        dist_left = math.sqrt((face_left[0] - nose_x) ** 2 + (face_right[1] - nose_y) ** 2)
        dist_side = (dist_left + dist_right) / 2

        # set new bbox dims
        y1 = int(center_y - dist_top)
        y2 = int(center_y + dist_bottom)
        x1 = int(center_x - dist_side)
        x2 = int(center_x + dist_side)

        # crop out face
        cropped_frame = aligned_frame[y1:y2, x1:x2, :]
        cv2.imshow('Cropped.png', cropped_frame)

        # Resize (tensorflow pre-processing)
        resize_res = 160
        resized_frame = cv2.resize(cropped_frame, (resize_res, resize_res), interpolation=cv2.INTER_CUBIC)

        # Get current user encoding
        encoding = self._get_encoding(resized_frame)
        return (encoding, landmarks) if get_landmarks else encoding

    def detect_and_encode_face(self, image):
        img_width, img_height, _ = image.shape

        faces = self.get_faces(image, None)

        if len(faces) == 0:
            return None

        _, face = faces[0]
        encoding = self.align_and_encode_face(image, face)
        return encoding

    def _calculate_similarity(self, users, current_user):
        diff = users - current_user
        squared = np.square(diff)
        added = np.sum(squared, axis=1)
        root = np.sqrt(added)
        return root

    def find_similarity(self, current_user):
        filenames = os.path.join('users', '*.txt')
        users = glob(filenames)
        encodings = np.zeros((len(users), 128))
        if len(users) == 0:
            return None
        for i in range(len(users)):
            file = open(users[i])
            encoding = file.read().split()
            encodings[i] = np.array(encoding, dtype=np.float32)
            file.close()
        similarity = self._calculate_similarity(encodings, current_user)
        candidate = np.argmin(similarity)
        print(similarity)
        if similarity[candidate] < self.rec_threshold:
            _, filename = os.path.split(users[candidate])
            #print('Cand:', similarity[candidate])
            return filename[:-4]
        else:
            return None

    def clean_up(self):
        self.sess.close()
