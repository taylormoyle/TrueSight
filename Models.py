import numpy as np
import tensorflow as tf
import math
import cv2
import dlib
import os

from glob import glob
from Operations import intersection_over_union as IoU

RES = 300
ENCODE_RES = 160
LEFTEYE = (0.3, 0.3)

class Model:

    def __init__(self,
                 detection_prototxt,
                 detection_model_file,
                 landmark_model_file,
                 encoder_meta,
                 encoder_ckpt,
                 conf_threshold=0.5,
                 rec_threshold=0.4):
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

    def get_encoding(self, faces):
        if len(faces.shape) < 4:
            faces = faces.reshape(-1, 160, 160, 3)

        gray_faces = np.zeros_like(faces)
        for f in range(len(faces)):
            face = faces[f]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray_faces[f, :, :, 0] = gray_faces[f, :, :, 1] = gray_faces[f, :, :, 2] = gray

        #mean = np.mean(gray_faces, axis=(1,2), keepdims=True)
        #std = np.std(gray_faces, axis=(1,2), keepdims=True)
        #std = np.maximum(std, 1.0/np.sqrt(gray_faces[0].size))
        #gray_faces = np.multiply((gray_faces - mean), 1/std)

        #gray_frame = gray_frame.reshape(-1, 160, 160, 3)

        feed_dict = {self.image_placeholder: gray_faces, self.phase_train_placeholder: False}
        embeddings = self.sess.run(self.encoder, feed_dict=feed_dict)
        return embeddings

    def align_and_encode_face(self, frame, box, get_landmarks=False):
        '''
        align face in image and encode it
        :return: currently, cropped and aligned face
                 future, encoding of face once implemented
        '''
        frame_width, frame_height, _ = frame.shape
        landmarks = self.get_landmarks(frame, box)

        # draw linear regression line between eyes
        left_eye = np.mean(landmarks['left_eye'], axis=0).astype('int')
        right_eye = np.mean(landmarks['right_eye'], axis=0).astype('int')
        eye_orientation = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        # get angle of regression line
        atan = np.arctan2(eye_orientation[1], eye_orientation[0]) - math.pi

        right_eye_x = 1.0 - LEFTEYE[0]
        eye_dist = np.sqrt(eye_orientation[0] ** 2 + eye_orientation[1] ** 2)
        dist = (right_eye_x - LEFTEYE[0]) * ENCODE_RES
        scale = dist / eye_dist

        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # build composite matrix
        cosine = math.cos(atan)
        sine = math.sin(atan)
        alpha = scale * cosine
        beta = scale * sine
        t_x = (1 - alpha) * eye_center[0] - beta * eye_center[1]
        t_y = beta * eye_center[0] + (1 - alpha) * eye_center[1]
        t_x += (ENCODE_RES*0.5 - eye_center[0])
        t_y += (ENCODE_RES*LEFTEYE[1] - eye_center[1])

        P = np.float64([[alpha, beta, t_x],
                        [-beta, alpha, t_y]])

        # apply composite matrix to image
        aligned_frame = cv2.warpAffine(frame, P, (ENCODE_RES, ENCODE_RES), flags=cv2.INTER_CUBIC)

        cv2.imshow('align', aligned_frame)

        # Get current user encoding
        #encoding = self._get_encoding(aligned_frame)
        return (aligned_frame, landmarks) if get_landmarks else (aligned_frame, _)

    def detect_and_encode_face(self, image):
        img_width, img_height, _ = image.shape

        faces = self.get_faces(image, None)

        if len(faces) == 0:
            return None

        _ , face = faces[0]
        aligned_face, _ = self.align_and_encode_face(image, face)
        return self.get_encoding(aligned_face)[0]


    def _calculate_similarity(self, users, current_user):
        diff = users - current_user
        squared = np.square(diff)
        added = np.sum(squared, axis=1)
        root = np.sqrt(added)
        return root

    def find_similarity(self, humans, boxes):
        filenames = os.path.join('users', '*.txt')
        users = glob(filenames)
        similarities = np.zeros((len(humans), len(users)))
        encodings = np.zeros((len(users), 128))
        candidates = [['UNKNOWN', b[1]] for b in boxes]
        names = []
        sims = np.zeros(len(candidates))

        if len(users) == 0:
            return candidates, sims

        for i in range(len(users)):
            file = open(users[i])
            encoding = file.read().split()
            encodings[i] = np.array(encoding, dtype=np.float32)
            file.close()
            _, filename = os.path.split(users[i])
            names.append(filename[:-4])

        for human in range(len(humans)):
            similarity = self._calculate_similarity(encodings, humans[human])
            similarities[human] = similarity

        for s in range(len(similarities)):
            i, j = np.unravel_index(similarities.argmin(), similarities.shape)
            if similarities[i, j] < self.rec_threshold:
                candidates[i][0] = names[j]
                sims[i] = similarities[i][j]
            similarities[i, :] = 10.
            similarities[:, j] = 10.

        return candidates, sims

    def clean_up(self):
        self.sess.close()
