import numpy as np
import tensorflow as tf
import math
import cv2
import dlib
import os

from glob import glob
from Operations import intersection_over_union as IoU
from tensorflow.python.platform import gfile

DETECT_RES = (300, 300) #(160, 120)
ENCODE_RES = 160
LEFTEYE = (0.3, 0.2)

LANDMARK_IDX = {'nose': (27,36),
                'right_eye': (36,42),
                'left_eye': (42,48),
                'right_brow': (17,22),
                'left_brow': (22,27),
                'mouth': (48,-1),
                'jaw': (0,17),
                'nose_tip': (33),
                'right_eye_corners': (36,39),
                'left_eye_corners': (45,42)
                }

class Model:

    def __init__(self,
                 detection_prototxt=None,
                 detection_model_file=None,
                 landmark_model_file=None,
                 encoder_pb=None,
                 encoder_meta=None,
                 encoder_ckpt=None,
                 conf_threshold=0.5,
                 rec_threshold=0.8):
        self._load_detection_model(detection_prototxt, detection_model_file)
        self._load_landmark_model(landmark_model_file)
        self._load_encoder_pb(encoder_pb)
        self.conf_threshold = conf_threshold
        self.rec_threshold = rec_threshold

    # Load Detection and Landmark Models
    def _load_detection_model(self, prototxt, detection_model_file):
        if prototxt is not None and detection_model_file is not None:
            self.detector = cv2.dnn.readNetFromCaffe(prototxt, detection_model_file)
        else:
            self.detector = None

    def _load_landmark_model(self, landmark_model_file):
        if landmark_model_file is not None:
            self.landmarker = dlib.shape_predictor(landmark_model_file)
        else:
            self.landmarker = None

    # Load Tensorflow Session
    def _load_encoder(self, encoder_meta, encoder_ckpt):
        if encoder_ckpt is not None and encoder_meta is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
            saver = tf.train.import_meta_graph(encoder_meta)
            saver.restore(self.sess, encoder_ckpt)
            self.encoder = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            self.image_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        else:
            self.encoder = None

    def _load_encoder_pb(self, pb_file):
        if pb_file is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
            with gfile.FastGFile(pb_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(init_op)
            self.encoder = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            self.image_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        else:
            self.encoder = None


    def get_faces(self, frame, crosshairs=None):
        if self.detector is None:
            return None
        h, w, _ = frame.shape
        resized_frame = cv2.resize(frame, DETECT_RES, interpolation=cv2.INTER_AREA)

        # Get the mean of the frame and convert to blob
        mean = [104., 177., 123.] #np.mean(resized_frame, axis=(0, 1))
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, DETECT_RES, mean)

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
            iou = IoU(crosshairs, box) if crosshairs is not None else None

            # package
            predictions.append((iou, box))

        return predictions

    def get_landmarks(self, frame, box):
        if self.landmarker is None:
            return None

        x1, y1, x2, y2 = box
        rect = dlib.rectangle(x1, y1, x2, y2)

        # get landmark shape and extract coordinates
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.landmarker(gray_frame, rect)
        landmark_coords = []
        for i in range(68):
            landmark_coords.append([shape.part(i).x, shape.part(i).y])

        return landmark_coords

    def get_encoding(self, faces):
        if self.encoder is None:
            return None

        if len(faces.shape) < 4:
            faces = faces.reshape(-1, 160, 160, 3)

        '''
        summed = np.sum(faces, axis=(1,2,3), keepdims=True)
        num_non_zeros = np.sum((faces != 0).astype(int), axis=(1,2,3), keepdims=True)
        mean = summed / num_non_zeros

        variance = np.square(np.sum(faces[(faces != 0)] - mean, axis=(1,2,3), keepdims=True)) / num_non_zeros
        std = np.sqrt(variance)
        std = np.maximum(std, 1.0/np.sqrt(faces[0].size))
        norm_faces = np.multiply((faces - mean), 1/std)
        '''

        # convert to RGB from opencv BGR
        rgb_faces = faces[:, :, :, ::-1]

        # normalize **** temp ****
        mean = np.mean(rgb_faces, axis=(1,2,3), keepdims=True)
        std = np.std(rgb_faces, axis=(1,2,3), keepdims=True)
        #std = np.maximum(std, 1.0/np.sqrt(faces[0].size))
        norm_face = np.multiply((rgb_faces - mean), 1/std)

        feed_dict = {self.image_placeholder: norm_face, self.phase_train_placeholder: False}
        embeddings = self.sess.run(self.encoder, feed_dict=feed_dict)
        return embeddings

    def _crop_face(self, aligned_frame, comp_matrix, landmarks):
        ldmk = np.ones((len(landmarks), 3), dtype=np.int32)
        ldmk[:, 0:2] = landmarks

        poly_pts = np.matmul(comp_matrix, ldmk.transpose())
        poly_pts = poly_pts.transpose().astype(np.int32)

        mask = np.zeros_like(aligned_frame)
        jaw_idx = LANDMARK_IDX['jaw']
        lb_idx = LANDMARK_IDX['left_brow']
        rb_idx = LANDMARK_IDX['right_brow']
        jaw = np.array(poly_pts[jaw_idx[0]:jaw_idx[1]], dtype=np.int32)
        left_brow = np.array(poly_pts[lb_idx[0]:lb_idx[1]][::-1], dtype=np.int32)
        right_brow = np.array(poly_pts[rb_idx[0]:rb_idx[1]][::-1], dtype=np.int32)

        pad = 5
        # pad jaw
        jaw[:9] -= [pad,0]
        jaw[9] += [0,pad]
        jaw[10:] += [pad,0]

        left_brow -= [0, pad]
        right_brow -= [0, pad]

        roi = np.vstack((jaw, left_brow[:-1], right_brow[1:])).reshape(1, -1, 2)
        mask_color = (255, 255, 255)
        cv2.fillPoly(mask, roi, mask_color)

        cropped_face = cv2.bitwise_and(aligned_frame, mask)
        cv2.imshow('cropped', cropped_face)

        return cropped_face

    def align_face(self, frame, box, get_landmarks=False):
        '''
        align face in image and encode it
        :return: currently, cropped and aligned face
                 future, encoding of face once implemented
        '''
        frame_width, frame_height, _ = frame.shape
        landmarks = self.get_landmarks(frame, box)

        if landmarks is None:
            return None

        # draw linear regression line between eyes
        le_idx = LANDMARK_IDX['left_eye']
        re_idx = LANDMARK_IDX['right_eye']
        left_eye = np.mean(landmarks[le_idx[0]:le_idx[1]], axis=0).astype('int')
        right_eye = np.mean(landmarks[re_idx[0]:re_idx[1]], axis=0).astype('int')
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

        #cropped_face = self._crop_face(norm_face, P, landmarks)
        cv2.imshow('aligned', aligned_frame)

        return (aligned_frame, landmarks) if get_landmarks else (aligned_frame, _)

    def detect_and_encode_face(self, image):
        img_width, img_height, _ = image.shape

        faces = self.get_faces(image, None)

        if len(faces) == 0:
            return None

        _ , face = faces[0]
        aligned_face, _ = self.align_face(image, face)
        return self.get_encoding(aligned_face)[0]


    def _calculate_similarity(self, users, current_user):
        diff = users - current_user
        squared = np.square(diff)
        added = np.sum(squared, axis=1)
        root = np.sqrt(added)
        return added

    def find_similarity(self, humans, boxes):
        filenames = os.path.join('users', '*.txt')
        users = glob(filenames)
        similarities = np.zeros((len(humans), len(users))) + self.rec_threshold
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
            print(similarity)

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
