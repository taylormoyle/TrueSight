import cv2
from tkinter import *
import os
import time
import numpy as np
import glob
import dlib
from Operations import intersection_over_union as IoU
import math
import tensorflow as tf

RES = 300
username = ""
password = ""
video_size = 960.0
conf_threshold = 0.7
iou_threshold = 0.4
rec_threshold = 0.5
frame_width = 0
frame_height = 0

crosshair_box = [int(frame_width / 3.5), int(frame_height / 5),
                 int(frame_width - frame_width / 3.5), int(frame_height - frame_height / 5)]

# files for the model
# Face Detector
prototxt = os.path.join('models', 'face_detector', 'deploy.prototxt.txt')
detection_model_file = os.path.join('models', 'face_detector', 'nn.caffemodel')

# Landmark Detector for alignment
landmark_model_file = os.path.join('models', 'landmark_detector', 'face_landmarks.dat')

# Encoder
encoder_meta = os.path.join('models', 'encoder', 'model-20170511-185253.meta')
encoder_ckpt = os.path.join('models', 'encoder', 'model-20170511-185253.ckpt-80000')

# Load Detection and Landmark Models
detection_net = cv2.dnn.readNetFromCaffe(prototxt, detection_model_file)
landmark_net = dlib.shape_predictor(landmark_model_file)

# Load Tensorflow Session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.import_meta_graph(encoder_meta)
saver.restore(sess, encoder_ckpt)
encoder = tf.get_default_graph().get_tensor_by_name('embeddings:0')
image_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')


def set_screen_dim():
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def close_window(window):
    window.destroy()
    window.quit()


# Create basic GUI with username and password capabilities
def login():
    error_text = "Enter correct ADMIN credentials..."
    while not (username == 'admin' and password == 'admin'):

        def retrieve_user_input():
            global username
            username = entry_username.get("1.0", "end-1c")
            global password
            password = entry_password.get("1.0", "end-1c")
            if username == 'admin' and password == 'admin':
                root.destroy()
            else:
                error = Toplevel()
                lbl_error = Label(error, text=error_text, height=0, width=40)
                lbl_error.pack()
            return username, password

        root = Tk()
        root.overrideredirect(1)
        root.bind('<Escape>', quit)

        eye_file = os.path.join('pics', 'eye.gif')
        bg_image = PhotoImage(file=eye_file)
        w = bg_image.width()
        h = bg_image.height()
        bg_label = Label(root, image=bg_image)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        root.wm_geometry("%dx%d+700+400" % (w, h))
        root.title('TrueSight')

        lbl_username = Label(root, text='Username: ')
        lbl_username.configure(background='black', foreground='white')
        lbl_username.pack(anchor=S, side=LEFT)
        entry_username = Text(root, height=1, width=10)
        entry_username.configure(background='black', foreground='white', insertbackground='white')
        entry_username.pack(anchor=S, side=LEFT)
        lbl_password = Label(root, text='Password: ')
        lbl_password.configure(background='black', foreground='white')
        lbl_password.pack(anchor=S, side=LEFT)
        entry_password = Text(root, height=1, width=10)
        entry_password.configure(background='black', foreground='white', insertbackground='white')
        entry_password.pack(anchor=S, side=LEFT)
        btn_submit = Button(root, text='Submit', width=15, command=lambda: retrieve_user_input())
        btn_submit.configure(background='black', foreground='white')
        btn_submit.pack(anchor=S, side=LEFT)
        btn_quit = Button(root, text='Quit', width=15, command=lambda: quit())
        btn_quit.configure(background='black', foreground='white')
        btn_quit.pack(anchor=S, side=RIGHT)

        entry_username.focus_set()
        # root.bind('<Return>', (lambda event: retrieve_user_input()))
        # root.bind('<Tab>', (lambda event: entry_password.focus_set()))
        root.protocol("WM_DELETE_WINDOW", (lambda: close_window(root)))
        root.mainloop()


def menu():
    root = Tk()
    root.bind('<Escape>', quit)

    menu_pic = os.path.join('pics', 'menu.gif')
    bg_image = PhotoImage(file=menu_pic)
    w = bg_image.width()
    h = bg_image.height()
    bg_label = Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    root.wm_geometry("%dx%d+70+450" % (w, h))
    root.title('TrueSight')

    def update_list(user_list):
        filenames = os.path.join('users', '*.txt')
        current_users = glob.glob(filenames)
        current_list_box = user_list.get(0, user_list.size())
        for user in current_users:
            _, filename = os.path.split(user)
            if not filename[: -4] in current_list_box:
                user_list.insert(END, filename[: -4].replace('_', ' '))

    def add_callback(entry_name, toplevel):
        name = entry_name.get()
        close_window(toplevel)
        close_window(root)
        display_video(mode='add_user', name=name)

    def add_user():
        name_text = 'Enter the User\'s Name: '
        toplevel = Toplevel()
        lbl_name = Label(toplevel, text=name_text, height=2, width=20)
        lbl_name.grid(column=0, row=0)
        entry_name = Entry(toplevel, width=20)
        entry_name.grid(column=1, row=0, padx=20)
        entry_name.focus_set()
        entry_name.bind('<Return>', (lambda event: add_callback(entry_name, toplevel)))

    def delete_user():
        selected = user_list.curselection()
        if selected:
            username = user_list.get(selected[0]).replace(' ', '_')
            filename = os.path.join('users', username + '.txt')
            os.remove(filename)
            user_list.delete(selected[0], selected[-1])

    def run_video():
        root.quit()
        root.destroy()
        display_video()

    scrollbar = Scrollbar(root)
    scrollbar.grid(column=5, row=0, sticky=N + S, pady=10, rowspan=7)
    user_list = Listbox(root, yscrollcommand=scrollbar.set)
    user_list.grid(column=0, row=0, padx=10, pady=10, rowspan=7, columnspan=4)
    user_list.config(width=65, height=26)
    scrollbar.config(command=user_list.yview)

    update_list(user_list)

    btn_add = Button(root, text='Add', width=12, command=lambda: add_user())
    btn_add.configure(background='black', foreground='white')
    btn_add.grid(column=6, row=0, padx=35, pady=10)

    btn_delete = Button(root, text='Delete', width=12, command=lambda: delete_user())
    btn_delete.configure(background='black', foreground='white')
    btn_delete.grid(column=6, row=1, padx=35, pady=10)

    btn_video = Button(root, text='Video', width=12, command=lambda: run_video())
    btn_video.configure(background='black', foreground='white')
    btn_video.grid(column=6, row=5, padx=35, pady=10)

    root.protocol("WM_DELETE_WINDOW", (lambda: close_window(root)))

    root.mainloop()


def draw_crosshairs(frame, width, height, color, thickness):
    line_length = 50
    cv2.line(frame, (int(width / 3.5), int(height / 5)),
             (int(width / 3.5) + line_length, int(height / 5)), color, thickness)
    cv2.line(frame, (int(width - width / 3.5) - line_length, int(height / 5)),
             (int(width - width / 3.5), int(height / 5)), color, thickness)
    cv2.line(frame, (int(width / 3.5), int(height - height / 5)),
             (int(width / 3.5) + line_length, int(height - height / 5)), color, thickness)
    cv2.line(frame, (int(width - width / 3.5) - line_length, int(height - height / 5)),
             (int(width - width / 3.5), int(height - height / 5)), color, thickness)

    cv2.line(frame, (int(width / 3.5), int(height / 5)),
             (int(width / 3.5), int(height / 5) + line_length), color, thickness)
    cv2.line(frame, (int(width - width / 3.5), int(height / 5)),
             (int(width - width / 3.5), int(height / 5) + line_length), color, thickness)
    cv2.line(frame, (int(width / 3.5), int(height - height / 5)),
             (int(width / 3.5), int(height - height / 5) - line_length), color, thickness)
    cv2.line(frame, (int(width - width / 3.5), int(height - height / 5)),
             (int(width - width / 3.5), int(height - height / 5) - line_length), color, thickness)


def detect_faces(frame, w, h):
    # Get the mean of the frame and convert to blob
    mean = np.mean(frame, axis=(0, 1))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (RES, RES), mean)

    # Set frame (blob) as input to network and get detections
    detection_net.setInput(blob)
    detections = detection_net.forward()

    predictions = []
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Ignore ones with conf below threshold
        if confidence < conf_threshold:
            continue

        # Get bounding box dims with respect to original frame
        box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
        x1, y1, x2, y2 = box

        # calculate the IoU
        iou = IoU(crosshair_box, box)

        # package
        predictions.append((iou, box))

    return predictions


def get_landmarks(frame, box, show_landmarks=False):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x1, y1, x2, y2 = box
    rect = dlib.rectangle(x1, y1, x2, y2)

    # get landmark shape and extract coordinates
    shape = landmark_net(grayscale_frame, rect)
    coordinates = []
    for i in range(68):
        coordinates.append((shape.part(i).x, shape.part(i).y))

        ### FOR DEBUGGING AND TESTING
        if show_landmarks:
            for (x, y) in coordinates:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # build dictionary with keypoints
    landmarks = {'nose': coordinates[27:35],
                 'right_eye': coordinates[36:41],
                 'left_eye': coordinates[42:47],
                 'right_brow': coordinates[17:21],
                 'left_brow': coordinates[22:26],
                 'mouth': coordinates[48:],
                 'jaw': coordinates[:16],
                 'nose_tip': coordinates[33],
                 'right_eye_corners': (coordinates[36], coordinates[39]),
                 'left_eye_corners': (coordinates[45], coordinates[42])
                 }
    return landmarks


def calculate_similarity(users, current_user):
    diff = users - current_user
    squared = np.square(diff)
    added = np.sum(squared, axis=1)
    root = np.sqrt(added)
    return root


def find_similarity(current_user):
    filenames = os.path.join('users', '*.txt')
    users = glob.glob(filenames)
    encodings = np.zeros((len(users), 128))
    if len(users) == 0:
        return None
    for i in range(len(users)):
        file = open(users[i])
        encoding = file.read().split()
        encodings[i] = np.array(encoding, dtype=np.float32)
        file.close()
    similarity = calculate_similarity(encodings, current_user)
    candidate = np.argmin(similarity)
    print(similarity)
    if similarity[candidate] < rec_threshold:
        _, filename = os.path.split(users[candidate])
        print('Cand:', similarity[candidate])
        return filename[:-4]
    else:
        return None


def get_encoding(frame):
    mean = np.mean(frame)
    std = np.std(frame)
    std = np.maximum(std, 1.0/np.sqrt(frame.size))
    norm_frame = (frame - mean) / std
    norm_frame = norm_frame.reshape(-1, 160, 160, 3)
    embeddings = sess.run(encoder, feed_dict={image_placeholder: norm_frame, phase_train_placeholder: False})
    return embeddings[0]

def align_and_encode_face(frame, box, show_landmarks=False):
    '''
    align face in image and encode it
    :return: currently, cropped and aligned face
             future, encoding of face once implemented
    '''
    landmarks = get_landmarks(frame, box, show_landmarks)
    nose_x, nose_y = landmarks['nose_tip']
    center_x, center_y = frame_width / 2, frame_height / 2

    # Shift image
    t_x = center_x - nose_x
    t_y = center_y - nose_y
    M = np.float32([[1, 0, t_x], [0, 1, t_y]])
    centered_frame = cv2.warpAffine(frame, M, (frame_width, frame_height))
    #cv2.imshow('Centered', centered_frame)

    '''
    # Rotate image (enhancement)
    R = cv2.getRotationMatrix2D((nose_x, nose_y), 10, 1)
    rotated_frame = cv2.warpAffine(centered_frame, R, (frame_width, frame_height))
    cv2.imshow('Rotated', rotated_frame)
    '''

    # Crop image
    pad = 10
    left_brow = landmarks['left_brow'][2][1]
    right_brow = landmarks['right_brow'][2][1]
    left_jaw = landmarks['jaw'][15][0]
    right_jaw = landmarks['jaw'][0][0]
    average_height = int((nose_y - left_brow) + (nose_y - right_brow) / 2)
    average_width = int((left_jaw - nose_x) + (nose_x - right_jaw) / 2)
    y1 = int((left_brow + right_brow) / 2) - (pad*2)
    y2 = int(landmarks['jaw'][8][1]) + pad
    x1 = int(landmarks['jaw'][0][0]) - pad
    x2 = int(landmarks['jaw'][-1][0]) + pad
    cropped_frame = frame[y1:y2, x1:x2, :]
    #cv2.imshow('Cropped.png', cropped_frame)

    # Resize (tensorflow pre-processing)
    resize_res = 160
    resized_frame = cv2.resize(cropped_frame, (resize_res, resize_res), interpolation=cv2.INTER_CUBIC)

    # Get current user encoding
    encoding = get_encoding(resized_frame)
    return encoding


# Capture video feed, frame by frame
def display_video(mode='normal', name=None):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size)
    global frame_width
    frame_width = int(cap.get(3))
    global frame_height
    frame_height = int(cap.get(4))
    success = True
    initial = True
    show_landmarks = False

    while success:
        if initial:
            cv2.moveWindow('TrueSight', int((screen_width - video_size) / 2), int((screen_height - video_size) / 2))
            initial = False
        success, frame = cap.read()
        og_frame = frame.copy()
        resized_frame = cv2.resize(frame, (RES, RES))

        crosshair_color = (0, 0, 255)
        thickness = 2

        faces = detect_faces(resized_frame, frame_width, frame_height)
        # Loop over the detections
        for face in faces:
            iou, box = face
            # check if inside crosshairs
            # if true change crosshair color and increase thickness else draw box around face
            if iou > iou_threshold:
                crosshair_color = (0, 255, 0)
                thickness = 4

                # Pre-process and get facial encodings
                encoding = align_and_encode_face(frame, box, show_landmarks)

                # Find simliarities between current user and all existing users
                human_name = find_similarity(encoding)

                # Display Recognized User's Name
                if human_name is None:
                    cv2.putText(frame, "UNKNOWN", (int(frame_width / 3.5), int(frame_height / 5) - 15),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 170, 0), 1)
                else:
                    human_name = human_name.replace('_', ' ')
                    cv2.putText(frame, human_name, (int(frame_width / 3.5), int(frame_height / 5) - 15),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 170, 0), 1)
            else:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(frame, confidence_level, (x1, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)


        # Legend
        cv2.putText(frame, "e: Menu", (frame_width - 80, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 150), 1)
        cv2.putText(frame, "s: Save", (frame_width - 80, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 150), 1)
        cv2.putText(frame, "q: Quit", (frame_width - 80, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 150), 1)

        if mode == 'add_user':
            cv2.putText(frame, "Position desired face in center of cross-hairs and press 'S'",
                        (int(frame_width / 5), frame_height - 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (100, 255, 0), 1)

        draw_crosshairs(frame, frame_width, frame_height, crosshair_color, thickness)
        cv2.imshow('TrueSight', frame)

        # Quit video feed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            if mode == 'add_user':
                display_video()
            else:
                break

        # Open Menu
        if mode == 'normal':
            if key & 0xFF == ord('e'):
                cap.release()
                cv2.destroyAllWindows()
                menu()
                break

        # Take a picture
        if key & 0xFF == ord('s'):
            if mode == 'add_user':
                user_name = name.replace(' ', '_')
                filename = os.path.join('users', user_name + '.txt')
                encoding = align_and_encode_face(frame, box)
                np.savetxt(filename, encoding)
                cap.release()
                cv2.destroyAllWindows()
                menu()
                break
            else:
                filename = os.path.join('frames', str(time.time() * 1000) + '.png')
                cv2.imwrite(filename, og_frame)

        if key & 0xFF == ord('l'):
            show_landmarks = not show_landmarks

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    # vs.stop


screen_width, screen_height = set_screen_dim()
login()
display_video()
sess.close()