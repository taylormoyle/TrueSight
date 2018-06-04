import cv2
from tkinter import *
from tkinter.filedialog import askopenfilename
import os
import time
import math
import numpy as np
import glob
import Models
import math
from PIL import ImageTk, Image

RES = 300
username = ""
password = ""
iou_threshold = 0.4
video_size = 960.0
frame_width = None
frame_height = None
DELAY = 3
count = 0


# files for the model
# Face Detector
prototxt = os.path.join('models', 'face_detector', 'deploy.prototxt.txt')
detection_model_file = os.path.join('models', 'face_detector', 'nn.caffemodel')

# Landmark Detector for alignment
landmark_model_file = os.path.join('models', 'landmark_detector', 'face_landmarks.dat')

# Encoder
encoder_meta = os.path.join('models', 'encoder', 'model-20170511-185253.meta')
encoder_ckpt = os.path.join('models', 'encoder', 'model-20170511-185253.ckpt-80000')

## create model object
model = Models.Model(prototxt,
                     detection_model_file,
                     landmark_model_file,
                     encoder_meta,
                     encoder_ckpt)

def set_screen_dim():
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def close_window(window):
    window.destroy()
    window.quit()


# Title Screen GUI with username and password capabilities
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
                lbl_error = Label(error, text=error_text, height=2, width=30, font=('Times New Roman', 13))
                lbl_error.pack()
                w = 250
                h = 30
                x = int((screen_width / 2) - (w / 2))
                y = int((screen_height / 2) - (h / 2))
                lbl_error.place()
                error.geometry('{}x{}+{}+{}'.format(w, h, x, y))
                entry_username.delete('1.0', END)
                entry_password.delete('1.0', END)
                entry_username.focus_set()
            return username, password

        def retrieve_user_input_enter(event):
            global username
            username = entry_username.get("1.0", "end-1c")
            global password
            password = entry_password.get("1.0", "end-1c")
            if username == 'admin' and password == 'admin':
                root.destroy()
            else:
                error = Toplevel()
                lbl_error = Label(error, text=error_text, height=2, width=30, font=('Times New Roman', 13))
                lbl_error.pack()
                w = 250
                h = 30
                x = int((screen_width / 2) - (w / 2))
                y = int((screen_height / 2) - (h / 2))
                lbl_error.place()
                error.geometry('{}x{}+{}+{}'.format(w, h, x, y))
                entry_username.delete('1.0', END)
                entry_password.delete('1.0', END)
                entry_username.focus_set()
            return username, password

        def focus_next_window(event):
            event.widget.tk_focusNext().focus()
            return ("break")

        root = Tk()
        root.overrideredirect(1)
        root.bind('<Escape>', quit)

        title_file = os.path.join('pics', 'title.gif')
        bg_image = PhotoImage(file=title_file)
        w = bg_image.width()
        h = bg_image.height()
        x = int((screen_width / 2) - (w / 2))
        y = int((screen_height / 2) - (h / 2))
        bg_label = Label(root, image=bg_image, text='TrueSight', font=('Times New Roman', 30), padx=300, pady=300)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        root.geometry('{}x{}+{}+{}'.format(w, h, x, y))
        root.title('TrueSight')

        lbl_username = Label(root, text='Username: ')
        lbl_username.configure(background='black', foreground='white', width=13, height=2, font=('Times New Roman', 15))
        lbl_username.pack(anchor=S, side=LEFT)
        entry_username = Text(root, height=1, width=10)
        entry_username.configure(background='black', foreground='white', insertbackground='white', width=13, height=2, font=('Times New Roman', 16))
        entry_username.pack(anchor=S, side=LEFT)
        entry_username.bind("<Tab>", focus_next_window)
        lbl_password = Label(root, text='Password: ')
        lbl_password.configure(background='black', foreground='white', width=13, height=2, font=('Times New Roman', 15))
        lbl_password.pack(anchor=S, side=LEFT)
        entry_password = Text(root, height=1, width=10)
        entry_password.configure(background='black', foreground='white', insertbackground='white', width=13, height=2, font=('Times New Roman', 16))
        entry_password.pack(anchor=S, side=LEFT)
        entry_password.bind("<Return>", retrieve_user_input_enter)

        btn_submit = Button(root, text='Engage', width=15, command=lambda: retrieve_user_input())
        btn_submit.configure(background='black', foreground='white', width=13, height=2, font=('Times New Roman', 17), borderwidth=3, relief='raised')
        btn_submit.pack(anchor=S, side=LEFT, padx=300)

        btn_quit = Button(root, text='Terminate', width=15, command=lambda: quit())
        btn_quit.configure(background='black', foreground='white', width=13, height=2, font=('Times New Roman', 17), borderwidth=3, relief='raised')
        btn_quit.pack(anchor=S, side=RIGHT)

        entry_username.focus_set()
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
    root.title('TrueSight')
    thumbnail = Label(root, image='', borderwidth=4, highlightthickness=3, relief='sunken')
    thumbnail.grid(column=0, row=5, padx=30, pady=10)
    thumbnail.grid_forget()

    def update_list(user_list):
        filenames = os.path.join('users', '*.txt')
        current_users = glob.glob(filenames)
        current_list_box = user_list.get(0, user_list.size())
        for user in current_users:
            _, filename = os.path.split(user)
            if not filename[: -4] in current_list_box:
                user_list.insert(END, filename[: -4].replace('_', ' '))

    def add_callback(entry_name, toplevel):
        user_name = entry_name.get()
        close_window(toplevel)
        close_window(root)
        display_video(mode='add_user', name=user_name)

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
            filename = os.path.join('users', username + '.png')
            os.remove(filename)
            user_list.delete(selected[0], selected[-1])
            thumbnail.grid_remove()

    def run_video():
        root.quit()
        root.destroy()
        display_video()

    def import_callback(entry_name, toplevel):
        user_name = entry_name.get()
        photo_filename = askopenfilename(title='Import')
        photo = cv2.imread(photo_filename)
        photo = cv2.resize(photo, (RES, RES))
        encoding = model.detect_and_encode_face(photo)
        user_name = user_name.replace(' ', '_')
        encoding_path = os.path.join('users', user_name + '.txt')
        np.savetxt(encoding_path, encoding)
        image_path = os.path.join('users', user_name + '.png')
        cv2.imwrite(image_path, photo)
        close_window(toplevel)
        close_window(root)
        menu()

    def import_photo():
        name_text = 'Enter the User\'s Name: '
        toplevel = Toplevel()
        lbl_name = Label(toplevel, text=name_text, height=2, width=20)
        lbl_name.grid(column=0, row=0)
        entry_name = Entry(toplevel, width=20)
        entry_name.grid(column=1, row=0, padx=20)
        entry_name.focus_set()
        entry_name.bind('<Return>', (lambda event: import_callback(entry_name, toplevel)))

    def select_user(evt):
        selected = user_list.curselection()
        filenames = os.path.join('users', '*.png')
        current_users = glob.glob(filenames)
        if selected:
            for user in current_users:
                _, filename = os.path.split(user)
                user_name = filename[:-4]
                selected_name = user_list.get(selected[0]).replace(' ', '_')
                if user_name == selected_name:
                    img = ImageTk.PhotoImage(file=user)
                    thumbnail.image = img
                    thumbnail.config(image=img, width=250, height=250)
                    thumbnail.grid()


    scrollbar = Scrollbar(root)
    scrollbar.grid(column=5, row=0, sticky=N + S, pady=10, rowspan=5)
    user_list = Listbox(root, yscrollcommand=scrollbar.set, exportselection=False)
    user_list.grid(column=0, row=0, padx=10, pady=10, rowspan=5, columnspan=4)
    user_list.config(width=50, height=15)
    scrollbar.config(command=user_list.yview)
    user_list.bind('<<ListboxSelect>>', select_user)

    update_list(user_list)

    btn_add = Button(root, text='Add', width=12, command=lambda: add_user())
    btn_add.configure(background='black', foreground='white')
    btn_add.grid(column=6, row=0, padx=35, pady=10)

    btn_import = Button(root, text='Import', width=12, command=lambda: import_photo())
    btn_import.configure(background='black', foreground='white')
    btn_import.grid(column=6, row=1, padx=35, pady=10)

    btn_delete = Button(root, text='Delete', width=12, command=lambda: delete_user())
    btn_delete.configure(background='black', foreground='white')
    btn_delete.grid(column=6, row=2, padx=35, pady=10)

    btn_video = Button(root, text='Video', width=12, command=lambda: run_video())
    btn_video.configure(background='black', foreground='white')
    btn_video.grid(column=6, row=4, padx=35, pady=10)

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


# Capture video feed, frame by frame
def display_video(mode='normal', name=None):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size)
    global frame_width
    frame_width = int(cap.get(3))
    global frame_height
    frame_height = int(cap.get(4))
    crosshair_box = [int(frame_width / 3.5), int(frame_height / 5),
                     int(frame_width - frame_width / 3.5), int(frame_height - frame_height / 5)]
    success, frame = cap.read()
    initial = True
    show_landmarks = False
    confidence_bar = False
    help_text = None

    print('video')
    while success:
        if initial:
            #cv2.moveWindow('TrueSight', int((screen_width - video_size) / 2), int((screen_height - video_size) / 2))
            initial = False
        success, frame = cap.read()
        og_frame = frame.copy()

        crosshair_color = (0, 0, 255)
        thickness = 2

        faces = model.get_faces(frame, crosshair_box)
        aligned_faces = np.zeros((len(faces), Models.ENCODE_RES, Models.ENCODE_RES, 3))
        encodings = np.zeros((len(faces), 128))

        if mode == 'normal':
            # Loop over the detections
            for f in range(len(faces)):
                # check if inside crosshairs
                # if true change crosshair color and increase thickness else draw box around face
                #if iou > iou_threshold or True:
                crosshair_color = (0, 255, 0)
                thickness = 4

                # Pre-process and get facial encodings
                aligned_faces[f], landmarks = model.align_face(frame, faces[f][1], get_landmarks=show_landmarks)

                if show_landmarks:
                    for (x, y) in landmarks:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if len(faces) > 0:
                encodings = model.get_encoding(aligned_faces.astype(np.uint8))

                # Find similarity between real-time user and list of users
                candidates, sims = model.find_similarity(encodings, faces)

                # Display predicted user's name, along with the network's confidence (bar)
                for h in range(len(candidates)):
                    x1, y1, x2, y2 = candidates[h][1]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.putText(frame, candidates[h][0], (x1, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
                    conf = sims[h]
                    min_conf = 0.1
                    max_conf = 0.4
                    diff = max_conf - min_conf
                    denom = diff * 0.1
                    num = math.floor(conf / denom)
                    if num > 7:
                        bar_color = (0, 0, 255)
                    if 7 >= num >= 3:
                        bar_color = (0, 255, 0)
                    if num < 3:
                        bar_color = (255, 0, 0)
                    if num <= 0:
                        length = 0
                    else:
                        length = math.floor(150 * (1 / num))
                    print(conf)
                    print(length)
                    x = x2 - length

                    if confidence_bar:
                        cv2.line(frame, (x, y), (x2, y), bar_color, thickness=5)

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
                help_text = ''
                if len(faces) > 1:
                    help_text += 'Too many faces detected. Please ask others to move.'
                elif len(faces) < 1:
                    help_text += 'No face detected.'
                else:
                    user_name = name.replace(' ', '_')
                    filename = os.path.join('users', user_name + '.txt')

                    aligned, _ = model.align_face(frame, faces[0][1], get_landmarks=False)
                    encoding = model.get_encoding(aligned)
                    np.savetxt(filename, encoding)

                    filename = os.path.join('users', str(user_name) + '.png')
                    cv2.imwrite(filename, cv2.resize(og_frame, (250, 250), interpolation=cv2.INTER_AREA))
                    cap.release()
                    cv2.destroyAllWindows()
                    menu()
                    break
            else:
                filename = os.path.join('frames', str(time.time() * 1000) + '.png')
                cv2.imwrite(filename, og_frame)

        if key & 0xFF == ord('l'):
            show_landmarks = not show_landmarks

        if key & 0xFF == ord('c'):
            confidence_bar = not confidence_bar

        # Legend
        cv2.putText(frame, "e: Menu", (frame_width - 80, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 100), 1)
        cv2.putText(frame, "s: Save", (frame_width - 80, 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 100), 1)
        cv2.putText(frame, "q: Quit", (frame_width - 80, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 100), 1)

        if mode == 'add_user':
            instruc = "Position desired face in center of cross-hairs and press 'S'"
            text_size = cv2.getTextSize(instruc, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 1)
            in_x, in_y = (int(frame_width / 5), frame_height - 40)
            cv2.putText(frame, instruc,(in_x, in_y) ,
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (100, 255, 0), 1)
            if help_text is not None:
                ht_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 1)
                ht_x, ht_y = (in_x, in_y + ht_size[0][1] + 10   )
                cv2.putText(frame, help_text, (ht_x, ht_y),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 170, 0), 1)

            draw_crosshairs(frame, frame_width, frame_height, crosshair_color, thickness)
        cv2.imshow('TrueSight', frame)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    # vs.stop


screen_width, screen_height = set_screen_dim()
#login()
display_video()
model.clean_up()
