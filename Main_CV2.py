import cv2
from tkinter import *
import os
import time
import numpy as np
import glob
from Operations import intersection_over_union as IoU

RES = 300
username = ""
password = ""
video_size = 960.0
conf_threshold = 0.7
iou_threshold = 0.4

# files for the model
prototxt = os.path.join('models', 'deploy.prototxt.txt')
model = os.path.join('models', 'nn.caffemodel')

# load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)


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
        #root.bind('<Return>', (lambda event: retrieve_user_input()))
        #root.bind('<Tab>', (lambda event: entry_password.focus_set()))
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
        filenames = os.path.join('users', '*.png')
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
            filename = os.path.join('users', username + '.png')
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
                 (int(width / 3.5) + line_length, int(height-height / 5)), color, thickness)
        cv2.line(frame, (int(width - width / 3.5) - line_length, int(height-height / 5)),
                 (int(width - width / 3.5), int(height-height / 5)), color, thickness)

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
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    crosshair_box = [int(frame_width/3.5), int(frame_height/5),
                     int(frame_width-frame_width/3.5), int(frame_height-frame_height/5)]
    success = True
    initial = True

    while success:
        if initial:
            #cv2.moveWindow('TrueSight', int((screen_width - video_size) / 2), int((screen_height - video_size) / 2))
            initial = False
        success, frame = cap.read()
        og_frame = frame.copy()
        resized_frame = cv2.resize(frame, (RES, RES))

        # Get the mean of the frame and convert to blob
        mean = np.mean(frame, axis=(0, 1))
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (RES, RES), mean)

        # Set frame (blob) as input to network and get detections
        net.setInput(blob)
        detections = net.forward()

        crosshair_color = (0, 0, 255)
        thickness = 2
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Ignore ones with conf below threshold
            if confidence < conf_threshold:
                continue

            # Get bounding box dims with respect to original frame
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height,
                                                       frame_width, frame_height])
            x1, y1, x2, y2 = box.astype("int")

            # Draw the bounding box and confidence level
            confidence_level = "%.2f" % (confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10

            if IoU(crosshair_box, [x1, y1, x2, y2]) > iou_threshold:
                crosshair_color = (0, 255, 0)
                thickness = 4
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, confidence_level, (x1, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

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
                filename = os.path.join('users', user_name + '.png')
                cv2.imwrite(filename, og_frame)
                cap.release()
                cv2.destroyAllWindows()
                menu()
                break
            else:
                filename = os.path.join('frames', str(time.time()*1000) + '.png')
                cv2.imwrite(filename, og_frame)

        '''
         if np.argmax(prediction) == 0:
             for i in range(shrug.shape[1]):
                 for j in range(shrug.shape[0]):
                     frame[425+j][500+i] = shrug[j][i]
         '''


    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    # vs.stop


screen_width, screen_height = set_screen_dim()
#login()
display_video()