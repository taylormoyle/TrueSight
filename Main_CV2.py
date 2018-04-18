import cv2
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import Controller_CV2 as con

RES = 300
username = ""
password = ""



# Create basic GUI with username and password capabilities
def login():
    error_text = "Enter correct ADMIN credentials..."
    while not (username == 'admin' and password == 'admin'):

        def close_window():
            root.destroy()

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

        bg_image = PhotoImage(file='pics\\eye.gif')
        w = bg_image.width()
        h = bg_image.height()
        bg_label = Label(root, image=bg_image)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        root.wm_geometry("%dx%d+800+450" % (w, h))
        root.title('TrueSight')

        lbl_username = Label(root, text='Username: ')
        lbl_username.configure(background='black', foreground='white')
        lbl_username.pack(anchor=S, side=LEFT)
        entry_username = Text(root, height=1, width=10)
        entry_username.configure(background='black', foreground='white')
        entry_username.pack(anchor=S, side=LEFT)
        lbl_password = Label(root, text='Password: ')
        lbl_password.configure(background='black', foreground='white')
        lbl_password.pack(anchor=S, side=LEFT)
        entry_password = Text(root, height=1, width=10)
        entry_password.configure(background='black', foreground='white')
        entry_password.pack(anchor=S, side=LEFT)
        btn_submit = Button(root, text='Submit', width=15, command=lambda: retrieve_user_input())
        btn_submit.configure(background='black', foreground='white')
        btn_submit.pack(anchor=S, side=RIGHT)

        #root.bind('<Return>', (lambda event: retrieve_user_input()))
        #root.bind('<Tab>', (lambda event: entry_password.focus_set()))
        #entry_username.focus_set()
        root.protocol("WM_DELETE_WINDOW", close_window)
        root.mainloop()


def menu():
    root = Tk()
    root.bind('<Escape>', quit)

    bg_image = PhotoImage(file='pics\\menu.gif')
    w = bg_image.width()
    h = bg_image.height()
    bg_label = Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    root.wm_geometry("%dx%d+800+450" % (w, h))
    root.title('TrueSight')

    def close_window():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()


# capture video feed, frame by frame
def display_video():
    cap = cv2.VideoCapture(0)
    success = True
    shrug = cv2.imread('pics\\shrug.png')
    i = 0
    while success:
        success, frame = cap.read()  # frame (640 x 480)
        resized_frame = cv2.resize(frame, (RES, RES), interpolation=cv2.INTER_AREA)

        cv2.imshow('TrueSight', frame)
        cv2.moveWindow('TrueSight', 800, 450)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        '''
         if np.argmax(prediction) == 0:
             for i in range(shrug.shape[1]):
                 for j in range(shrug.shape[0]):
                     frame[425+j][500+i] = shrug[j][i]
         '''

    cap.release()
    cv2.destroyAllWindows()


login()
menu()

