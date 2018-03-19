import cv2
from tkinter import *
import tensorflow as tf
import Neural_Network as nn
import os
import time

RES = 208
username = ""
password = ""
ERROR_TEXT = "Enter correct ADMIN credentials..."


while not (username == 'admin' and password == 'admin'):

    root = Tk()
    root.overrideredirect(1)
    root.bind('<Escape>', quit)

    bg_image = PhotoImage(file='eye.gif')
    w = bg_image.width()
    h = bg_image.height()
    bg_label = Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    root.wm_geometry("%dx%d+800+450" % (w, h))
    root.title('TrueSight')


    def retrieve_input():
        global username
        username = entry_username.get("1.0", "end-1c")
        global password
        password = entry_password.get("1.0", "end-1c")
        if username == 'admin' and password == 'admin':
            root.quit()
            root.destroy()
        toplevel = Toplevel()
        lbl_error = Label(toplevel, text=ERROR_TEXT, height=0, width=100)
        lbl_error.pack()
        return username, password


    def close_window():
        root.quit()
        root.destroy()


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
    btn_submit = Button(root, text='Submit', width=15, command=lambda: retrieve_input())
    btn_submit.configure(background='black', foreground='white')
    btn_submit.pack(anchor=S, side=RIGHT)

    root.protocol("WM_DELETE_WINDOW", close_window)

    root.mainloop()

model_file = 'models\\'
architecture = {'conv1': [16, 3, 3, 3, 1, 1], 'pool1': [2, 2, 2, 0],    # output shape 104
                'conv2': [32, 16, 3, 3, 1, 1], 'pool2': [2, 2, 2, 0],    # output shape 52
                'conv3': [64, 32, 3, 3, 1, 1], 'pool3': [2, 2, 2, 0],    # output shape 26
                'conv4': [128, 64, 3, 3, 1, 1], 'pool4': [2, 2, 2, 0],   # output shape 13
                'conv5': [256, 128, 3, 3, 1, 1],
                'full':  [13*13*256, 2]
                }

sess = tf.InteractiveSession()
inp = tf.placeholder(tf.float32, shape=[RES, RES, 3])
inp_transposed = tf.transpose(inp, perm=[2, 0, 1])
fac_rec = nn.create_facial_rec(inp, architecture)
nn.load_model(sess, model_file)

cap = cv2.VideoCapture(0)
success, image = cap.read()
success = True

while success:
    ret, frame = cap.read()
    cv2.imshow('TrueSight', frame)
    cv2.moveWindow('TrueSight', 800, 450)
    success, image = cap.read()
    resized_image = cv2.resize(image, (RES, RES), interpolation=cv2.INTER_AREA)

    cv2.imwrite('frames\\test.jpg', image)
    cv2.imwrite('frames\\test_resized.jpg', resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

