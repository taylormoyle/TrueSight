import cv2
from tkinter import *
import tensorflow as tf

RES = 208
username = ""
password = ""
ERROR_TEXT = "Enter correct ADMIN credentials..."


while not (username == 'admin' and password == 'admin'):

    root = Tk()

    bg_image = PhotoImage(file='eye.gif')
    w = bg_image.width()
    h = bg_image.height()
    bg_label = Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    root.wm_geometry("%dx%d+0+0" % (w, h))
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


    lbl_username = Label(root, text='Username: ')
    lbl_password = Label(root, text='Password: ')
    entry_username = Text(root, height=2, width=10)
    entry_username.pack()
    entry_password = Text(root, height=2, width=10)
    entry_password.pack()
    btn_submit = Button(root, text='Submit', width=15, command=lambda: retrieve_input())
    btn_submit.pack()

    root.mainloop()


cap = cv2.VideoCapture(0)
success, image = cap.read()
success = True

while success:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    success, image = cap.read()
    resized_image = cv2.resize(image, (RES, RES), interpolation=cv2.INTER_AREA)

    cv2.imwrite('frames\\test.jpg', image)
    cv2.imwrite('frames\\test_resized.jpg', resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

