import cv2
from tkinter import *

username = ""
password = ""


root = Tk()

bg_image = PhotoImage(file='bg.gif')
w = bg_image.width()
h = bg_image.height()
bg_label = Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
root.wm_geometry("%dx%d+0+0" % (w, h))
root.title('TrueSight')

lbl_username = Label(root, text='Username: ').grid(row=0)
lbl_password = Label(root, text='Password: ').grid(row=1)
entry_username = Entry(root).grid(row=0, column=1)
entry_password = Entry(root).grid(row=1, column=1)

def on_click():
    if entry_username.get() == 'admin' and entry_password.get() == 'admin':
        root.quit()

btn_submit = Button(root, text='Submit', width=15, command=on_click).grid(row=2, column=1)

root.mainloop()


cap = cv2.VideoCapture(1)
success, image = cap.read()
count = 0
success = True
while success:
    ret, frame = cap.read()
    print('Read a new frame: ', success)
    cv2.imwrite("frames\\frame%d.jpg" % count, image)
    cv2.imshow('frame', frame)
    success, image = cap.read()
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
