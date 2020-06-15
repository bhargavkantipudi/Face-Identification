import numpy as np
import cv2
import os
import time 
import argparse

 
from allFunctions import *
from tkinter import *
main = Tk()

main.title("Facial Attendence")

main.geometry('300x200')

f1 = LabelFrame(main, bg="red")

label = Label(f1, text = "HELP",fg = "red", bg = "yellow" )
label.pack()

def callingCapture():
    name=E1.get()
    if E1.get()=='':
        messagebox.showinfo('Return','You will now return to the application screen')
    else:
        capture_video(name,50)
def Updatetodb():
    print("button clicked upfate to db")
    name=E1.get()
    if E1.get()=='':
        messagebox.showinfo('Return','You will now return to the application screen')
    else:
        upload_embedings(name)
def Create_model():
    create_model("model/")
def Live_model():
    live_detection()

L1 = Label(main, text="Name")
L1.pack( side = LEFT)
E1 = Entry(main, bd =5)
E1.pack(side = RIGHT)
B1 = Button(main, text ="Capture ", command = callingCapture)
B1.pack(side=BOTTOM)
B2 = Button(main, text ="Upload", command = Updatetodb)
B2.pack(side=BOTTOM)
B3 = Button(main, text ="Create Model", command = Create_model)
B3.pack(side=BOTTOM)
B4 = Button(main, text ="Live Detection", command = Live_model)
B4.pack(side=BOTTOM)

main.mainloop()



































 



 
 