from settings import Settings

import tkinter as tk
from tkinter import *
from tkinter import filedialog

import numpy
from PIL import ImageTk, Image
from keras.models import load_model


class GUI:
    def __init__(self):
        self.model = load_model('traffic_signal_classifier.h5')

        # initialise GUI
        self.top = tk.Tk()
        self.top.geometry('800x600')
        self.top.title('Traffic sign classification')
        self.top.configure(background='#CDCDCD')

        self.label = Label(self.top, background='#CDCDCD', font=('arial', 15, 'bold'))
        self.sign_image = Label(self.top)

        upload = Button(self.top, text="Upload an image", command=self.upload_image, padx=10, pady=5)
        upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

        upload.pack(side=BOTTOM, pady=50)
        self.sign_image.pack(side=BOTTOM, expand=True)
        self.label.pack(side=BOTTOM, expand=True)
        heading = Label(self.top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
        heading.configure(background='#CDCDCD', foreground='#364156')
        heading.pack()

    def classify(self, file_path):
        image = Image.open(file_path)
        image = image.resize((30, 30))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        print(image.shape)
        pred = self.model.predict([image])[0]
        sign = Settings.classes[pred + 1]
        print(sign)
        self.label.configure(foreground='#011638', text=sign)

    def show_classify_button(self, file_path):
        classify_b = Button(self.top, text="Classify Image", command=lambda: self.classify(file_path), padx=10, pady=5)
        classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
        classify_b.place(relx=0.79, rely=0.46)

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail((self.top.winfo_width() / 2.25, self.top.winfo_height() / 2.25))

            im = ImageTk.PhotoImage(uploaded)
            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label.configure(text='')
            self.show_classify_button(file_path)
        except Exception:
            pass


if __name__ == '__main__':
    gui = GUI()
    gui.top.mainloop()
