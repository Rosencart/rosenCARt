from settings import Settings

import os
import random
import tkinter as tk
import warnings
from tkinter import *
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageTk, Image
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from keras.models import load_model
from matplotlib.image import imread
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_dir = 'recourse'
train_path = 'recourse/Train'
test_path = 'Test'

# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

NUM_CATEGORIES = len(os.listdir(train_path))
NUM_CATEGORIES

folders = os.listdir(train_path)

train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(f'{train_path}/' + folder)
    train_number.append(len(train_files))
    class_num.append(Settings.classes[int(folder)])

# Sorting the dataset on the basis of number of images in each class
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [list(tuple) for tuple in tuples]

# Plotting the number of images in each class
plt.figure(figsize=(21, 10))
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()

# Visualizing 25 random images from test data

test = pd.read_csv(f'{data_dir}/Test.csv')
imgs = test["Path"].values

plt.figure(figsize=(25, 25))

for i in range(1, 26):
    plt.subplot(5, 5, i)
    random_img_path = f'{data_dir}/' + random.choice(imgs)
    rand_img = imread(random_img_path)
    plt.imshow(rand_img)
    plt.grid(visible=None)
    plt.xlabel(rand_img.shape[1], fontsize=20)  # width of image
    plt.ylabel(rand_img.shape[0], fontsize=20)  # height of image

imgs_path = "recourse/Train"
data_list = []
labels_list = []
classes_list = 43
for i in range(classes_list):
    i_path = os.path.join(imgs_path, str(i))  # 0-42
    for img in os.listdir(i_path):
        im = Image.open(i_path + '/' + img)
        im = im.resize((30, 30))
        im = np.array(im)
        data_list.append(im)
        labels_list.append(i)
data = np.array(data_list)
labels = np.array(labels_list)
print("Done")

path = "recourse/Train/0/00000_00004_00029.png"
img = Image.open(path)
img = img.resize((30, 30))
sr = np.array(img)
plt.imshow(img)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=10)

print("training_shape: ", x_train.shape, y_train.shape)
print("testing_shape: ", x_test.shape, y_test.shape)

y_train = tf.one_hot(y_train, 43)
y_test = tf.one_hot(y_test, 43)

training_shape: (35288, 30, 30, 3)
testing_shape: (3921, 30, 30, 3)

model = tf.keras.Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu", input_shape=x_train.shape[1:]))
model.add((Conv2D(filters=64, kernel_size=(5, 5), activation="relu")))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add((Conv2D(filters=128, kernel_size=(3, 3), activation="relu")))
model.add((MaxPool2D(pool_size=(2, 2))))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(rate=0.40))
model.add(Dense(43, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 5
history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test))

plt.figure(0)
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.title("Accuracy Graph")
plt.xlabel("epochs")
plt.ylabel("accuracy (0,1)")
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title("Loss Graph")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

test = pd.read_csv("recourse/Test.csv")
test_labels = test['ClassId'].values
test_img_path = "recourse"
test_imgs = test['Path'].values

test_data = []
test_labels = []

for img in test_imgs:
    im = Image.open(f'{test_img_path}/' + img)
    im = im.resize((30, 30))
    im = np.array(im)
    test_data.append(im)

test_data = np.array(test_data)
print(test_data.shape)

warnings.filterwarnings("ignore")
test_labels = test['ClassId'].values
test_labels

# predictions = model.predict_classes(test_data)
predictions = np.argmax(model.predict(test_data), axis=1)
print("accuracy: ", (accuracy_score(test_labels, predictions) * 100))

model.save('traffic_signal_classifier.h5')

model = load_model('./traffic_signal_classifier.h5')
# dictionary to label all traffic signs class.

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict([image])[0]
    sign = Settings.classes[pred + 1]
    print(sign)
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
