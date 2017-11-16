import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def show_keypoints_on_data(X, Y, figsize=(12, 12)):
    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10, c='r')

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(X.shape[0]):
        ax = fig.add_subplot(int(np.sqrt(X.shape[0])), int(np.sqrt(Y.shape[0])), i + 1, xticks=[], yticks=[])
        plot_sample(X[i], Y[i], ax)
    plt.show()

def get_image(data, mode, size, color):
    img = Image.new( mode, size, color) 
    pixels = img.load() # create the pixel map
    
    # cot=[int(i) for i in each.split(' ')]
    for i in range(img.size[0]):    # for every pixel:
        for j in range(img.size[1]):
            pixels[i,j] = (data[i+j*size[0]], data[i+j*size[0]], data[i+j*size[0]]) # set the colour accordingly
    return img

def find_faces_by_opencv(image):
    temp = np.array(image)[:, :, ::-1].copy()
    face_classifier = cv2.CascadeClassifier('./opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    temp1 = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(temp1, 1.3, 5)
    return faces 