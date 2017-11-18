import cv2
import numpy as np
from PIL import Image
from utils.load import load_test_data
from utils.models import load_models_with_weights
from utils.constant import FILEPATH_TEST, ID_LOOKUP_TABLE
from utils.constant import IMAGE_SIZE
from pandas.io.parsers import read_csv

model_name = 'model_20171117_1723'

X_test = load_test_data(FILEPATH_TEST)




# def get_image(data):
#     img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "black") 
#     pixels = img.load()
#     for i in range(IMAGE_SIZE):
#         for j in range(IMAGE_SIZE):
#             pixels[i,j] = (data[i+j*IMAGE_SIZE], data[i+j*IMAGE_SIZE], data[i+j*IMAGE_SIZE])
#     return img

# def find_faces_by_opencv(image):
#     img_copied = np.array(image)[:, :, ::-1].copy()
#     face_classifier = cv2.CascadeClassifier('../opencv/haarcascades/haarcascade_frontalface_default.xml')
#     img_copied_grey = cv2.cvtColor(img_copied, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(img_copied_grey, 1.3, 5)
#     return faces


# reader = read_csv(ID_LOOKUP_TABLE)
# grouped = reader.groupby('ImageId')
# for name, group in grouped:
#     print(name, group)


model = np.array(load_models_with_weights(model_name))
print(model.shape)
for submodel in model:
    for cnn in submodel:
        keypoints = cnn.predict(X_test)
        print(len(keypoints))