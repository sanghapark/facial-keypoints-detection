import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pandas as pd
from pandas.io.parsers import read_csv
import datetime as dt
import time
from PIL import Image
from .models import load_models_with_weights
from .constant import *
from .load import load_train_data_and_split, load_test_data
from .data_augment_generator import DataAugmentGenerator

COLS = [
    'left_eye_inner_corner_x',      'left_eye_inner_corner_y',
    'left_eye_outer_corner_x',      'left_eye_outer_corner_y',
    'right_eye_inner_corner_x',     'right_eye_inner_corner_y',
    'right_eye_outer_corner_x',     'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x',     'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x',     'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x',    'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x',    'right_eyebrow_outer_end_y',
    'mouth_left_corner_x',          'mouth_left_corner_y',
    'mouth_right_corner_x',         'mouth_right_corner_y',
    'mouth_center_top_lip_x',       'mouth_center_top_lip_y',
    'Image'
]
FLIP_INDICES = [(0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15), (16, 18), (17, 19)]


cols01 = [
    'left_eye_center_x',            'left_eye_center_y',
    'right_eye_center_x',           'right_eye_center_y',
    'nose_tip_x',                   'nose_tip_y',
    'mouth_center_bottom_lip_x',    'mouth_center_bottom_lip_y'
]
cols02 = [

    'left_eye_inner_corner_x',      'left_eye_inner_corner_y',
    'left_eye_outer_corner_x',      'left_eye_outer_corner_y',
    'right_eye_inner_corner_x',     'right_eye_inner_corner_y',
    'right_eye_outer_corner_x',     'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x',     'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x',     'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x',    'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x',    'right_eyebrow_outer_end_y',

    'mouth_left_corner_x',          'mouth_left_corner_y',
    'mouth_right_corner_x',         'mouth_right_corner_y',
    'mouth_center_top_lip_x',       'mouth_center_top_lip_y'
]

keypoints = [
    'left_eye_center_x', 'left_eye_center_y',
    'right_eye_center_x', 'right_eye_center_y',
    'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
    'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y',
    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
]

keypoints_indexes = {'left_eye_center_x': 0, 'left_eye_center_y': 1,
                  'right_eye_center_x': 2, 'right_eye_center_y': 3,
                  'left_eye_inner_corner_x': 4, 'left_eye_inner_corner_y': 5,
                  'left_eye_outer_corner_x': 6, 'left_eye_outer_corner_y': 7,
                  'right_eye_inner_corner_x': 8, 'right_eye_inner_corner_y': 9,
                  'right_eye_outer_corner_x': 10, 'right_eye_outer_corner_y': 11,
                  'left_eyebrow_inner_end_x': 12, 'left_eyebrow_inner_end_y': 13,
                  'left_eyebrow_outer_end_x': 14, 'left_eyebrow_outer_end_y': 15,
                  'right_eyebrow_inner_end_x': 16, 'right_eyebrow_inner_end_y': 17,
                  'right_eyebrow_outer_end_x': 18, 'right_eyebrow_outer_end_y': 19,
                  'nose_tip_x': 20, 'nose_tip_y': 21,
                  'mouth_left_corner_x': 22, 'mouth_left_corner_y': 23,
                  'mouth_right_corner_x': 24, 'mouth_right_corner_y': 25,
                  'mouth_center_top_lip_x': 26, 'mouth_center_top_lip_y': 27,
                  'mouth_center_bottom_lip_x': 28, 'mouth_center_bottom_lip_y': 29}

def plot_data(img, landmarks, axis):
    axis.imshow(np.squeeze(img), cmap='gray')
    landmarks = landmarks * 48 + 48
    axis.scatter(landmarks[0::2], landmarks[1::2], marker='o', c='r', s=20)



def loop(values, group, df):
    for index, row in group.iterrows():
        values.append((df[int(row['ImageId']) - 1][keypoints_indexes[row['FeatureName']]]))
    return values

def predict(X, model):
    Y_hat01, Y_hat02 = np.zeros((X.shape[0], 8)), np.zeros((X.shape[0], 22))
    for submodel in model:
        for idx, cnn in enumerate(submodel):
            Y_hat = cnn.predict(X)
            if Y_hat.shape[1] == 8:
                Y_hat01 += Y_hat
            elif Y_hat.shape[1] == 22:
                Y_hat02 += Y_hat
    if model[0][0].outputs[0].shape[1] == 8:
        Y_hat01 = Y_hat01 / len(model[0])
    if model[1][0].outputs[0].shape[1] == 22:
        Y_hat02 = Y_hat02 / len(model[1])

    Y_hat01 = Y_hat01 * 48 + 48
    Y_hat02 = Y_hat02 * 48 + 48
    
    df01 = pd.DataFrame(Y_hat01, columns=cols01)
    df02 = pd.DataFrame(Y_hat02, columns=cols02)
    df_merged = pd.concat([df01, df02], axis=1)
    df_merged = df_merged[keypoints]
    return df_merged.values

def predict_with_cv2(X, model):
    face_classifier = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    
    Y_hat01, Y_hat02 = np.zeros((X.shape[0], 8)), np.zeros((X.shape[0], 22))
    for idx, x in enumerate(X):
        x = x.reshape(1, 96, 96, 1)

        img = to_img(x.reshape(-1)*255)
        cv2_img = np.array(img)[:, :, ::-1].copy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(cv2_img, 1.01, 9)
        
        y_hat01, y_hat02 = np.zeros((1, 8)), np.zeros((1, 22))
        for submodel in model:
            for cnn in submodel:
                
                if len(faces) == 1:
                    box_x, box_y, box_w, box_h = faces[0]

                    detected_face = cv2_img[box_y:box_y+box_h, box_x:box_x+box_w]
                    scale = box_w/96
                    reshaped = np.reshape(cv2.resize(detected_face, (96, 96)), (1, 96, 96, 1))
                    normalized = reshaped / 255
                    
                    cropped_y_hat = cnn.predict(normalized)
                    cropped_y_hat = (cropped_y_hat * 48 + 48) * scale
                    cropped_y_hat[0, 0::2] += box_x
                    cropped_y_hat[0, 1::2] += box_y
                    if cropped_y_hat.shape[1] == 8:
                        y_hat01 += cropped_y_hat
                    elif cropped_y_hat.shape[1] == 22:
                        y_hat02 += cropped_y_hat
                else:
                    y_hat = cnn.predict(x.reshape(-1, 96, 96, 1)) * 48 + 48
                    if y_hat.shape[1] == 8:
                        y_hat01 += y_hat
                    elif y_hat.shape[1] == 22:
                        y_hat02 += y_hat
        Y_hat01[idx] = y_hat01
        Y_hat02[idx] = y_hat02

    if model[0][0].outputs[0].shape[1] == 8:
        Y_hat01 = Y_hat01 / len(model[0])
    if model[1][0].outputs[0].shape[1] == 22:
        Y_hat02 = Y_hat02 / len(model[1])
    df01 = pd.DataFrame(Y_hat01, columns=cols01)
    df02 = pd.DataFrame(Y_hat02, columns=cols02)
    df_merged = pd.concat([df01, df02], axis=1)
    df_merged = df_merged[keypoints]
    return df_merged.values


def create_submission(predicted):
    reader = read_csv('./data/IdLookUpTable.csv')
    grouped = reader.groupby('ImageId')
    values = []
    for name, group in grouped:
        loop(values, group, predicted)
    submission = pd.DataFrame({'Location': values})
    submission.index += 1
    return submission

def pick_cnns(model, indexes_dataset01, indexes_dataset02):
    cnns_dataset01 = [model[0][i] for i in indexes_dataset01]
    cnns_dataset02 = [model[1][i] for i in indexes_dataset02]
    return [cnns_dataset01, cnns_dataset02]

def plot_loss_history(model_name, submodel_name, count):
    for i in range(count):
        cnn_name = '{}_{:02}'.format(submodel_name, i)
        loss_history = pd.read_csv('K/models/{}/{}/{}_{:02}.csv'.format(model_name, submodel_name, submodel_name, i))
        plt.plot(loss_history['rmse'])
        plt.plot(loss_history['val_rmse'])
        plt.title(cnn_name)
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        axes = plt.gca()
        # axes.set_ylim([0,3])
        plt.show()
    
def to_img(x):
    img_size = 96
    img = Image.new("RGB", (img_size, img_size), "black") 
    pixels = img.load()
    for i in range(img_size):
        for j in range(img_size):
            pixels[i,j] = (x[i+j*img_size], x[i+j*img_size], x[i+j*img_size])
    return img

def generate_augmented_images(X, Y, flip=True, rotate=True, contrast=True, perspective_transform=True, elastic_transform=True):
    generator = DataAugmentGenerator(X,
                                    Y,
                                    batchsize=100,
                                    flip_indices=FLIP_INDICES,
                                    flip_ratio=0.5,
                                    rotate_ratio=0.5,
                                    contrast_ratio=0.5,
                                    perspective_transform_ratio=0.5,
                                    elastic_transform_ratio=0.5)
    batch = generator.generate(batchsize=100, flip=True, rotate=True, contrast=True, perspective_transform=perspective_transform, elastic_transform=elastic_transform)
    for X, Y in batch:
        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i in range(16):
            rnd_idx = random.randint(0, 99)
            ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
            plot_data(X[rnd_idx, :, :, 0], Y[rnd_idx], ax)
        break

def predict_and_make_submission_file(X, model):
    predicted = predict(X, model)
    submission = create_submission(predicted)
    create_submission_file(submission)
    return predicted

def predict_with_cv2_and_make_submission_file(X, model):
    predicted = predict_with_cv2(X, model)
    submission = create_submission(predicted)
    create_submission_file(submission)
    return predicted

def create_submission_file(submission):
    datetime = dt.datetime.now().strftime("%Y%m%d_%H%M")
    filename = 'submission_' + datetime + '.csv'
    submission.to_csv(filename, index_label='RowId')
    print('{} created for submission.'.format(filename))

def plot_image_keypoints_with_cv2(x, model):

    img = to_img(x.reshape(-1)*255)

    cv2_img = np.array(img)[:, :, ::-1].copy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    image_with_detections = np.copy(cv2_img)
        
    fig = plt.figure(figsize = (5,5))
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('image')    

    landmarks = predict_with_cv2(np.array([x]), model)
    ax1.scatter(landmarks[0, 0::2], landmarks[0, 1::2], marker='o', c='c', s=15)
    ax1.imshow(image_with_detections, cmap='gray')