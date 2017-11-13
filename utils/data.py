import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

train_data = './data/training.csv'
test_data = './data/test.csv'
lookup_data = './data/IdLookupTable.csv'

def output_for_kaggle_submission(Y_pred, filename):
    nImages = Y_pred.shape[0]
    ImageId = []
    FeatureName = []
    for i in range(0, nImages):
        for j in range(0, 2*15):
            ImageId.append(i+1)
            if j == 0:
                FeatureName.append('left_eye_center_x')
            if j == 1:
                FeatureName.append('left_eye_center_y')
            if j == 2:
                FeatureName.append('right_eye_center_x')
            if j == 3:
                FeatureName.append('right_eye_center_y')
            if j == 4:
                FeatureName.append('left_eye_inner_corner_x')
            if j == 5:
                FeatureName.append('left_eye_inner_corner_y')
            if j == 6:
                FeatureName.append('left_eye_outer_corner_x')
            if j == 7:
                FeatureName.append('left_eye_outer_corner_y')
            if j == 8:
                FeatureName.append('right_eye_inner_corner_x')
            if j == 9:
                FeatureName.append('right_eye_inner_corner_y')
            if j == 10:
                FeatureName.append('right_eye_outer_corner_x')
            if j == 11:
                FeatureName.append('right_eye_outer_corner_y')
            if j == 12:
                FeatureName.append('left_eyebrow_inner_end_x')
            if j == 13:
                FeatureName.append('left_eyebrow_inner_end_y')
            if j == 14:
                FeatureName.append('left_eyebrow_outer_end_x')
            if j == 15:
                FeatureName.append('left_eyebrow_outer_end_y')
            if j == 16:
                FeatureName.append('right_eyebrow_inner_end_x')
            if j == 17:
                FeatureName.append('right_eyebrow_inner_end_y')
            if j == 18:
                FeatureName.append('right_eyebrow_outer_end_x')
            if j == 19:
                FeatureName.append('right_eyebrow_outer_end_y')
            if j == 20:
                FeatureName.append('nose_tip_x')
            if j == 21:
                FeatureName.append('nose_tip_y')
            if j == 22:
                FeatureName.append('mouth_left_corner_x')
            if j == 23:
                FeatureName.append('mouth_left_corner_y')
            if j == 24:
                FeatureName.append('mouth_right_corner_x')
            if j == 25:
                FeatureName.append('mouth_right_corner_y')
            if j == 26:
                FeatureName.append('mouth_center_top_lip_x')
            if j == 27:
                FeatureName.append('mouth_center_top_lip_y')
            if j == 28:
                FeatureName.append('mouth_center_bottom_lip_x')
            if j == 29:
                FeatureName.append('mouth_center_bottom_lip_y')

    df1= pd.DataFrame()
    df1['ImageId']= ImageId
    df1['FeatureName']= FeatureName
    df1["Location"]= Y_pred.reshape(-1,1)
    df1["Location"] = (df1["Location"]*48) + 48
    
    df1.loc[df1.Location > 96, 'Location'] = 95
    df1.loc[df1.Location < 0, 'Location'] = 0

    df_b = pd.read_csv(lookup_data,header=0)

    df_b = df_b.drop('Location',axis=1)
    merged = df_b.merge(df1, on=['ImageId','FeatureName'] )
    
    merged.to_csv(filename, index=0, columns = ['RowId','Location'] )

def batch_output_for_kaggle_submission(Y_pred, batch_index, batch_size):
    nImages = Y_pred.shape[0]
    ImageId = []
    FeatureName = []
    for i in range((batch_index-1)*batch_size, batch_index*batch_size):
        for j in range(0, 2*15):
            ImageId.append(i+1)
            if j == 0:
                FeatureName.append('left_eye_center_x')
            if j == 1:
                FeatureName.append('left_eye_center_y')
            if j == 2:
                FeatureName.append('right_eye_center_x')
            if j == 3:
                FeatureName.append('right_eye_center_y')
            if j == 4:
                FeatureName.append('left_eye_inner_corner_x')
            if j == 5:
                FeatureName.append('left_eye_inner_corner_y')
            if j == 6:
                FeatureName.append('left_eye_outer_corner_x')
            if j == 7:
                FeatureName.append('left_eye_outer_corner_y')
            if j == 8:
                FeatureName.append('right_eye_inner_corner_x')
            if j == 9:
                FeatureName.append('right_eye_inner_corner_y')
            if j == 10:
                FeatureName.append('right_eye_outer_corner_x')
            if j == 11:
                FeatureName.append('right_eye_outer_corner_y')
            if j == 12:
                FeatureName.append('left_eyebrow_inner_end_x')
            if j == 13:
                FeatureName.append('left_eyebrow_inner_end_y')
            if j == 14:
                FeatureName.append('left_eyebrow_outer_end_x')
            if j == 15:
                FeatureName.append('left_eyebrow_outer_end_y')
            if j == 16:
                FeatureName.append('right_eyebrow_inner_end_x')
            if j == 17:
                FeatureName.append('right_eyebrow_inner_end_y')
            if j == 18:
                FeatureName.append('right_eyebrow_outer_end_x')
            if j == 19:
                FeatureName.append('right_eyebrow_outer_end_y')
            if j == 20:
                FeatureName.append('nose_tip_x')
            if j == 21:
                FeatureName.append('nose_tip_y')
            if j == 22:
                FeatureName.append('mouth_left_corner_x')
            if j == 23:
                FeatureName.append('mouth_left_corner_y')
            if j == 24:
                FeatureName.append('mouth_right_corner_x')
            if j == 25:
                FeatureName.append('mouth_right_corner_y')
            if j == 26:
                FeatureName.append('mouth_center_top_lip_x')
            if j == 27:
                FeatureName.append('mouth_center_top_lip_y')
            if j == 28:
                FeatureName.append('mouth_center_bottom_lip_x')
            if j == 29:
                FeatureName.append('mouth_center_bottom_lip_y')

    df1= pd.DataFrame()
    df1['ImageId']= ImageId
    df1['FeatureName']= FeatureName
    df1["Location"]= Y_pred.reshape(-1,1)
    df1["Location"] = (df1["Location"]*48) + 48
    
    df1.loc[df1.Location > 96, 'Location'] = 95
    df1.loc[df1.Location < 0, 'Location'] = 0

    df_b = pd.read_csv(lookup_data,header=0)

    df_b = df_b.drop('Location',axis=1)
    merged = df_b.merge(df1, on=['ImageId','FeatureName'] )
    
    # merged.to_csv(filename, index=0, columns = ['RowId','Location'] )
    return merged

def show_predictions_on_test_data(X, Y_predicted):
    def plot_sample(x, y, axis):
        img = x.reshape(96, 96)
        axis.imshow(img, cmap='gray')
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(49):
        ax = fig.add_subplot(7, 7, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], Y_predicted[i], ax)

    plt.show()

def load_data_with_image_in_1D(test=False):
    fname = test_data if test else train_data
    df = pd.read_csv(fname)
    cols = df.columns[:-1]
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0)
    df = df.dropna()

    X = np.array(df['Image'].values.tolist())
    if not test:
        y = (df[cols].values - 48.0) / 48.0
    else:
        y = None
    return X, y


def load(test=False, cols=None):
    fname = test_data if test else train_data
    df = pd.read_csv(fname)

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    # test에서 하면 안됨. 하지만 테스트 파일은 null이 없어서 dropna 할게 없음
    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only train_data has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def fetch_batch(total_data, total_target, batch_index, batch_size):
    X_batch = total_data[batch_index:batch_index + batch_size]
    y_batch = total_target[batch_index:batch_index + batch_size,]
    return X_batch.astype(np.float32), y_batch.astype(np.float32)
