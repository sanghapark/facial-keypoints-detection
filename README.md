
# Facial Keypoints Detection
## Deep Learning Project by Sang-Ha Park

## Table of Contents
### [1. Data Exploration](#1)

### [2. Problem Approach](#2)

### [3. CNN Modelling](#3)
  * [Dimensionality of Output](#3.1)
  * [Kernel Size](#3.2)
  * [Kernel Initialization](#3.3)
  * [Batch Normalization](#3.4)
  * [Activation Function](#3.5)
  * [Regularization](#3.6)
  
### [4. Learning Method: Gradient Descent Optimization](#4)

### [5. Training](#5)
  * [Cost Function](#5.1)
  * [Mini-batch Training](#5.2)
  * [Callbacks at the End of Each Batch](#5.3)
  * [Batch Generator & Training-time Data Augmentation](#5.4)
  
### [6. More on Model Optimization](#6)
  * [Best Single CNN Model vs. Ensemble Model](#6.1)
  * [Face Detection with OpenCV](#6.2)
  * [Advanced Image Augmentation](#6.3)
  
### [7. Final Kaggle Submission](#7)
  * [Final Score](#7.1)
  


```python
%matplotlib inline
from K.utils.utils import *
import pandas as pd
```

<a id="1"></a>
# 1. Data Exploration


```python
df = pd.read_csv('./data/training.csv')
```


```python
sns.heatmap(df.isnull() == False, cbar=False, yticklabels=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x127e4c908>




![png](/mdoutputs/output_5_1.png)


The entire dataset is drawn as a heat map and the white part is where the data is missing. For some of the keypoints we have only about 2000 labels, while other keypoints have more than 7000 labels available for training.


```python
df.head(1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_eye_center_x</th>
      <th>left_eye_center_y</th>
      <th>right_eye_center_x</th>
      <th>right_eye_center_y</th>
      <th>left_eye_inner_corner_x</th>
      <th>left_eye_inner_corner_y</th>
      <th>left_eye_outer_corner_x</th>
      <th>left_eye_outer_corner_y</th>
      <th>right_eye_inner_corner_x</th>
      <th>right_eye_inner_corner_y</th>
      <th>...</th>
      <th>nose_tip_y</th>
      <th>mouth_left_corner_x</th>
      <th>mouth_left_corner_y</th>
      <th>mouth_right_corner_x</th>
      <th>mouth_right_corner_y</th>
      <th>mouth_center_top_lip_x</th>
      <th>mouth_center_top_lip_y</th>
      <th>mouth_center_bottom_lip_x</th>
      <th>mouth_center_bottom_lip_y</th>
      <th>Image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>66.033564</td>
      <td>39.002274</td>
      <td>30.227008</td>
      <td>36.421678</td>
      <td>59.582075</td>
      <td>39.647423</td>
      <td>73.130346</td>
      <td>39.969997</td>
      <td>36.356571</td>
      <td>37.389402</td>
      <td>...</td>
      <td>57.066803</td>
      <td>61.195308</td>
      <td>79.970165</td>
      <td>28.614496</td>
      <td>77.388992</td>
      <td>43.312602</td>
      <td>72.935459</td>
      <td>43.130707</td>
      <td>84.485774</td>
      <td>238 236 237 238 240 240 239 241 241 243 240 23...</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 31 columns</p>
</div>



15 facial keypoints are given in x-y coordinates. The last column, Image, is the data, a gray-scale image of 96x96x1 size. Let's visualize a subset of the training data.


```python
X_train, _, Y_train, _ = load_train_data_and_split("data/training.csv", COLS, 0.1)
```


```python
import matplotlib.pyplot as plt
def plot_data(img, landmarks, axis):
    axis.imshow(np.squeeze(img), cmap='gray')
    landmarks = landmarks * 48 + 48
    axis.scatter(landmarks[0::2], landmarks[1::2], marker='o', c='r', s=20)
    
fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(3):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_train[i], Y_train[i], ax)
```


![png](/mdoutputs/output_10_0.png)


<a id="2"></a>
# 2. Problem Approach

A total of 30 correct position (the x and y coordinates) of 15 facial keypoints must be predicted. Instead of dropping images with partial labels, I decided to train the CNN model separately by dividing the dataset into two. The size of dataset with 4 keypoints is much smaller than the dataset with 11 keypoints. I will handle this problem with data augmentation.
<br>
### Splitting Dataset
* Dataset 1 with 4 Keypoints
  * left_eye_center
  * right_eye_center
  * nose_tip
  * mouth_center_bottomlip
* Dataset 2 with 11 Keypoints
  * left_eye_inner_corner
  * left_eye_outer_corner
  * right_eye_inner_corner
  * right_eye_outer_corner
  * left_eyebrow_inner_corner
  * left_eyebrow_outer_corner
  * right_eyebrow_inner_corner
  * right_eyebrow_outer_corner
  * mouth_left
  * mouth_right
  * mouth_center
<br>

### Pre-processing
I will also use OpenCV for pre-processing and face detection. OpenCV will detect the face and the trained CNN model will predict the facial keypoints within that location. The more data there is, the less overfitting can be achieved. Using OpenCV, images will be augmented during training before every batch. This allows training with unlimited data.

### Training
I decided to use <strong>mean squared error</strong> as the cost function because this is a non-linear regression problem. For a learning method, there are several options such as stochastic gradient descent, adagrad, and RMSProp. All methods will be tested and I will see which one works the best.

### Testing
I will predict the test data and upload it to kaggle to see how competitive my model is.

<a id="3"></a>
# 3. CNN  Modelling

I have applied various CNN architectures for fast error convergence time and cost function minimization. However, the deeper the layers, the longer the training took. Despite having more parameters that can be trained
MSE did not decrease significantly, but rather worse.
I had to test the model multiple times, changing the hyper parameters. If the cost function converges quickly and MSEs are similar, I have chosen a simpler architecture.

<table>
    <tr>
        <!--<td><img src="img/cnn_model.png" width=300 /></td> -->
        <td><img src="img/summary.png" width=500/></td>
    </tr>
</table>


<a id="3.1"></a>
* ### Dimensionality of Output
  * As convolution layers deepen, the number of kernels increases like 24 → 36 → 48 → 64 
  
<a id="3.2"></a>
* ### Kernel Size
  * 3x3 vs 5x5
  * Small filter sizes captures fine details of the image and a bigger filter size leaves out minute details in the image.
  * I chose filter size 5 because it is a fairly simple problem compared to other image classification problems.

<a id="3.3"></a>
* ### Kernel Initialization
  * The initial value of the kernel has a very large effect on the training speed and convergence, so I used the He Normal, which is the most recently used.
  * He Normal
    ![He Normal](https://latex.codecogs.com/gif.latex?%5Ctext%7BTruncated%20Normal%7D%280%2C%20%5Csqrt%7B%5Ctext%7B%23%20of%20input%20units%7D%7D%29)

<a id="3.4"></a>
* ### Batch Normalization
  * To precent gradient vanishing and exploding
  * Since the output value is normalized for each learning, it is less influenced by the initialization.

<a id="3.5"></a>
* ### Activation Function
  * Rectified Linear Units (ReLU): $\max(0,Wx+b)$
  * To avoid gradient vanishing, I used ReLU over sigmoid for the activation function.

<a id="3.6"></a>
* ### Regularization
  * Lasso(L1) & Ridge(L2)
  * Dropout
  * <strong>No Regularizations</strong>
    * During training, data augmentation is applied to every mini-batch, so it will train based on randomly augmented set of data. I do not apply normalization because I judged that it is unlikely to overfit.

<a id="4"></a>
# 4. Learning Method: Gradient Descent Optimization

* Stochastic Gradient Descent
$$
\theta \leftarrow \theta - \eta \nabla_{\theta}J(\theta)
$$

* Adagrad
$$
G_t \leftarrow G_{t-1} + (\nabla_{\theta}J(\theta_t))^2
$$

$$
\theta_{t+1} \leftarrow \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}}\cdot \nabla_{\theta}J(\theta_t)
$$

* RMSProp
$$
G_t \leftarrow \gamma{G_{t-1}} + (1-\gamma)(\nabla_{\theta}J(\theta_t))^2
$$

$$
\theta_{t+1} \leftarrow \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}}\cdot \nabla_{\theta}J(\theta_t)
$$

<img src="img/opts.gif" width=350/>
Ref: http://i.imgur.com/2dKCQHh.gif?1

It is difficult to say which optimization method is the best. The performance of each method varies depending on the problem, data set, and network structure. In my case, <strong>RMSProp</strong> showed the best performance.

<a id="5"></a>
# 5. Training

<a id="5.1"></a>
### Cost Function
  * Mean Squared Error
$$
    \text{MSE} = \frac{1}{n}\sum^{n}_{i=1}{(Y_i - \hat{Y}_i)^2}
$$

<a id="5.2"></a>
### Mini-Batch Training
  * Dataset 1
    * N = 7000
    * Batch Size = 256
  * Dataset 2
    * N = 2155
    * Batch Size = 128

<a id="5.3"></a>
### Callbacks at the End of Each Batch
  * ModelCheckpoint
    * Stored weight values for every epoch
    * During training on AWS spot instance, the instance can be terminated without notice. I have to save the parameter values that were previously trained to start again.
  * EarlyStopping
    * If there is no performance improvement over the last 50 epochs, training is automatically stopped.

<a id="5.4"></a>
### Batch Generator & Real-time Data Augmentation
  * The image data is newly augmented for each batch.
  * Pros and Cons
    * Pros: Image data that can be augmented in an infinite way. This is good to avoid overfitting.
    * Cons: It slows down the training by augmenting images every batch.
  * Image Augmentation
    * Horizontal Flip
    * Contrast
    * Rotation
$$
    \text{Transforming Matrix of Counter-clockwise Rotation}
$$
<br>
$$
\begin{bmatrix} 
x^{'} \\
y^{'} 
\end{bmatrix}
=
\begin{bmatrix} 
\cos{\theta} & -\sin{\theta} \\
\sin{\theta} & \cos{\theta} 
\end{bmatrix}
\begin{bmatrix} 
x \\
y 
\end{bmatrix}
$$


```python
X_train, _, Y_train, _ = load_train_data_and_split("data/training.csv", COLS, 0.1)
```


```python
generate_augmented_images(X_train, Y_train, True, True, True, False, False)
```


![png](/mdoutputs/output_31_0.png)


<a id="6"></a>
# 6. More on Model Optimization

<a id="6.1"></a>
## Best Single CNN Model vs. Ensemble
* Not all models produce the same error for given test data.
* Let's reduce generalization rrror through model averaging.

### Ensemble


```python
model_name = 'model_20171117_1723'
X_test = load_test_data('./data/test.csv')
```


```python
plot_loss_history(model_name, 'cnn2_dataset01', 20)
plot_loss_history(model_name, 'cnn2_dataset02', 30)
```


![png](/mdoutputs/output_36_0.png)



![png](/mdoutputs/output_36_1.png)



![png](/mdoutputs/output_36_2.png)



![png](/mdoutputs/output_36_3.png)



![png](/mdoutputs/output_36_4.png)



![png](/mdoutputs/output_36_5.png)



![png](/mdoutputs/output_36_6.png)



![png](/mdoutputs/output_36_7.png)



![png](/mdoutputs/output_36_8.png)



![png](/mdoutputs/output_36_9.png)



![png](/mdoutputs/output_36_10.png)



![png](/mdoutputs/output_36_11.png)



![png](/mdoutputs/output_36_12.png)



![png](/mdoutputs/output_36_13.png)



![png](/mdoutputs/output_36_14.png)



![png](/mdoutputs/output_36_15.png)



![png](/mdoutputs/output_36_16.png)



![png](/mdoutputs/output_36_17.png)



![png](/mdoutputs/output_36_18.png)



![png](/mdoutputs/output_36_19.png)



![png](/mdoutputs/output_36_20.png)



![png](/mdoutputs/output_36_21.png)



![png](/mdoutputs/output_36_22.png)



![png](/mdoutputs/output_36_23.png)



![png](/mdoutputs/output_36_24.png)



![png](/mdoutputs/output_36_25.png)



![png](/mdoutputs/output_36_26.png)



![png](/mdoutputs/output_36_27.png)



![png](/mdoutputs/output_36_28.png)



![png](/mdoutputs/output_36_29.png)



![png](/mdoutputs/output_36_30.png)



![png](/mdoutputs/output_36_31.png)



![png](/mdoutputs/output_36_32.png)



![png](/mdoutputs/output_36_33.png)



![png](/mdoutputs/output_36_34.png)



![png](/mdoutputs/output_36_35.png)



![png](/mdoutputs/output_36_36.png)



![png](/mdoutputs/output_36_37.png)



![png](/mdoutputs/output_36_38.png)



![png](/mdoutputs/output_36_39.png)



![png](/mdoutputs/output_36_40.png)



![png](/mdoutputs/output_36_41.png)



![png](/mdoutputs/output_36_42.png)



![png](/mdoutputs/output_36_43.png)



![png](/mdoutputs/output_36_44.png)



![png](/mdoutputs/output_36_45.png)



![png](/mdoutputs/output_36_46.png)



![png](/mdoutputs/output_36_47.png)



![png](/mdoutputs/output_36_48.png)



![png](/mdoutputs/output_36_49.png)



```python
model = load_models_with_weights(model_name)
```


```python
_ = predict_and_make_submission_file(X_test, model)
```

    submission_20180324_0016.csv created for submission.


* Score (RMSE) of submission_20171120_0110.csv
  * public:  <strong>2.57497</strong>
  * private: <strong>2.52040</strong>

### Best CNN Model


```python
best_cnns = pick_cnns(model, [4], [25])
_ = predict_and_make_submission_file(X_test, best_cnns)
```

    submission_20180324_0016.csv created for submission.


* Score (RMSE) of submission_20171120_0116.csv
  * public:  <strong>2.25358</strong>
  * private: <strong>2.02075</strong>

The ensemble model shows worse performance. This is because the ensemble model includes overfitted CNNs with bad validation error as you can see above. Let's just select some of the good models and construct the ensemble again.

### Ensemble of CNNs with Good Performance


```python
indexes08 = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19] 
indexes22 = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15, 18, 21, 22, 25, 27, 29]
selected_ensemble = pick_cnns(model, indexes08, indexes22)
_ = predict_and_make_submission_file(X_test, selected_ensemble)
```

    submission_20180324_0020.csv created for submission.


* Score (RMSE) of submission_20171120_0132.csv
  * public:  <strong>2.14311</strong>
  * private: <strong>1.92491</strong>

The ensemble model of well-trained CNNs shows better performance than the  single best model. It improved performance by approximately <strong>5%</strong>.

<a id="6.2"></a>
## Face Detection with OpenCV
  * For a better performance, let's detect a face with OpenCV2 and preditct the facial keypoints in it.


```python
_ = predict_with_cv2_and_make_submission_file(X_test, selected_ensemble)
```

    submission_20180324_0026.csv created for submission.


* score (RMSE): submission_20171120_1719.csv
  * public:  <strong>2.83592</strong>
  * private: <strong>2.51648</strong>
  
The performance got worse. This is because of frequent false positives. Let's see a prediction with OpenCV below.


```python
plot_keypoints_with_cv2(X_test[1137], selected_ensemble)
```


![png](/mdoutputs/output_51_0.png)


Since images are not clear profile pictures, there are many cases where OpenCV can not capture the exact position of the face. I decided not to use OpenCV2.

<a id="6.3"></a>
## Advanced Image Augmentation

In order to further optimize the model, I decided to apply data augmentation more variously.

  * Elastic Transformation
    * It moves each pixel individually around based on distortion fields.
  * Perspective Transformation
    * It transforms like viewing an object at another point.


```python
generate_augmented_images(X_train, Y_train, True, True, True, True, True)
```


![png](/mdoutputs/output_56_0.png)



```python
final_model_name = 'final_model'
final_model = load_models_with_weights(final_model_name)
```

Let's train again with the advanced data augmentation and construct the ensemble model by selecting well-trained CNN models.


```python
plot_loss_history(final_model_name, 'cnn2_dataset01', len(final_model[0]))
plot_loss_history(final_model_name, 'cnn2_dataset02', len(final_model[1]))
```


![png](/mdoutputs/output_59_0.png)



![png](/mdoutputs/output_59_1.png)



![png](/mdoutputs/output_59_2.png)



![png](/mdoutputs/output_59_3.png)



![png](/mdoutputs/output_59_4.png)



![png](/mdoutputs/output_59_5.png)



![png](/mdoutputs/output_59_6.png)



![png](/mdoutputs/output_59_7.png)



![png](/mdoutputs/output_59_8.png)



![png](/mdoutputs/output_59_9.png)



![png](/mdoutputs/output_59_10.png)



![png](/mdoutputs/output_59_11.png)



![png](/mdoutputs/output_59_12.png)



![png](/mdoutputs/output_59_13.png)



![png](/mdoutputs/output_59_14.png)



![png](/mdoutputs/output_59_15.png)



![png](/mdoutputs/output_59_16.png)



![png](/mdoutputs/output_59_17.png)



![png](/mdoutputs/output_59_18.png)



![png](/mdoutputs/output_59_19.png)



![png](/mdoutputs/output_59_20.png)



![png](/mdoutputs/output_59_21.png)



![png](/mdoutputs/output_59_22.png)



![png](/mdoutputs/output_59_23.png)



![png](/mdoutputs/output_59_24.png)



![png](/mdoutputs/output_59_25.png)



![png](/mdoutputs/output_59_26.png)



![png](/mdoutputs/output_59_27.png)



![png](/mdoutputs/output_59_28.png)



![png](/mdoutputs/output_59_29.png)



![png](/mdoutputs/output_59_30.png)



![png](/mdoutputs/output_59_31.png)



![png](/mdoutputs/output_59_32.png)



![png](/mdoutputs/output_59_33.png)



![png](/mdoutputs/output_59_34.png)



![png](/mdoutputs/output_59_35.png)



![png](/mdoutputs/output_59_36.png)



![png](/mdoutputs/output_59_37.png)



![png](/mdoutputs/output_59_38.png)



![png](/mdoutputs/output_59_39.png)



![png](/mdoutputs/output_59_40.png)



![png](/mdoutputs/output_59_41.png)



![png](/mdoutputs/output_59_42.png)



![png](/mdoutputs/output_59_43.png)



![png](/mdoutputs/output_59_44.png)



![png](/mdoutputs/output_59_45.png)



![png](/mdoutputs/output_59_46.png)



![png](/mdoutputs/output_59_47.png)



![png](/mdoutputs/output_59_48.png)



![png](/mdoutputs/output_59_49.png)



![png](/mdoutputs/output_59_50.png)



![png](/mdoutputs/output_59_51.png)



![png](/mdoutputs/output_59_52.png)



![png](/mdoutputs/output_59_53.png)



![png](/mdoutputs/output_59_54.png)



![png](/mdoutputs/output_59_55.png)



![png](/mdoutputs/output_59_56.png)



![png](/mdoutputs/output_59_57.png)



![png](/mdoutputs/output_59_58.png)



![png](/mdoutputs/output_59_59.png)



![png](/mdoutputs/output_59_60.png)



![png](/mdoutputs/output_59_61.png)



![png](/mdoutputs/output_59_62.png)



![png](/mdoutputs/output_59_63.png)



![png](/mdoutputs/output_59_64.png)



![png](/mdoutputs/output_59_65.png)



![png](/mdoutputs/output_59_66.png)



![png](/mdoutputs/output_59_67.png)



![png](/mdoutputs/output_59_68.png)



![png](/mdoutputs/output_59_69.png)



![png](/mdoutputs/output_59_70.png)



![png](/mdoutputs/output_59_71.png)



![png](/mdoutputs/output_59_72.png)



![png](/mdoutputs/output_59_73.png)



![png](/mdoutputs/output_59_74.png)



![png](/mdoutputs/output_59_75.png)



![png](/mdoutputs/output_59_76.png)



![png](/mdoutputs/output_59_77.png)



![png](/mdoutputs/output_59_78.png)



![png](/mdoutputs/output_59_79.png)



![png](/mdoutputs/output_59_80.png)



![png](/mdoutputs/output_59_81.png)



![png](/mdoutputs/output_59_82.png)



![png](/mdoutputs/output_59_83.png)



![png](/mdoutputs/output_59_84.png)



![png](/mdoutputs/output_59_85.png)



![png](/mdoutputs/output_59_86.png)



![png](/mdoutputs/output_59_87.png)



![png](/mdoutputs/output_59_88.png)



![png](/mdoutputs/output_59_89.png)



![png](/mdoutputs/output_59_90.png)



![png](/mdoutputs/output_59_91.png)



![png](/mdoutputs/output_59_92.png)



![png](/mdoutputs/output_59_93.png)



![png](/mdoutputs/output_59_94.png)



![png](/mdoutputs/output_59_95.png)



![png](/mdoutputs/output_59_96.png)



![png](/mdoutputs/output_59_97.png)



![png](/mdoutputs/output_59_98.png)



![png](/mdoutputs/output_59_99.png)



![png](/mdoutputs/output_59_100.png)



![png](/mdoutputs/output_59_101.png)



![png](/mdoutputs/output_59_102.png)



![png](/mdoutputs/output_59_103.png)



![png](/mdoutputs/output_59_104.png)



![png](/mdoutputs/output_59_105.png)



![png](/mdoutputs/output_59_106.png)



![png](/mdoutputs/output_59_107.png)



![png](/mdoutputs/output_59_108.png)



![png](/mdoutputs/output_59_109.png)



![png](/mdoutputs/output_59_110.png)



![png](/mdoutputs/output_59_111.png)



![png](/mdoutputs/output_59_112.png)



![png](/mdoutputs/output_59_113.png)



```python
final_indexes08 = [7, 9, 15, 17, 22, 23, 27, 30, 40, 43, 55, 57] 
final_indexes22 = [0, 1, 2, 4, 8, 14, 17, 26, 27, 32, 34, 40, 41, 42, 46, 49]
final_selected_ensemble = pick_cnns(final_model, final_indexes08, final_indexes22)
_ = predict_and_make_submission_file(X_test, final_selected_ensemble)
```

    submission_20180324_0034.csv created for submission.


<a id="7"></a>
# 7. Final Kaggle Submission
<a id="7.1"></a>
## Final Score
* score (RMSE): submission_20171122_0152.csv
  * public:  <strong>1.87261</strong>
  * private: <strong>1.64578</strong>

The additional data augmentation techniques improved the performance by <strong>14.5%</strong>. This shows that more and more data is good for performance improvement.
<br>
<br>
One thing that impressed me was that there was no single overfitted CNN model out of 110 trained models even though I did not apply any normalization. This is because training-time data augmentation has allowed the model to be trained with almost infinite data.

<img src="img/finalscore.png" />

### Private Leaderboard
<img src="img/private.png" />

### Public Leaderboard
<img src="img/public.png" />

My final score is about sixth and seventh out of the 175 participating teams according to the public and private learderboards, respectively.
