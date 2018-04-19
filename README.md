
# Facial Keypoints Detection
## Deep Learning Project by Sang-Ha Park

## Table of Contents

### [Analysis Description](#desc)
* [English](#eng)
* [한국어](#kr)

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
  

<a id="desc"></a>
# Analysis Description

- - -

<a id="eng"></a>
## English

### Data
The problem is to detect 15 facial key points in black-and-white face images. The training data consists of clean background profile images and face images cropped from normal photos. In 7,000 training data, approximately 2000 images are labeled for all 15 key points, while the rest are labeled for only 4 keypoints. Instead of throwing the partially labeled picture, I split the dataset into two, 11 key points and 4 key points, and trained two CNN models for each dataset.

### CNN Modeling
Various CNN architectures are tested. Deeper models have more variables for fitting but took longer to train and did not significantly reduce MSE(Mean Squared Error). Since it is a relatively simple problem of finding facial key-points, I chose a simple model if the errors do not differ much. I tried a different number of layers, kernel size, kernel initializers, batch normalization and activation functions and chose the one with the best performance and fast convergence. 
 
### Training
MSE(Mean Squared Error) is used as the error metric. I trained using mini-batch gradient descent. For each batch, images are randomly augmented in training time. The advantage of this was that image data could be generated in almost infinite ways, but the training speed got slower. I used AWS Spot Instance to train on GPU and stored the model's weights after each batch and applied early-stopping if there was no improvement in performance. I used a variety of optimization methods and selected RMSProp which performed best. 

### Data Augmentation
The initial image augmentation methods were horizontal flip, contrast, and rotation. Parameters for each transformation were chosen randomly every batch. To further improve the performance, I added two other augmentation methods, elastic transformation and perspective transformation. These two methods improved performance by 14%. Elastic transformation moves pixels one or two spaces around and degrades the image quality. Perspective transformation transforms images as seen in other viewpoints.

### Overfitting and Regularization
Drop-out or L1, L2 regularization is not applied. I thought that by feeding randomly augmented images to each batch, the model could learn from near-infinite data and avoid over-fitting. When the first three augmentation methods were applied, the overfitting occurred in the probability of one out of ten. After adding two more augmentation methods, the training error and validation error of all models converged to same MSE with very little difference.

### Model Tuning
Before a model detects facial key points, I used OpenCV to find a face and look for the points in the detected area. Many false positives occurred in face detection because there are many non-frontal face images. So I stopped using OpenCV. In addition, I trained as many CNN models as possible to create an ensemble model and calculate the average value. There was 4% performance improvement. 

### Result
My score is about 6th among 175 teams. The images with large errors were either partial or small faces, making it difficult to detect faces. I can over-sample those unusual data to learn these rare cases for better performance.

- - -

<a id="kr"></a>
## 한국어
### Data
딥러닝 프로젝트로 캐글에서 제공되는 얼굴 특징점 찾기를 선택하였습니다. 데이터들은 배경이 깨끗한 흑백 프로필 이미지와 일반 사진에서 얼굴만 크롭 된 사진들로 이루어져 있었습니다. 7000개의 트레이닝 데이터에서 대략 2000개의 이미지는 모든 15개의 특징점에 대해 레이블링 되어 있었지만, 나머지는 오직 11개의 특징점만이 레이블링 돼 있었습니다. 부분적으로 레이블링 된 사진을 버리지 않고 11개의 특징점 데이터와 4개의 특징점 데이터로 나누어 두 개의 모델을 트레이닝시키기고 합치기로 하였습니다.

### CNN Modeling
다양한 CNN 구조를 테스트하였습니다. 레이어를 깊게 가져보았지만, 트레이닝 시간은 오래 걸리고 변수가 많음에도 불구하고 MSE(Mean Squared Error)가 크게 줄어들지 않았습니다. 얼굴 특징점을 찾는 상대적으로 간단한 문제이기에 에러가 크게 차이가 없다면 단순한 모델을 선택하기로 하였습니다. Layer의 개수, 커널 사이즈, 커널 초기화, Batch Normalization, Activation Function을 다양하게 적용해보고 가장 좋은 퍼포먼스를 내는 것들로 선택하였습니다. 

### Training
Error Metric으로는 MSE(Mean Squared Error)를 사용 하였습니다. Mini-Batch Gradient Descent를 사용하여 트레이닝시켰습니다. 배치마다 사진들을 랜덤하게 증강해 트레이닝시켰고 이미지를 무한대로 증강할 수 있지만 트레이닝의 속도가 느려진다는 단점이 있었습니다. AWS Spot Instance를 통해 GPU Training을 시키면서 배치마다 모델의 변숫값들을 저장하고 퍼포먼스의 향상이 없다면 Early Stop을 적용하였습니다. 최적화 방법은 다양한 방법을 사용해보았고 RMSProp이 가장 좋은 성능을 보여 선택하였습니다.

### Data Augmentation
처음에 적용한 이미지 증강방법은 Horizontal Flip, Contrast, Rotation 이였습니다. 배치마다 증강변수들을 랜덤하게 적용하였습니다. 랜덤하게 변형된 데이터들을 트레이닝하니 데이터의 수가 적은 문제는 해소되었습니다. 퍼포먼스를 더 높이기 위해 더 다양한 증강방법을 적용하기로 하였습니다. Elastic Transformation과 Perspective Transformation을 추가적 적용하니 퍼포먼스가 14.5% 향상하였습니다. Elastic Transformation은 픽셀을 랜덤하게 한두 칸씩 이동시켜 화질을 떨어뜨리는 효과를 주었고 Perspective Transformation을 통해 다른 시점에서 보는 듯한 효과를 주었습니다.

### Overfitting and Regularization
Drop-out이나 L1, L2 정규화는 적용하지 않았습니다. 미니배치마다 데이터를 랜덤하게 증강 시키면 무한대에 가까운 데이터로 학습시킬 수 있고 과적합을 피할 수 있다고 생각했기 때문입니다. 처음 3가지 증강방법을 적용했을 때는 과적합이 열개 중 한개의 확률로 발생하였습니다. 증강방법 2가지를 추가하였더니 모든 모델의 train 에러와 validation 에러가 작은 오차로 수렴하였습니다.

### Model Tuning
특징점을 예측하기 전에 OpenCV를 사용하여 얼굴을 인식하고 인식된 부분에서 특징점을 찾도록 하였습니다. 나쁜 화질 및 얼굴 정면 모습이 아닌 이미지들이 많아서 얼굴 인식에서 false positives가 많이 발생해 퍼포먼스는 오히려 나빠졌습니다. 그래서 OpenCV 사용을 중지하였습니다. 추가로 다른 초깃값으로 세팅된 CNN 모델들을 최대한 많이 트레이닝시키고 평균값을 구하는 앙상블 모델을 구현하였습니다. 이것으로 인해 5%의 퍼포먼스 향상이 있는 것을 확인할 수 있었습니다. 또한, 위에 언급한 더 다양한 데이터 증강 방법을 적용하였습니다.

### Result
175팀 중에 6등에 해당하는 점수를 기록하였습니다. 에러가 크게 발생하는 이미지들은 얼굴의 부분이 잘려있거나 이미지 크기보다 얼굴 사이즈가 너무 작아 인식하기 힘든 사진들이었습니다. 이런 문제는 특이한 사진들을 좀 더 오버샘플링하여 트레이닝 시켜 볼 수 있을 것 같습니다.


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
  * He Normal: <br>
    ![He Normal](https://latex.codecogs.com/gif.latex?%5Ctext%7BTruncated%20Normal%7D%280%2C%20%5Csqrt%7B%5Ctext%7B%23%20of%20input%20units%7D%7D%29)

<a id="3.4"></a>
* ### Batch Normalization
  * To precent gradient vanishing and exploding
  * Since the output value is normalized for each learning, it is less influenced by the initialization.

<a id="3.5"></a>
* ### Activation Function
  * Rectified Linear Units (ReLU): <br>
  ![ReLU](https://latex.codecogs.com/gif.latex?%5Cmax%280%2CWx&plus;b%29)
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

![SGD](https://latex.codecogs.com/gif.latex?%5Ctheta%20%5Cleftarrow%20%5Ctheta%20-%20%5Ceta%20%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta%29)

* Adagrad

![adagrad1](https://latex.codecogs.com/gif.latex?G_t%20%5Cleftarrow%20G_%7Bt-1%7D%20&plus;%20%28%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta_t%29%29%5E2)

![adagrad2](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bt&plus;1%7D%20%5Cleftarrow%20%5Ctheta_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7BG_t&plus;%5Cepsilon%7D%7D%5Ccdot%20%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta_t%29)

* RMSProp

![rmsprop1](https://latex.codecogs.com/gif.latex?G_t%20%5Cleftarrow%20%5Cgamma%7BG_%7Bt-1%7D%7D%20&plus;%20%281-%5Cgamma%29%28%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta_t%29%29%5E2)

![rmsprop2](https://latex.codecogs.com/gif.latex?%5Ctheta_%7Bt&plus;1%7D%20%5Cleftarrow%20%5Ctheta_t%20-%20%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7BG_t&plus;%5Cepsilon%7D%7D%5Ccdot%20%5Cnabla_%7B%5Ctheta%7DJ%28%5Ctheta_t%29)

<img src="img/opts.gif" width=350/>
Ref: http://i.imgur.com/2dKCQHh.gif?1

It is difficult to say which optimization method is the best. The performance of each method varies depending on the problem, data set, and network structure. In my case, <strong>RMSProp</strong> showed the best performance.

<a id="5"></a>
# 5. Training

<a id="5.1"></a>
### Cost Function
  * Mean Squared Error

![mse](https://latex.codecogs.com/gif.latex?%5Ctext%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum%5E%7Bn%7D_%7Bi%3D1%7D%7B%28Y_i%20-%20%5Chat%7BY%7D_i%29%5E2%7D)

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
       * Transforming Matrix of Counter-clockwise Rotation: ![rotation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20x%5E%7B%27%7D%20%5C%5C%20y%5E%7B%27%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Ccos%7B%5Ctheta%7D%20%26%20-%5Csin%7B%5Ctheta%7D%20%5C%5C%20%5Csin%7B%5Ctheta%7D%20%26%20%5Ccos%7B%5Ctheta%7D%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5Cend%7Bbmatrix%7D)


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
* Let's reduce generalization error through model averaging.

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
<br><br>
One thing that impressed me was that there was no single overfitted CNN model out of 110 trained models even though I did not apply any normalization. This is because training-time data augmentation has allowed the model to be trained with almost infinite data.
<br><br>
My final score is <strong>6th</strong> and <strong>7th</strong> out of the 175 participating teams according to the public and private learderboards, respectively.


<img src="img/finalscore.png" />

### Private Leaderboard
<img src="img/private.png" />

### Public Leaderboard
<img src="img/public.png" />

