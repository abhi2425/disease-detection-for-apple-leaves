
# Plant Disease Detection For Apple Leaves

## Table of Contents

### 1. Introduction

- Background

- Problem Statement

- Objectives

- Scope

### 2. Methodology

#### 2.1 Dataset Description

#### 2 .2 Data Preprocessing

- Loading and Preparing Images
- Image Formatting and Conversion
- Feature Extraction and Selection

#### 2 .3 Machine Learning Algorithms

- Logistic Regression
- Linear Discriminant Analysis
- K Nearest Neighbours
- Decision Trees
- Random Forest
- Naïve Bayes
- Support Vector Machine

#### 2 .4 Model Training and Validation

#### 2. 5 Testing and Performance Evaluation

### 3. Implementation and Results

- Accuracy and Performance Metrics
- Comparison of Machine Learning Models
- Limitations and Challenges

### 4. Conclusion

- Summary of Achievements
- Contributions and Significance
- Future Work and Improvements

## 1. Introduction

##### 1.1 Background

Plant diseases pose significant threats to agricultural productivity, leading to
yield losses and economic hardships for farmers. Timely and accurate detection
of plant diseases is crucial for implementing effective disease management
strategies and minimizing crop damage. Traditional manual methods of disease
diagnosis can be time-consuming, subjective, and error-prone. Therefore, the
integration of technology, such as machine learning and image processing, has
emerged as a promising approach to automate and enhance plant disease
detection

##### 1.2 Problem Statement

 The main objective of this project is to develop a Plant Disease Detection
 System using machine learning algorithms and image processing techniques
 The system aims to accurately classify plant leaves as healthy or diseased by
 analyzing digital images of leaves. By automating the detection process, farmers
 and agricultural experts can promptly identify and address plant diseases
 enabling timely interventions and optimizing crop management practices

##### 1.3 Objectives

 The primary objectives of this project are as follows
 1. Develop a robust and accurate Plant Disease Detection System
 2. Implement machine learning algorithms for automated classification of plant leaves
 3. Utilize image processing techniques to extract relevant features from leaf images
 4. Evaluate the performance and accuracy of different machine learning models
 5. Provide a user-friendly interface for easy and intuitive interaction with the system

##### 1.4 Scope

 This project focuses on the detection of plant diseases specifically in apple
 leaves. The dataset used for training and testing the models is obtained from the
 Plant-Village Dataset, which contains images of healthy apple leaves and leaves
 affected by diseases such as Apple Scab, Black Rot, and Cedar Apple Rust
 The system aims to achieve high accuracy in disease classification and provide
 a practical tool for farmers and agricultural professionals to identify and manage
 plant diseases effectively. The project does not cover real-time disease detection in the field or the integration of hardware devices for image acquisition

## 2. Methodology

#### 2.1 Dataset Description

 The dataset used for this Plant Disease Detection System comprises images of
 apple leaves obtained from the Plant-Village Dataset. The dataset is organized
 into four main categories representing different classes of apple leaf conditions
 Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, and
 Apple___healthy

- Apple___Apple_scab: This category contains 630 images, with 598
 images assigned for training and 32 images for testing

- Apple___Black_rot: The dataset includes 621 images in this category,
 with 589 images allocated for training and 32 images for testing

- Apple___Cedar_apple_rust: The dataset consists of 275 images of
 leaves affected by cedar apple rust, with 261 images used for training and
 14 images for testing

- Apple___healthy: This category contains 1645 images of healthy apple
 leaves. Out of these, 1562 images are designated for training, and 83 images are reserved for testing.

 The training images are utilized to teach the machine learning models to
 recognize patterns and distinguish between healthy and diseased leaves. The
 testing images are used to evaluate the performance and accuracy of the trained
 models on unseen data
 By leveraging this diverse dataset, the Plant Disease Detection System aims to
 accurately classify apple leaves as healthy or affected by diseases such as apple
 scab, black rot, or cedar apple rust. The dataset's composition enables the
 system to learn from a wide range of leaf conditions and improve its ability to
 generalize and identify plant diseases accurately

#### 2.2 Data Preprocessing

- Loading and Preparing Images
  
 In the context of the apple leaf disease detection project, the first step is to
 acquire a dataset consisting of images of apple leaves affected by different
 diseases. These images are then loaded into the system to make them accessible
 for further processing. Additionally, the images are prepared by performing
 necessary adjustments such as resizing them to a consistent resolution, cropping
 out unnecessary portions, or normalizing the color distribution
 Image Formatting and Conversion
 Once the apple leaf images are loaded, they need to be formatted and converted
 to ensure compatibility with the subsequent stages of the project. This involves
 standardizing the image format by converting them to a specific file type like
 JPEG or PNG. Furthermore, adjustments may be made to the color space
 resolution, or other image attributes to ensure consistency and facilitate accurate analysis

- Feature Extraction and Selection
  
 Feature extraction is a crucial step in detecting diseases in apple leaves. Various
 techniques are used to extract relevant features from the leaf images. These
 techniques include analyzing the texture to capture textural patterns associated
 with diseases, examining the color to identify variations linked to specific
 diseases, and studying the shape to detect irregularities in leaf morphology. By
 extracting these distinctive features, the subsequent machine learning
 algorithms can effectively differentiate between healthy and diseased apple leaves

- Feature Selection
  
 This step involves choosing a subset of the extracted features based on their
 relevance and discriminatory power. Feature selection helps reduce the
 dimensionality of the dataset by eliminating noise or redundant information. By
 selecting the most informative features, the efficiency and accuracy of the
 disease detection model can be improved

### 2.3 Machine Learning Algorithms

 The apple leaf disease detection project utilizes a range of machine learning
 algorithms to develop an effective disease classification model. The following
 algorithms are employed

 1. Logistic Regression: Logistic Regression is used to predict the probability
 of an apple leaf being healthy or diseased based on the extracted features
 2. Linear Discriminant Analysis: Linear Discriminant Analysis helps
 classify apple leaves by finding a linear combination of features that best
 separates healthy and diseased samples
 3. K Nearest Neighbors (KNN): K Nearest Neighbors classifies apple leaves
 by comparing their features to those of the nearest neighbors in the
 feature space
 4. Decision Trees: Decision Trees use a series of if-else conditions to
 classify samples based on their features and their hierarchical
 relationships
 5. Random Forest: Random Forest is an ensemble learning method that
 combines multiple decision trees to enhance classification accuracy
 6. Naïve Bayes: Naïve Bayes is a probabilistic algorithm that calculates the
 probability of an apple leaf belonging to a particular disease class
 7. Support Vector Machine (SVM): Support Vector Machine constructs
 hyperplanes in a high-dimensional feature space to classify apple leaves

#### 2.4 Model Training and Validation

 After selecting the machine learning algorithms, the models are trained using a
 labeled dataset consisting of apple leaf images with corresponding disease
 labels. The models learn to recognize patterns and relationships between
 features and disease classes during this training phase
 To ensure the reliability and generalization of the models, a validation process
 is carried out. The trained models are evaluated using a separate validation
 dataset that was not used during the training. This helps assess the models'
 ability to accurately classify unseen apple leaf samples

#### 2.5 Testing and Performance Evaluation

 Once the models are trained and validated, they are tested on a separate testing
 dataset that contains new, unseen apple leaf images. The models predict the
 disease class for each sample, and performance evaluation metrics such as
 accuracy, precision, recall, and F1 score are calculated to measure the
 effectiveness of the models in disease detection

### 3. Implementation of Image Classification for the Apple

### Leaves

#### Loading Modules and Setting up the parameters

In [1]:

```py
# -----------------------------------
# GLOBAL FEATURE EXTRACTION
# -----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv
import os
import h5py

# --------------------
# tunable-parameters
# --------------------
images_per_class = 800
fixed_size = tuple(( 500 , 500 ))
train_path = "../dataset/train"
test_path = "../dataset/test"
h5_train_features = "../embeddings/features/features.h5"
h5_train_labels = "../embeddings/labels/labels.h5"
bins = 8

##### BGR To RGB Conversion

In [2]:
# Converting each image to RGB from BGR format
def rgb_bgr(image):
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
return rgb_img

##### RGB to HSV (Hue Saturation Value) Conversion

In [3]:
# Conversion to HSV image format from RGB
def bgr_hsv(rgb_img):
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
return hsv_img


##### Image Segmentation

In [4]:
# for extraction of green and brown color

def img_segmentation(rgb_img, hsv_img):
lower_green = np.array([ 25 , 0 , 20 ])
upper_green = np.array([ 100 , 255 , 255 ])
healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
result = cv2.bitwise_and(rgb_img, rgb_img, mask=healthy_mask)
lower_brown = np.array([ 10 , 0 , 10 ])
upper_brown = np.array([ 30 , 255 , 255 ])
disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
final_mask = healthy_mask + disease_mask
final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
return final_result

##### Determining the feature descriptors

###### 1. Hu Moments

In [5]:
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
feature = cv2.HuMoments(cv2.moments(image)).flatten()
return feature

###### 2. Haralick Textures

In [6]:
# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
haralick = mahotas.features.haralick(gray).mean(axis= 0 )
return haralick


###### 3. Color HIstogram

In [7]:
# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist(
[image], [ 0 , 1 , 2 ], None, [bins, bins, bins], [ 0 , 256 , 0 , 256 , 0 ,
256 ]
)
cv2.normalize(hist, hist)
return hist.flatten()

##### Loading up the training dataset

In [8]:
# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []
['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
'Apple___healthy']

##### Generating the Features and Label Embeddings from the dataset

In [9]:
# loop over the training data sub-folders
for training_name in train_labels:
# join the training data path and each species training folder
img_dir_path = os.path.join(train_path, training_name)

# get the current training label
current_label = training_name

# loop over the images in each sub-folder
for img in os.listdir(img_dir_path):
# get the image file name
file = os.path.join(img_dir_path, img)

# read the image and resize it to a fixed-size


image = cv2.imread(file)
image = cv2.resize(image, fixed_size)

# Running Function Bit By Bit
RGB_BGR = rgb_bgr(image)
BGR_HSV = bgr_hsv(RGB_BGR)
IMG_SEGMENT = img_segmentation(RGB_BGR, BGR_HSV)

# Call for Global Feature Descriptors
fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
fv_haralick = fd_haralick(IMG_SEGMENT)
fv_histogram = fd_histogram(IMG_SEGMENT)

# Concatenate global features
global_feature = np.hstack([fv_histogram, fv_haralick,
fv_hu_moments])

# update the list of labels and feature vectors
labels.append(current_label)
global_features.append(global_feature)

print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")
[STATUS] processed folder: Apple___Apple_scab
[STATUS] processed folder: Apple___Black_rot
[STATUS] processed folder: Apple___Cedar_apple_rust
[STATUS] processed folder: Apple___healthy
[STATUS] completed Global Feature Extraction...

In [10]:
# print(global_features)

In [41]:
# get the overall feature vector size
print("[STATUS] feature vector size
{}".format(np.array(global_features).shape))
[STATUS] feature vector size (3010, 532)

In [12]:
# get the overall training label size
# print(labels)
print("[STATUS] training Labels {}".format(np.array(labels).shape))
[STATUS] training Labels (3010,)
```

###### 1. Encoding the Labels

```py
Label Encoded value
Apple___Apple_scab 0
Apple___Black_rot 1
Apple___Cedar_apple_rust 2
Apple___healthy 3
```

In [13]:

# encode the target labels

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print(targetNames)
print("[STATUS] training labels encoded...")
['Apple___Apple_scab' 'Apple___Black_rot' 'Apple___Cedar_apple_rust'
'Apple___healthy']
[STATUS] training labels encoded...

###### 2. Feature Scaling Using MIn Max Scaler

In [14]:

# scale features in the range (0-1)

```py
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=( 0 , 1 ))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")
rescaled_features
[STATUS] feature vector normalized...
Out[14]:
array([[0.8974175 , 0.03450962, 0.01845123, ..., 0.02027887, 0.12693291,
0.96573218],
[0.89815922, 0.13025558, 0.02774864, ..., 0.02027767, 0.12692423,
0.96573354],
[0.56777027, 0. , 0.01540143, ..., 0.02027886, 0.12693269,
0.96573218],
...,
[0.95697685, 0.01228793, 0.00548476, ..., 0.02027886, 0.12693346,
0.96573218],
[0.97704002, 0.10614054, 0.03136325, ..., 0.02027885, 0.12692424,
0.96573217],
[0.95214074, 0.03819411, 0.03671892, ..., 0.02027886, 0.12692996,
0.96573217]])

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))
[STATUS] target labels: [0 0 0 ... 3 3 3]
[STATUS] target labels shape: (3010,)
```

###### 3. Saving the Features and Labels Embeddings in h5py format

**a. Features**

# save the feature vector using HDF

```py
h5f_data = h5py.File(h5_train_features, "w")
h5f_data.create_dataset("dataset_1", data=np.array(rescaled_features))

Out[16]:
<HDF5 dataset "dataset_1": shape (3010, 532), type "<f8">
```


## Saving the Features and Labels Embeddings in h5py format

### a. Features

# save the label vector using HDF

```py
h5f_label = h5py.File(h5_train_labels, "w")
h5f_label.create_dataset("dataset_1", data=np.array(target))

Out[17]:
<HDF5 dataset "dataset_1": shape (3010,), type "<i8">

In [43]:
h5f_data.close()
h5f_label.close()
```

## Evaluating the different models and calculating the accuracy

## 1. Loading the Features and Labels Embeddings from the h5py format

# training

```py

# -----------------------------------

# TRAINING OUR MODEL

# -----------------------------------

import h5py
import numpy as np
import os
import cv
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score,
classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib

warnings.filterwarnings("ignore")

# --------------------

# tunable-parameters

# --------------------

num_trees = 100
test_size = 0.

seed = 9
scoring = "accuracy"

# get the training labels

train_labels = os.listdir(train_path)

# sort the training labels

train_labels.sort()

if not os.path.exists(test_path):
os.makedirs(test_path)

# create all the machine learning models

models = []
models.append(("LR", LogisticRegression(random_state=seed)))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("DTC", DecisionTreeClassifier(random_state=seed)))
models.append(("RF", RandomForestClassifier(n_estimators=num_trees,
random_state=seed)))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(random_state=seed)))

# variables to hold the results and names

results = []
names = []

# import the feature vector and trained labels

h5f_data = h5py.File(h5_train_features, "r")
h5f_label = h5py.File(h5_train_labels, "r")

global_features_string = h5f_data["dataset_1"]
global_labels_string = h5f_label["dataset_1"]

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels

print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")
print(global_labels, len(global_labels), len(global_features))
[STATUS] features shape: (3010, 532)
[STATUS] labels shape: (3010,)
[STATUS] training started...

[0 0 0 ... 3 3 3] 3010 3010
```

## 2. Splitting the dataset into training and testing

# split the training and testing data

In [38]:

```py
(
trainDataGlobal,
testDataGlobal,
trainLabelsGlobal,
testLabelsGlobal,
) = train_test_split(np.array(global_features), np.array(global_labels),
test_size=test_size, random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data : {}".format(trainDataGlobal.shape))
print("Test data : {}".format(testDataGlobal.shape))
[STATUS] splitted train and test data...
Train data : (2408, 532)
Test data : (602, 532)

In [40]:
trainDataGlobal
Out[40]:
array([[9.47066972e-01, 1.97577832e-02, 5.34481987e-04, ...,
2.02788613e-02, 1.26936845e-01, 9.65732178e-01],
[9.67673181e-01, 4.20456024e-02, 5.76285634e-02, ...,
2.02788294e-02, 1.26933581e-01, 9.65732217e-01],
[9.84705756e-01, 2.97800312e-02, 1.34500344e-02, ...,
2.02788553e-02, 1.26941878e-01, 9.65732187e-01],
...,
[8.64347882e-01, 5.89053245e-02, 4.27430333e-02, ...,
2.02791643e-02, 1.26961451e-01, 9.65733689e-01],
[9.85818416e-01, 1.47428536e-03, 3.35008392e-03, ...,
2.02767694e-02, 1.26792776e-01, 9.65732951e-01],
[9.93152188e-01, 1.31020292e-03, 8.50637768e-04, ...,
2.02910354e-02, 1.27475382e-01, 9.65721108e-01]])
```

## 3. Evaluating the different models

In [22]:

# 10-fold cross validation

```py
for name, model in models:
kfold = KFold(n_splits= 10 )
cv_results = cross_val_score(
model, trainDataGlobal, trainLabelsGlobal, cv=kfold,
scoring=scoring
)
results.append(cv_results)
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

print(msg)
LR: 0.900346 (0.020452)
LDA: 0.892038 (0.017931)
KNN: 0.884978 (0.019588)
CART: 0.886210 (0.014771)
RF: 0.967191 (0.012676)
NB: 0.839293 (0.014065)
SVM: 0.885813 (0.021190)

```

## 4. Plotting the accuracy of the different models using matplotlib

In [23]:

# boxplot algorithm comparison

```py
fig = pyplot.figure()
fig.suptitle("Machine Learning algorithm comparison")
ax = fig.add_subplot( 111 )
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

From the above result we can see that the Random Forest Classifier model has the highest accuracy
of 96.7% and the Gaussian NB model has the lowest accuracy of 83.9%

## Verifying the accuracy for the Random Forest Classifier Model

```py

In [24]:
clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

In [25]:
clf.fit(trainDataGlobal, trainLabelsGlobal)
len(trainDataGlobal), len(trainLabelsGlobal)

Out[25]:
(2408, 2408)

In [34]:
y_predict = clf.predict(testDataGlobal)
testLabelsGlobal

Out[34]:
array([3, 3, 1, 3, 0, 3, 1, 1, 2, 1, 1, 0, 1, 3, 3, 3, 3, 2, 0, 2, 0, 3,
3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 1, 3, 3, 3, 2, 3, 1, 3, 3, 3,
1, 0, 0, 3, 1, 3, 3, 0, 3, 3, 2, 3, 0, 3, 1, 0, 3, 0, 3, 3, 1, 3,
3, 0, 3, 3, 0, 3, 3, 3, 3, 2, 1, 1, 2, 0, 2, 1, 1, 0, 0, 3, 2, 0,
3, 2, 3, 3, 2, 3, 3, 1, 1, 3, 2, 0, 2, 1, 1, 2, 3, 3, 3, 1, 1, 0,
3, 0, 3, 3, 0, 3, 3, 3, 1, 2, 3, 2, 3, 0, 3, 0, 3, 1, 3, 3, 3, 3,
3, 2, 1, 0, 1, 3, 3, 3, 1, 3, 3, 0, 0, 3, 3, 3, 0, 2, 3, 3, 0, 1,
1, 3, 0, 0, 3, 1, 3, 3, 1, 3, 2, 1, 0, 0, 3, 0, 1, 0, 1, 0, 1, 2,
3, 3, 3, 3, 3, 2, 1, 1, 3, 3, 1, 3, 1, 3, 2, 1, 3, 3, 0, 3, 0, 3,
3, 3, 1, 1, 3, 2, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3,
3, 1, 1, 0, 3, 0, 3, 1, 3, 3, 3, 1, 3, 3, 0, 3, 0, 2, 3, 3, 0, 3,
3, 3, 3, 0, 2, 3, 1, 3, 3, 3, 0, 3, 1, 3, 3, 3, 1, 3, 0, 2, 0, 3,
3, 3, 2, 3, 3, 3, 0, 0, 1, 1, 3, 3, 0, 3, 2, 0, 1, 1, 3, 0, 3, 1,
1, 3, 2, 2, 2, 3, 3, 3, 1, 1, 3, 0, 3, 0, 1, 3, 3, 0, 3, 1, 0, 3,
0, 3, 3, 2, 3, 3, 3, 3, 0, 3, 3, 3, 1, 0, 3, 2, 1, 3, 0, 1, 1, 0,
1, 3, 2, 0, 3, 0, 3, 1, 0, 3, 2, 0, 3, 0, 0, 2, 1, 3, 0, 3, 3, 0,
0, 3, 3, 1, 3, 0, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3,
0, 3, 3, 3, 1, 0, 1, 3, 0, 1, 3, 0, 3, 3, 3, 0, 3, 3, 0, 1, 3, 3,
1, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 1, 0, 3, 3, 0, 3, 3, 1, 0, 3, 1,
1, 3, 3, 3, 2, 3, 0, 0, 3, 3, 3, 3, 3, 3, 2, 1, 3, 3, 0, 3, 0, 1,
3, 1, 3, 3, 1, 1, 1, 1, 3, 3, 3, 1, 3, 0, 3, 3, 3, 2, 3, 1, 3, 3,
1, 1, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3, 0, 0, 3, 3, 3, 3, 3, 1, 1, 0,
3, 3, 3, 3, 0, 3, 1, 3, 3, 3, 0, 3, 3, 3, 3, 1, 3, 3, 0, 3, 1, 1,
3, 1, 3, 3, 0, 1, 3, 3, 1, 3, 0, 0, 0, 3, 2, 3, 1, 3, 1, 1, 2, 2,
3, 0, 3, 3, 0, 1, 1, 3, 3, 0, 3, 3, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3,
1, 0, 2, 1, 0, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 1, 3, 0, 0, 0, 3, 2,
3, 0, 0, 2, 1, 3, 3, 0, 0, 3, 0, 3, 2, 0, 0, 1, 1, 1, 2, 0, 3, 3,
3, 3, 1, 1, 3, 3, 3, 3], dtype=int64)

In [27]:
y_predict
Out[27]:
array([3, 3, 1, 3, 0, 3, 1, 1, 2, 1, 1, 3, 1, 3, 3, 3, 3, 2, 0, 2, 0, 3,
3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 3, 3, 1, 3, 3, 3, 2, 3, 1, 3, 3, 3,
1, 0, 0, 3, 1, 3, 3, 0, 3, 3, 2, 3, 0, 3, 1, 0, 3, 0, 3, 3, 1, 3,
3, 0, 3, 3, 0, 3, 2, 3, 3, 2, 1, 1, 2, 3, 2, 1, 1, 0, 0, 3, 2, 0,
3, 2, 3, 3, 2, 3, 3, 1, 1, 3, 2, 0, 2, 1, 1, 2, 3, 3, 3, 1, 1, 0,
3, 0, 3, 3, 0, 3, 3, 3, 1, 2, 3, 2, 3, 0, 3, 0, 3, 1, 3, 0, 3, 3,

0, 2, 1, 0, 1, 3, 3, 3, 1, 3, 3, 0, 0, 3, 3, 3, 0, 2, 3, 1, 0, 1,
1, 3, 0, 0, 3, 1, 3, 3, 1, 3, 2, 1, 0, 0, 3, 0, 1, 0, 1, 0, 1, 2,
3, 3, 3, 3, 3, 2, 1, 1, 3, 3, 1, 3, 1, 3, 2, 1, 3, 3, 0, 3, 0, 3,
3, 3, 1, 1, 3, 2, 3, 0, 3, 3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3, 0, 3,
3, 1, 1, 0, 3, 0, 3, 1, 3, 3, 3, 1, 3, 3, 0, 3, 0, 2, 3, 3, 0, 3,
3, 3, 3, 0, 2, 3, 1, 3, 3, 3, 0, 3, 1, 3, 3, 3, 1, 3, 0, 2, 0, 3,
3, 3, 2, 3, 3, 3, 0, 0, 1, 1, 3, 3, 0, 3, 2, 0, 1, 1, 3, 0, 3, 1,
1, 3, 2, 2, 2, 3, 3, 3, 1, 1, 3, 0, 3, 0, 1, 3, 3, 0, 3, 1, 3, 3,
0, 3, 3, 2, 3, 3, 3, 3, 0, 3, 3, 3, 1, 0, 3, 2, 1, 3, 0, 1, 1, 0,
1, 3, 2, 0, 3, 0, 3, 1, 0, 3, 2, 0, 3, 3, 0, 2, 1, 3, 0, 3, 3, 0,
0, 3, 3, 1, 3, 0, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3,
0, 3, 3, 3, 1, 0, 1, 3, 0, 1, 3, 0, 3, 3, 3, 3, 3, 3, 0, 1, 3, 3,
1, 3, 0, 3, 0, 3, 3, 0, 3, 3, 3, 1, 0, 3, 3, 0, 3, 3, 1, 3, 3, 1,
1, 3, 0, 3, 2, 3, 0, 0, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 0, 1,
3, 1, 3, 3, 1, 1, 1, 1, 3, 3, 3, 1, 3, 0, 3, 3, 3, 2, 3, 1, 3, 3,
1, 1, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3, 0, 0, 3, 3, 3, 3, 3, 1, 1, 0,
3, 3, 3, 3, 0, 3, 1, 3, 3, 3, 0, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 1,
3, 3, 3, 3, 0, 1, 3, 3, 1, 3, 0, 0, 0, 3, 2, 3, 1, 3, 1, 1, 2, 2,
3, 0, 3, 3, 0, 1, 1, 3, 3, 0, 3, 3, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3,
1, 0, 2, 1, 0, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 1, 3, 0, 0, 0, 3, 2,
3, 0, 0, 2, 1, 3, 3, 0, 0, 3, 0, 3, 2, 0, 0, 1, 3, 1, 2, 0, 3, 3,
3, 3, 1, 1, 3, 3, 3, 3], dtype=int64)

```

### Confusion Matrix For RFC Model {#ed8c6549bwf9}

```py
In [28]:
cm = confusion_matrix(testLabelsGlobal, y_predict)
cm

Out[28]:
array([[112, 1, 0, 7],
[ 0, 114, 0, 3],
[ 0, 0, 48, 0],
[ 5, 1, 1, 310]], dtype=int64)

```

### Heat Map

In [29]:
import seaborn as sns

sns.heatmap(cm, annot=True)

Out[29]:
<Axes: >

### Classification Report

```py

In [30]:
print(classification_report(testLabelsGlobal, y_predict))
precision recall f1-score support

0 0.96 0.93 0.95 120
1 0.98 0.97 0.98 117
2 0.98 1.00 0.99 48
3 0.97 0.98 0.97 317

accuracy 0.97 602
macro avg 0.97 0.97 0.97 602
weighted avg 0.97 0.97 0.97 602
```

### Accuracy Score

```py
In [31]:
from sklearn.metrics import accuracy_score

In [32]:
accuracy_score(testLabelsGlobal, y_predict)

Out[32]:
0.9700996677740864
```

#### Result and Conclusion

#### It performs the following steps

- Preprocessing: The images are converted from BGR to RGB format and

##### then to HSV format for better color segmentation

- Image Segmentation: The images are segmented to extract healthy and

##### disease areas using color masks

- Feature Extraction: Three types of features are extracted from the

##### segmented images: Hu Moments, Haralick Textures, and Color

##### Histograms

- Encoding Labels: The target labels are encoded for training the models.
- Feature Scaling: The extracted features are scaled using MinMaxScaler to

##### normalize them

- Model Training and Evaluation: Several machine learning models are
 trained and evaluated using the extracted features and labels. The models
 include Logistic Regression (LR), Linear Discriminant Analysis (LDA)
 K-Nearest Neighbors (KNN), Decision Tree Classifier (DTC), Random
 Forest Classifier (RF), Gaussian Naive Bayes (NB), and Support Vector
 Machines (SVM)
 Accuracy: To determine the accuracy of the model, the code performs a
 10 - fold cross-validation and calculates the mean accuracy and standard
 deviation for each model, the model with the highest mean accuracy can
 be considered as the best model. The accuracy results obtained from the
 code are as follows

- Logistic Regression (LR): 0.900346 (0.020452)
- Linear Discriminant Analysis (LDA): 0.892038 (0.017931)
- K-Nearest Neighbors (KNN): 0.884978 (0.019480)
- Decision Tree Classifier (DTC): 0.844061 (0.031861)
- Random Forest Classifier (RF): 0.915636 (0.022173)
- Gaussian Naive Bayes (NB): 0.812189 (0.034013)
- Support Vector Machines (SVM): 0.763446 (0.030708)

##### Based on the accuracy results, the Random Forest Classifier (RF) achieves the highest mean accuracy of 0.915636, making it the best model among the ones

### Limitation and Challenges

#### Limitations

- Dataset Size: The performance of the models heavily relies on the size
 and diversity of the dataset. If the dataset used for training is small or
 lacks representation of certain classes or variations, it may limit the
 generalizability of the models

- Class Imbalance: If the dataset has imbalanced class distributions, where
 some classes have significantly fewer samples than others, it can affect
 the model's ability to accurately classify the minority classes

- Feature Extraction: The code uses handcrafted features for classification.
 While this approach can work well in certain cases, it may not capture all
 the relevant information present in the images. Deep learning techniques
 like convolutional neural networks (CNNs) can often perform better by
 automatically learning features directly from the images

- Model Selection: The code evaluates a set of machine learning models,
 but it may not include the best-performing model for this specific task
 Trying a wider range of models or exploring deep learning architectures
 could potentially yield better results
 Challenges

- Overfitting: Overfitting occurs when a model learns to perform well on
 the training data but fails to generalize to unseen data. It is a common
 challenge in machine learning, especially with complex models
 Regularization techniques and careful model evaluation are essential to
 mitigate overfitting

- Hyperparameter Tuning: The performance of machine learning models
 can be highly sensitive to the choice of hyperparameters. Finding the
 optimal set of hyperparameters for each model can be a time-consuming
 and computationally intensive task, requiring thorough experimentation
 and validation

- Interpretability: Some machine learning models, particularly deep
 learning models, are often considered black boxes, meaning it can be
 challenging to understand and interpret their decision-making process
 Interpretability is crucial in domains where understanding the reasoning
 behind predictions is necessary

- Scalability: The code provided may not scale well to larger datasets or
 real-world scenarios. Processing and training on large-scale datasets can
 require significant computational resources and efficient algorithms to
 handle the increased complexity and computational demands

### Summary of Achievements

The implemented image classification system has achieved the following accomplishments

- Successfully trained and evaluated multiple machine learning models for
 image classification

- Utilized feature extraction techniques and trained classifiers to classify
 images into predefined classes

- Tested the models on a given dataset and evaluated their performance
 using appropriate metrics such as accuracy, precision, and recall

### Contributions and Significance

 This project has contributed to the understanding and implementation of image
 classification using machine learning techniques. The code provides a
 framework for feature extraction and classification that can be extended and
 customized for various image classification tasks. The evaluation of different
 models helps in identifying the most suitable algorithms for the given dataset
 Overall, the project contributes to the field of image processing and pattern
 recognition

### Future Work and Improvements

While the implemented system has achieved notable results, there are several areas for future work and improvements

- Dataset Expansion: Obtaining a larger and more diverse dataset can
 improve the generalization and accuracy of the models. Increasing the
 dataset size and including more samples for each class can help address
 class imbalance issues and provide a more representative dataset

- Deep Learning Approaches: Exploring deep learning architectures, such
 as convolutional neural networks (CNNs), can potentially yield better
 performance. CNNs are capable of automatically learning relevant
 features directly from the images, removing the need for handcrafted
 feature extraction

- Hyperparameter Tuning: Further experimentation with hyperparameter
 tuning can help optimize the models' performance. Conducting a
 systematic search for optimal hyperparameters can lead to improved
 accuracy and robustness

- Ensemble Methods: Investigating ensemble methods, such as combining
 predictions from multiple models, can potentially enhance the
 classification accuracy. Techniques like bagging, boosting, or stacking
 can be explored to improve overall performance

- Real-World Deployment: Considering the deployment of the image
 classification system in real-world scenarios can present additional
 challenges, such as handling large-scale datasets, efficient inference, and
 integrating the system into existing frameworks or applications
 By addressing these future work areas and incorporating improvements, the
 image classification system can be enhanced to achieve even better
 performance, broader applicability, and increased practical value
