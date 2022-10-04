from skimage import io
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, resample
import pandas as pd
import cv2
from datetime import datetime

import pickle
from joblib import dump, load

from skimage.color import rgb2gray

from skimage.util import crop


IMG_WIDTH = 640
IMG_HEIGHT = 352
RESIZE_FACTOR = 5 # 5 seems about right

RESIZE_WIDTH = int(IMG_WIDTH/RESIZE_FACTOR)
RESIZE_HEIGHT = int(IMG_HEIGHT/RESIZE_FACTOR)


# #############################################################################
#                              READING IMAGES
# #############################################################################

def read_images(crop_images = False):

    def imread_convert(f):
        if crop_images:
            return crop(cv2.resize(rgb2gray(io.imread(f)).reshape(352, 640), (RESIZE_WIDTH, RESIZE_HEIGHT)), ((15, 20), (50, 30)), copy=True)
        return cv2.resize(rgb2gray(io.imread(f)).reshape(352, 640), (RESIZE_WIDTH, RESIZE_HEIGHT))

    imgs_aanwezig = io.ImageCollection("data/aanwezig/*.jpg", load_func=imread_convert).concatenate()
    imgs_niets = io.ImageCollection("data/niets/*.jpg", load_func=imread_convert).concatenate()
    imgs_buiten = io.ImageCollection("data/buiten/*.jpg", load_func=imread_convert).concatenate()

    return imgs_aanwezig, imgs_niets, imgs_buiten


def get_shortest_len(a, b):
    return a if a <= b else b 



# confusion matrix
# [[298  13   1]
#  [115 142  49]
#  [  5  15 295]]


def get_shortest_len_ternary(a, b, c):
    s = a
    if b < s:
        s = b
    if c < s:
        s = c
    return s 


def create_dataset(imgs_aanwezig, imgs_niets, imgs_buiten, use_full_length=False, r_state=8):

    if use_full_length:
        X_aanwezig = [[i, 1] for i in imgs_aanwezig]
        X_niets = [[i, 0] for i in imgs_niets]
        X_buiten = [[i, 2] for i in imgs_buiten]
    else:
        s_len = get_shortest_len_ternary(len(imgs_aanwezig), len(imgs_niets), len(imgs_buiten))

        X_aanwezig = [[i, 1] for i in imgs_aanwezig[:s_len]] 
        X_niets = [[i, 0] for i in imgs_niets[:s_len]] 
        X_buiten = [[i, 2] for i in imgs_buiten[:s_len]] 

    X_sorted = X_aanwezig + X_niets + X_buiten
    X = shuffle(X_sorted, random_state=r_state) # 8 or 22 (22 has a bit of everything)

    return X


# #############################################################################

imgs_aanwezig, imgs_niets, imgs_buiten = read_images()
X = create_dataset(imgs_aanwezig, imgs_niets, imgs_buiten)

# #############################################################################
#                              CREATE TEST SET
# #############################################################################
from sklearn.model_selection import train_test_split


train_set, test_set = train_test_split(X, test_size=0.2, random_state=42)

# #############################################################################
#                              TRAIN MODEL
# #############################################################################


def train_model(implementation):
    X_train = [np.ravel(i[0]) for i in train_set]
    y_train = [i[1] for i in train_set]

    print(f"Start: {datetime.now()}")
    model = implementation()
    model.fit(X_train, y_train)
    print(f"End: {datetime.now()}")

    return model, X_train, y_train



def persist_model(model, name: str):
    dump(model, name)
    # dump(svm_clf, "model_bean_small.joblib")
    # model = load("model_bean_small.joblib")

# #############################################################################
#                              TEST MODEL
# #############################################################################

def test_model_manually(model, amount):
    
    tested_images = []
    # predicitons = []

    for i in range(amount):
        instance = test_set[i]

        prediction = model.predict([np.ravel(instance[0])])
        correct = "correct" if prediction[0] == instance[1] else "not correct"
        print(f"For image {i+1} got {prediction[0]=} \tactual={instance[1]} \t{correct}")

        tested_images.append(instance[0]) 

    return tested_images



from sklearn.svm import SVC
model, X_train, y_train = train_model(SVC)

fizz = test_model_manually(model, 9)

# #############################################################################
#                            CROSS VALIDATION SCORE
# #############################################################################
from sklearn.model_selection import cross_val_score


print(f"getting the_cross_val_score")
the_cross_val_score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
print(f"{the_cross_val_score=}\n")



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"getting the_cross_val_score_preprocessed")
the_cross_val_score_preprocessed = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(f"{the_cross_val_score_preprocessed=}\n")


# #############################################################################
#                              CROSS VALIDATION PREDICTION
# #############################################################################
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

print(f"getting cross_vall_predict")
y_train_pred = cross_val_predict(model, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print("confusion matrix")
print(conf_mx)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# #############################################################################
#                              PLOT IMAGES
# #############################################################################

# CREATE FIGURE
def plot_mustis(imgs):
    fig = plt.figure(figsize=(RESIZE_WIDTH, RESIZE_HEIGHT))
    columns = 3
    rows = 3

    for i in range(1, columns*rows +1):
        img = imgs[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap="gray")



plot_mustis(fizz)
# io.imshow(bar)

plt.show(block=True)

# plt.show(block=False)


row_sums = conf_mx.sum(axis=1, keepdims=True)  # returns a 1-dimensional array with row totals
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('actual')
ax1.set_title('predicted')



ax = fig.add_subplot(111)
ax.matshow(norm_conf_mx, cmap=plt.cm.gray)

# alpha = ['predicted\nniets', 'predicted\naanwezig', 'predicted buiten']
# beta = ['actual niets', 'actual aanwezig', 'actual buiten']
# ax.set_xlabels("predicted")
# ax.set_ylabels("actual")
# ax.set_yticklabels(['']+beta)

plt.show()

# #############################################################################
print("Finished.")





















































# #############################################################################
# SOURCES:

# For img reading
# https://scikit-image.org/docs/dev/user_guide/getting_started.html
# https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread_collection



# #############################################################################
