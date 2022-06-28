from os import listdir
from os.path import join, isfile, abspath, dirname
from tkinter import image_names
from tqdm import tqdm
import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DIR_BACKEND = abspath(join(dirname(abspath(__file__))))

def get_quadrants(arr):
    quadrants = [M for SubA in np.split(arr,2, axis = 0) for M in np.split(SubA,2, axis = 1)]
    return quadrants

def get_features_from_image(image):
    np_img = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))        
    quadrants = [*get_quadrants(np_img[:, :, 0]), *get_quadrants(np_img[:, :, 1]), *get_quadrants(np_img[:, :, 2])]
    features = [np.mean(quadrant) for quadrant in quadrants]
    return np.array(features)

def get_features_array(dir_path):
    image_names = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    images = [cv2.imread(name) for name in tqdm(image_names)]
    features = [get_features_from_image(img) for img in tqdm(images)]
    return features

def main():
    ap_inactive_base_path = join(DIR_BACKEND, 'data_ap', '0')
    ap_active_base_path = join(DIR_BACKEND, 'data_ap', '1')

    inactive_features = get_features_array(ap_inactive_base_path)
    active_features = get_features_array(ap_active_base_path)

    data = np.array([*inactive_features, *active_features])
    labels = np.array([0] * len(inactive_features) + [1] * len(active_features))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)

    preds = np.round(bst.predict(dtest)).astype(int)
    print(f"Accuracy: {accuracy_score(y_test, preds)}, Got {np.sum(np.abs(preds - y_test) > 0)}/{len(y_test)} wrong predictions in the testset")

    bst.save_model(join(DIR_BACKEND, 'ap_model.xgb'))

if __name__ == "__main__":
    main()