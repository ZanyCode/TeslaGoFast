from os import listdir
from os.path import join, isfile, abspath, dirname
from tkinter import image_names
from tqdm import tqdm
import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from detector import get_features_from_image


DIR_BACKEND = abspath(join(dirname(abspath(__file__))))

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
    all_data = xgb.DMatrix(data, label=labels)

    param = {'max_depth':10, 'eta':1, 'objective':'binary:logistic' }
    num_round = 2
    # bst = xgb.train(param, all_data, num_round)
    bst = xgb.Booster(model_file=join(DIR_BACKEND, 'ap_model.xgb'))   

    # preds = np.round(bst.predict(dtest)).astype(int)
    # print(f"Accuracy: {accuracy_score(y_test, preds)}, Got {np.sum(np.abs(preds - y_test) > 0)}/{len(y_test)} wrong predictions in the testset")
    preds = np.round(bst.predict(all_data, ntree_limit=bst.best_ntree_limit)).astype(int)
    print(f"Accuracy: {accuracy_score(labels, preds)}, Got {np.sum(np.abs(preds - labels) > 0)}/{len(labels)} wrong predictions in the testset")

    bst.save_model(join(DIR_BACKEND, 'ap_model.xgb'))

if __name__ == "__main__":
    main()