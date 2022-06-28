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

def get_features_array(image_names):
    print("Loading Images...")
    images = [cv2.imread(name) for name in tqdm(image_names)]
    print("Getting Features...")
    features = [get_features_from_image(img) for img in tqdm(images)]
    return features

def main():
    ap_inactive_base_path = join(DIR_BACKEND, 'data_ap', '0')
    ap_active_base_path = join(DIR_BACKEND, 'data_ap', '1')

    inactive_image_names = [join(ap_inactive_base_path, f) for f in listdir(ap_inactive_base_path) if isfile(join(ap_inactive_base_path, f))]
    active_image_names = [join(ap_active_base_path, f) for f in listdir(ap_active_base_path) if isfile(join(ap_active_base_path, f))]

    inactive_features = get_features_array(inactive_image_names)
    active_features = get_features_array(active_image_names)

    data = np.array([*inactive_features, *active_features])
    labels = np.array([0] * len(inactive_features) + [1] * len(active_features))
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dtest = xgb.DMatrix(X_test, label=y_test)
    all_data = xgb.DMatrix(data, label=labels)

    param = {'max_depth':10, 'eta':1, 'objective':'binary:logistic' }
    num_round = 2
    bst = xgb.train(param, all_data, num_round)
    bst.save_model(join(DIR_BACKEND, 'ap_model.xgb'))

    bst_loaded = xgb.Booster(model_file=join(DIR_BACKEND, 'ap_model.xgb'))      
    preds = np.round(bst_loaded.predict(all_data, ntree_limit=bst_loaded.best_ntree_limit)).astype(int)

    print("Wrong images:")
    wrong_pred_indices = np.where(preds != labels)[0]
    wrong_pred_names = [[*inactive_image_names, *active_image_names][i] for i in wrong_pred_indices]
    for name in wrong_pred_names:
        print(name)

    print(f"Accuracy: {accuracy_score(labels, preds)}, Got {np.sum(np.abs(preds - labels) > 0)}/{len(labels)} wrong predictions")


if __name__ == "__main__":
    main()