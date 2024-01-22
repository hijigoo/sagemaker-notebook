import os
import joblib


# 모델 로드 (Predict 할 때 사용)
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
