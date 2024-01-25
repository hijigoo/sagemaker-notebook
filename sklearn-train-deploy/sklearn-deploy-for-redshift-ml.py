import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# 모델 로드 (Predict 할 때 사용)
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, request_content_type):
    print("# Start input_fn")
    print(request_body)
    data = []
    lines = [line for line in request_body.splitlines() if line.strip()]
    for line in lines:
        values = [x for x in line.split(",") if x]
        data.append(np.array(values, dtype=int))

    arr = np.array(data)
    print("# End input_fn")
    return data


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    print("# Start predict_fn")
    print(input_object)

    scaler = StandardScaler()
    scaler.fit(input_object)
    x_norm = scaler.transform(input_object)

    prediction = model.predict(x_norm)
    print(prediction)
    print("# End predict_fn")
    return prediction


# Serialize the prediction result into the desired response content type
# def output_fn(prediction, content_type):
#     pass
