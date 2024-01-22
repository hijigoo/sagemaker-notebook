import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


# Deserialize the Invoke request body into an object we can perform prediction on
# def input_fn(request_body, request_content_type):
#     pass


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    scaler = StandardScaler()
    scaler.fit(input_object)
    x_norm = scaler.transform(input_object)

    prediction = model.predict(x_norm)
    pred_prob = model.predict_proba(x_norm)
    pred_prob = np.amax(pred_prob, axis=1)
    print(np.array([prediction, pred_prob]))
    return np.array([prediction, pred_prob])


# Serialize the prediction result into the desired response content type
# def output_fn(prediction, content_type):
#     pass


# x_data = [[80, 8, 7], [60, 8, 9], [25, 3, 2], [19, 4, 5], [30, 3, 3]]
# model = model_fn("./model")
# predict_fn(x_data, model)