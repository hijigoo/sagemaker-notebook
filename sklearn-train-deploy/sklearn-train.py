import csv
import sys
import argparse

import numpy as np
from datetime import datetime
import os

import joblib
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def prepare_datasets(filepath):
    csv_data = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            csv_data.append(row)

    x_data = []
    y_data = []
    for row in csv_data[1:]:
        x_data.append(row[1:])
        y_data.append(1 if row[0] == 'yes' else 0)

    x_data = np.asarray(x_data, dtype='float32')
    y_data = np.asarray(y_data, dtype='float32').ravel()

    # StandardScaler 인스턴스 생성
    scaler = StandardScaler()
    scaler.fit(x_data)

    # transform() 메서드로 정규화 적용
    x_norm = scaler.transform(x_data)

    return x_norm, y_data


def train(args, train_x, train_y) -> MLPClassifier:
    print(" --- Start train --- ")

    print("MLP fit 시작")
    start_time = datetime.now()
    mlp_model = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', solver='sgd', batch_size=args.batch_size,
                              learning_rate_init=args.learning_rate, max_iter=args.epochs, verbose=True)
    mlp_model.fit(train_x, train_y)
    end_time = datetime.now()
    print("MLP fit 실행 시간 : ", end_time - start_time)
    # 고객 코드 끝

    return mlp_model


def test(model, filepath="data/sample_buy_test.csv"):
    print(" --- TEST --- ")
    x, y = prepare_datasets(filepath=filepath)
    predict_y = model.predict(x)
    print('Accuracy: {:.2f}'.format(accuracy_score(y, predict_y)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameter sent by the client
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)

    # Input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    # Load datasets
    train_x, train_y = prepare_datasets(filepath=os.path.join(args.train, "sample_buy_train.csv"))

    # Train
    model = train(args=args, train_x=train_x, train_y=train_y)

    # Test
    test(model=model, filepath=os.path.join(args.test, "sample_buy_test.csv"))

    # Store
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
