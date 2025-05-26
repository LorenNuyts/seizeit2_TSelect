import warnings

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sktime.transformations.panel.rocket import MiniRocket
from tensorflow import keras

from net.DL_config import Config


class MiniRocketLR:
    def __init__(self, model_load_path=None):
        self.rocket = MiniRocket()
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_fitted = False

        if model_load_path:
            self.load(model_load_path)

    def fit(self, config, gen_train, gen_val, model_save_path):
        X_train = gen_train.data_segs
        y_train = gen_train.labels[:, 0]  # Assuming binary classification and one-hot encoding

        # If validation data is provided, append it
        if gen_val is not None:
            X_val = gen_val.data_segs
            y_val = gen_val.labels[:, 0]

            X_train = np.concatenate([X_train, X_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)

        # Ensure a correct shape
        X_train = self._ensure_shape(X_train)

        # Fit and transform
        self.rocket.fit(X_train)
        X_transformed = self.rocket.transform(X_train)
        self.classifier.fit(X_transformed, y_train)
        self.is_fitted = True

        # Save model
        self.save(model_save_path)

    def predict(self, gen_test):
        y_aux = []
        for j in range(len(gen_test)):
            _, y = gen_test[j]
            y_aux.append(y)
        true_labels = np.vstack(y_aux)

        y_true = np.empty(len(true_labels), dtype='uint8')
        for j in range(len(y_true)):
            y_true[j] = true_labels[j][1]

        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        X = gen_test.data_segs
        X = self._ensure_shape(X)
        X_transformed = self.rocket.transform(X)
        return self.classifier.predict(X_transformed), y_true

    def transform(self, gen_data):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        X = gen_data.data_segs
        X = self._ensure_shape(X)
        return self.rocket.transform(X)

    def _ensure_shape(self, X):
        if X.ndim == 2:
            return X[:, np.newaxis, :]
        return X

    def save(self, model_save_path):
        joblib.dump({
            "rocket": self.rocket,
            "classifier": self.classifier
        }, model_save_path)
        print(f"Model saved to {model_save_path}")

    def load(self, model_load_path):
        model_data = joblib.load(model_load_path)
        self.rocket = model_data["rocket"]
        self.classifier = model_data["classifier"]
        self.is_fitted = True
        print(f"Model loaded from {model_load_path}")
