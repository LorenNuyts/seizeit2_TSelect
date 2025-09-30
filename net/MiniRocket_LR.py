import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sktime.datatypes._panel._convert import from_3d_numpy_to_multi_index
from sktime.transformations.panel.rocket import MiniRocketMultivariate


class MiniRocketLR:
    def __init__(self, model_load_path=None):
        self.rocket = MiniRocketMultivariate()
        self.collect_all_features = True
        if self.collect_all_features:
            self.classifier = LogisticRegression(max_iter=1000)
        else:
            self.classifier = SGDClassifier()
        self.is_fitted = False

        if model_load_path:
            self.load(model_load_path)

    def fit(self, config, gen_train, gen_val, model_save_path):
        all_x: List[pd.DataFrame] = []
        all_y: List[np.ndarray] = []
        for batch_x, batch_y in gen_train:
            batch_x = batch_x.numpy()
            batch_x = self._ensure_3Dshape(batch_x)
            batch_x = np.transpose(batch_x, (0, 2, 1))  # (n_samples, n_channels, n_timestamps)
            # print("Batch x shape:", batch_x.shape)
            batch_x = from_3d_numpy_to_multi_index(batch_x)
            # print("Converted batch x shape:", batch_x.index.shape, "columns:", batch_x.columns)
            batch_y = batch_y.numpy()

            if not self.rocket.is_fitted:
                self.rocket.fit(batch_x)
            X_transformed = self.rocket.transform(batch_x)
            if self.collect_all_features:
                all_x.append(X_transformed)
                all_y.append(batch_y)
            else:
                y_batch = batch_y[:, 0]  # Assuming binary classification and one-hot encoding
                self.classifier.partial_fit(X_transformed, y_batch, classes=np.array([0, 1]))

        if self.collect_all_features:
            X = pd.concat(all_x)
            y = np.vstack(all_y)
            y = y[:, 0]  # Assuming binary classification and one-hot encoding
            self.classifier.fit(X, y)

        self.is_fitted = True

        # Save model
        self.save(model_save_path)

    def predict(self, gen_test):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        y_aux = []
        for j in range(len(gen_test)):
            _, y = gen_test[j]
            y_aux.append(y)
        true_labels = np.vstack(y_aux)

        y_true = np.empty(len(true_labels), dtype='uint8')
        for j in range(len(y_true)):
            y_true[j] = true_labels[j][1]

        X = gen_test.data_segs
        X = self._ensure_3Dshape(X)
        X = np.transpose(X, (0, 2, 1))  # (n_samples, n_channels, n_timestamps)
        X_transformed = self.rocket.transform(X)
        return self.classifier.predict(X_transformed), y_true

    def _ensure_3Dshape(self, X: np.ndarray):
        if X.ndim == 2:
            return X[:, np.newaxis, :]
        elif X.ndim == 4:
            return X.reshape(X.shape[0], X.shape[1], -1)
        return X

    def save(self, model_save_path):
        model_save_path = os.path.join(model_save_path, 'MiniRocketLR_model.joblib')
        joblib.dump({
            "rocket": self.rocket,
            "classifier": self.classifier
        }, model_save_path)
        print(f"Model saved to {model_save_path}")

    def load(self, model_load_path):
        model_load_path = os.path.join(model_load_path, 'MiniRocketLR_model.joblib')
        model_data = joblib.load(model_load_path)
        self.rocket = model_data["rocket"]
        self.classifier = model_data["classifier"]
        self.is_fitted = True
        print(f"Model loaded from {model_load_path}")
