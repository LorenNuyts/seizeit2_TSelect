import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate


class MiniRocketLR:
    def __init__(self, model_load_path=None):
        self.rocket = MiniRocketMultivariate()
        # self.classifier = LogisticRegression(max_iter=1000)
        self.classifier = SGDClassifier()
        self.is_fitted = False

        if model_load_path:
            self.load(model_load_path)

    def fit(self, config, gen_train, gen_val, model_save_path):
        for batch_x, batch_y in gen_train:
            batch_x = batch_x.numpy()
            batch_x = self._ensure_shape(batch_x)
            batch_y = batch_y.numpy()

            if not self.rocket.is_fitted:
                self.rocket.fit(batch_x)
            X_transformed = self.rocket.transform(batch_x)
            y_batch = batch_y[:, 0]  # Assuming binary classification and one-hot encoding
            self.classifier.partial_fit(X_transformed, y_batch, classes=np.array([0, 1]))

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
        X = self._ensure_shape(X)
        X_transformed = self.rocket.transform(X)
        return self.classifier.predict(X_transformed), y_true

    def transform(self, gen_data):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        X = gen_data.data_segs
        X = self._ensure_shape(X)
        return self.rocket.transform(X)

    def _ensure_shape(self, X: np.ndarray):
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
