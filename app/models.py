from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class Model:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        """Trains the model on training set"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predicts the sentiments for test set"""
        self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluates the model accuracy"""
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)
