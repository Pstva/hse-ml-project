from src.read_prepare_data import (
    read_cancer_dataset,
    read_spam_dataset,
    train_test_split,
    normalize,
)
from src.metrics import plot_precision_recall, plot_roc_curve


X, y = read_cancer_dataset("cancer.csv")
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
X_train, X_test = normalize(X_train, X_test)
plot_precision_recall(X_train, y_train, X_test, y_test)
plot_roc_curve(X_train, y_train, X_test, y_test, max_k=10)

X, y = read_spam_dataset("spam.csv")
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
X_train, X_test = normalize(X_train, X_test)
plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)
