import numpy as np
from classifiers.multinominal_logistic_regression import MultinominalLogisticRegression
from evaluation.metrics import accuracy_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import train_test_split, shuffle_train_test_split
from utils.display_helpers import to_accuracy_text


def main():
    dataset = np.genfromtxt(fname="data/winequality_white.csv", delimiter=";", dtype=str)
    # Remove header row
    dataset = dataset[1:]
    dataset = dataset.astype(float)
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    classes = np.sort(np.unique(Y))

    train_X, train_Y, test_X, test_Y = shuffle_train_test_split(
        X=X, Y=Y, test_split_ratio=0.3
    )

    # Initialize the model
    weights = np.asarray([[0 for _ in classes] for _ in range(0, train_X.shape[1])])
    bias = np.asarray([0.05 for _ in classes])
    model = MultinominalLogisticRegression(
        weights=weights, bias=bias, unique_classes=classes
    )

    # Train the model
    model.train(train_X=train_X, train_Y=train_Y, iterations=3000, print_losses=False)

    pred_probabilities = model.predict(test_X=test_X)
    pred_Y = [classes[np.argmax(pbb)] for pbb in pred_probabilities]

    # Printing the scores
    header_labels = ["Actual Y", "Prediction"] + [str(c) for c in classes]
    template = " ".join(["%12s"] * len(header_labels))
    headers = template % tuple(header_labels)
    print(headers)
    for i, y in enumerate(test_Y):
        row = (y, pred_Y[i]) + tuple(pred_probabilities[i])
        print(" ".join(["%12.2f"] * len(row)) % tuple(row))

    # Confusion Matrix
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="White Wine Quality/Logistic Regression (0.3 test split ratio)",
        info=to_accuracy_text(accuracy=accuracy),
    )


if __name__ == "__main__":
    main()
