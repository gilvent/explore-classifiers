import numpy as np
from classifiers.logistic_regression import LogisticRegression
from evaluation.accuracy import accuracy_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import train_test_split, shuffle_train_test_split
from utils.display_helpers import to_accuracy_text
from evaluation.roc_curve import print_roc_curve, predictions_by_threshold


def main():
    dataset = np.loadtxt(
        fname="data/heart_disease_cleveland_processed.txt", delimiter=",", dtype=str
    )

    # Remove rows with missing values
    filtered_dataset = dataset[~np.char.equal(dataset, "?").any(axis=1)]
    filtered_dataset = filtered_dataset.astype(float)
    X = filtered_dataset[:, 0:-1]
    Y = filtered_dataset[:, -1]

    # Since all target > 0 is classified as having heart disease, we convert the value into 1
    Y[Y > 0] = 1

    # Initialize the model
    classes = np.unique(Y)
    train_X, train_Y, test_X, test_Y = shuffle_train_test_split(
        X=X, Y=Y, test_split_ratio=0.3
    )
    weights = np.asarray([0 for _ in range(0, train_X.shape[1])])
    model = LogisticRegression(weights=weights)

    model.train(train_X=train_X, train_Y=train_Y, iterations=3800)

    pred_probabilities = model.predict(test_X=test_X)
    pred_Y = predictions_by_threshold(probabilities=pred_probabilities, threshold=0.5)

    # Confusion matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="Heart Disease Dataset/Logistic Regression",
        info=to_accuracy_text(accuracy=accuracy),
    )

    # ROC Curve
    # We use posterior probabilities for affirmative class (where heart disease classified as PRESENT)
    tpr = []
    fpr = []

    for t in np.arange(0, 1.05, 0.05):
        thresholded_pred = predictions_by_threshold(
            probabilities=pred_probabilities, threshold=t
        )
        conf_matrix = confusion_matrix(
            classes=classes, actual_Y=test_Y, pred_Y=thresholded_pred
        )
        # print(conf_matrix)
        tp = conf_matrix[1][1]
        fp = conf_matrix[0][1]
        tn = conf_matrix[0][0]
        fn = conf_matrix[1][0]

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    print_roc_curve(fpr, tpr)


if __name__ == "__main__":
    main()
