import numpy as np
from classifiers.naive_bayes import NaiveBayes
from evaluation.metrics import accuracy_score, f1_score, recall_score, precision_score
from evaluation.confusion_matrix import confusion_matrix, display_confusion_matrix
from utils.data_preprocess import shuffle_train_test_split
from utils.enums import FeatureType
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
    feature_types = [
        FeatureType.NUMERICAL.name,  # age
        FeatureType.CATEGORICAL.name,  # sex
        FeatureType.CATEGORICAL.name,  # cp
        FeatureType.NUMERICAL.name,  # trestbps
        FeatureType.NUMERICAL.name,  # chol
        FeatureType.CATEGORICAL.name,  # fbs
        FeatureType.CATEGORICAL.name,  # restecg
        FeatureType.NUMERICAL.name,  # thalach
        FeatureType.CATEGORICAL.name,  # exang
        FeatureType.NUMERICAL.name,  # oldpeak
        FeatureType.CATEGORICAL.name,  # slope
        FeatureType.NUMERICAL.name,  # ca
        FeatureType.CATEGORICAL.name,  # thal
    ]
    classifier = NaiveBayes(unique_classes=classes, feature_types=feature_types)

    # Fit to the model
    train_X, train_Y, test_X, test_Y = shuffle_train_test_split(X=X, Y=Y)
    classifier.train(train_X=train_X, train_Y=train_Y)

    output = classifier.output(test_X=test_X)
    pred_Y = output["predictions"]

    # Confusion Matrix
    conf_matrix = confusion_matrix(classes=classes, actual_Y=test_Y, pred_Y=pred_Y)
    accuracy = accuracy_score(actual_Y=test_Y, pred_Y=pred_Y)
    recall = recall_score(tp=conf_matrix[1][1], fn=conf_matrix[1][0])
    precision = precision_score(tp=conf_matrix[1][1], fp=conf_matrix[0][1])
    f1 = f1_score(recall=recall, precision=precision)
    info_text = f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}"

    display_confusion_matrix(
        conf_matrix=conf_matrix,
        classes=classes,
        title="Heart Disease/Naive Bayes",
        info=info_text,
    )

    # ROC Curve
    # We use posterior probabilities for affirmative class (where heart disease classified as PRESENT)
    posteriors = np.array(output["posteriors"])
    affirmative_class_posteriors = posteriors[:, 1]
    tpr = []
    fpr = []

    for t in np.arange(0, 1.05, 0.05):
        thresholded_pred = predictions_by_threshold(
            probabilities=affirmative_class_posteriors, threshold=t
        )
        conf_matrix = confusion_matrix(
            classes=classes, actual_Y=test_Y, pred_Y=thresholded_pred
        )
        tp = conf_matrix[1][1]
        fp = conf_matrix[0][1]
        tn = conf_matrix[0][0]
        fn = conf_matrix[1][0]

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    print_roc_curve(fpr, tpr)


if __name__ == "__main__":
    main()
