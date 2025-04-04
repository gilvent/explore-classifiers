import numpy as np


def confusion_matrix(classes, actual_Y, pred_Y):
    counts = {}
    conf_matrix = []

    for actual in classes:
        for pred in classes:
            counts[(actual, pred)] = 0

    for actual, pred in zip(actual_Y, pred_Y):
        counts[(actual, pred)] += 1

    for row, actual_val in enumerate(classes):
        conf_matrix.append([])
        for pred_val in classes:
            conf_matrix[row].append(counts[(actual_val, pred_val)])

    return conf_matrix

def print_confusion_matrix(conf_matrix, classes, template = "%12s"):
    
    template = " ".join([template] * (len(classes) + 1))
    header_labels = ["Predicted"] + [str(cl) for cl in classes]
    header = template % tuple(header_labels)
    secondary_header_labels = ["Actual"] + [""] * len(classes)
    secondary_header = template % tuple(secondary_header_labels)

    print("Confusion Matrix:")
    print(header)
    print(secondary_header)

    for row, counts in enumerate(conf_matrix):
        row_header_label = str(classes[row])
        labels = [row_header_label] + [str(val) for val in counts]
        row = template % tuple(labels)
        print(row)
