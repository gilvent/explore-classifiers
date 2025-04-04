import numpy as np

def train_test_split(X, Y, unique_classes):
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    for c in unique_classes:
        # Indexes where c is the target
        indexes_of_c_target = np.where(Y == c)
        features_list = X[indexes_of_c_target]
        targets = Y[indexes_of_c_target]
        test_data_count = int(0.3 * len(features_list))
        split_index = len(features_list) - test_data_count

        train_X.extend(features_list[:split_index])
        test_X.extend(features_list[split_index:])
        train_Y.extend(targets[:split_index])
        test_Y.extend(targets[split_index:])

    return (np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y))