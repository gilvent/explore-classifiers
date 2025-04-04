import numpy as np
import math
from utils.enums import FeatureType


class NaiveBayes:

    # Use 1-D Gaussian PDF
    def __gaussian_pdf(self, mean: float, std: float, x: float):
        coefficient = 1 / (math.sqrt(2 * math.pi) * std)
        variance = math.pow(std, 2)
        exponent = math.exp(-1 / (2 * variance) * math.pow(x - mean, 2))
        return coefficient * exponent

    def __category_probability_in_class(self, feature_index, cls, x):
        # if the key is not in the map, it means the value of this feature has no occurence in this class during training

        if (feature_index, cls, x) not in self.categorical_feature_freq:
            return 0

        return (
            self.categorical_feature_freq[(feature_index, cls, x)]
            / self.class_freq[cls]
        )

    # nd_inputs = n-dimension inputs, a tuple of input values for multiple features
    def __classify(self, nd_inputs: tuple):
        all_posteriors = []

        # Calculate posterior probability for each class
        for cl in self.classes:
            posterior = self.priori_probabilities[cl]

            # Multiply p(x|w) for each feature in the given class
            for feature_index, x in enumerate(nd_inputs):
                if self.feature_types[feature_index] == FeatureType.NUMERICAL.name:
                    mean, std = self.pdf_params[(feature_index, cl)]
                    posterior *= self.__gaussian_pdf(mean, std, x)

                if self.feature_types[feature_index] == FeatureType.CATEGORICAL.name:
                    posterior *= self.__category_probability_in_class(
                        feature_index, cl, x
                    )

            all_posteriors.append(posterior)

        # Get the index of max posterior and return associated class
        return self.classes[np.argmax(all_posteriors)]

    def train(self, train_X: np.ndarray, train_Y: np.ndarray, feature_types: list):
        if len(train_X) <= 0:
            print("Training failed: Input data required")
            return False
        if len(train_Y) <= 0:
            print("Training failed: Output data required")
            return False
        if len(train_X[0]) != len(feature_types):
            print("Type for each feature required")
            return False

        self.classes = np.unique(train_Y)
        self.feature_types = feature_types
        self.training_train_X = train_X
        self.pdf_params = {}
        self.categorical_feature_freq = {}
        self.priori_probabilities = {}
        self.class_freq = {}

        # Calculate parameters for probability distribution function
        for cl in self.classes:
            rows_with_cl_output = self.training_train_X[train_Y == cl]
            transposed_rows = np.transpose(rows_with_cl_output)

            for feature_index, values in enumerate(transposed_rows):

                # For numerical feature:
                # Store the mean and standard deviation in given class
                if feature_types[feature_index] == FeatureType.NUMERICAL.name:
                    self.pdf_params[(feature_index, cl)] = (
                        np.mean(values),
                        np.std(values),
                    )

                # For categorical feature:
                # Store the frequency of each category in given class
                if feature_types[feature_index] == FeatureType.CATEGORICAL.name:
                    for val in values:
                        if (feature_index, cl, val) in self.categorical_feature_freq:
                            self.categorical_feature_freq[(feature_index, cl, val)] += 1
                        else:
                            self.categorical_feature_freq[(feature_index, cl, val)] = 1

        # Count the classes frequency for priori probabilities P(Y)
        for cl in train_Y:
            if cl in self.class_freq:
                self.class_freq[cl] += 1
            else:
                self.class_freq[cl] = 1

        # Calculate priori probabilities for each class
        for cl in self.classes:
            self.priori_probabilities[cl] = self.class_freq[cl] / len(train_Y)

        return True

    def test(self, test_X: np.ndarray):
        predictions = []

        for nd_inputs in test_X:
            prediction = self.__classify(nd_inputs)
            predictions.append(prediction)

        return predictions
