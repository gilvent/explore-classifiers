import numpy as np
import math
from utils.enums import FeatureType


class NaiveBayes:

    # Use 1-D Gaussian PDF formula
    def __gaussian_pdf(self, mean: float, std: float, x: float):
        coefficient = 1 / (math.sqrt(2 * math.pi) * std)
        variance = math.pow(std, 2)
        exponent = math.exp(-1 / (2 * variance) * math.pow(x - mean, 2))
        return coefficient * exponent

    def __probability_by_frequency(self, feature_index, cls, x):
        # If the key is not in the map, then this feature has no occurence in this class during training.
        if (feature_index, cls, x) not in self.feature_freq:
            return 0

        return self.feature_freq[(feature_index, cls, x)] / self.class_freq[cls]

    def __likelihood(self, target_class, feature_index, x):
        if self.feature_types[feature_index] == FeatureType.NUMERICAL.name:
            mean, std = self.pdf_params[(feature_index, target_class)]
            return self.__gaussian_pdf(mean, std, x)

        if self.feature_types[feature_index] == FeatureType.CATEGORICAL.name:
            return self.__probability_by_frequency(feature_index, target_class, x)

        return 0

    def __scaling_factor(self, nd_features):
        scaling_factor = 0

        for cl in self.classes:
            map_key = self.__get_likelihoods_map_key(
                target_class=cl, nd_features=nd_features
            )
            scaling_factor += self.prioris[cl] * self.calculated_likelihoods[map_key]

        return scaling_factor

    # nd_features = n-dimension features, a tuple of multiple features for single prediction
    def __classify(self, nd_features: tuple):
        all_posteriors = []

        # Calculate posterior probability for each class
        for cl in self.classes:
            likelihood_product = 1

            # Multiply likelihoods when there are multple features
            for feature_index, x in enumerate(nd_features):
                likelihood = self.__likelihood(
                    target_class=cl, feature_index=feature_index, x=x
                )
                likelihood_product *= likelihood

            # Store the likelihood product for scaling factor calculation
            map_key = self.__get_likelihoods_map_key(
                target_class=cl, nd_features=nd_features
            )
            self.calculated_likelihoods[map_key] = likelihood_product

            # The actual Posterior Probability requires division by Scaling Factor.
            # We ignore Scaling Factor because we only want to compare the result between classes,
            # and the scaling factor will be the same for all classes
            posterior = self.prioris[cl] * likelihood_product
            all_posteriors.append(posterior)

        # Get the index of max posterior and return associated class
        return self.classes[np.argmax(all_posteriors)]

    def __get_likelihoods_map_key(self, target_class, nd_features):
        return tuple([target_class] + nd_features)

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
        self.pdf_params = {}
        self.feature_freq = {}
        self.prioris = {}
        self.class_freq = {}
        self.calculated_likelihoods = {}

        # Calculate parameters for probability distribution function
        for cl in self.classes:
            rows_with_cl_output = train_X[train_Y == cl]
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
                        if (feature_index, cl, val) in self.feature_freq:
                            self.feature_freq[(feature_index, cl, val)] += 1
                        else:
                            self.feature_freq[(feature_index, cl, val)] = 1

        # Count the classes frequency for priori probabilities P(Y)
        for cl in train_Y:
            if cl in self.class_freq:
                self.class_freq[cl] += 1
            else:
                self.class_freq[cl] = 1

        # Calculate priori probabilities for each class
        for cl in self.classes:
            self.prioris[cl] = self.class_freq[cl] / len(train_Y)

        return True

    def test(self, test_X: np.ndarray):
        predictions = []

        for nd_features in test_X:
            prediction = self.__classify(nd_features)
            predictions.append(prediction)

        return predictions

    def posterior_probabilities(self, test_X: np.ndarray, target_class: float):
        if len(self.calculated_likelihoods) <= 0:
            print("Please run test() first")
            return []

        posteriors = []

        for nd_features in test_X:
            map_key = self.__get_likelihoods_map_key(
                target_class=target_class, nd_features=nd_features
            )

            posterior = (
                self.prioris[target_class]
                * self.calculated_likelihoods[map_key]
                / self.__scaling_factor(nd_features=nd_features)
            )

            posteriors.append(posterior)

        return posteriors
