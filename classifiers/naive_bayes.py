import numpy as np
import math
import sys


class NaiveBayes:

    # Use 1-D Gaussian PDF
    def __gaussian_pdf(self, mean: float, std: float, x: float):
        coefficient = 1 / (math.sqrt(2 * math.pi) * std)
        variance = math.pow(std, 2)
        exponent = math.exp(-1 / (2 * variance) * math.pow(x - mean, 2))
        return coefficient * exponent

    # nd_inputs = n-dimension inputs, a tuple of input values for multiple features
    def __classify(self, nd_inputs: tuple):
        all_posteriors = []

        # Calculate posterior probability for each class
        for cl in self.classes:
            posterior = self.priori_probabilities[cl]

            # Multiply p(x|w) for each feature in the given class
            for feature_index, x in enumerate(nd_inputs):
                mean, std = self.pdf_params[(feature_index, cl)]

                posterior *= self.__gaussian_pdf(mean, std, x)

            all_posteriors.append(posterior)

        # Get the index of max posterior and return associated class
        return self.classes[np.argmax(all_posteriors)]

    def train(self, train_X: np.ndarray, train_Y: np.ndarray):
        if len(train_X) <= 0:
            print("Training failed: Input data required")
            return False
        if len(train_Y) <= 0:
            print("Training failed: Output data required")
            return False

        self.classes = np.unique(train_Y)
        self.training_train_X = train_X
        self.pdf_params = {}
        self.priori_probabilities = {}
        classes_frequency = {}

        # Calculate the parameters for 1-D Gaussian PDF.
        # Here we need the mean and standard deviation.
        for cl in self.classes:
            rows_with_cl_output = self.training_train_X[train_Y == cl]
            transposed_rows = np.transpose(rows_with_cl_output)

            # Store the mean and standard deviation for each feature in a given class
            for feature_index, values in enumerate(transposed_rows):
                self.pdf_params[(feature_index, cl)] = (np.mean(values), np.std(values))

        # Count the classes frequency
        for cl in train_Y:
            if cl in classes_frequency:
                classes_frequency[cl] += 1
            else:
                classes_frequency[cl] = 1

        # Calculate priori probabilities for each class
        for cl in self.classes:
            self.priori_probabilities[cl] = classes_frequency[cl] / len(train_Y)

        return True

    def test(self, test_X: np.ndarray):
        predictions = []

        for nd_inputs in test_X:
            prediction = self.__classify(nd_inputs)
            predictions.append(prediction)

        return predictions
