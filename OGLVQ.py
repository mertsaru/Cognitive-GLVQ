"""
The model is Optimized GLVQ (OGLVQ) model.
Turning any LVQ model to optimized version of it introduced by Kohonen (1995, pp. 175-189) in "Self-Orgazing Maps" (DOI:https://doi.org/10.1007/978-3-642-97610-0), please refer to the paper when needed.
Optimization effects the model's learning rate update.

the model includes two performance measures:

- Accuracy
- F-Score (weighed average)

To use the model please import the file and use the class CGLVQ. Then use class method train with the following parameters:
    num_epochs: train time
    training_set: adjust the training set as: list[tuple[np.array, np.array]],
    test_set: adjust the test set as: list[tuple[np.array, np.array]],
    validation_set: if you want to use validation set adjust the validation set as: list[tuple[np.array, np.array]] = None,
    f_score_beta: beta value of the F score, default = 1 any float value can be used,
    sample_number: Number of training samples each class uses. It is needed to calculate the weighted F scores

One can use the following methods to see the results:
    lr_graph: shows the learning rate graph for each prototype
    acc_graph: shows the accuracy graph
    f1_graph: shows the f1 score graph
    
    methods use matplotlib.pyplot library. Title can be added to the graphs as string by adding the title in the method as parameter.
"""

import numpy as np
import copy
import matplotlib.pyplot as plt

__author__ = " Mert Saruhan "
__maintainer__ = " Mert Saruhan "
__email__ = " mertsaruhn@gmail.com "


class OGLVQ:
    def __init__(self, prototypes: list, learning_rate: float):
        self.feature_size = len(prototypes[0][0])
        prototypes_copy = copy.deepcopy(prototypes)
        self.prototypes = self.create_prototype_dict(prototypes_copy, learning_rate)
        self.datatype = prototypes[0][0].dtype
        self.labeltype = prototypes[0][1].dtype
        self.epoch = 0
        self.history = {
            "lr": {i: [] for i in range(len(prototypes))},
            "loss": [],
            "accuracy": [],
            "f_score": [],
        }
        self.classes = self.get_class(prototypes)
        self.colors = self.get_colors(prototypes)

    def get_colors(self, prototypes) -> dict:
        """
        Divides prototypes into color groups by classes in dictionary form
        For now there are 3 colors: blue, red, green
        The function used in __init__
        """
        color_list = ["#5171fF", "#fF7151", "#519951"]
        unique_class = self.get_class(prototypes)
        return {unique_class[i]: color_list[i % 3] for i in range(len(unique_class))}

    def get_class(self, prototypes) -> np.ndarray:
        """
        Gets the distinct class groups.
        The function used in __init__
        """
        list_labels = []
        for p in prototypes:
            list_labels.append(p[1][0])
        unique_class = list(set(list_labels))  # get rid of duplicates
        unique_class.sort()
        unique_class = np.array(unique_class, dtype=self.labeltype)
        return unique_class

    def create_prototype_dict(self, prototypes, learning_rate) -> dict:
        """
        Creates each prototype's local values in __init__ part.
        """
        prototypes_dict = {}
        for i, p in enumerate(prototypes):
            prototypes_dict[i] = {"feature": p[0], "label": p[1], "lr": learning_rate}
        return prototypes_dict

    def sigmoid(self, x) -> float:
        """
        Activation function for loss
        """
        return 1 / (1 + np.exp(-x))

    def prediction(self, x) -> tuple:
        """
        Function has one parameter, test features
        Returns tuple of (winner prototype, winner class)

        Test features should be same lenght as the prototypes

        Winner prototype is the closest prototype to the parameter entered
        Winner class is the class of the winner prototype

        Function has different distance functions for real values and complex values

        Real values: sum of square of feature diffrences
        Complex values: sum of absolute value of feature diffrences
        """
        distance = None
        for prototype, values in self.prototypes.items():
            if self.datatype == np.csingle:
                dist_p_x = np.sum(np.abs(values["feature"] - x) ** 2)
            else:
                dist_p_x = np.sum((values["feature"] - x) ** 2)

            if distance is None:
                distance = dist_p_x
                winner_class = values["label"]
                winner_prototype = prototype
            elif dist_p_x < distance:
                distance = dist_p_x
                winner_class = values["label"]
                winner_prototype = prototype
        return winner_prototype, winner_class

    def local_loss(self, x) -> tuple:
        """
        Local loss used in model training
        The model is GLVQ model, so we calculate two winners: winner_true, winner_false

        Winner_true: closest prototype to the sample with same class than the sample
        Winner_false: closest prototype to the sample with different class than the sample

        Function returns loss, winner_true to sample distance, winner_true, winner_false to sample distance, winner_false as tuple
        All these values used in prototype update
        """
        x_feature, x_label = x
        d_1 = None
        d_2 = None
        for prototype, values in self.prototypes.items():
            if self.datatype == np.csingle:
                dist_p_x = np.sum(np.abs(values["feature"] - x_feature) ** 2)
            else:
                dist_p_x = np.sum((values["feature"] - x_feature) ** 2)

            if values["label"] == x_label:
                if d_1 is None:
                    d_1 = dist_p_x
                    winner_true = prototype
                elif dist_p_x < d_1:
                    d_1 = dist_p_x
                    winner_true = prototype
            else:
                if d_2 is None:
                    d_2 = dist_p_x
                    winner_false = prototype
                elif dist_p_x < d_2:
                    d_2 = dist_p_x
                    winner_false = prototype

        loss = self.sigmoid((d_1 - d_2) / (d_1 + d_2))
        return loss, d_1, winner_true, d_2, winner_false

    def train(
        self,
        num_epochs: int,
        training_set: list[tuple[np.array, np.array]],
        test_set: list[tuple[np.array, np.array]],
        f_score_beta: float = 1.0,
        sample_number: dict = None,
    ) -> dict:
        """
        Trains the model returns history of the model as dictionary
        history = {
            history of learning rate for each prototype,
            history of loss,
            history of accuracy,
            history of f-score (weighted f-score)
        }
        To reach history of any prototype's learning rate use history["lr"][prototype_number]

        Parameters:
        - num_epochs: number of epochs
        - training_set: list of tuples (feature, label)
        - test_set: list of tuples (feature, label)
        - f_score_beta: beta value for f-score calculation default = 1
        - sample_number: dictionary of sample numbers for each class (class_name: sample_number)

        sample number is used for weighted f-score calculation
        """
        if len(self.classes) == 1:
            print("Error: there is only one class in the prototypes")
            return

        if sample_number is None:
            print("Error: sample_number is None")
            return

        sum_samples = sum(sample_number.values())
        sample_weight = {
            class_num: sample / sum_samples
            for class_num, sample in sample_number.items()
        }

        if f_score_beta == int(f_score_beta):
            f_name = int(f_score_beta)
        else:
            f_name = f_score_beta

        for epoch in range(num_epochs):
            # Clear loss
            global_loss = 0
            # Tranining
            for x in training_set:
                x_feature, x_label = x
                loss, d_1, winner_true, d_2, winner_false = self.local_loss(x)
                _, x_prediction = self.prediction(x_feature)

                # Update global_loss
                global_loss += loss

                common_multiplier = loss * (1 - loss) / ((d_1 + d_2) ** 2)

                # Update learning_rate
                self.prototypes[winner_true]["lr"] = self.prototypes[winner_true][
                    "lr"
                ] / (
                    1
                    + (
                        1
                        * self.prototypes[winner_true]["lr"]
                        * 4
                        * common_multiplier
                        * d_2
                    )
                )

                self.prototypes[winner_false]["lr"] = self.prototypes[winner_false][
                    "lr"
                ] / (
                    1
                    + (
                        -1
                        * self.prototypes[winner_false]["lr"]
                        * 4
                        * common_multiplier
                        * d_1
                    )
                )

                # Update prototypes
                ## update winner_true
                self.prototypes[winner_true]["feature"] += (
                    self.prototypes[winner_true]["lr"]
                    * 4
                    * common_multiplier
                    * d_2
                    * (x_feature - self.prototypes[winner_true]["feature"])
                )

                ## update winner_false
                self.prototypes[winner_false]["feature"] -= (
                    self.prototypes[winner_false]["lr"]
                    * 4
                    * common_multiplier
                    * d_1
                    * (x_feature - self.prototypes[winner_false]["feature"])
                )

            # Calculate f-score and accuracy
            correct = 0
            f_dict = {}
            for x in self.classes:
                f_dict[x] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

            for x in test_set:
                x_feature, x_label = x
                _, x_prediction = self.prediction(x_feature)

                ## accuracy counter
                if x_prediction == x_label:
                    correct += 1

                ## f-score counter
                for class_name, value in f_dict.items():
                    if x_prediction == x_label:
                        if x_prediction == class_name:
                            value["TP"] += 1
                        else:
                            value["TN"] += 1
                    else:
                        if x_prediction == class_name:
                            value["FP"] += 1
                        else:
                            value["FN"] += 1

            ## calculate accuracy
            acc = correct / len(test_set)

            ## calculate f-score
            for class_name, value in f_dict.items():
                if value["TP"] == 0:
                    score = 0
                else:
                    precision = value["TP"] / (value["TP"] + value["FP"])
                    recall = value["TP"] / (value["TP"] + value["FN"])
                    score = (
                        (1 + (f_score_beta**2))
                        * precision
                        * recall
                        / (((f_score_beta**2) * precision) + recall)
                    )
                f_dict[class_name] = score
            weighted_f_score = 0
            for class_name, value in f_dict.items():
                weighted_f_score += value * sample_weight[class_name]

            self.epoch += 1

            # Update history
            ## Update learning rate history
            for i, values in enumerate(self.prototypes.values()):
                self.history["lr"][i].append(values["lr"])
            ## Update loss history
            self.history["loss"].append(global_loss)
            ## Update accuracy history
            self.history["accuracy"].append(acc)
            ## Update f-score history
            self.history["f_score"].append(weighted_f_score)

            if epoch % 10 == 0 or epoch == num_epochs:
                print(
                    f"Epoch: {self.epoch}, Loss: {global_loss:.4f}, Accuracy: {acc*100:.2f} %, F_{f_name}_score: {weighted_f_score*100:.2f} %"
                )
        return self.history

    def lr_graph(self, title: str = None, marker: str = None) -> plt.figure:
        """
        Shows learning rate graph for each prototype in combined graph
        Prototypes are grouped by their class with different colors (for now max 3 colors)

        Function uses matplotlib.pyplot library so use markers according to matplotlib.pyplot library
        Parameters:
        - title: title of the graph
        - marker: marker of the graph
        """
        used_labels = []
        fig, ax = plt.subplots(figsize=(10, 10))
        for prototype_name, lr in self.history["lr"].items():
            if self.prototypes[prototype_name]["label"][0] in used_labels:
                label = None
            else:
                label = self.prototypes[prototype_name]["label"][0]
                used_labels.append(label)
            ax.plot(
                range(self.epoch),
                lr,
                label=label,
                color=self.colors[self.prototypes[prototype_name]["label"][0]],
                linestyle="dashed",
                marker=marker,
            )
        plt.xlabel("Epoch (t)", fontsize=25, weight="bold")
        plt.legend()
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        if title:
            plt.title(title, fontsize=40)
        plt.show()
        return fig

    def acc_graph(self, title: str = None):
        """
        Shows accuracy graph of the model

        Function uses matplotlib.pyplot library so use markers according to matplotlib.pyplot library
        Parameters:
        - title: title of the graph
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(
            range(self.epoch),
            self.history["accuracy"],
        )
        plt.xlabel("Epoch (t)", fontsize=25, weight="bold")
        plt.ylim(0, 1.01)
        plt.yticks(
            np.arange(0, 1.01, step=0.2),
            ["0%", "20%", "40%", "60%", "80%", "100%"],
            fontsize=20,
        )
        plt.xticks(fontsize=20)
        if title:
            plt.title(title, fontsize=40)
        plt.show()
        return fig

    def f1_graph(self, title: str = None):
        """
        Shows weighted f-score graph of the model

        Function uses matplotlib.pyplot library so use markers according to matplotlib.pyplot library
        Parameters:
        - title: title of the graph
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(
            range(self.epoch),
            self.history["f_score"],
        )
        plt.xlabel("Epoch (t)", fontsize=25, weight="bold")
        plt.ylim(0, 1.01)
        plt.yticks(
            np.arange(0, 1.01, step=0.2),
            ["0%", "20%", "40%", "60%", "80%", "100%"],
            fontsize=20,
        )
        plt.xticks(fontsize=20)
        if title:
            plt.title(title, fontsize=40)
        plt.show()
        return fig
