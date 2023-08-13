import numpy as np
import copy
import matplotlib.pyplot as plt


class GLVQ:
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

    def get_colors(self, prototypes):
        color_list = [
            "red",
            "green",
            "blue",
            "yellow",
            "black",
            "orange",
            "purple",
            "pink",
        ]
        unique_class = self.get_class(prototypes)
        return {unique_class[i]: color_list[i] for i in range(len(unique_class))}

    def get_class(self, prototypes):
        list_labels = []
        for p in prototypes:
            list_labels.append(p[1][0])
        unique_class = list(set(list_labels))  # get rid of duplicates
        unique_class.sort()
        unique_class = np.array(unique_class, dtype=self.labeltype)
        return unique_class

    def create_prototype_dict(self, prototypes, learning_rate):
        prototypes_dict = {}
        for i, p in enumerate(prototypes):
            prototypes_dict[i] = {"feature": p[0], "label": p[1], "lr": learning_rate}
        return prototypes_dict

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def prediction(self, x):
        distance = None
        for prototype, values in self.prototypes.items():
            if self.datatype == np.csingle:
                dist_p_x = np.sum(np.abs(values["feature"] - x))
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

    def local_loss(self, x):
        x_feature, x_label = x
        d_1 = None
        d_2 = None
        for prototype, values in self.prototypes.items():
            if self.datatype == np.csingle:
                dist_p_x = np.sum(np.abs(values["feature"] - x_feature))
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
        training_set: list,
        test_set: list,
        f_score_beta: float = 1.0,
        sample_number: dict = None,
        optimizer: str = "glvq",
    ):
        """
        num_epochs: number of epochs
        training_set: list of tuples (feature: (np.array), label: (np.array)))
        test_set: list of tuples (feature: (np.array), label: (np.array)))
        optimizer: lvq1 or glvq
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

        for epoch in range(num_epochs):
            # Clear loss
            global_loss = 0
            # Tranining
            for x in training_set:
                x_feature, x_label = x
                loss, d_1, winner_true, d_2, winner_false = self.local_loss(x)
                winner_prototype, x_prediction = self.prediction(x_feature)

                # Update global_loss
                global_loss += loss

                common_multiplier = loss * (1 - loss) / ((d_1 + d_2) ** 2)

                # Update learning_rate
                if optimizer == "lvq1":
                    if x_prediction == x_label:
                        s = 1
                    else:
                        s = -1
                    self.prototypes[winner_prototype]["lr"] = self.prototypes[
                        winner_prototype
                    ]["lr"] / (1 + (s * self.prototypes[winner_prototype]["lr"]))
                elif optimizer == "glvq":
                    self.prototypes[winner_true]["lr"] = self.prototypes[winner_true][
                        "lr"
                    ] / (
                        1
                        + (
                            1
                            * self.prototypes[winner_true]["lr"]
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

            # # Append learning rate to lr_history
            # for i, values in enumerate(self.prototypes.values()):
            #     self.lr_hist[i].append(values["lr"])

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
                        / ((f_score_beta**2) * (precision + recall))
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
                    f"Epoch: {self.epoch}, Loss: {global_loss:.4f}, Accuracy: {acc*100:.2f} %, F_{f_score_beta}_score: {weighted_f_score*100:.2f} %"
                )
        return self.history

    def lr_graph(self, title: str = None, marker: str = None):
        used_labels = []
        for prototype_name, lr in self.history["lr"].items():
            if self.prototypes[prototype_name]["label"][0] in used_labels:
                label = None
            else:
                label = self.prototypes[prototype_name]["label"][0]
                used_labels.append(label)
            plt.plot(
                range(self.epoch),
                lr,
                label=label,
                color=self.colors[self.prototypes[prototype_name]["label"][0]],
                marker=marker,
                linestyle="dashed",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.legend()
        if title:
            plt.title(title)
        plt.show()

    def acc_graph(self, title: str = None):
        plt.plot(
            range(self.epoch),
            self.history["accuracy"],
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        if title:
            plt.title(title)
        plt.show()

    def f1_graph(self, title: str = None):
        plt.plot(
            range(self.epoch),
            self.history["f_score"],
        )
        plt.xlabel("Epoch")
        plt.ylabel("F Score")
        if title:
            plt.title(title)
        plt.show()
