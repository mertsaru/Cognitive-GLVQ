import numpy as np
import copy
import matplotlib.pyplot as plt

class GLVQ():

    def __init__(self, prototypes: list, learning_rate: float):
        self.feature_size = len(prototypes[0][0])
        prototypes_copy = copy.deepcopy(prototypes) 
        self.prototypes = self.create_prototype_dict(prototypes_copy, learning_rate)
        self.datatype = prototypes[0][0].dtype
        self.labeltype = prototypes[0][1].dtype
        self.epoch = 0
        self.history = []
        self.classes = self.get_class(prototypes)
        self.colors = {
            0: "red",
            1: "green",
            2: "blue",
            3: "yellow",
            4: "black",
            5: "orange",
            6: "purple",
            7: "pink",
        }
        self.lr_hist = self.lr_list()

    def lr_list(self):
        lr_hist = {}
        for i in self.classes:
            lr_hist[i] = []
        return lr_hist

    def get_class(self,prototypes):
        list_labels = []
        for p in prototypes:
            list_labels.append(p[1][0])
        unique_class = list(set(list_labels)) # get rid of duplicates
        unique_class.sort()
        unique_class = np.array(unique_class, dtype=self.labeltype)
        return unique_class
    
    def create_prototype_dict(self, prototypes, learning_rate):
        prototypes_dict = {}
        for i, p in enumerate(prototypes):
            prototypes_dict[i] = {
                "feature": p[0],
                "label": p[1],
                "lr": learning_rate
            }
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
        
        loss = self.sigmoid((d_1 - d_2)/(d_1 + d_2))
        return loss, d_1, winner_true, d_2, winner_false
    
    def train(self, num_epochs: int, training_set: list, test_set: list, f_score_beta: float = 1.0):
        """
        num_epochs: number of epochs
        training_set: list of tuples (feature: (np.array), label: (np.array)))
        test_set: list of tuples (feature: (np.array), label: (np.array)))
        """

        if len(self.classes) == 1:
            print("Error: there is only one class in the prototypes")
            return

        for epoch in range(num_epochs):
            
            # Clear loss
            global_loss = 0

            for x in training_set:
                x_feature, x_label = x
                loss, d_1, winner_true, d_2, winner_false = self.local_loss(x)
                winner_prototype, x_prediction = self.prediction(x_feature)
                
                # Update global_loss
                global_loss += loss
                
                # Update learning_rate of winner_prototype
                if x_prediction == x_label:
                    s = 1
                else:
                    s = -1
                self.prototypes[winner_prototype]["lr"] = (
                    self.prototypes[winner_prototype]["lr"] 
                    / (1 + (s * self.prototypes[winner_prototype]["lr"]))
                )

                # Update prototypes
                common_multiplier = (4 * loss * (1-loss) / ((d_1 + d_2) ** 2))

                ## update winner_true
                self.prototypes[winner_true]["feature"] += (self.prototypes[winner_true]["lr"] 
                                                            * common_multiplier 
                                                            * d_2 
                                                            * (x_feature - self.prototypes[winner_true]["feature"]))

                ## update winner_false
                self.prototypes[winner_false]["feature"] -= (self.prototypes[winner_false]["lr"] 
                                                            * common_multiplier 
                                                            * d_1 
                                                            * (x_feature - self.prototypes[winner_false]["feature"]))
                
            # Append learning rate to lr_history
            stored_classes = []
            for values in self.prototypes.values():
                if values["label"] not in stored_classes:
                    self.lr_hist[values["label"][0]].append(values["lr"])
                    stored_classes.append(values["label"])
            
            # Calculate f-score and accuracy    
            correct = 0
            f_dict = {}
            for x in self.classes:
                f_dict[x] = {"TP":0,"FP":0,"FN":0,"TN":0}

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
                    score = (1+(f_score_beta**2)) * precision * recall / ((f_score_beta**2) * (precision + recall))
                f_dict[class_name] = score

            self.epoch += 1
            hist = {"epoch": self.epoch, "loss": global_loss, "accuracy": acc, "f_score": f_dict, "prototypes": self.prototypes}
            self.history.append(hist)
            
            #if epoch % 10 == 9 or epoch == num_epochs:
            #    print(f"Epoch: {self.epoch}, Loss: {global_loss:.4f}, Accuracy: {acc*100:.2f} %, F_{f_score_beta}_score: {[{class_name: f'{round(val*100,2)} %'} for class_name, val in f_dict.items()]}")
        return self.history
    
    def lr_graph(self, title: str = None):
        for class_name, lr_list in self.lr_hist.items():
            plt.plot(range(self.epoch),lr_list, label = class_name, color = self.colors[class_name])
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()




                    
