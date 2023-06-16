import numpy as np

class GLVQ():

    def __init__(self, prototypes: list, learning_rate: float):
        self.feature_size = len(prototypes[0][0])
        self.prototypes = self.create_prototype_dict(prototypes, learning_rate)
        self.datatype = prototypes[0][0].dtype
    
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
            if self.datatype == np.float64\
                or self.datatype == np.int32:
                dist_p_x = np.sum((values["feature"] - x) ** 2)
            elif self.datatype == np.csingle:
                dist_p_x = np.sum(np.abs(values["feature"] - x))
            if distance is None:
                distance = dist_p_x
                winner = values["label"]
                winner_prototype = prototype
            elif dist_p_x < distance:
                distance = dist_p_x
                winner = values["label"]
                winner_prototype = prototype
        return winner_prototype, winner

    def accuracy(self, test_set):
        correct = 0
        for x in test_set:
            x_feature = x[0]
            x_label = x[1]
            x_prediction = self.prediction(x_feature)
            if x_prediction == x_label:
                correct += 1
        return correct / len(test_set)
    
    def f1_score(self, test_set, f1_beta: float):
        """
        test_set: test set
        f1_beta: parameter for the f1_score"""
        TP = FP = FN = TN = 0
        for x in test_set:
            x_feature = x[0]
            x_label = x[1]
            x_prediction = self.prediction(x_feature)
            if x_prediction == x_label:
                if x_prediction == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if x_prediction == 1:
                    FP += 1
                else:
                    FN += 1

        if TP == 0:
            return 0
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return (1+(f1_beta**2)) * precision * recall / ((f1_beta**2) * (precision + recall))

    def test_measure(self, test_set, f1_beta: float):
        """
        test_measure calculates both the accuracy and the f1_score
        test_set: test set
        f1_beta: parameter for the f1_score"""
        correct = TP = FP = TN = FN = 0
        for x in test_set:
            x_feature = x[0]
            x_label = x[1]
            x_prediction = self.prediction(x_feature)
            if x_prediction == x_label:
                correct += 1
                if x_prediction == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if x_prediction == 1:
                    FP += 1
                else:
                    FN += 1
        
        accuracy = correct / len(test_set)
        if TP == 0:
            return accuracy, 0
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = (1+(f1_beta**2)) * precision * recall / ((f1_beta**2) * (precision + recall))
        return accuracy, f1_score

    def local_loss(self, x):
        x_feature = x[0]
        x_label = x[1]
        d_1 = None
        d_2 = None
        for prototype, values in self.prototypes.items():
            if self.datatype == np.float64\
                or self.datatype == np.int32:
                dist_p_x = np.sum((values["feature"] - x_feature) ** 2)
            elif self.datatype == np.csingle:
                dist_p_x = np.sum(np.abs(values["feature"] - x_feature))
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
    
    def train(self, num_epochs: int, training_set: list, test_set: list, measure = "test_measure"):
        history = []
        for epoch in range(num_epochs):
            
            # Clear loss
            global_loss = 0

            for x in training_set:
                x_feature = x[0]
                x_label = x[1]
                loss, d_1, winner_true, d_2, winner_false = self.local_loss(x)
                winner_prototype, x_prediction = self.prediction(x_feature)
                
                # Update learning_rate update_sum
                if self.prototypes[winner_prototype]["label"] == x_label:
                    s = 1
                else:
                    s = -1
                self.prototypes[winner_prototype]["lr"] = self.prototypes[winner_prototype]["lr"] / (1 + (s * self.prototypes[winner_prototype]["lr"]))

                # Update global_loss
                global_loss += loss

                # Update prototypes update_sum
                common_multiplier = (4 * loss * (1-loss) / ((d_1 + d_2) ** 2))

                ## update winner_true update_sum
                self.prototypes[winner_true]["loss_update_sum"] += common_multiplier * d_2 * (x_feature - self.prototypes[winner_true]["feature"])

                ## update winner_false update_sum
                self.prototypes[winner_false]["loss_update_sum"] -= common_multiplier * d_1 * (x_feature - self.prototypes[winner_false]["feature"])
            
            # Train prototypes
            for values in self.prototypes.values():
                
                ## Update prototypes
                values["feature"] += values["lr"] * values["loss_update_sum"]

            if measure == "accuracy":
                acc = self.accuracy(test_set)
                f1 = None
            elif measure == "f1_score":
                acc = None
                f1 = self.f1_score(test_set, 1)
            elif measure == "test_measure":
                acc, f1 = self.test_measure(test_set, 1)

            hist = {"loss": global_loss / len(training_set),"accuracy": acc, "f1_score": f1 ,"prototypes": self.prototypes}
            history.append(hist)

            if epoch % 10 == 9 or epoch == num_epochs - 1:
                print("Epoch: ", epoch+1, " Loss: ", global_loss / len(training_set), " Accuracy: ", acc, " F1_score: ", f1)
        return history




                    
