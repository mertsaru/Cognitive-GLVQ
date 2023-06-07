import numpy as np

import helper_func as hf

class GLVQ():

    def __init__(self, prototypes, training_set, learning_rate):
        self.training = training_set
        self.feature_size = len(training_set[0][0])
        self.prototypes = self.create_prototype_dict(prototypes, learning_rate)
    
    def create_prototype_dict(self, prototypes, learning_rate):
        prototypes_dict = {}
        for i, p in enumerate(prototypes):
            prototypes_dict[i] = {
                "feature": p[0],
                "label": p[1],
                "lr": learning_rate
            }
        return prototypes_dict
        
    def prediction(self, x):
        distance = None
        for prototype, values in self.prototypes.items():
            dist_p_x = np.sum((values["feature"] - x) ** 2)
            if distance is None:
                distance = dist_p_x
                winner = values["label"]
            elif dist_p_x < distance:
                distance = dist_p_x
                winner = values["label"]
        return prototype, winner

    def local_loss(self, x):
        x_feature = x[0]
        x_label = x[1]
        d_1 = None
        d_2 = None
        for prototype, values in self.prototypes.items():
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
        
        loss = hf.sigmoid((d_1 - d_2)/(d_1 + d_2))
        return loss, d_1, winner_true, d_2, winner_false
    
    def train(self):

        # Clear loss
        global_loss = 0

        for x in self.training:
            x_feature = x[0]
            x_label = x[1]
            loss, d_1, winner_true, d_2, winner_false = self.local_loss(x)
            winner_prototype, x_prediction = self.prediction(x_feature)
            
            # Update learning_rate
            if self.prototypes[winner_prototype]["label"] == x_label:
                s = 1
            else:
                s = -1
            self.prototypes[winner_prototype]["lr"] += self.prototypes[winner_prototype]["lr"] / (1 + (s * self.prototypes[winner_prototype]["lr"]))

            # Update global_loss
            global_loss += loss

            # Update prototypes 
            common_multiplier = (4 * loss * (1-loss) / ((d_1 + d_2) ** 2))

            ## update winner_true 
            self.prototypes[winner_true]["feature"] += self.prototypes[winner_true]["lr"] * common_multiplier * d_2 * (x_feature - self.prototypes[winner_true]["feature"])

            ## update winner_false 
            self.prototypes[winner_false]["feature"] -= self.prototypes[winner_false]["lr"] * common_multiplier * d_1 * (x_feature - self.prototypes[winner_false]["feature"])

        history = {"loss": global_loss / len(self.training), "prototypes": self.prototypes}
        return history




                    
