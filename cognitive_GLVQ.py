import numpy as np

import helper_func as hf

class GLVQ_c():

    def __init__(self, prototypes, training_set):
        self.training = training_set
        self.feature_size = len(training_set[0][0])
        self.prototypes = self.create_prototype_dict(prototypes)
    
    def create_prototype_dict(self, prototypes):
        prototypes_dict = {}
        for i, p in enumerate(prototypes):
            prototypes_dict[i] = {
                "feature": p[0],
                "label": p[1],
                "lr": 0
            }
        return prototypes_dict
        
    def prediction(self, x):
        distance = None
        for values in self.prototypes.values():
            dist_p_x = np.sum((values["feature"] - x) ** 2)
            if distance is None:
                distance = dist_p_x
                winner = values["label"]
            elif dist_p_x < distance:
                distance = dist_p_x
                winner = values["label"]
        return winner

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
    
    #def update_lr(self, values):
    """
    This is an example of a learning rate update function.
    Every congnitive GLVQ model should write an individual learning rate update function.
    """
    #    R = values["a"] / (values["a"] + values["b"])
    #    values.update({"lr": 1 - R})
    #    return values["lr"]
    
    def train(self):
        
        # Clear accurence_frequncy
        for values in self.prototypes.values():
            values.update({
                        "a": 0,
                        "b": 0,
                        "c": 0,
                        "d": 0,
                        "update_sum": np.zeros(self.feature_size)
            })

        # Clear loss
        global_loss = 0

        for x in self.training:
            x_feature = x[0]
            x_label = x[1]
            loss, d_1, winner_true, d_2, winner_false = self.local_loss(x)
            x_prediction = self.prediction(x_feature)
            
            # Update global_loss
            global_loss += loss

            # Update accurence_frequncy
            for values in self.prototypes.values():
                if values["label"] == x_prediction and x_label == x_prediction:
                    values["a"] += 1
                elif values["label"] == x_prediction and x_label != x_prediction:
                    values["b"] += 1
                elif values["label"] != x_prediction and x_label == x_prediction:
                    values["c"] += 1
                elif values["label"] != x_prediction and x_label != x_prediction:
                    values["d"] += 1

            # Update prototypes update_sum
            common_multiplier = (4 * loss * (1-loss) / ((d_1 + d_2) ** 2))

            ## update winner_true update_sum
            self.prototypes[winner_true]["update_sum"] += common_multiplier * d_2 * (x_feature - self.prototypes[winner_true]["feature"])

            ## update winner_false update_sum
            self.prototypes[winner_false]["update_sum"] -= common_multiplier * d_1 * (x_feature - self.prototypes[winner_false]["feature"])
        
        # Train prototypes
        for values in self.prototypes.values:
            
            ## Update learning rate
            self.update_lr(values)
            
            ## Update prototypes
            values["feature"] += values["lr"] * values["update_sum"]

        history = {"loss": global_loss / len(self.training), "prototypes": self.prototypes}
        return history




                    
