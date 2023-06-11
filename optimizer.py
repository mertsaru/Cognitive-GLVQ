import numpy as np

"""
values = {
    "feature" : narray,
    "label" : int,
    "lr" : float,
    "a": int,
    "b": int,
    "c": int,
    "d": int,
    "update_sum": np.zeros(len(training_set[0][0]))
    }"""


# Update the learning rate of the prototypes
def middle_symmetry(values: dict, lr_alpha: float, lr_beta: float):
    R = (
        (values["a"] + (lr_beta * values["d"])) 
        / 
        (values["a"] + (lr_beta * values["d"]) + values["b"] + (lr_alpha * values["c"]))
    )
    values.update({"lr": 1 - R})


def conditional_probability(values: dict, lr_alpha: float, lr_beta: float):
    R = values["a"] / (values["a"] + values["b"])
    values.update({"lr": 1 - R})


def dual_factor_heuristic(values: dict, lr_alpha: float, lr_beta: float):
    R = values["a"] / np.sqrt((values["a"] + values["b"]) * (values["a"] + values["c"]))
    values.update({"lr": 1 - R})


def loose_symmetry(values: dict, lr_alpha: float, lr_beta: float):
    R = (
        (values["a"] + (values["b"] * values["d"] / (values["b"] + values["d"]))) 
        / 
        (values["a"] + (values["b"] * values["d"] / (values["b"] + values["d"]))
        + values["b"] + (values["c"] * values["a"] / (values["c"] + values["a"])))
    )
    values.update({"lr": 1 - R})


def loose_symmetry_rarity(values: dict, lr_alpha: float, lr_beta: float):
    R = (
        (values["a"] + values["b"]) 
        / 
        (values["a"] + (2 * values["b"])
        + ((values["a"] * values["c"]) / (values["a"] + values["c"])))
    )
    values.update({"lr": 1 - R})
