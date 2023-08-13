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
def middle_symmetry(
    values: dict, global_lr: float, lr_alpha: float = 1, lr_beta: float = 0
):
    if values["a"] == 0:
        R = 0
    else:
        R = (values["a"] + (lr_beta * values["d"])) / (
            values["a"]
            + (lr_beta * values["d"])
            + values["b"]
            + (lr_alpha * values["c"])
        )
    updated_lr = global_lr * (1 - R)
    values.update({"lr": updated_lr})


def conditional_probability(values: dict, global_lr: float):
    if values["a"] == 0:
        R = 0
    else:
        R = values["a"] / (values["a"] + values["b"])
    updated_lr = global_lr * (1 - R)
    values.update({"lr": updated_lr})


def dual_factor_heuristic(values: dict, global_lr: float):
    if values["a"] == 0:
        R = 0
    else:
        R = values["a"] / np.sqrt(
            (values["a"] + values["b"]) * (values["a"] + values["c"])
        )
    updated_lr = global_lr * (1 - R)
    values.update({"lr": updated_lr})


def loose_symmetry(values: dict, global_lr: float):
    if values["a"] == 0:
        if values["b"] == 0:
            R = 0
        else:
            R = (values["b"] * values["d"] / (values["b"] + values["d"])) / (
                (values["b"] * values["d"] / (values["b"] + values["d"])) + values["b"]
            )
    elif values["b"] == 0:
        if values["c"] == 0:
            R = 1
        else:
            R = values["a"] / (
                values["a"] + (values["c"] * values["a"] / (values["c"] + values["a"]))
            )

    else:
        R = (
            values["a"] + (values["b"] * values["d"] / (values["b"] + values["d"]))
        ) / (
            values["a"]
            + (values["b"] * values["d"] / (values["b"] + values["d"]))
            + values["b"]
            + (values["c"] * values["a"] / (values["c"] + values["a"]))
        )
    updated_lr = global_lr * (1 - R)
    values.update({"lr": updated_lr})


def loose_symmetry_rarity(values: dict, global_lr: float):
    if values["a"] == 0:
        if values["b"] == 0:
            R = 0
        else:
            R = 0.5

    elif values["b"] == 0:
        if values["c"] == 0:
            R = 1
        else:
            R = values["a"] / (
                values["a"] + (values["c"] * values["a"] / (values["c"] + values["a"]))
            )
    elif values["c"] == 0:
        R = (values["a"] + values["b"]) / (values["a"] + (2 * values["b"]))

    else:
        R = (values["a"] + values["b"]) / (
            values["a"]
            + (2 * values["b"])
            + ((values["a"] * values["c"]) / (values["a"] + values["c"]))
        )
    updated_lr = global_lr * (1 - R)
    values.update({"lr": updated_lr})
