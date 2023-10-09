"""
File contains optimizer functions for the learning rate of cognitive GLVQ (CGLVQ) model. More information about the optimizers can be found in:
- Takahashi, T., Nakano, M., & Shinohara, S. (2010). Cognitive Symmetry: Illogical but Rational Biases. Symmetry Culture and Science. 21. 1-3. https://www.researchgate.net/publication/285850238_Cognitive_Symmetry_Illogical_but_Rational_Biases.
- Shinohara, S., Taguchi, R., Katsurada, K., & Nitta, T. (2007). A Model of Belief Formation Based on Causality and Application to N-armed Bandit Problem. Transactions of the Japanese Society for Artificial Intelligence, 22(1), 58–68. (in Japanese) https://doi.org/10.1527/tjsai.22.58.
- Taniguchi, H., Sato, H., & Shirakawa, T. (2018). A machine learning model with human cognitive biases capable of learning from small and biased datasets. Scientific Reports, 8(1). https://doi.org/10.1038/s41598-018-25679-z.
- Manome, N., Shinohara, S., Takahashi, T., Chen, Y., & Chung, U. (2021). Self-incremental learning vector quantization with human cognitive biases. Scientific Reports, 11(1). https://doi.org/10.1038/s41598-021-83182-4.
"""

import numpy as np

__author__ = " Mert Saruhan "
__maintainer__ = " Mert Saruhan "
__email__ = " mertsaruhn@gmail.com "


# Update the learning rate of the prototypes
def middle_symmetry(
    values: dict, global_lr: float, lr_alpha: float = 1, lr_beta: float = 0
) -> None:
    """
    updates the learning rate of the prototypes based on the middle symmetry with alpha = 1 and beta = 0
    """
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


def conditional_probability(values: dict, global_lr: float) -> None:
    """
    updates the learning rate of the prototypes based on the conditional probability
    """
    if values["a"] == 0:
        R = 0
    else:
        R = values["a"] / (values["a"] + values["b"])
    updated_lr = global_lr * (1 - R)
    values.update({"lr": updated_lr})


def dual_factor_heuristic(values: dict, global_lr: float) -> None:
    """
    updates the learning rate of the prototypes based on the dual factor heuristic
    """
    if values["a"] == 0:
        R = 0
    else:
        R = values["a"] / np.sqrt(
            (values["a"] + values["b"]) * (values["a"] + values["c"])
        )
    updated_lr = global_lr * (1 - R)
    values.update({"lr": updated_lr})


def loose_symmetry(values: dict, global_lr: float) -> None:
    """
    updates the learning rate of the prototypes based on the loose symmetry
    """
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


def loose_symmetry_rarity(values: dict, global_lr: float) -> None:
    """
    updates the learning rate of the prototypes based on the loose symmetry with rarity
    Loose symmetry with rarity is loose symmetry when d -> infinty
    """
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
