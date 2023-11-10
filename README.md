# cognitive GLVQ

The project is used in my master thesis.
This code adapts cognitive-bias learning methods to GLVQ(Generalized Learning Vector Quantization) as learning rate optimizers.

The CGLVQ models can be imported from CGLVQ.py file and be used with requested cognitive-bias model optimizer.

The OGLVQ model can be imported from OGLVQ.py file. OGLVQ is optimized GLVQ and the optimization is made using euclidian distance as distance metric and sigmoid function as activation function.

Both use models euclidian distance as distance metric and sigmoid function as activation function. For both of the models, it is possible to use complex space.

 optimizer.py contains 5 different cognitive learning rates, namely:

- conditional probability
- dual factor heuristic
- middle symmetry
- loose symmetry
- lose symmetry with rarity

## Datasets

The SP and NSP datasets used in the training in Jupyter Notebook "model_testing(NSP_F).ipynb" and "model_testing(SP_F).ipynb" are private datasets.

Other datasets can be found in various sources:

Breast Cancer Wisconsin dataset:

- Kaggle: <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>
- UW CS FTP server:
  - ftp: ftp.cs.wisc.edu
  - cd math-prog/cpo-dataset/machine-learn/WDBC/

- UCI Machine Learning Repository: <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29>

Iris dataset:

- Kaggle: <https://www.kaggle.com/datasets/sims22/irisflowerdatasets>

Ionosphere dataset:

- Kaggle: <https://www.kaggle.com/datasets/prashant111/ionosphere>
- UC Irvine Machine Learning Repository: <https://archive.ics.uci.edu/dataset/52/ionosphere>

Sonar dataset:

- Kaggle: <https://www.kaggle.com/datasets/rupakroy/sonarcsv>

## Model Testing

To run the model_testing(*).py files, run the notebook in root folder.
