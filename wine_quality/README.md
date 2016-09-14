

## Purpose:

The notebook is used to pass in datasets and run them through a series of instantiated off-the-shelf visualizations and algorithms in order to help get a first-pass understanding of the data and how different untuned estimators perform. The results are then used to provide initial directions of approach for more in-depth analysis.

The notebook begins with data visualization, preprocessing methods, and moves onto a number of different estimators and methods of evaluating estimator performance. A selection of favorite techniques receive in-depth explanation. [NOTE: If viewing on github, some browser+OS combinations render iPython LaTeX incorrectly (too small or with weird black lines, so the math in these analysis sections may disappear or look funny. Safari OSX seems to be one exception. This is an open problem.]

Currently, this notebook tests a large number of different statistical and machine learning methods on a dataset of red wines which contains, for each wine in row i, eleven descriptive characteristics and a score of overall wine quality as determined by a group of wine experts. We experiment with and compare the performance of a wide range of supervised learning algorithms (both regression and classification) since the labels are an integer score of 1-10.

#### Data viz:

- Pairplots features against quality
- Pairplots all features against all features
- Violinplot
- Correlation matrix heatmap
- Scaled parallel coordinates, mean scaled parallel coordinates
- Radviz
- PCA biplot of components
- Gridsearch results
- Score against # of folds, score against # neighbors in KNN

#### Algorithms:

- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso (L1 regularization)
- Elastic Net (L1 + L2 regularization)
- KNN Regression
- KNN Classification
- Decision Tree Regression
- Decision Tree Classification
- Random Forest Classifier
- Adaboost
- SVM Regression (multiple kernels)
- SVM Classification (multiple kernels)

#### Metrics & other:

- Custom test/train split
- Mean 0 unit variance scaling
- MinMax scaling
- Data normalization
- PCA
- K-folds CV
- Stratified K-folds CV
- Null Accuracy
- Confusion Matrix
- Polynomial Preprocessing for Regression
- Grid Search
-Iterate over number of K-folds
-Iterate over number of neighbors for KNN
-Iterate KNN gridsearch max score over # of folds
