# Machine Learning Path

This section is now organized as one flat topic folder per module.

- Each folder starts with a sortable numeric prefix.
- Notebooks, datasets, PDFs, and images for a topic now live together in the same folder.
- The original teaching sequence is preserved in a continuous numbered order so the folders sort cleanly from start to finish.

## Module Map

- `01-Regression`
  Covers simple linear regression, multiple regression, polynomial regression, and regularized regression. Includes the regression datasets plus reference notes for OLS, metrics, overfitting, regularization, and cross-validation.

- `02-Classification`
  Starts the classification path with handling imbalanced datasets.

- `03-Logistic-Regression`
  Covers logistic regression fundamentals, performance metrics, one-vs-rest strategy, and the main implementation notebook.

- `04-SVM`
  Includes support vector classification, support vector regression, and kernel-based SVM practicals.

- `05-Naive-Bayes`
  Includes classifier intuition, common variants, and a practical implementation notebook.

- `06-K-Nearest-Neighbor`
  Covers KNN classification, KNN regression, and KD-tree / Ball-tree support material.

- `07-Decision-Tree`
  Includes decision tree classifier and regressor notes plus practical notebooks.

- `08-Random-Forest`
  Includes classification and regression implementations, supporting notes, ROC image output, and the required datasets.

- `09-AdaBoost`
  Includes classification and regression practicals, core boosting notes, and the required datasets.

- `10-Gradient-Boosting`
  Includes classification and regression notebooks with colocated datasets and plot assets.

- `11-XGBoost`
  Includes classifier and regressor notebooks, supporting PDFs, ROC image output, and the regression dataset needed by the regressor notebook.

- `12-Unsupervised-Machine-Learning`
  Introductory theory material for unsupervised learning.

- `13-PCA`
  Includes the PCA note set and the practical PCA implementation notebook.

- `14-K-Means-Clustering`
  Includes K-means theory notes and the implementation notebook.

- `15-Hierarchical-Clustering`
  Includes hierarchical clustering notes, comparison material against K-means, and the implementation notebook.

- `16-DBSCAN-Clustering`
  Includes DBSCAN theory and the implementation notebook.

- `17-Silhouette-Clustering`
  Includes silhouette-based clustering evaluation notes.

- `18-Anomaly-Detection`
  Includes anomaly detection notes, the isolation forest notebook, the DBSCAN anomaly notebook, and the healthcare dataset.

- `19-Time-Series-Forecasting`
  Starter expansion folder for ARIMA, SARIMA, Prophet-style workflows, feature-based forecasting, and backtesting discipline.

- `20-Recommender-Systems`
  Starter expansion folder for collaborative filtering, content-based recommenders, ranking metrics, retrieval, and re-ranking ideas.

- `21-Causal-Inference-and-AB-Testing`
  Starter expansion folder for treatment effects, confounding, uplift thinking, experiment design, and A/B testing interpretation.

## Layout Notes

- There are no nested subfolders left inside the topic folders.
- Files that were previously under `Projects`, `Handwritten Notes`, `reference_notes`, or `data` now sit directly in their topic folder.
- Dataset paths inside the moved notebooks were updated to match the flatter layout.
- The last three folders are expansion scaffolds for high-value classical ML topics that fit naturally after the current path.
