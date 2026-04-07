
# All Python code samples for each ML algorithm tab
# Each entry is a list of {"title": str, "code": str} dicts

CODE_SAMPLES = {

# ══════════════════════════════════════════════════════
"linear_regression": [
  {
    "title": "Basic OLS Linear Regression",
    "code": """import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print(f"Coefficients : {model.coef_}")
print(f"Intercept    : {model.intercept_:.4f}")
print(f"R² Score     : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE         : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")"""
  },
  {
    "title": "Ridge & Lasso Regularization",
    "code": """from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=20, n_informative=5,
                        noise=20, random_state=42)

for name, model in [
    ("Ridge   (L2)", Ridge(alpha=1.0)),
    ("Lasso   (L1)", Lasso(alpha=0.1, max_iter=5000)),
    ("ElasticNet  ", ElasticNet(alpha=0.1, l1_ratio=0.5)),
]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    print(f"{name} | CV R² = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
  {
    "title": "Polynomial Regression + Feature Engineering",
    "code": """import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Non-linear data
np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 150)).reshape(-1, 1)
y = 0.5 * X.ravel()**3 - 2 * X.ravel()**2 + np.random.normal(0, 2, 150)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for degree in [1, 2, 3, 5]:
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=0.1)),
    ])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"Degree {degree:2d} | Test R² = {r2:.4f}")"""
  },
  {
    "title": "Residual Diagnostics",
    "code": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from scipy import stats

X, y = make_regression(n_samples=200, n_features=5, noise=20, random_state=42)
model = LinearRegression().fit(X, y)
residuals = y - model.predict(X)

# Normality test
stat, p_value = stats.shapiro(residuals[:50])  # Shapiro on sample
print(f"Shapiro-Wilk: stat={stat:.4f}, p={p_value:.4f}")
print(f"Residual mean : {residuals.mean():.4f} (should be ~0)")
print(f"Residual std  : {residuals.std():.4f}")

# Durbin-Watson (autocorrelation)
dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"Durbin-Watson : {dw:.4f} (2=no autocorr, <2=positive, >2=negative)")"""
  },
],

# ══════════════════════════════════════════════════════
"logistic_regression": [
  {
    "title": "Binary Classification",
    "code": """from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (critical for LR with regularization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, model.predict(X_test)))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")"""
  },
  {
    "title": "Multiclass (Softmax / OvR)",
    "code": """from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

iris = load_iris()
X, y = iris.data, iris.target

for strategy in ['ovr', 'multinomial']:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(multi_class=strategy, max_iter=1000,
                                   solver='lbfgs', C=1.0)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"{strategy:15s} | CV Acc = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
  {
    "title": "Threshold Tuning & Precision-Recall",
    "code": """import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

X, y = make_classification(n_samples=1000, weights=[0.85, 0.15], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
for thresh in np.arange(0.2, 0.9, 0.1):
    preds = (probs >= thresh).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)
    print(f"{thresh:>10.1f} {p:>10.4f} {r:>8.4f} {f:>8.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"decision_tree": [
  {
    "title": "Classification Tree with Visualization",
    "code": """from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
dt.fit(X_train, y_train)

print(f"Train Accuracy : {accuracy_score(y_train, dt.predict(X_train)):.4f}")
print(f"Test  Accuracy : {accuracy_score(y_test,  dt.predict(X_test)):.4f}")
print(f"Tree Depth     : {dt.get_depth()}")
print(f"Leaf Nodes     : {dt.get_n_leaves()}")
print("\\nFeature Importances:")
for name, imp in zip(iris.feature_names, dt.feature_importances_):
    print(f"  {name:25s}: {imp:.4f}")
print("\\nText Tree:")
print(export_text(dt, feature_names=iris.feature_names))"""
  },
  {
    "title": "Regression Tree",
    "code": """import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=5, noise=25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Try different depths
print(f"{'Max Depth':>10} {'Train R²':>10} {'Test R²':>10} {'RMSE':>10}")
for depth in [2, 4, 6, 8, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    tr = r2_score(y_train, model.predict(X_train))
    te = r2_score(y_test,  model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"{str(depth):>10} {tr:>10.4f} {te:>10.4f} {rmse:>10.4f}")"""
  },
  {
    "title": "Cost-Complexity Pruning (Post-Pruning)",
    "code": """from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# Get effective alphas from a full tree
full_tree = DecisionTreeClassifier(random_state=42)
path = full_tree.cost_complexity_pruning_path(X, y)
ccp_alphas = path.ccp_alphas[::5]  # sample every 5th

print(f"{'Alpha':>12} {'CV Score':>10} {'Std':>8}")
for alpha in ccp_alphas[:10]:
    dt = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(dt, X, y, cv=5)
    print(f"{alpha:>12.6f} {scores.mean():>10.4f} {scores.std():>8.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"svm": [
  {
    "title": "SVC with RBF Kernel",
    "code": """from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=400, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM requires feature scaling
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc',    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])

# Grid search for best C and gamma
param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 'auto', 0.01]}
gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

print(f"Best params  : {gs.best_params_}")
print(f"Best CV score: {gs.best_score_:.4f}")
print(classification_report(y_test, gs.predict(X_test)))"""
  },
  {
    "title": "Kernel Comparison",
    "code": """from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

datasets = {
    "Moons"   : make_moons(n_samples=300, noise=0.1, random_state=42),
    "Circles" : make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
}
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for ds_name, (X, y) in datasets.items():
    X = StandardScaler().fit_transform(X)
    print(f"\\n{ds_name}:")
    for kernel in kernels:
        svc = SVC(kernel=kernel, C=1.0, degree=3, gamma='scale')
        scores = cross_val_score(svc, X, y, cv=5)
        print(f"  {kernel:8s} | {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
  {
    "title": "Multi-class SVM (OvR + OvO)",
    "code": """from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_iris(return_X_y=True)

strategies = {
    "SVC (built-in OvO)": SVC(kernel='rbf', decision_function_shape='ovo'),
    "OvR wrapper"       : OneVsRestClassifier(SVC(kernel='rbf')),
    "OvO wrapper"       : OneVsOneClassifier(SVC(kernel='rbf')),
}

for name, clf in strategies.items():
    pipe = Pipeline([('sc', StandardScaler()), ('clf', clf)])
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"{name:25s} | CV Acc = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"svr": [
  {
    "title": "SVR for Regression",
    "code": """import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 150)).reshape(-1, 1)
y = np.sin(X.ravel()) * 2 + np.random.normal(0, 0.3, 150)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "SVR (RBF)  ": SVR(kernel='rbf',  C=5,   epsilon=0.1, gamma='scale'),
    "SVR (Linear)": SVR(kernel='linear', C=1,  epsilon=0.1),
    "SVR (Poly) ": SVR(kernel='poly', C=5,   epsilon=0.1, degree=3, gamma='scale'),
}

for name, svr in models.items():
    pipe = Pipeline([('sc', StandardScaler()), ('svr', svr)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{name} | RMSE={rmse:.4f}  R²={r2:.4f}")"""
  },
  {
    "title": "SVR Hyperparameter Tuning",
    "code": """from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)

pipe = Pipeline([('sc', StandardScaler()), ('svr', SVR())])

param_grid = {
    'svr__kernel' : ['rbf', 'poly'],
    'svr__C'      : [0.1, 1.0, 10.0],
    'svr__epsilon': [0.01, 0.1, 0.5],
    'svr__gamma'  : ['scale', 'auto'],
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
gs.fit(X, y)
print(f"Best R²    : {gs.best_score_:.4f}")
print(f"Best params: {gs.best_params_}")"""
  },
],

# ══════════════════════════════════════════════════════
"knn": [
  {
    "title": "KNN Classifier",
    "code": """from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find optimal k
print(f"{'k':>5} {'CV Acc':>10} {'Std':>8}")
for k in [1, 3, 5, 7, 10, 15, 20]:
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance',
                                      metric='euclidean', n_jobs=-1))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"{k:>5} {scores.mean():>10.4f} {scores.std():>8.4f}")"""
  },
  {
    "title": "KNN Regressor with Different Distance Metrics",
    "code": """from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_regression(n_samples=400, n_features=8, noise=20, random_state=42)

configs = [
    ('Euclidean k=5 ',   5,  'euclidean', 'uniform'),
    ('Manhattan k=5 ',   5,  'manhattan', 'uniform'),
    ('Euclidean k=10',  10,  'euclidean', 'distance'),
    ('Chebyshev k=7 ',   7,  'chebyshev', 'distance'),
]

for label, k, metric, weights in configs:
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights))
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    print(f"{label} | R² = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
  {
    "title": "Ball Tree vs KD-Tree (Speed Comparison)",
    "code": """import time, numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Simulate large dataset
np.random.seed(42)
X_train = np.random.randn(5000, 20)
y_train = (X_train[:, 0] > 0).astype(int)
X_test  = np.random.randn(500, 20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

for algo in ['auto', 'ball_tree', 'kd_tree', 'brute']:
    knn = KNeighborsClassifier(n_neighbors=10, algorithm=algo)
    t0 = time.time()
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    elapsed = time.time() - t0
    print(f"algorithm={algo:12s} | Time = {elapsed*1000:.2f} ms")"""
  },
],

# ══════════════════════════════════════════════════════
"naive_bayes": [
  {
    "title": "Gaussian NB for Continuous Features",
    "code": """from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

gnb = GaussianNB(var_smoothing=1e-9)  # Laplace-like smoothing for numerics
gnb.fit(X_train, y_train)

print("Class priors:", gnb.class_prior_)
print("Feature means per class:")
for cls, means in enumerate(gnb.theta_):
    print(f"  Class {cls}: {np.round(means[:5], 3)} ...")
print(classification_report(y_test, gnb.predict(X_test)))"""
  },
  {
    "title": "Multinomial NB for Text Classification",
    "code": """from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Toy text dataset
texts = [
    "Python machine learning neural network deep",
    "sklearn classification regression model",
    "sports football basketball team win",
    "match goal player score field",
    "finance stock market investment portfolio",
    "bitcoin crypto trading price bull",
] * 20
labels = ([0]*2 + [1]*2 + [2]*2) * 20

for name, vectorizer, clf in [
    ("CountVec + MultinomialNB", CountVectorizer(), MultinomialNB()),
    ("TF-IDF   + ComplementNB ", TfidfVectorizer(),ComplementNB()),
    ("CountVec + BernoulliNB  ", CountVectorizer(binary=True), BernoulliNB()),
]:
    pipe = Pipeline([("vec", vectorizer), ("clf", clf)])
    scores = cross_val_score(pipe, texts, labels, cv=5)
    print(f"{name} | CV Acc = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"bagging": [
  {
    "title": "BaggingClassifier with Custom Base Estimator",
    "code": """from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single unpruned tree (high variance)
single = DecisionTreeClassifier(max_depth=None, random_state=42)
single.fit(X_train, y_train)
print(f"Single DT  | Train={accuracy_score(y_train, single.predict(X_train)):.4f}"
      f"  Test={accuracy_score(y_test, single.predict(X_test)):.4f}")

# Bagging ensemble
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100,
    max_samples=0.8,         # 80% of data per tree
    max_features=0.8,        # 80% of features per tree
    bootstrap=True,          # with replacement
    oob_score=True,          # free validation
    random_state=42, n_jobs=-1
)
bag.fit(X_train, y_train)
print(f"Bagging    | OOB={bag.oob_score_:.4f}"
      f"  Test={accuracy_score(y_test, bag.predict(X_test)):.4f}")"""
  },
  {
    "title": "BaggingRegressor",
    "code": """from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_regression(n_samples=400, n_features=10, noise=25, random_state=42)

configs = [
    ("DT Bagging  ", BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=100)),
    ("Ridge Bagging", BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=50)),
]

for name, model in configs:
    pipe = Pipeline([('sc', StandardScaler()), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    print(f"{name} | R² = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"random_forest": [
  {
    "title": "Random Forest Classifier — Production Pattern",
    "code": """from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                            n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,     # more trees = more stable (diminishing returns after ~100)
    max_depth=None,       # grow full trees; regularize via min_samples_leaf
    max_features='sqrt',  # sqrt(n_features) for classification
    min_samples_leaf=2,
    oob_score=True,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print(f"OOB Score : {rf.oob_score_:.4f}")
print(f"Test AUC  : {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")
print(classification_report(y_test, rf.predict(X_test)))

# Top 5 features
top5 = np.argsort(rf.feature_importances_)[::-1][:5]
for rank, idx in enumerate(top5, 1):
    print(f"  Rank {rank}: Feature_{idx} = {rf.feature_importances_[idx]:.4f}")"""
  },
  {
    "title": "Random Forest Regressor with Permutation Importance",
    "code": """from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_features='sqrt',
                            min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print(f"Test R²: {r2_score(y_test, rf.predict(X_test)):.4f}")

# Permutation importance (more reliable than MDI)
result = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=42)
print("\\nPermutation Importance (top 5):")
order = result.importances_mean.argsort()[::-1][:5]
for i in order:
    print(f"  {data.feature_names[i]:15s}: {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")"""
  },
  {
    "title": "Extremely Randomized Trees (ExtraTrees)",
    "code": """from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import time

X, y = make_classification(n_samples=2000, n_features=30, random_state=42)

for name, model in [
    ("RandomForest  ", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ("ExtraTrees    ", ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
]:
    t0 = time.time()
    scores = cross_val_score(model, X, y, cv=5)
    elapsed = time.time() - t0
    print(f"{name} | CV={scores.mean():.4f} ± {scores.std():.4f} | Time={elapsed:.2f}s")"""
  },
],

# ══════════════════════════════════════════════════════
"gradient_boosting": [
  {
    "title": "Sklearn GradientBoosting",
    "code": """from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbm = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,    # smaller lr = better generalization (needs more trees)
    max_depth=4,
    subsample=0.8,         # stochastic gradient boosting
    min_samples_leaf=10,
    max_features='sqrt',
    validation_fraction=0.15,
    n_iter_no_change=20,   # early stopping
    random_state=42
)
gbm.fit(X_train, y_train)

print(f"Trees used (early stop): {gbm.n_estimators_}")
print(f"Train Log-Loss : {log_loss(y_train, gbm.predict_proba(X_train)):.4f}")
print(f"Test  AUC-ROC  : {roc_auc_score(y_test, gbm.predict_proba(X_test)[:,1]):.4f}")"""
  },
  {
    "title": "XGBoost — Production Pattern",
    "code": """# pip install xgboost
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=2000, n_features=30, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,         # L2 regularization
    eval_metric='auc',
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

print(f"Best iteration : {model.best_iteration}")
print(f"Test AUC       : {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")"""
  },
  {
    "title": "LightGBM — Production Pattern",
    "code": """# pip install lightgbm
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=5000, n_features=40, n_informative=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

params = {
    'objective'       : 'binary',
    'metric'          : 'auc',
    'learning_rate'   : 0.05,
    'num_leaves'      : 63,        # 2^max_depth - 1
    'max_depth'       : -1,        # no limit (leaf-wise growth)
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq'    : 5,
    'reg_alpha'       : 0.1,
    'reg_lambda'      : 1.0,
    'verbose'         : -1,
}

callbacks = [lgb.early_stopping(30), lgb.log_evaluation(period=50)]
model = lgb.train(params, train_data, num_boost_round=500,
                  valid_sets=[val_data], callbacks=callbacks)

preds = model.predict(X_val)
print(f"Best iteration : {model.best_iteration}")
print(f"Val AUC        : {roc_auc_score(y_val, preds):.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"xgboost": [
  {
    "title": "XGBoost with Optuna Hyperparameter Tuning",
    "code": """# pip install xgboost optuna
import xgboost as xgb
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

def objective(trial):
    params = {
        'n_estimators'   : trial.suggest_int('n_estimators', 50, 300),
        'learning_rate'  : trial.suggest_float('lr', 0.01, 0.3, log=True),
        'max_depth'      : trial.suggest_int('max_depth', 2, 8),
        'subsample'      : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
        'reg_alpha'      : trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'reg_lambda'     : trial.suggest_float('lambda', 1e-5, 1.0, log=True),
        'random_state'   : 42, 'n_jobs': -1, 'use_label_encoder': False,
    }
    model = xgb.XGBClassifier(**params, eval_metric='logloss')
    return cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=False)
print(f"Best AUC   : {study.best_value:.4f}")
print(f"Best params: {study.best_params}")"""
  },
  {
    "title": "CatBoost for Categorical Features",
    "code": """# pip install catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Synthetic dataset with categorical features
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age'      : np.random.randint(18, 70, n),
    'income'   : np.random.exponential(50000, n),
    'city'     : np.random.choice(['Mumbai','Delhi','Hyderabad','Bengaluru'], n),
    'education': np.random.choice(['BSc','MSc','PhD','MBA'], n),
    'employed' : np.random.choice([0, 1], n),
})
y = ((df['income'] > 60000) & (df['employed'] == 1)).astype(int).values

cat_features = ['city', 'education']
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)

model = CatBoostClassifier(iterations=200, learning_rate=0.05,
                            depth=5, verbose=0)
model.fit(train_pool, eval_set=test_pool)
print(f"AUC: {roc_auc_score(y_test, model.predict_proba(test_pool)[:,1]):.4f}")"""
  },
  {
    "title": "Feature Importance: XGBoost vs LightGBM vs RF",
    "code": """import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10,
                            n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    'XGBoost'  : xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM' : lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
}

feat_names = [f'F{i}' for i in range(10)]
print(f"{'Feature':>8}", end='')
for name in models: print(f" | {name:>12}", end='')
print()

for fi, feat in enumerate(feat_names):
    print(f"{feat:>8}", end='')
    for name, m in models.items():
        m.fit(X_train, y_train)
        imp = m.feature_importances_[fi]
        print(f" | {imp:>12.4f}", end='')
    print()"""
  },
],

# ══════════════════════════════════════════════════════
"adaboost": [
  {
    "title": "AdaBoost Classifier",
    "code": """from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# AdaBoost with decision stumps (default)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # weak learner = stump
    n_estimators=200,
    learning_rate=0.5,
    algorithm='SAMME.R',  # real-valued probabilities (better than SAMME)
    random_state=42
)
ada.fit(X_train, y_train)

print(f"Train Acc  : {ada.score(X_train, y_train):.4f}")
print(f"Test  AUC  : {roc_auc_score(y_test, ada.predict_proba(X_test)[:,1]):.4f}")

# Staged predictions (accuracy at each boosting round)
staged_accs = [acc for acc in ada.staged_score(X_test, y_test)]
best_n = np.argmax(staged_accs) + 1
print(f"Best #estimators: {best_n} (staged test acc = {staged_accs[best_n-1]:.4f})")"""
  },
  {
    "title": "AdaBoost Sensitivity to Noise",
    "code": """import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# Add label noise
def add_noise(y, rate):
    y_noisy = y.copy()
    idx = np.random.choice(len(y), int(len(y)*rate), replace=False)
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy

print(f"{'Noise %':>8} {'AdaBoost':>12} {'GBM':>12}")
for noise in [0.0, 0.05, 0.1, 0.2]:
    y_noisy = add_noise(y, noise)
    ada_sc = cross_val_score(AdaBoostClassifier(n_estimators=100, random_state=42), X, y_noisy, cv=5).mean()
    gbm_sc = cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42), X, y_noisy, cv=5).mean()
    print(f"{noise*100:>7.0f}% {ada_sc:>12.4f} {gbm_sc:>12.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"ensemble_comparison": [
  {
    "title": "Voting Classifier (Hard + Soft)",
    "code": """from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=600, n_features=15, random_state=42)

estimators = [
    ('lr',  LogisticRegression(max_iter=1000)),
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, kernel='rbf')),
]

for voting in ['hard', 'soft']:
    vc = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
    pipe = Pipeline([('sc', StandardScaler()), ('vc', vc)])
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"Voting ({voting}) | CV = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
  {
    "title": "Stacking Classifier",
    "code": """from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=600, n_features=15, random_state=42)

base_learners = [
    ('rf',  RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gbm', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=7)),
    ('gnb', GaussianNB()),
]

stack = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),  # meta-learner
    cv=5,                                   # cross-val predictions for meta-learner
    passthrough=False,                      # set True to also pass original features
    n_jobs=-1
)
pipe = Pipeline([('sc', StandardScaler()), ('stack', stack)])
scores = cross_val_score(pipe, X, y, cv=5)
print(f"Stacking | CV = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"kmeans": [
  {
    "title": "K-Means Clustering with Optimal K Selection",
    "code": """from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.2, random_state=42)
X = StandardScaler().fit_transform(X)

print(f"{'k':>5} {'Inertia':>12} {'Silhouette':>12} {'BIC approx':>12}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    # Approximate BIC: inertia / (n * d) + k * log(n)
    n, d = X.shape
    bic = km.inertia_ / (n * d) + k * np.log(n)
    print(f"{k:>5} {km.inertia_:>12.2f} {sil:>12.4f} {bic:>12.4f}")"""
  },
  {
    "title": "Mini-Batch K-Means (Large Scale)",
    "code": """from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import time, numpy as np

X, y_true = make_blobs(n_samples=50_000, centers=8, random_state=42)
X = StandardScaler().fit_transform(X)

for name, cls in [("KMeans      ", KMeans), ("MiniBatchKM ", MiniBatchKMeans)]:
    kwargs = dict(n_clusters=8, random_state=42) if cls == KMeans else dict(n_clusters=8, batch_size=1024, random_state=42)
    t0 = time.time()
    model = cls(**kwargs).fit(X)
    elapsed = time.time() - t0
    print(f"{name} | Inertia={model.inertia_:.1f} | Time={elapsed:.3f}s")"""
  },
  {
    "title": "K-Means for Image Color Quantization",
    "code": """# Demonstrates K-Means for real use: compressing image colors
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Simulate an image (H x W x 3 RGB)
np.random.seed(42)
H, W = 100, 100
image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

# Reshape to (N_pixels, 3)
pixels = image.reshape(-1, 3).astype(float)

for n_colors in [4, 8, 16, 32, 64]:
    km = MiniBatchKMeans(n_clusters=n_colors, batch_size=1024, random_state=42)
    labels = km.fit_predict(pixels)
    compressed = km.cluster_centers_[labels]
    # Compression ratio (bits)
    original_bits = H * W * 3 * 8
    compressed_bits = H * W * int(np.log2(n_colors)) + n_colors * 3 * 8
    ratio = original_bits / compressed_bits
    mse = np.mean((pixels - compressed)**2)
    print(f"Colors={n_colors:3d} | MSE={mse:6.2f} | Compression ratio≈{ratio:.2f}x")"""
  },
],

# ══════════════════════════════════════════════════════
"dbscan": [
  {
    "title": "DBSCAN — Arbitrary Shape Clustering",
    "code": """from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import numpy as np

datasets = {
    "Moons"  : make_moons(n_samples=300, noise=0.07, random_state=42),
    "Circles": make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
}

for name, (X, y_true) in datasets.items():
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.2, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    ari        = adjusted_rand_score(y_true, labels)
    print(f"{name:8s} | Clusters={n_clusters} | Noise={n_noise} | ARI={ari:.4f}")"""
  },
  {
    "title": "Finding Optimal eps via k-distance Plot",
    "code": """import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=400, noise=0.08, random_state=42)
X = StandardScaler().fit_transform(X)

# k-distance plot: sort k-NN distances, look for the "knee"
k = 5  # min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, -1])[::-1]  # sorted descending

# Find the knee automatically (max curvature)
diffs = np.diff(k_distances)
knee_idx = np.argmax(np.diff(diffs)) + 1
eps_optimal = k_distances[knee_idx]
print(f"Optimal eps ≈ {eps_optimal:.4f} (knee at index {knee_idx})")

# Apply DBSCAN with found eps
db = DBSCAN(eps=eps_optimal, min_samples=k).fit(X)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters found: {n_clusters}")
print(f"Noise points  : {np.sum(labels == -1)}")"""
  },
  {
    "title": "HDBSCAN (Variable Density)",
    "code": """# pip install hdbscan
# from hdbscan import HDBSCAN

# Simulated example using sklearn DBSCAN as fallback
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Two clusters with very different densities
X1, _ = make_blobs(n_samples=200, centers=[[0,0]], cluster_std=0.3, random_state=0)
X2, _ = make_blobs(n_samples=100, centers=[[5,5]], cluster_std=1.5, random_state=1)
X = np.vstack([X1, X2])
X = StandardScaler().fit_transform(X)

for eps in [0.1, 0.2, 0.4, 0.6]:
    db = DBSCAN(eps=eps, min_samples=5).fit(X)
    n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = np.sum(db.labels_ == -1)
    print(f"eps={eps} | clusters={n_cl} | noise={n_noise}")
# Note: HDBSCAN handles this automatically — install via: pip install hdbscan"""
  },
],

# ══════════════════════════════════════════════════════
"pca": [
  {
    "title": "PCA for Dimensionality Reduction",
    "code": """from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

X, y = load_digits(return_X_y=True)  # 1797 samples, 64 features

print(f"Original shape: {X.shape}")
print(f"{'n_components':>14} {'Var Explained':>14} {'CV Acc':>10}")

for n in [2, 5, 10, 20, 32, 64]:
    pipe = Pipeline([
        ('sc',  StandardScaler()),
        ('pca', PCA(n_components=min(n, 64))),
        ('lr',  LogisticRegression(max_iter=2000)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5)
    # Get explained variance
    pca = PCA(n_components=min(n, 64)).fit(StandardScaler().fit_transform(X))
    var = pca.explained_variance_ratio_.sum()
    print(f"{n:>14} {var:>13.3%} {scores.mean():>10.4f}")"""
  },
  {
    "title": "PCA — Reconstruction & Information Loss",
    "code": """import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

for n_comp in [2, 5, 10, 20, 40, 64]:
    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(X_scaled)            # encode
    X_rec = sc.inverse_transform(pca.inverse_transform(Z))  # decode
    mse = np.mean((X - X_rec)**2)
    var = pca.explained_variance_ratio_.sum()
    print(f"n={n_comp:3d} | Var explained={var:.3%} | Reconstruction MSE={mse:.4f}")"""
  },
  {
    "title": "Kernel PCA (Non-Linear)",
    "code": """from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_circles(n_samples=400, noise=0.05, factor=0.3, random_state=42)

kernels = ['linear', 'rbf', 'poly', 'sigmoid', 'cosine']
for kernel in kernels:
    pipe = Pipeline([
        ('sc',   StandardScaler()),
        ('kpca', KernelPCA(n_components=2, kernel=kernel, gamma=10)),
        ('lr',   LogisticRegression()),
    ])
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"Kernel={kernel:8s} | CV Acc = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"tsne": [
  {
    "title": "t-SNE for High-Dimensional Visualization",
    "code": """from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

# t-SNE is slow — use PCA first to reduce to 50 dims
from sklearn.decomposition import PCA
X_pca = PCA(n_components=50).fit_transform(X)  # pre-reduce for speed

print("Running t-SNE with different perplexity values...")
for perp in [5, 15, 30, 50]:
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate='auto',
        init='pca',           # PCA init (more stable than random)
        max_iter=1000,
        random_state=42
    )
    embedding = tsne.fit_transform(X_pca)
    kl_div = tsne.kl_divergence_
    print(f"  perplexity={perp:3d} | KL divergence={kl_div:.4f} | shape={embedding.shape}")"""
  },
  {
    "title": "UMAP (Faster + Better Global Structure)",
    "code": """# pip install umap-learn
# import umap
# from sklearn.datasets import load_digits
# from sklearn.preprocessing import StandardScaler

# UMAP example (uncomment after installing umap-learn):
# X, y = load_digits(return_X_y=True)
# X = StandardScaler().fit_transform(X)
#
# reducer = umap.UMAP(
#     n_components=2,
#     n_neighbors=15,       # local neighborhood size
#     min_dist=0.1,         # how tightly to pack points
#     metric='euclidean',
#     random_state=42
# )
# embedding = reducer.fit_transform(X)
# print(f"UMAP embedding shape: {embedding.shape}")

# Simulated comparison using scikit-learn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import time

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

for name, method in [("PCA", PCA(n_components=2)), ("t-SNE", TSNE(n_components=2, random_state=42, max_iter=500))]:
    t0 = time.time()
    emb = method.fit_transform(X)
    print(f"{name:6s} | Time={time.time()-t0:.2f}s | Shape={emb.shape}")"""
  },
],

# ══════════════════════════════════════════════════════
"clustering_advanced": [
  {
    "title": "Agglomerative Clustering",
    "code": """from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)

linkages = ['ward', 'complete', 'average', 'single']
print(f"{'Linkage':>10} {'ARI':>8} {'Silhouette':>12}")
for linkage in linkages:
    ag = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    labels = ag.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X, labels)
    print(f"{linkage:>10} {ari:>8.4f} {sil:>12.4f}")"""
  },
  {
    "title": "Gaussian Mixture Model (EM Algorithm)",
    "code": """from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)
X = StandardScaler().fit_transform(X)

print("Model selection via BIC (lower is better):")
print(f"{'n_components':>14} {'covariance_type':>18} {'BIC':>12} {'ARI':>8}")
for n in [2, 3, 4, 5, 6]:
    for cov in ['full', 'tied', 'diag', 'spherical']:
        gm = GaussianMixture(n_components=n, covariance_type=cov,
                              n_init=5, random_state=42)
        gm.fit(X)
        labels = gm.predict(X)
        bic = gm.bic(X)
        ari = adjusted_rand_score(y_true, labels)
        print(f"{n:>14} {cov:>18} {bic:>12.2f} {ari:>8.4f}")"""
  },
  {
    "title": "Dendrogram Cutting",
    "code": """from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import numpy as np

X, y_true = make_blobs(n_samples=150, centers=4, random_state=42)
X = StandardScaler().fit_transform(X)

# Build linkage matrix (Ward)
Z = linkage(X, method='ward')

print("Cutting dendrogram at different heights:")
print(f"{'Height':>8} {'# Clusters':>12} {'ARI':>8}")
for height in [2, 4, 6, 8, 10, 15]:
    labels = fcluster(Z, t=height, criterion='distance')
    n_cl = len(np.unique(labels))
    ari  = adjusted_rand_score(y_true, labels)
    print(f"{height:>8} {n_cl:>12} {ari:>8.4f}")

# Cut to get exactly 4 clusters
labels_4 = fcluster(Z, t=4, criterion='maxclust')
print(f"\\n4-cluster cut ARI: {adjusted_rand_score(y_true, labels_4):.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"autoencoder": [
  {
    "title": "Autoencoder with PyTorch",
    "code": """# pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Data
X, _ = load_digits(return_X_y=True)
X = MinMaxScaler().fit_transform(X).astype(np.float32)
X_tensor = torch.tensor(X)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(input_dim=64, latent_dim=8)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    out = model(X_tensor)
    loss = criterion(out, X_tensor)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:4d} | Loss = {loss.item():.6f}")

# Extract latent codes
with torch.no_grad():
    latent = model.encoder(X_tensor).numpy()
print(f"\\nLatent space shape: {latent.shape}")"""
  },
  {
    "title": "Anomaly Detection with Autoencoder",
    "code": """import torch, torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

# Create normal vs anomaly data
X_normal = np.random.randn(800, 20).astype(np.float32)
X_anomaly = (np.random.randn(200, 20) * 3 + 4).astype(np.float32)

sc = StandardScaler()
X_train = torch.tensor(sc.fit_transform(X_normal))
X_test_normal  = torch.tensor(sc.transform(X_normal[:100]))
X_test_anomaly = torch.tensor(sc.transform(X_anomaly[:100]))

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(20,10), nn.ReLU(), nn.Linear(10,4))
        self.dec = nn.Sequential(nn.Linear(4,10),  nn.ReLU(), nn.Linear(10,20))
    def forward(self, x): return self.dec(self.enc(x))

ae = AE()
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
for _ in range(200):
    loss = nn.MSELoss()(ae(X_train), X_train)
    opt.zero_grad(); loss.backward(); opt.step()

# Reconstruction error as anomaly score
with torch.no_grad():
    score_normal  = ((ae(X_test_normal)  - X_test_normal)**2).mean(1).numpy()
    score_anomaly = ((ae(X_test_anomaly) - X_test_anomaly)**2).mean(1).numpy()

scores = np.concatenate([score_normal, score_anomaly])
labels = np.array([0]*100 + [1]*100)
print(f"Anomaly Detection AUC: {roc_auc_score(labels, scores):.4f}")
print(f"Normal  recon error mean : {score_normal.mean():.4f}")
print(f"Anomaly recon error mean : {score_anomaly.mean():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"anomaly_detection": [
  {
    "title": "Isolation Forest — Production Pattern",
    "code": """from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

np.random.seed(42)
# Normal data
X_normal = np.random.randn(1000, 5)
# Anomalies (5% contamination)
X_anomaly = np.random.uniform(-6, 6, (50, 5))
X = np.vstack([X_normal, X_anomaly])
y_true = np.array([0]*1000 + [1]*50)  # 0=normal, 1=anomaly

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Note: contamination tells the model what % to flag as outliers
ifo = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples='auto',
    random_state=42, n_jobs=-1
)
ifo.fit(X_scaled[y_true == 0])  # train on normal only (semi-supervised)
scores = ifo.decision_function(X_scaled)  # higher = more normal
preds  = ifo.predict(X_scaled)  # 1=normal, -1=outlier

auc = roc_auc_score(y_true, -scores)  # negate: lower score = more anomalous
print(f"AUC-ROC: {auc:.4f}")
print(f"Flagged as anomalies: {(preds==-1).sum()} (true: {y_true.sum()})")"""
  },
  {
    "title": "LOF + Elliptic Envelope + OCSVM",
    "code": """from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np

np.random.seed(42)
X_normal  = np.random.randn(500, 3)
X_anomaly = np.random.uniform(-5, 5, (25, 3))
X = np.vstack([X_normal, X_anomaly])
y = np.array([0]*500 + [1]*25)
X = StandardScaler().fit_transform(X)

detectors = {
    'LOF (n=20)': LocalOutlierFactor(n_neighbors=20, contamination=0.05),
    'Elliptic Env': EllipticEnvelope(contamination=0.05, random_state=42),
    'One-Class SVM': OneClassSVM(nu=0.05, kernel='rbf', gamma='scale'),
}

for name, det in detectors.items():
    if isinstance(det, LocalOutlierFactor):
        preds = det.fit_predict(X)
        scores = -det.negative_outlier_factor_
    else:
        det.fit(X[y == 0])
        preds  = det.predict(X)
        scores = -det.decision_function(X)
    auc = roc_auc_score(y, scores)
    flagged = (preds == -1).sum()
    print(f"{name:20s} | AUC={auc:.4f} | Flagged={flagged}")"""
  },
],

# ══════════════════════════════════════════════════════
"gaussian_process": [
  {
    "title": "GP Regression with Uncertainty",
    "code": """import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

np.random.seed(42)
# Sparse observations
X_train = np.array([-2, -1, 0, 1.5, 2.5, 3]).reshape(-1, 1)
y_train = np.sin(X_train.ravel()) + np.random.normal(0, 0.1, 6)
X_pred  = np.linspace(-4, 5, 200).reshape(-1, 1)

# Kernel: signal * RBF + noise
kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,  # restart to avoid local optima
    normalize_y=True,
    random_state=42
)
gpr.fit(X_train, y_train)

mu, sigma = gpr.predict(X_pred, return_std=True)

print(f"Optimized kernel: {gpr.kernel_}")
print(f"Log-marginal-likelihood: {gpr.log_marginal_likelihood_value_:.4f}")
print(f"\\nPrediction at x=0: μ={mu[100]:.4f}, σ={sigma[100]:.4f}")
print(f"Prediction at x=4: μ={mu[180]:.4f}, σ={sigma[180]:.4f} (extrapolation — high uncertainty)")"""
  },
  {
    "title": "GP for Bayesian Optimization",
    "code": """import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Simulate Bayesian Optimization of a black-box function
def black_box(x):
    return -np.sin(3*x) - x**2 + 0.7*x + np.random.normal(0, 0.02)

def expected_improvement(X_candidates, gpr, y_best, xi=0.01):
    from scipy.stats import norm
    mu, sigma = gpr.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

np.random.seed(42)
# Initial observations
X_obs = np.array([-2, 0, 2]).reshape(-1, 1)
y_obs = np.array([black_box(x[0]) for x in X_obs])

kernel = Matern(nu=2.5)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

X_candidates = np.linspace(-3, 3, 200).reshape(-1, 1)

for iteration in range(8):
    gpr.fit(X_obs, y_obs)
    ei = expected_improvement(X_candidates, gpr, y_obs.max())
    x_next = X_candidates[np.argmax(ei)]
    y_next = black_box(x_next[0])
    X_obs = np.vstack([X_obs, x_next])
    y_obs = np.append(y_obs, y_next)
    print(f"Iter {iteration+1:2d} | x_next={x_next[0]:6.3f} | y={y_next:7.4f} | best={y_obs.max():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"neural_network": [
  {
    "title": "PyTorch MLP — Full Training Loop",
    "code": """import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train = torch.tensor(sc.fit_transform(X_train), dtype=torch.float32)
X_test  = torch.tensor(sc.transform(X_test),      dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model = MLP()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(100):
    model.train()
    loss = criterion(model(X_train), y_train)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
    if (epoch+1) % 25 == 0:
        model.eval()
        with torch.no_grad():
            acc = ((model(X_test) > 0.5).float() == y_test).float().mean()
        print(f"Epoch {epoch+1:4d} | Loss={loss:.4f} | Test Acc={acc:.4f}")"""
  },
  {
    "title": "Keras/TF MLP with Callbacks",
    "code": """# pip install tensorflow keras
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# X, y = make_classification(n_samples=2000, n_features=25, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)
#
# model = keras.Sequential([
#     keras.layers.Dense(256, activation='relu', input_shape=(25,)),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.4),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(64, activation='swish'),
#     keras.layers.Dense(1, activation='sigmoid'),
# ])
# model.compile(optimizer=keras.optimizers.AdamW(1e-3),
#               loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
# callbacks = [
#     keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
#     keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
#     keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
# ]
# history = model.fit(X_train, y_train, validation_split=0.15,
#                     epochs=200, batch_size=64, callbacks=callbacks, verbose=0)
# print(f"Test AUC: {model.evaluate(X_test, y_test, verbose=0)[2]:.4f}")

# Sklearn MLPClassifier as runnable substitute
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    solver='adam', learning_rate_init=1e-3, max_iter=300,
                    early_stopping=True, validation_fraction=0.15, random_state=42)
mlp.fit(X_train, y_train)
print(f"Test Accuracy: {mlp.score(X_test, y_test):.4f}")
print(f"Best val loss: {mlp.best_loss_:.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"activations": [
  {
    "title": "Activation Functions Comparison",
    "code": """import numpy as np
import torch
import torch.nn as nn

# All standard activation functions in PyTorch
activations = {
    'ReLU'      : nn.ReLU(),
    'LeakyReLU' : nn.LeakyReLU(0.01),
    'ELU'       : nn.ELU(alpha=1.0),
    'GELU'      : nn.GELU(),
    'Swish/SiLU': nn.SiLU(),
    'Mish'      : nn.Mish(),
    'Sigmoid'   : nn.Sigmoid(),
    'Tanh'      : nn.Tanh(),
    'Softplus'  : nn.Softplus(),
}

x = torch.tensor([-3., -1., 0., 1., 3.])
print(f"{'Activation':>12} | {'f(-3)':>8} {'f(-1)':>8} {'f(0)':>8} {'f(1)':>8} {'f(3)':>8}")
print("-" * 65)
for name, act in activations.items():
    vals = act(x).detach().numpy()
    print(f"{name:>12} | {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f} {vals[4]:>8.4f}")"""
  },
  {
    "title": "Dying ReLU Problem & Solutions",
    "code": """import torch
import torch.nn as nn
import numpy as np

def check_dead_neurons(model, X_tensor):
    \"\"\"Count neurons that output 0 for ALL inputs (dead ReLU)\"\"\"
    with torch.no_grad():
        acts = {}
        def hook(name):
            def fn(mod, inp, out): acts[name] = out
            return fn
        handles = [layer.register_forward_hook(hook(f'layer_{i}'))
                   for i, layer in enumerate(model.modules())
                   if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.ELU))]
        _ = model(X_tensor)
        for h in handles: h.remove()
    dead = {name: (act == 0).all(dim=0).sum().item() for name, act in acts.items()}
    return dead

X = torch.randn(200, 10)

# ReLU with poor init — many dead neurons
model_relu = nn.Sequential(
    nn.Linear(10, 50), nn.ReLU(),
    nn.Linear(50, 20), nn.ReLU(),
)
# Initialize with very negative bias → dead neurons
for m in model_relu.modules():
    if isinstance(m, nn.Linear): nn.init.constant_(m.bias, -3.0)

dead = check_dead_neurons(model_relu, X)
for layer, count in dead.items():
    print(f"ReLU    {layer}: {count} dead neurons (out of ~50/20)")

# LeakyReLU — no dead neurons
model_lrelu = nn.Sequential(nn.Linear(10,50), nn.LeakyReLU(0.01), nn.Linear(50,20), nn.LeakyReLU(0.01))
for m in model_lrelu.modules():
    if isinstance(m, nn.Linear): nn.init.constant_(m.bias, -3.0)
dead2 = check_dead_neurons(model_lrelu, X)
for layer, count in dead2.items():
    print(f"Leaky   {layer}: {count} dead neurons")"""
  },
],

# ══════════════════════════════════════════════════════
"attention_transformer": [
  {
    "title": "Scaled Dot-Product Attention from Scratch",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    \"\"\"
    Q: (batch, heads, seq, d_k)
    K: (batch, heads, seq, d_k)
    V: (batch, heads, seq, d_v)
    \"\"\"
    d_k = Q.shape[-1]
    # Scaled dot-product: QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        def split_heads(t):
            return t.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        Q = split_heads(self.Wq(x))
        K = split_heads(self.Wk(x))
        V = split_heads(self.Wv(x))
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.Wo(out), attn

# Test
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
mha = MultiHeadAttention(d_model=512, n_heads=8)
out, weights = mha(x)
print(f"Input  shape : {x.shape}")
print(f"Output shape : {out.shape}")
print(f"Attn weights : {weights.shape}  (batch, heads, seq, seq)")"""
  },
  {
    "title": "Full Transformer Encoder Block",
    "code": """import torch
import torch.nn as nn
import math

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Pre-LN (modern: normalize before sublayer)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                 key_padding_mask=src_key_padding_mask)
        x = x + self.drop(attn_out)          # residual connection
        x = x + self.drop(self.ff(self.norm2(x)))  # FFN + residual
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Build a 4-layer encoder
encoder = nn.Sequential(
    PositionalEncoding(256),
    *[TransformerEncoderBlock(256, 8, 1024) for _ in range(4)]
)
x = torch.randn(4, 32, 256)  # batch=4, seq_len=32, d_model=256
out = encoder(x)
print(f"Encoder input  : {x.shape}")
print(f"Encoder output : {out.shape}")
total_params = sum(p.numel() for p in encoder.parameters())
print(f"Total params   : {total_params:,}")"""
  },
],

# ══════════════════════════════════════════════════════
"feature_engineering": [
  {
    "title": "Complete Feature Engineering Pipeline",
    "code": """import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(42)
n = 300
df = pd.DataFrame({
    'age'     : np.random.randint(18, 70, n).astype(float),
    'income'  : np.random.exponential(50000, n),
    'score'   : np.random.normal(600, 100, n),
    'city'    : np.random.choice(['Mumbai','Delhi','Hyderabad'], n),
    'edu'     : np.random.choice(['BSc','MSc','PhD'], n, p=[0.5,0.35,0.15]),
})
# Inject missing values
df.loc[np.random.choice(n, 30), 'age']    = np.nan
df.loc[np.random.choice(n, 20), 'income'] = np.nan
y = (df['income'].fillna(50000) > 55000).astype(int).values

num_cols = ['age', 'income', 'score']
cat_cols = ['city']
ord_cols = ['edu']

num_pipe = Pipeline([
    ('impute', KNNImputer(n_neighbors=5)),
    ('scale',  StandardScaler()),
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
ord_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ord',    OrdinalEncoder(categories=[['BSc','MSc','PhD']])),
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
    ('ord', ord_pipe, ord_cols),
])

full_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('select',     SelectKBest(f_classif, k=5)),
    ('model',      RandomForestClassifier(n_estimators=100, random_state=42)),
])

scores = cross_val_score(full_pipe, df, y, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
  {
    "title": "SMOTE for Imbalanced Classes",
    "code": """# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

# Heavily imbalanced dataset (5% minority)
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
print(f"Class distribution: {np.bincount(y)}")

strategies = {
    'No resampling'      : ImbPipeline([('clf', RandomForestClassifier(random_state=42))]),
    'class_weight=bal'   : ImbPipeline([('clf', RandomForestClassifier(class_weight='balanced', random_state=42))]),
    'SMOTE'              : ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', RandomForestClassifier(random_state=42))]),
    'ADASYN'             : ImbPipeline([('ada', ADASYN(random_state=42)),  ('clf', RandomForestClassifier(random_state=42))]),
    'Undersample'        : ImbPipeline([('rus', RandomUnderSampler(random_state=42)), ('clf', RandomForestClassifier(random_state=42))]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, pipe in strategies.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1')
    print(f"{name:22s} | F1 = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"regularization": [
  {
    "title": "L1/L2/ElasticNet Comparison",
    "code": """import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression

# High-dimensional, sparse: many irrelevant features
X, y, coef_true = make_regression(n_samples=200, n_features=100,
                                   n_informative=10, noise=20,
                                   coef=True, random_state=42)

models = {
    'OLS (no reg)': LinearRegression(),
    'Ridge  L2 α=1': Ridge(alpha=1.0),
    'Lasso  L1 α=0.1': Lasso(alpha=0.1, max_iter=5000),
    'ElasticNet α=0.1': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
}

for name, model in models.items():
    pipe = Pipeline([('sc', StandardScaler()), ('m', model)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    pipe.fit(X, y)
    coefs = pipe.named_steps['m'].coef_ if hasattr(pipe.named_steps['m'], 'coef_') else np.zeros(100)
    nonzero = np.sum(np.abs(coefs) > 1e-6)
    print(f"{name:22s} | R²={scores.mean():.4f} | Non-zero coefs={nonzero}")"""
  },
  {
    "title": "Dropout Regularization in Neural Networks",
    "code": """import torch, torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=600, n_features=20, random_state=42)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
Xtr = torch.tensor(X_tr, dtype=torch.float32)
Xte = torch.tensor(X_te, dtype=torch.float32)
ytr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
yte = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

def build_model(dropout_rate):
    return nn.Sequential(
        nn.Linear(20,256), nn.ReLU(), nn.Dropout(dropout_rate),
        nn.Linear(256,128), nn.ReLU(), nn.Dropout(dropout_rate),
        nn.Linear(128,1), nn.Sigmoid()
    )

def train_eval(model, epochs=150):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCELoss()
    for _ in range(epochs):
        model.train(); loss = crit(model(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(Xtr)>0.5)==ytr).float().mean().item()
        te_acc = ((model(Xte)>0.5)==yte).float().mean().item()
    return tr_acc, te_acc

print(f"{'Dropout':>10} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}")
for p in [0.0, 0.2, 0.4, 0.6]:
    tr, te = train_eval(build_model(p))
    print(f"{p:>10.1f} {tr:>10.4f} {te:>10.4f} {tr-te:>8.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"bias_variance": [
  {
    "title": "Bias-Variance Decomposition (Empirical)",
    "code": """import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

np.random.seed(42)
n_datasets = 100
n_train, n_test = 50, 200
n_features = 5

# True function
def true_f(X): return X[:, 0]**2 + np.sin(X[:, 1] * 3)

def empirical_bias_variance(max_depth):
    X_test = np.random.randn(n_test, n_features)
    y_test_true = true_f(X_test)
    all_preds = []
    for _ in range(n_datasets):
        X_train = np.random.randn(n_train, n_features)
        y_train = true_f(X_train) + np.random.normal(0, 0.5, n_train)
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        all_preds.append(model.predict(X_test))
    preds = np.array(all_preds)  # (100, 200)
    mean_pred = preds.mean(axis=0)
    bias2    = np.mean((mean_pred - y_test_true)**2)
    variance = np.mean(preds.var(axis=0))
    noise    = 0.25  # σ² = 0.5² = 0.25
    return bias2, variance, noise, bias2 + variance + noise

print(f"{'Max Depth':>10} {'Bias²':>10} {'Variance':>10} {'Noise':>8} {'Total':>8}")
for depth in [1, 2, 3, 5, 8, None]:
    b, v, n, total = empirical_bias_variance(depth)
    print(f"{str(depth):>10} {b:>10.4f} {v:>10.4f} {n:>8.4f} {total:>8.4f}")"""
  },
  {
    "title": "Learning Curves — Diagnose Bias vs Variance",
    "code": """from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = make_classification(n_samples=800, n_features=20, random_state=42)

configs = {
    "DT depth=1 (High Bias) ": DecisionTreeClassifier(max_depth=1),
    "DT depth=20 (High Var) ": DecisionTreeClassifier(max_depth=20),
    "SVC RBF (Balanced)     ": SVC(kernel='rbf', C=1.0, gamma='scale'),
}

for name, clf in configs.items():
    pipe = Pipeline([('sc', StandardScaler()), ('clf', clf)])
    train_sizes, tr_scores, val_scores = learning_curve(
        pipe, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    final_train = tr_scores[-1].mean()
    final_val   = val_scores[-1].mean()
    gap         = final_train - final_val
    print(f"{name} | Train={final_train:.4f} Val={final_val:.4f} Gap={gap:.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"model_evaluation": [
  {
    "title": "Comprehensive Classification Metrics",
    "code": """from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=1000, weights=[0.8, 0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
TP, FN, FP, TN = cm[1,1], cm[1,0], cm[0,1], cm[0,0]

print("=" * 50)
print(f"Confusion Matrix:\\n{cm}")
print(f"\\nTP={TP}  FP={FP}  FN={FN}  TN={TN}")
print("=" * 50)
print(f"Accuracy            : {(TP+TN)/(TP+FP+FN+TN):.4f}")
print(f"Precision           : {TP/(TP+FP):.4f}")
print(f"Recall (Sensitivity): {TP/(TP+FN):.4f}")
print(f"Specificity         : {TN/(TN+FP):.4f}")
print(f"F1 Score            : {2*TP/(2*TP+FP+FN):.4f}")
print(f"AUC-ROC             : {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR (Avg Prec)   : {average_precision_score(y_test, y_prob):.4f}")
print(f"MCC                 : {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen's Kappa       : {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"Log Loss            : {log_loss(y_test, y_prob):.4f}")
print(f"Brier Score         : {brier_score_loss(y_test, y_prob):.4f}")"""
  },
  {
    "title": "Regression Metrics",
    "code": """from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2   = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print(f"MSE                : {mse:.4f}")
print(f"RMSE               : {rmse:.4f}")
print(f"MAE                : {mae:.4f}")
print(f"MAPE               : {mape:.2f}%")
print(f"R²                 : {r2:.4f}")
print(f"Adjusted R²        : {adj_r2:.4f}")
print(f"Explained Variance : {explained_variance_score(y_test, y_pred):.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"cross_validation": [
  {
    "title": "All CV Strategies",
    "code": """from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedStratifiedKFold,
    LeaveOneOut, GroupKFold, TimeSeriesSplit, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=300, n_features=10, random_state=42)
groups = np.repeat(np.arange(30), 10)  # 30 groups of 10 samples
model = RandomForestClassifier(n_estimators=50, random_state=42)

strategies = {
    "KFold (k=5)           ": KFold(n_splits=5, shuffle=True, random_state=42),
    "StratifiedKFold (k=5) ": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    "RepeatedStratKF (3x5) ": RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
    "TimeSeriesSplit (k=5) ": TimeSeriesSplit(n_splits=5),
}

for name, cv in strategies.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"{name} | AUC = {scores.mean():.4f} ± {scores.std():.4f} ({len(scores)} folds)")

# GroupKFold (no group leakage)
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='roc_auc')
print(f"GroupKFold (k=5)        | AUC = {scores.mean():.4f} ± {scores.std():.4f}")"""
  },
  {
    "title": "Nested Cross-Validation (Gold Standard)",
    "code": """from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=300, n_features=15, random_state=42)

# Inner CV: hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# Outer CV: performance evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}

# GridSearchCV as the inner loop
gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1
)

# Outer loop evaluates the nested system
nested_scores = cross_val_score(gs, X, y, cv=outer_cv, scoring='roc_auc', n_jobs=-1)

# Non-nested (optimistic bias)
gs.fit(X, y)
non_nested = cross_val_score(gs.best_estimator_, X, y, cv=outer_cv, scoring='roc_auc')

print(f"Nested CV AUC     : {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
print(f"Non-nested CV AUC : {non_nested.mean():.4f} ± {non_nested.std():.4f}")
print(f"Optimism bias     : {non_nested.mean() - nested_scores.mean():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"hyperparameter_tuning": [
  {
    "title": "Optuna — Bayesian Hyperparameter Optimization",
    "code": """# pip install optuna
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = make_classification(n_samples=800, n_features=20, random_state=42)

def objective(trial):
    params = {
        'n_estimators'   : trial.suggest_int('n_estimators', 50, 300),
        'learning_rate'  : trial.suggest_float('lr', 0.01, 0.3, log=True),
        'max_depth'      : trial.suggest_int('max_depth', 2, 7),
        'subsample'      : trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_leaf': trial.suggest_int('min_leaf', 1, 20),
        'max_features'   : trial.suggest_categorical('max_f', ['sqrt', 'log2', None]),
        'random_state'   : 42
    }
    model = GradientBoostingClassifier(**params)
    return cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1).mean()

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=40)

print(f"Best AUC   : {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print(f"\\nTop 3 trials:")
for t in study.trials[:3]: print(f"  Trial {t.number}: AUC={t.value:.4f}")"""
  },
  {
    "title": "Halving Grid Search (Resource-Efficient)",
    "code": """from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np, time

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth'   : [3, 5, 7, None],
    'max_features': ['sqrt', 'log2', 0.5],
    'min_samples_leaf': [1, 5, 10],
}

# Standard GridSearch
# (would test 4*4*3*3=144 combos — expensive!)

# Successive Halving — prunes bad configs early
t0 = time.time()
hs = HalvingGridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, factor=3, min_resources=100,
    scoring='roc_auc', n_jobs=-1, random_state=42
)
hs.fit(X, y)
elapsed = time.time() - t0

print(f"HalvingGridSearch time  : {elapsed:.2f}s")
print(f"Best AUC                : {hs.best_score_:.4f}")
print(f"Best params             : {hs.best_params_}")
print(f"Total candidate configs evaluated: {len(hs.cv_results_['params'])}")"""
  },
],

# ══════════════════════════════════════════════════════
"optimization": [
  {
    "title": "Optimizer Comparison in PyTorch",
    "code": """import torch
import torch.nn as nn
import numpy as np

# Fit a simple network, compare optimizers
torch.manual_seed(42)
X = torch.randn(500, 10)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

def make_model(): return nn.Sequential(nn.Linear(10,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())

def train(optimizer_fn, epochs=200):
    model = make_model()
    opt = optimizer_fn(model.parameters())
    crit = nn.BCELoss()
    losses = []
    for e in range(epochs):
        out = model(X); loss = crit(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    acc = ((model(X) > 0.5).float() == y).float().mean().item()
    return losses[-1], acc

configs = {
    'SGD (lr=0.1)      ': lambda p: torch.optim.SGD(p, lr=0.1),
    'SGD+Momentum      ': lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    'RMSprop           ': lambda p: torch.optim.RMSprop(p, lr=1e-3),
    'Adam              ': lambda p: torch.optim.Adam(p, lr=1e-3),
    'AdamW             ': lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-4),
    'AdaGrad           ': lambda p: torch.optim.Adagrad(p, lr=0.05),
}

print(f"{'Optimizer':25s} | {'Final Loss':>12} | {'Accuracy':>10}")
for name, opt_fn in configs.items():
    loss, acc = train(opt_fn)
    print(f"{name:25s} | {loss:>12.6f} | {acc:>10.4f}")"""
  },
  {
    "title": "Learning Rate Schedulers",
    "code": """import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

model = nn.Linear(10, 1)
base_lr = 0.1

schedulers = {
    'StepLR (step=10, γ=0.5)' : (optim.SGD(model.parameters(), lr=base_lr),
                                   lambda opt: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)),
    'CosineAnnealing T=50'    : (optim.SGD(model.parameters(), lr=base_lr),
                                   lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)),
    'OneCycleLR'              : (optim.SGD(model.parameters(), lr=base_lr/10),
                                   lambda opt: optim.lr_scheduler.OneCycleLR(opt, max_lr=base_lr, total_steps=50)),
    'ExponentialLR γ=0.95'   : (optim.SGD(model.parameters(), lr=base_lr),
                                   lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)),
}

print(f"{'Scheduler':28s} | {'LR at 1':>10} {'LR at 10':>10} {'LR at 25':>10} {'LR at 50':>10}")
for name, (optimizer, sched_fn) in schedulers.items():
    sched = sched_fn(optimizer)
    lrs = {}
    for step in range(1, 51):
        lrs[step] = optimizer.param_groups[0]['lr']
        sched.step()
    print(f"{name:28s} | {lrs[1]:>10.6f} {lrs[10]:>10.6f} {lrs[25]:>10.6f} {lrs[50]:>10.6f}")"""
  },
],

# ══════════════════════════════════════════════════════
"interpretability": [
  {
    "title": "SHAP Values for Tree Models",
    "code": """# pip install shap
import shap
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP TreeExplainer (fast for tree models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:50])

print("SHAP analysis for first test sample:")
print(f"Base value (expected output): {explainer.expected_value:.4f}")
print(f"Model output for sample[0] : {model.predict_proba(X_test[:1])[0,1]:.4f}")
print(f"SHAP values sum + base      : {shap_values[0].sum() + explainer.expected_value:.4f}")
print("\\nTop 5 feature contributions (sample 0):")
top5 = np.abs(shap_values[0]).argsort()[::-1][:5]
for i in top5:
    print(f"  {data.feature_names[i]:35s}: SHAP={shap_values[0][i]:+.4f}  value={X_test[0,i]:.4f}")"""
  },
  {
    "title": "LIME — Local Explanations",
    "code": """# pip install lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = LimeTabularExplainer(
    X_train,
    feature_names=data.feature_names,
    class_names=data.target_names,
    mode='classification',
    discretize_continuous=True
)

# Explain a single prediction
idx = 5
exp = explainer.explain_instance(
    X_test[idx], model.predict_proba, num_features=10, num_samples=1000)

print(f"True label      : {data.target_names[y_test[idx]]}")
print(f"Predicted proba : {model.predict_proba(X_test[idx:idx+1])[0]}")
print("\\nTop 10 LIME feature contributions:")
for feat, weight in exp.as_list():
    print(f"  {feat:50s}: {weight:+.4f}")"""
  },
  {
    "title": "Permutation Importance + Partial Dependence",
    "code": """from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

data = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)

# Permutation importance
pi = permutation_importance(model, X_te, y_te, n_repeats=20, random_state=42, n_jobs=-1)
order = pi.importances_mean.argsort()[::-1]

print("Permutation Importance (test set):")
print(f"{'Feature':>20} {'Mean Drop':>12} {'Std':>8}")
for i in order[:5]:
    print(f"{data.feature_names[i]:>20} {pi.importances_mean[i]:>12.4f} {pi.importances_std[i]:>8.4f}")

# Interaction between top 2 features
top2 = order[:2].tolist()
print(f"\\nTop 2 features for PDP: {[data.feature_names[i] for i in top2]}")"""
  },
],

# ══════════════════════════════════════════════════════
"time_series": [
  {
    "title": "ARIMA / SARIMA Forecasting",
    "code": """# pip install statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import numpy as np

np.random.seed(42)
# Simulate ARMA(1,1) + trend + seasonal
t = np.arange(200)
trend = 0.02 * t
seasonal = 3 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 0.5, 200)
ts = trend + seasonal + noise

# Stationarity test (ADF)
result = adfuller(ts)
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f} ({'stationary' if result[1]<0.05 else 'NON-stationary'})")

train, test = ts[:180], ts[180:]

# ARIMA(1,1,1) — differencing for trend
arima = ARIMA(train, order=(1,1,1)).fit()
fc_arima = arima.forecast(steps=20)

# SARIMA(1,1,1)(1,1,0,12) — seasonal differencing
sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,0,12)).fit(disp=False)
fc_sarima = sarima.forecast(steps=20)

from sklearn.metrics import mean_squared_error
rmse_a = np.sqrt(mean_squared_error(test, fc_arima))
rmse_s = np.sqrt(mean_squared_error(test, fc_sarima))
print(f"\\nARIMA   RMSE: {rmse_a:.4f}")
print(f"SARIMA  RMSE: {rmse_s:.4f}")"""
  },
  {
    "title": "LSTM for Time Series Forecasting",
    "code": """import torch, torch.nn as nn
import numpy as np

# Create sliding window sequences
def make_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

np.random.seed(42)
t = np.linspace(0, 8*np.pi, 500)
ts = np.sin(t) + 0.1*np.random.randn(500)

X_seq, y_seq = make_sequences(ts, seq_len=30)
split = int(0.8 * len(X_seq))
X_tr = torch.tensor(X_seq[:split], dtype=torch.float32).unsqueeze(-1)  # (N, seq, 1)
X_te = torch.tensor(X_seq[split:], dtype=torch.float32).unsqueeze(-1)
y_tr = torch.tensor(y_seq[:split], dtype=torch.float32).unsqueeze(-1)
y_te = torch.tensor(y_seq[split:], dtype=torch.float32).unsqueeze(-1)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep

model = LSTMForecaster()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

for epoch in range(200):
    model.train()
    loss = crit(model(X_tr), y_tr)
    opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    preds = model(X_te).numpy().ravel()
    rmse  = np.sqrt(((preds - y_seq[split:])**2).mean())
print(f"LSTM RMSE: {rmse:.6f}")"""
  },
],

# ══════════════════════════════════════════════════════
"dimensionality_methods": [
  {
    "title": "PCA + t-SNE + UMAP Pipeline Comparison",
    "code": """from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

print(f"{'Method':20s} | {'Dims':>5} | {'Time(s)':>8} | {'KNN-5 CV Acc':>14}")
print("-" * 60)

# PCA
for n in [2, 10, 30]:
    t0 = time.time()
    Xr = PCA(n_components=n).fit_transform(X)
    elapsed = time.time() - t0
    acc = cross_val_score(KNeighborsClassifier(5), Xr, y, cv=5).mean()
    print(f"{'PCA':20s} | {n:>5} | {elapsed:>8.3f} | {acc:>14.4f}")

# t-SNE (visualization only — not suitable as features)
t0 = time.time()
Xr_tsne = TSNE(n_components=2, perplexity=30, max_iter=500, init='pca',
                random_state=42).fit_transform(X)
elapsed = time.time() - t0
acc_tsne = cross_val_score(KNeighborsClassifier(5), Xr_tsne, y, cv=5).mean()
print(f"{'t-SNE':20s} | {'2':>5} | {elapsed:>8.3f} | {acc_tsne:>14.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"feature_engineering2": [
  {
    "title": "Production ML Pipeline with MLflow Tracking",
    "code": """# pip install mlflow
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("ml-pipeline-demo")

configs = [
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
]

for cfg in configs:
    with mlflow.start_run():
        mlflow.log_params(cfg)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingClassifier(**cfg, random_state=42))
        ])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std",  cv_scores.std())
        mlflow.log_metric("test_auc",    test_auc)
        mlflow.sklearn.log_model(pipe, "model")

        print(f"Params: {cfg}")
        print(f"  CV AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test AUC: {test_auc:.4f}")"""
  },
  {
    "title": "Model Monitoring — Data Drift Detection",
    "code": """import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import StandardScaler

# Simulate training distribution (reference)
np.random.seed(42)
train_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 5))

# Simulate production data (with drift on feature 0 and 2)
prod_data_t1 = np.random.normal(loc=0.0, scale=1.0, size=(500, 5))  # no drift
prod_data_t2 = np.random.normal(loc=0.0, scale=1.0, size=(500, 5))  # drift
prod_data_t2[:, 0] += 1.5   # mean shift on feature 0
prod_data_t2[:, 2] *= 2.0   # variance shift on feature 2

def detect_drift(reference, production, alpha=0.05):
    results = {}
    for feat_idx in range(reference.shape[1]):
        stat, p = ks_2samp(reference[:, feat_idx], production[:, feat_idx])
        results[f"feature_{feat_idx}"] = {"ks_stat": stat, "p_value": p, "drift": p < alpha}
    return results

print("Time 1 — No drift expected:")
r1 = detect_drift(train_data, prod_data_t1)
for f, v in r1.items(): print(f"  {f}: KS={v['ks_stat']:.4f} p={v['p_value']:.4f} drift={v['drift']}")

print("\\nTime 2 — Drift on feature 0 and 2:")
r2 = detect_drift(train_data, prod_data_t2)
for f, v in r2.items(): print(f"  {f}: KS={v['ks_stat']:.4f} p={v['p_value']:.4f} drift={v['drift']}")"""
  },
],

# ══════════════════════════════════════════════════════
"eval_classification": [
  {
    "title": "Full Classification Metrics Suite",
    "code": """import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    classification_report, confusion_matrix
)

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                            weights=[0.8, 0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
y_prob  = model.predict_proba(X_test)[:, 1]

print("=== Classification Metrics ===")
print(f"Accuracy      : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision     : {precision_score(y_test, y_pred):.4f}")
print(f"Recall        : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score      : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC       : {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC        : {average_precision_score(y_test, y_prob):.4f}")
print(f"Log-Loss      : {log_loss(y_test, y_prob):.4f}")
print(f"MCC           : {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen Kappa   : {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"Brier Score   : {brier_score_loss(y_test, y_prob):.4f}")
print("\\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))"""
  },
  {
    "title": "ROC & Precision-Recall Curves with Threshold Tuning",
    "code": """import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=3000, n_features=15, weights=[0.85, 0.15], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LogisticRegression(C=1.0, max_iter=500)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc:.4f}")

# PR curve
prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)
auc_pr = average_precision_score(y_test, y_prob)
print(f"AUC-PR : {auc_pr:.4f}")

# Find optimal F1 threshold
f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
best_idx = np.argmax(f1_scores)
best_thresh = pr_thresh[best_idx]
y_pred_opt = (y_prob >= best_thresh).astype(int)
print(f"Optimal threshold: {best_thresh:.3f}")
print(f"F1 at default 0.5: {f1_score(y_test, (y_prob >= 0.5).astype(int)):.4f}")
print(f"F1 at optimal    : {f1_score(y_test, y_pred_opt):.4f}")"""
  },
  {
    "title": "Calibration Analysis & Probability Calibration",
    "code": """import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Uncalibrated model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_prob_raw = rf.predict_proba(X_test)[:, 1]

# Calibrated with Platt scaling (sigmoid)
rf_platt = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42),
                                   method='sigmoid', cv=5)
rf_platt.fit(X_train, y_train)
y_prob_platt = rf_platt.predict_proba(X_test)[:, 1]

# Calibrated with isotonic regression
rf_iso = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42),
                                 method='isotonic', cv=5)
rf_iso.fit(X_train, y_train)
y_prob_iso = rf_iso.predict_proba(X_test)[:, 1]

print(f"Brier Score (uncalibrated) : {brier_score_loss(y_test, y_prob_raw):.4f}")
print(f"Brier Score (Platt)        : {brier_score_loss(y_test, y_prob_platt):.4f}")
print(f"Brier Score (Isotonic)     : {brier_score_loss(y_test, y_prob_iso):.4f}")

# Expected Calibration Error (ECE) computation
def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece_val += mask.sum() * abs(acc - conf)
    return ece_val / len(y_true)

print(f"ECE (uncalibrated) : {ece(y_test, y_prob_raw):.4f}")
print(f"ECE (Platt)        : {ece(y_test, y_prob_platt):.4f}")
print(f"ECE (Isotonic)     : {ece(y_test, y_prob_iso):.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"eval_regression": [
  {
    "title": "Complete Regression Metrics Suite",
    "code": """import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats

X, y = make_regression(n_samples=1000, n_features=15, n_informative=8,
                        noise=30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

n, p = len(y_test), X_test.shape[1]
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# SMAPE
smape = 200 * np.mean(np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred) + 1e-9))

print(f"MAE        : {mae:.4f}")
print(f"MSE        : {mse:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R²         : {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
print(f"MAPE       : {mape:.2f}%")
print(f"SMAPE      : {smape:.2f}%")

# Residual normality test
residuals = y_test - y_pred
stat, p_val = stats.shapiro(residuals[:50])
print(f"\\nShapiro-Wilk normality test: W={stat:.4f}, p={p_val:.4f}")
print(f"Residuals appear {'normal' if p_val > 0.05 else 'non-normal'} at α=0.05")"""
  },
  {
    "title": "Huber Loss & Quantile Regression for Robust Estimation",
    "code": """import numpy as np
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
n = 500
X = np.random.randn(n, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(n) * 2

# Add 10% outliers
outlier_idx = np.random.choice(n, size=50, replace=False)
y[outlier_idx] += np.random.choice([-30, 30], size=50)

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)

models = {
    "OLS (sensitive to outliers)": LinearRegression(),
    "Huber (robust)"             : HuberRegressor(epsilon=1.35, max_iter=500),
    "Quantile 0.10"              : QuantileRegressor(quantile=0.10, solver='highs'),
    "Quantile 0.50 (median)"     : QuantileRegressor(quantile=0.50, solver='highs'),
    "Quantile 0.90"              : QuantileRegressor(quantile=0.90, solver='highs'),
}

for name, m in models.items():
    m.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, m.predict(X_te))
    print(f"{name:35s} MAE={mae:.3f}")"""
  },
  {
    "title": "AIC/BIC Model Selection & Residual Diagnostics",
    "code": """import numpy as np
import statsmodels.api as sm
from scipy import stats

np.random.seed(42)
n = 300
X_raw = np.random.randn(n, 5)
y = 3*X_raw[:, 0] + 2*X_raw[:, 1] + np.random.randn(n) * 2

results = {}
for k in [1, 2, 3, 5]:  # number of features
    X_k = sm.add_constant(X_raw[:, :k])
    ols = sm.OLS(y, X_k).fit()
    results[k] = {"aic": ols.aic, "bic": ols.bic, "r2": ols.rsquared_adj,
                  "resid": ols.resid}
    print(f"k={k}: AIC={ols.aic:.2f}  BIC={ols.bic:.2f}  Adj-R²={ols.rsquared_adj:.4f}")

print("\\nBest model by AIC:", min(results, key=lambda k: results[k]["aic"]), "features")
print("Best model by BIC:", min(results, key=lambda k: results[k]["bic"]), "features")

# Residual diagnostics for best model
best_resid = results[2]["resid"]
stat_sw, p_sw = stats.shapiro(best_resid[:50])
print(f"\\nResidual Shapiro-Wilk: p={p_sw:.4f}")
stat_dw = sm.stats.durbin_watson(best_resid)
print(f"Durbin-Watson (autocorr): {stat_dw:.3f} (should be ~2)")"""
  },
],

# ══════════════════════════════════════════════════════
"cluster_evaluation": [
  {
    "title": "Internal Clustering Metrics — Silhouette, Davies-Bouldin, Calinski-Harabasz",
    "code": """import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

X, y_true = make_blobs(n_samples=500, n_features=2, centers=4,
                        cluster_std=0.8, random_state=42)
X = StandardScaler().fit_transform(X)

print("=== Elbow Method + Internal Metrics ===")
print(f"{'K':>3} {'Inertia':>10} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>20}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    sil   = silhouette_score(X, labels)
    db    = davies_bouldin_score(X, labels)
    ch    = calinski_harabasz_score(X, labels)
    print(f"{k:>3} {inertia:>10.1f} {sil:>12.4f} {db:>16.4f} {ch:>20.1f}")

print("\\nBest K by Silhouette: higher is better")
print("Best K by Davies-Bouldin: lower is better")
print("Best K by Calinski-Harabasz: higher is better")"""
  },
  {
    "title": "External Metrics — ARI, NMI, V-Measure",
    "code": """import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    v_measure_score, fowlkes_mallows_score, homogeneity_score,
    completeness_score
)

X, y_true = make_blobs(n_samples=600, centers=5, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)

clusterers = {
    "KMeans (k=5)"       : KMeans(n_clusters=5, n_init=10, random_state=42),
    "KMeans (k=3)"       : KMeans(n_clusters=3, n_init=10, random_state=42),
    "Agglomerative (k=5)": AgglomerativeClustering(n_clusters=5),
}

print(f"{'Method':25s} {'ARI':>8} {'NMI':>8} {'V-meas':>8} {'FM':>8}")
for name, clf in clusterers.items():
    labels = clf.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    vm  = v_measure_score(y_true, labels)
    fm  = fowlkes_mallows_score(y_true, labels)
    print(f"{name:25s} {ari:8.4f} {nmi:8.4f} {vm:8.4f} {fm:8.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"cnn": [
  {
    "title": "CNN for Image Classification (PyTorch)",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Simple CNN architecture
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Global Average Pooling
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Count parameters
model = ConvNet(num_classes=10)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Demo forward pass
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
x_demo = torch.randn(4, 1, 28, 28).to(device)
out = model(x_demo)
print(f"Input shape : {x_demo.shape}")
print(f"Output shape: {out.shape}")"""
  },
  {
    "title": "ResNet-Style Skip Connections from Scratch",
    "code": """import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    '''Basic ResNet residual block with skip connection.'''
    def __init__(self, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

        # Shortcut if dimensions change
        self.shortcut = nn.Identity() if stride == 1 else nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)   # skip connection adds identity
        return out

class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = ResidualBlock(64)
        self.layer2 = ResidualBlock(64)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = MiniResNet()
x = torch.randn(2, 3, 64, 64)
print("Output shape:", model(x).shape)
print("Parameters: ", sum(p.numel() for p in model.parameters()))"""
  },
  {
    "title": "Transfer Learning with Pretrained CNN (torchvision)",
    "code": """import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# Load pretrained ResNet-18
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Strategy 1: Feature extraction — freeze everything except head
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer for new task (e.g., 5-class dog breed classifier)
num_classes = 5
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

# Only head parameters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Strategy 2: Fine-tune with discriminative learning rates
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Different LRs for different layer groups
param_groups = [
    {"params": model.layer1.parameters(), "lr": 1e-5},
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
print("Optimizer LR groups:", [g['lr'] for g in optimizer.param_groups])
print("All params trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))"""
  },
],

# ══════════════════════════════════════════════════════
"rnn_lstm": [
  {
    "title": "LSTM Time Series Forecasting (PyTorch)",
    "code": """import torch
import torch.nn as nn
import numpy as np

# Generate synthetic sine wave time series
np.random.seed(42)
t = np.linspace(0, 100, 2000)
y = np.sin(0.5 * t) + 0.3 * np.sin(1.5 * t) + 0.1 * np.random.randn(2000)

def make_sequences(data, seq_len=50):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

seq_len = 50
X, Y = make_sequences(y, seq_len)
X = torch.tensor(X).unsqueeze(-1)  # (N, T, 1)
Y = torch.tensor(Y).unsqueeze(-1)

train_size = int(0.8 * len(X))
X_tr, X_te = X[:train_size], X[train_size:]
Y_tr, Y_te = Y[:train_size], Y[train_size:]

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # last timestep

model = LSTMForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

for epoch in range(20):
    model.train()
    pred = model(X_tr)
    loss = loss_fn(pred, Y_tr)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_te), Y_te)
        print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")"""
  },
  {
    "title": "Bidirectional GRU for Text Classification",
    "code": """import torch
import torch.nn as nn
import numpy as np

# Simulated sentiment classification setup
torch.manual_seed(42)
VOCAB_SIZE = 5000
EMBED_DIM  = 64
HIDDEN     = 128
N_CLASSES  = 2

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, n_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)   # *2 for bidirectional

    def forward(self, x, lengths=None):
        emb = self.dropout(self.embedding(x))
        # Pack padded sequence for efficiency
        if lengths is not None:
            emb = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(),
                                                      batch_first=True, enforce_sorted=False)
        out, h = self.gru(emb)
        # Concatenate final forward and backward hidden states
        h_last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.dropout(h_last))

model = BiGRUClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN, N_CLASSES)

# Demo batch
batch_size, seq_len = 8, 30
x_demo   = torch.randint(1, VOCAB_SIZE, (batch_size, seq_len))
lengths  = torch.randint(10, seq_len, (batch_size,))
output   = model(x_demo)
print(f"Input shape : {x_demo.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")"""
  },
],

# ══════════════════════════════════════════════════════
"gan": [
  {
    "title": "DCGAN on MNIST (PyTorch)",
    "code": """import torch
import torch.nn as nn

LATENT_DIM = 100

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim -> 7x7 feature map
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),   # 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)

G = Generator(); D = Discriminator()
print(f"Generator params    : {sum(p.numel() for p in G.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

# Training step demo
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

batch = 16
real  = torch.randn(batch, 1, 28, 28)   # simulated real images
z     = torch.randn(batch, LATENT_DIM)

# D step
optD.zero_grad()
d_real = criterion(D(real).squeeze(), torch.ones(batch))
d_fake = criterion(D(G(z).detach()).squeeze(), torch.zeros(batch))
(d_real + d_fake).backward(); optD.step()

# G step
optG.zero_grad()
g_loss = criterion(D(G(z)).squeeze(), torch.ones(batch))
g_loss.backward(); optG.step()
print(f"D loss: {(d_real + d_fake).item():.4f}  G loss: {g_loss.item():.4f}")"""
  },
  {
    "title": "Wasserstein GAN with Gradient Penalty (WGAN-GP)",
    "code": """import torch
import torch.nn as nn

def gradient_penalty(D, real, fake, device):
    '''Compute gradient penalty for WGAN-GP.'''
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp     = D(interpolated)
    grad = torch.autograd.grad(d_interp, interpolated,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

class Critic(nn.Module):
    '''WGAN critic (no sigmoid — outputs unbounded real value).'''
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64,  4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128*7*7, 1),  # no sigmoid
        )
    def forward(self, x): return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
critic = Critic().to(device)
LAMBDA_GP = 10   # gradient penalty coefficient
N_CRITIC   = 5   # critic updates per generator update

print("WGAN-GP training loop:")
print("  - Critic trained N_CRITIC=5 times per generator step")
print("  - Loss = E[D(fake)] - E[D(real)] + lambda * GP")
print("  - No sigmoid on critic output (Earth Mover distance approximation)")
print("  - Gradient penalty enforces 1-Lipschitz constraint")
print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")"""
  },
],

# ══════════════════════════════════════════════════════
"vae": [
  {
    "title": "Variational Autoencoder (PyTorch)",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1  = nn.Linear(input_dim, hidden)
        self.fc_mu    = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        # Decoder
        self.fc3  = nn.Linear(latent_dim, hidden)
        self.fc4  = nn.Linear(hidden, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        '''z = mu + eps * std, eps ~ N(0,I) — differentiable sampling.'''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.fc4(F.relu(self.fc3(z))))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    '''ELBO = Reconstruction loss + beta * KL divergence.'''
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL(N(mu, sigma^2) || N(0,1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl

vae = VAE()
x_demo = torch.randn(32, 1, 28, 28)   # simulated MNIST batch
recon, mu, logvar = vae(x_demo)
loss = vae_loss(recon, x_demo.view(-1, 784), mu, logvar, beta=1.0)
print(f"Input shape     : {x_demo.shape}")
print(f"Latent mu shape : {mu.shape}")
print(f"Recon shape     : {recon.shape}")
print(f"VAE loss        : {loss.item():.2f}")
print(f"Parameters      : {sum(p.numel() for p in vae.parameters()):,}")"""
  },
  {
    "title": "VAE Latent Space Interpolation & Anomaly Detection",
    "code": """import torch
import torch.nn as nn
import numpy as np

# Simplified VAE encoder/decoder for demo
torch.manual_seed(42)
LATENT = 2  # 2D latent for visualization

class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
        self.fc_mu  = nn.Linear(32, LATENT)
        self.fc_lv  = nn.Linear(32, LATENT)
        self.dec = nn.Sequential(nn.Linear(LATENT, 32), nn.ReLU(), nn.Linear(32, 10))

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z): return self.dec(z)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
        return self.decode(z), mu, lv

model = ToyVAE()
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)

# Latent space interpolation
with torch.no_grad():
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)
    print("Latent space interpolation (z1 -> z2):")
    for alpha in np.linspace(0, 1, 5):
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        decoded  = model.decode(z_interp)
        print(f"  alpha={alpha:.2f} | decoded norm = {decoded.norm().item():.4f}")

# Anomaly detection: high reconstruction error = anomaly
normal_samples = torch.randn(50, 10) * 0.5
anom_samples   = torch.randn(10, 10) * 5.0   # out-of-distribution

with torch.no_grad():
    for name, samples in [("Normal", normal_samples), ("Anomaly", anom_samples)]:
        recon, mu, lv = model(samples)
        recon_err = ((recon - samples)**2).mean(dim=1)
        print(f"{name} avg reconstruction error: {recon_err.mean():.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"nlp_fundamentals": [
  {
    "title": "TF-IDF Text Classification Pipeline",
    "code": """from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Load text data
categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics', 'talk.politics.guns']
train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers','quotes'))
test  = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers','footers','quotes'))

# TF-IDF + Logistic Regression pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,    # log(TF) instead of raw TF
        min_df=3,
        max_df=0.95,
    )),
    ('clf', LogisticRegression(C=5.0, max_iter=1000, solver='lbfgs', multi_class='multinomial')),
])

pipe.fit(train.data, train.target)
y_pred = pipe.predict(test.data)

print(f"Test Accuracy: {accuracy_score(test.target, y_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(test.target, y_pred, target_names=categories))

# Top features per class
vec = pipe.named_steps['tfidf']
clf = pipe.named_steps['clf']
for i, cat in enumerate(categories):
    top_idx = np.argsort(clf.coef_[i])[-8:]
    top_words = [vec.get_feature_names_out()[j] for j in top_idx]
    print(f"Top words [{cat}]: {top_words}")"""
  },
  {
    "title": "Word Embeddings with Word2Vec (gensim)",
    "code": """from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

# Sample corpus (in practice: Wikipedia, news, etc.)
corpus_raw = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with many layers",
    "natural language processing handles text and speech",
    "computer vision processes images and videos",
    "reinforcement learning trains agents through rewards",
    "transformers changed natural language processing forever",
    "attention mechanism is the core of transformer models",
    "convolutional networks excel at image classification tasks",
    "gradient descent optimizes neural network parameters",
    "regularization prevents overfitting in machine learning",
]

sentences = [simple_preprocess(s) for s in corpus_raw]

# Train Word2Vec
model = Word2Vec(
    sentences,
    vector_size=50,   # embedding dimension
    window=3,         # context window
    min_count=1,      # minimum word frequency
    workers=2,
    epochs=100,
    sg=1,             # skip-gram (1) vs CBOW (0)
)

print("Vocabulary size:", len(model.wv))
print("Embedding dim  :", model.wv.vector_size)

# Semantic similarity
words_to_test = [("machine", "learning"), ("neural", "networks"), ("image", "text")]
for w1, w2 in words_to_test:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        print(f"Similarity({w1}, {w2}) = {sim:.4f}")

# Most similar words
if "learning" in model.wv:
    print("\\nMost similar to 'learning':", model.wv.most_similar("learning", topn=5))"""
  },
  {
    "title": "BERT Fine-Tuning for Text Classification",
    "code": """from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import torch

# Minimal example — SST-2 style binary sentiment
texts  = ["I love this movie!", "Terrible film, waste of time.",
          "Absolutely brilliant performance.", "Boring and predictable plot."] * 10
labels = [1, 0, 1, 0] * 10

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

ds = Dataset.from_dict({"text": texts, "label": labels})
ds = ds.map(tokenize, batched=True)
ds = ds.train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    report_to="none",
)

trainer = Trainer(model=model, args=args,
                  train_dataset=ds["train"], eval_dataset=ds["test"])
print("Model params:", sum(p.numel() for p in model.parameters()))
print("Fine-tuning setup complete — call trainer.train() to start")"""
  },
],

# ══════════════════════════════════════════════════════
"transfer_learning": [
  {
    "title": "Feature Extraction vs Full Fine-Tuning Comparison",
    "code": """import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split

def build_model(strategy: str, num_classes: int = 5):
    '''strategy: 'frozen' | 'finetune' | 'scratch''''
    if strategy == 'scratch':
        model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    if strategy == 'frozen':
        for p in model.parameters():
            p.requires_grad = False

    # Replace head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return model, trainable, total

for strategy in ['frozen', 'finetune', 'scratch']:
    model, trainable, total = build_model(strategy)
    print(f"[{strategy:10s}] trainable={trainable:>7,} / {total:>8,} ({100*trainable/total:5.1f}%)")

# LoRA-style low-rank adaptation
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01, requires_grad=False)
        self.bias   = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        # Low-rank adaptation: W_delta = B @ A
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.rank = rank

    def forward(self, x):
        W = self.weight + self.B @ self.A  # original + low-rank delta
        return nn.functional.linear(x, W, self.bias)

lora = LoRALinear(512, 256, rank=4)
full = nn.Linear(512, 256)
print(f"\\nLoRA trainable: {sum(p.numel() for p in lora.parameters() if p.requires_grad):,}")
print(f"Full trainable: {sum(p.numel() for p in full.parameters() if p.requires_grad):,}")"""
  },
  {
    "title": "PEFT LoRA Fine-Tuning with Hugging Face",
    "code": """from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load a small base model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Base model params: {sum(p.numel() for p in model.parameters()):,}")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # rank
    lora_alpha=32,                # scaling factor
    lora_dropout=0.1,
    target_modules=["c_attn"],    # which layers to apply LoRA
    bias="none",
)

# Wrap model with LoRA adapters
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Quick forward pass
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(["Transfer learning is", "Machine learning helps"],
                    return_tensors="pt", padding=True)
with torch.no_grad():
    out = peft_model(**inputs, labels=inputs["input_ids"])
print(f"LoRA fine-tune loss: {out.loss.item():.4f}")
print("Only LoRA adapter parameters are updated during training")"""
  },
],

# ══════════════════════════════════════════════════════
"reinforcement_learning": [
  {
    "title": "Q-Learning on GridWorld (from scratch)",
    "code": """import numpy as np

# Simple GridWorld: 5x5 grid, goal at (4,4), start at (0,0)
# Actions: 0=up, 1=down, 2=left, 3=right
GRID_SIZE = 5
GOAL = (4, 4)

def step(state, action):
    r, c = state
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    nr = max(0, min(GRID_SIZE-1, r + moves[action][0]))
    nc = max(0, min(GRID_SIZE-1, c + moves[action][1]))
    next_state = (nr, nc)
    reward = 10.0 if next_state == GOAL else -0.1
    done   = next_state == GOAL
    return next_state, reward, done

# Q-table: (5, 5, 4)
Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))
alpha   = 0.1   # learning rate
gamma   = 0.9   # discount factor
epsilon = 1.0   # exploration rate

rewards_per_ep = []
for episode in range(2000):
    state = (0, 0)
    total_r = 0
    for _ in range(100):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)
        # Q-learning update
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state    = next_state
        total_r += reward
        if done: break

    epsilon = max(0.05, epsilon * 0.999)
    rewards_per_ep.append(total_r)

print(f"Last 100 episodes avg reward: {np.mean(rewards_per_ep[-100:]):.2f}")
print(f"Final epsilon: {epsilon:.4f}")
print("Greedy path from (0,0):")
state = (0, 0); path = [state]
for _ in range(20):
    action = np.argmax(Q[state])
    state, _, done = step(state, action)
    path.append(state)
    if done: break
print(" -> ".join(str(s) for s in path))"""
  },
  {
    "title": "Deep Q-Network (DQN) with Experience Replay",
    "code": """import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)
    def push(self, *transition): self.buf.append(transition)
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return [torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*batch)]
    def __len__(self): return len(self.buf)

# Setup (CartPole-like: 4 state dims, 2 actions)
STATE_DIM, ACTION_DIM = 4, 2
GAMMA, LR, BATCH = 0.99, 1e-3, 64

online_net = DQN(STATE_DIM, ACTION_DIM)
target_net = DQN(STATE_DIM, ACTION_DIM)
target_net.load_state_dict(online_net.state_dict())  # sync target
optimizer  = torch.optim.Adam(online_net.parameters(), lr=LR)
buffer     = ReplayBuffer()

def compute_loss(batch):
    states, actions, rewards, next_states, dones = batch
    q_values    = online_net(states).gather(1, actions.long().unsqueeze(1)).squeeze()
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
    target  = rewards + GAMMA * next_q * (1 - dones)
    return nn.functional.smooth_l1_loss(q_values, target)

print("DQN setup complete.")
print(f"Online net params: {sum(p.numel() for p in online_net.parameters()):,}")
print("Key ideas: experience replay + target network stabilize training")
print("Target network synced every N steps (hard) or with EMA (soft)")"""
  },
],

# ══════════════════════════════════════════════════════
"bandits_active": [
  {
    "title": "Multi-Armed Bandit: UCB vs Thompson Sampling vs Epsilon-Greedy",
    "code": """import numpy as np

np.random.seed(42)
K     = 10    # arms
N     = 5000  # rounds
TRUE_MEANS = np.random.uniform(0, 1, K)  # unknown true means
BEST  = TRUE_MEANS.max()

class BanditAgent:
    def __init__(self, k):
        self.k = k
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

class EpsilonGreedy(BanditAgent):
    def __init__(self, k, eps=0.1): super().__init__(k); self.eps = eps
    def choose(self, t=None):
        return np.random.randint(self.k) if np.random.rand() < self.eps else np.argmax(self.values)

class UCB1(BanditAgent):
    def choose(self, t):
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored): return unexplored[0]
        ucb = self.values + np.sqrt(2 * np.log(t) / self.counts)
        return np.argmax(ucb)

class ThompsonSampling:
    def __init__(self, k): self.alpha = np.ones(k); self.beta_ = np.ones(k)
    def choose(self, t=None): return np.argmax(np.random.beta(self.alpha, self.beta_))
    def update(self, arm, reward):
        r = int(reward > 0.5)  # binarize
        self.alpha[arm] += r
        self.beta_[arm] += 1 - r

agents = {"EpsGreedy":  EpsilonGreedy(K, 0.1),
          "UCB1":        UCB1(K),
          "Thompson":    ThompsonSampling(K)}

for name, agent in agents.items():
    cum_regret = 0
    for t in range(1, N+1):
        arm    = agent.choose(t)
        reward = np.random.binomial(1, TRUE_MEANS[arm])
        agent.update(arm, reward)
        cum_regret += BEST - TRUE_MEANS[arm]
    print(f"{name:12s} | Cumulative Regret: {cum_regret:.1f} | Avg Regret/round: {cum_regret/N:.4f}")"""
  },
  {
    "title": "Active Learning with Uncertainty Sampling",
    "code": """import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initial labeled pool (small)
labeled_idx = np.random.choice(len(X), size=20, replace=False)
unlabeled_idx = np.array(list(set(range(len(X))) - set(labeled_idx)))
test_idx = np.random.choice(len(X), size=200, replace=False)

def uncertainty_sampling(model, X_pool, n_query=10):
    '''Select most uncertain samples (least confidence strategy).'''
    probs = model.predict_proba(X_pool)
    max_probs = probs.max(axis=1)
    return np.argsort(max_probs)[:n_query]

print("Active Learning with Uncertainty Sampling:")
print(f"{'Labels':>8} {'Accuracy':>12}")

for query_round in range(10):
    X_lab = X[labeled_idx]
    y_lab = y[labeled_idx]
    model = LogisticRegression(C=1.0, max_iter=500)
    model.fit(X_lab, y_lab)
    acc = accuracy_score(y[test_idx], model.predict(X[test_idx]))
    print(f"{len(labeled_idx):>8} {acc:>12.4f}")

    # Query 10 most uncertain samples
    X_unlab = X[unlabeled_idx]
    query = uncertainty_sampling(model, X_unlab, n_query=10)
    new_labeled = unlabeled_idx[query]
    labeled_idx = np.concatenate([labeled_idx, new_labeled])
    unlabeled_idx = np.array(list(set(unlabeled_idx) - set(new_labeled)))"""
  },
],

# ══════════════════════════════════════════════════════
"semi_self_supervised": [
  {
    "title": "SimCLR Contrastive Learning (PyTorch)",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRProjectionHead(nn.Module):
    def __init__(self, feature_dim=512, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, proj_dim)
        )
    def forward(self, x): return F.normalize(self.net(x), dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    '''NT-Xent contrastive loss for SimCLR.'''
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # (2N, D)
    sim = torch.mm(z, z.T) / temperature    # (2N, 2N)

    # Mask self-similarity
    mask = torch.eye(2*N, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)])
    loss = F.cross_entropy(sim, labels)
    return loss

# Demo
torch.manual_seed(42)
batch_size = 32
feature_dim = 512

encoder = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, feature_dim)
)
head = SimCLRProjectionHead(feature_dim, 128)

# Two augmented views of the same batch
view1 = torch.randn(batch_size, 784)
view2 = view1 + 0.1 * torch.randn_like(view1)  # simulated augmentation

h1 = encoder(view1);  z1 = head(h1)
h2 = encoder(view2);  z2 = head(h2)
loss = nt_xent_loss(z1, z2, temperature=0.5)
print(f"SimCLR NT-Xent Loss: {loss.item():.4f}")
print(f"z1 shape: {z1.shape}  (batch, proj_dim)")
print(f"Positive pairs are augmented views of same image")
print(f"Negatives are all other images in the batch ({batch_size*2-2} per sample)")"""
  },
  {
    "title": "Semi-supervised with Pseudo-labels & FixMatch",
    "code": """import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X, y = make_classification(n_samples=3000, n_features=20, n_informative=12, random_state=42)
X = StandardScaler().fit_transform(X)

N_LABELED   = 50    # very small labeled set
THRESHOLD   = 0.95  # confidence threshold for pseudo-labels

labeled_mask = np.zeros(len(X), dtype=bool)
labeled_idx  = np.random.choice(len(X), N_LABELED, replace=False)
labeled_mask[labeled_idx] = True
test_idx = np.random.choice(np.where(~labeled_mask)[0], 300, replace=False)

print("=== Pseudo-label Semi-supervised Training ===")
print(f"{'Round':>6} {'Labeled':>9} {'Pseudo':>9} {'Test Acc':>10}")

X_lab = X[labeled_mask]
y_lab = y[labeled_mask]

for round_i in range(8):
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_lab, y_lab)
    acc = accuracy_score(y[test_idx], model.predict(X[test_idx]))

    # Generate pseudo-labels for unlabeled data
    X_unlab_mask = ~labeled_mask
    X_unlab_mask[test_idx] = False
    X_unlab = X[X_unlab_mask]
    probs = model.predict_proba(X_unlab)
    confident = probs.max(axis=1) >= THRESHOLD

    # Add confident pseudo-labels
    pseudo_X = X_unlab[confident]
    pseudo_y = probs[confident].argmax(axis=1)
    X_lab = np.vstack([X_lab, pseudo_X])
    y_lab = np.concatenate([y_lab, pseudo_y])
    print(f"{round_i+1:>6} {N_LABELED+round_i:>9} {confident.sum():>9} {acc:>10.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"pgm": [
  {
    "title": "Hidden Markov Model (Viterbi Decoding)",
    "code": """import numpy as np

# POS tagging HMM example
# States: Noun(0), Verb(1), Adj(2)
# Observations: 'the','dog','runs','fast','big','small' = 0..5
N_STATES = 3
N_OBS    = 6

# HMM parameters
pi = np.array([0.6, 0.3, 0.1])  # initial probabilities

A = np.array([                    # transition matrix
    [0.4, 0.4, 0.2],  # Noun -> {Noun, Verb, Adj}
    [0.5, 0.1, 0.4],  # Verb -> ...
    [0.6, 0.3, 0.1],  # Adj  -> ...
])

B = np.array([                    # emission matrix
    [0.4, 0.4, 0.0, 0.1, 0.0, 0.1],  # Noun emits
    [0.0, 0.1, 0.7, 0.1, 0.0, 0.1],  # Verb emits
    [0.1, 0.0, 0.0, 0.1, 0.5, 0.3],  # Adj  emits
])

def viterbi(obs, pi, A, B):
    T, K = len(obs), len(pi)
    dp    = np.zeros((T, K))
    psi   = np.zeros((T, K), dtype=int)
    dp[0] = np.log(pi + 1e-10) + np.log(B[:, obs[0]] + 1e-10)
    for t in range(1, T):
        for k in range(K):
            scores = dp[t-1] + np.log(A[:, k] + 1e-10) + np.log(B[k, obs[t]] + 1e-10)
            psi[t, k] = np.argmax(scores)
            dp[t,  k] = scores[psi[t, k]]
    # Backtrack
    path  = [np.argmax(dp[-1])]
    for t in range(T-1, 0, -1):
        path.insert(0, psi[t, path[0]])
    return path, dp[-1].max()

# "the dog runs fast"
obs = [0, 1, 2, 3]
path, score = viterbi(obs, pi, A, B)
state_names = ['Noun', 'Verb', 'Adj']
obs_names   = ['the', 'dog', 'runs', 'fast', 'big', 'small']
print("Viterbi POS tagging:")
for o, s in zip(obs, path):
    print(f"  {obs_names[o]:8s} -> {state_names[s]}")
print(f"Log-probability of best path: {score:.4f}")"""
  },
  {
    "title": "Gaussian Mixture Model with EM Algorithm",
    "code": """import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

np.random.seed(42)
X, y_true = make_blobs(n_samples=500, centers=3,
                        cluster_std=[0.5, 1.0, 0.7], random_state=42)

# GMM via sklearn (uses EM under the hood)
for cov_type in ['full', 'tied', 'diag', 'spherical']:
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type,
                          max_iter=200, n_init=5, random_state=42)
    gmm.fit(X)
    print(f"cov_type={cov_type:12s} | BIC={gmm.bic(X):.1f}  AIC={gmm.aic(X):.1f}  "
          f"converged={gmm.converged_}")

# Best model
best = GaussianMixture(n_components=3, covariance_type='full',
                        max_iter=200, random_state=42)
best.fit(X)
print(f"\\nLearned means:\\n{best.means_}")
print(f"Mixing weights: {best.weights_}")

# Bayesian Information Criterion for model selection (n_components)
print("\\nBIC for different K (lower is better):")
for k in range(1, 7):
    g = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(X)
    print(f"  K={k}: BIC={g.bic(X):.1f}")"""
  },
],

# ══════════════════════════════════════════════════════
"object_detection": [
  {
    "title": "IoU, NMS & mAP Implementation",
    "code": """import numpy as np

def box_iou(box1, box2):
    '''Compute IoU between two boxes [x1, y1, x2, y2].'''
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def nms(boxes, scores, iou_thresh=0.5):
    '''Non-Maximum Suppression.'''
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        ious = np.array([box_iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious < iou_thresh]
    return keep

def compute_ap(recalls, precisions):
    '''Compute area under precision-recall curve (11-point interpolation).'''
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precisions[recalls >= thr]
        ap += (prec_at_thr.max() if len(prec_at_thr) else 0.0) / 11
    return ap

# Demo NMS
np.random.seed(42)
boxes  = np.array([[10,10,50,50],[15,15,55,55],[100,100,150,150],[105,105,148,148]])
scores = np.array([0.9, 0.75, 0.85, 0.6])
kept   = nms(boxes, scores, iou_thresh=0.5)
print(f"Boxes before NMS: {len(boxes)}")
print(f"Boxes after  NMS: {len(kept)} -> indices {kept}")

# IoU examples
print("\\nIoU Examples:")
print(f"  Identical boxes : {box_iou([0,0,10,10], [0,0,10,10]):.4f}")
print(f"  50% overlap     : {box_iou([0,0,10,10], [5,0,15,10]):.4f}")
print(f"  No overlap      : {box_iou([0,0,5,5],   [6,0,10,10]):.4f}")"""
  },
  {
    "title": "Focal Loss & Feature Pyramid Network (PyTorch)",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Focal Loss for dense object detection (RetinaNet).'''
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p     = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t   = p * targets + (1 - p) * (1 - targets)
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl    = alpha * (1 - p_t) ** self.gamma * bce
        return fl.mean()

class FPN(nn.Module):
    '''Minimal Feature Pyramid Network: combines multi-scale backbone features.'''
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        self.output  = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list]
        )

    def forward(self, features):
        # features: list of tensors from backbone (small to large resolution)
        laterals = [l(f) for l, f in zip(self.lateral, features)]
        # Top-down pathway
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
        return [o(l) for o, l in zip(self.output, laterals)]

# Test
fpn = FPN([64, 128, 256], out_channels=256)
features = [torch.randn(2, 64, 64, 64),
            torch.randn(2, 128, 32, 32),
            torch.randn(2, 256, 16, 16)]
outs = fpn(features)
print("FPN output shapes:")
for o in outs: print(f"  {tuple(o.shape)}")

fl = FocalLoss(alpha=0.25, gamma=2.0)
logits = torch.randn(100, 80)  # 100 anchors, 80 classes
targets = torch.randint(0, 2, (100, 80)).float()
print(f"\\nFocal Loss: {fl(logits, targets):.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"gnn": [
  {
    "title": "Graph Convolution Network from Scratch (NumPy)",
    "code": """import numpy as np

# Simple GCN layer: H' = sigma(D^{-1/2} A_hat D^{-1/2} H W)
def gcn_layer(A, H, W):
    N = A.shape[0]
    A_hat = A + np.eye(N)          # add self-loops
    D_hat = np.diag(A_hat.sum(1))  # degree matrix
    D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D_hat) + 1e-6))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return np.maximum(0, A_norm @ H @ W)  # ReLU activation

# Toy citation network: 6 nodes, 2 features, 3 classes
np.random.seed(42)
N, F_in, F_out = 6, 4, 3

# Adjacency matrix (undirected)
A = np.array([
    [0,1,1,0,0,0],
    [1,0,1,1,0,0],
    [1,1,0,0,1,0],
    [0,1,0,0,1,1],
    [0,0,1,1,0,1],
    [0,0,0,1,1,0],
], dtype=float)

H0 = np.random.randn(N, F_in)      # node features
W1 = np.random.randn(F_in, 8)  * 0.1
W2 = np.random.randn(8, F_out) * 0.1

H1 = gcn_layer(A, H0, W1)       # first GCN layer
H2 = gcn_layer(A, H1, W2)       # second GCN layer

# Softmax for node classification
logits = H2 - H2.max(1, keepdims=True)
probs  = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
preds  = probs.argmax(1)

print(f"Input  shape: {H0.shape}  (N nodes, F features)")
print(f"Hidden shape: {H1.shape}")
print(f"Output shape: {H2.shape}  (N nodes, C classes)")
print(f"Node predictions: {preds}")
print(f"\\nKey: GCN aggregates neighbor features via normalized adjacency")
print("Each layer expands 1-hop receptive field")"""
  },
  {
    "title": "Graph Neural Network with PyTorch Geometric",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal GCN implementation using sparse message passing (no PyG dependency)
# Demonstrates core GNN message passing framework

class MessagePassingGNN(nn.Module):
    '''Hand-rolled GNN showing message passing, aggregation, update.'''
    def __init__(self, in_dim, hidden, out_dim, n_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])]
        )
        self.n_layers = n_layers

    def forward(self, x, adj):
        '''adj: normalized adjacency (N, N) sparse or dense.'''
        for i, layer in enumerate(self.layers):
            # Aggregate: sum neighbor features
            x_agg = adj @ x
            # Update: linear transform + activation
            x = layer(x_agg)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        return x

# Cora-like setup
torch.manual_seed(42)
N, F_in, N_classes = 50, 16, 7

model = MessagePassingGNN(F_in, 32, N_classes, n_layers=2)
x   = torch.randn(N, F_in)

# Symmetric normalized adj (D^-1/2 A D^-1/2)
A = (torch.rand(N, N) > 0.85).float()
A = (A + A.T).clamp(0, 1)
A_hat = A + torch.eye(N)
D_inv_sqrt = torch.diag(1.0 / A_hat.sum(1).sqrt())
adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

out = model(x, adj_norm)
print(f"Input  shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Avg degree  : {A.sum(1).mean():.1f}")"""
  },
],

# ══════════════════════════════════════════════════════
"loss_functions": [
  {
    "title": "Loss Functions Comparison: Regression",
    "code": """import torch
import torch.nn as nn
import numpy as np

# Generate predictions and targets with outliers
torch.manual_seed(42)
target  = torch.zeros(100)
outlier = torch.zeros(100)
outlier[90:] = 10.0   # 10% outliers
pred    = target + torch.randn(100) * 0.5 + outlier

# Regression losses
mse_loss  = nn.MSELoss()
mae_loss  = nn.L1Loss()
huber     = nn.HuberLoss(delta=1.0)
smooth_l1 = nn.SmoothL1Loss(beta=1.0)

print("=== Regression Loss Comparison ===")
print(f"MSE (L2)     : {mse_loss(pred, target):.4f}  (sensitive to outliers)")
print(f"MAE (L1)     : {mae_loss(pred, target):.4f}  (robust to outliers)")
print(f"Huber (d=1)  : {huber(pred, target):.4f}  (quadratic small, linear large)")
print(f"SmoothL1     : {smooth_l1(pred, target):.4f}")

# Quantile loss
def quantile_loss(pred, target, q=0.9):
    e = target - pred
    return torch.mean(torch.max(q * e, (q - 1) * e))

for q in [0.1, 0.5, 0.9]:
    print(f"Quantile q={q} : {quantile_loss(pred, target, q):.4f}")

# Gradient comparison: how large is gradient for outlier?
pred_r = pred.detach().requires_grad_(True)
mse_loss(pred_r, target).backward()
print(f"\\nMSE gradient for outlier point: {pred_r.grad[95].item():.4f}  (large!)")

pred_r2 = pred.detach().requires_grad_(True)
mae_loss(pred_r2, target).backward()
print(f"MAE gradient for outlier point: {pred_r2.grad[95].item():.4f}  (constant)")"""
  },
  {
    "title": "Loss Functions for Classification & Metric Learning",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Classification Losses ----
batch_size, n_classes = 32, 10
logits  = torch.randn(batch_size, n_classes)
targets = torch.randint(0, n_classes, (batch_size,))

ce_loss = nn.CrossEntropyLoss()
print(f"Cross-Entropy Loss    : {ce_loss(logits, targets):.4f}")

# Label smoothing
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
print(f"Label Smoothing (0.1) : {ce_smooth(logits, targets):.4f}  (usually lower, more calibrated)")

# Focal Loss
def focal_loss(logits, targets, gamma=2.0):
    p   = F.softmax(logits, dim=1)
    pt  = p.gather(1, targets.unsqueeze(1)).squeeze()
    log_pt = torch.log(pt + 1e-8)
    return (-((1 - pt) ** gamma) * log_pt).mean()

print(f"Focal Loss (g=2)      : {focal_loss(logits, targets):.4f}")

# ---- Triplet Loss ----
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()

dim = 64
anchor   = F.normalize(torch.randn(16, dim), dim=1)
positive = anchor + 0.1 * torch.randn_like(anchor)  # similar
negative = F.normalize(torch.randn(16, dim), dim=1)  # random
print(f"\\nTriplet Loss: {triplet_loss(anchor, positive, negative):.4f}")

# ---- KL Divergence ----
p_dist = F.softmax(torch.randn(batch_size, n_classes), dim=1)
q_dist = F.softmax(torch.randn(batch_size, n_classes), dim=1)
kl = F.kl_div(q_dist.log(), p_dist, reduction='batchmean')
print(f"KL Divergence D(P||Q) : {kl:.4f}")"""
  },
  {
    "title": "Dice Loss & IoU Loss for Segmentation",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        '''pred, target: (N, H, W) with pred being probabilities.'''
        pred   = torch.sigmoid(pred).view(-1)
        target = target.view(-1).float()
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class CombinedSegLoss(nn.Module):
    '''BCE + Dice — common for medical image segmentation.'''
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce  = self.bce(pred, target.float())
        dice = self.dice(pred, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice, bce, dice

# Test with binary segmentation mask
torch.manual_seed(42)
N, H, W = 4, 64, 64
pred   = torch.randn(N, H, W)                        # logits
target = torch.randint(0, 2, (N, H, W))              # binary mask

dice_loss = DiceLoss()
comb_loss  = CombinedSegLoss(bce_weight=0.5)

total, bce, dice = comb_loss(pred, target)
print(f"BCE Loss      : {bce:.4f}")
print(f"Dice Loss     : {dice:.4f}")
print(f"Combined Loss : {total:.4f}")

# Why Dice? For imbalanced segmentation (e.g., 5% lesion pixels)
imbalanced_target = torch.zeros(N, H, W)
imbalanced_target[:, :4, :4] = 1  # tiny foreground
print(f"\\nImbalanced (tiny lesion):")
print(f"  BCE loss  : {nn.BCEWithLogitsLoss()(pred, imbalanced_target.float()):.4f}")
print(f"  Dice loss : {DiceLoss()(pred, imbalanced_target):.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"data_preprocessing": [
  {
    "title": "Complete Preprocessing Pipeline with sklearn",
    "code": """import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder,
    OrdinalEncoder, RobustScaler, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age'       : np.random.normal(40, 12, n),
    'income'    : np.random.lognormal(10, 0.8, n),  # skewed
    'score'     : np.random.uniform(0, 100, n),
    'education' : np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n),
    'city'      : np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'SF'], n),
    'gender'    : np.random.choice(['M', 'F'], n),
    'target'    : np.random.randint(0, 2, n)
})

# Introduce missing values
for col in ['age', 'income', 'education']:
    df.loc[np.random.choice(n, 40, replace=False), col] = np.nan

numeric_features  = ['age', 'income', 'score']
ordinal_features  = ['education']
nominal_features  = ['city', 'gender']

numeric_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('power',   PowerTransformer(method='yeo-johnson')),  # handle skew
    ('scaler',  StandardScaler()),
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[['HS','Bachelor','Master','PhD']])),
])

nominal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, drop='first')),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('ord', ordinal_transformer, ordinal_features),
    ('cat', nominal_transformer, nominal_features),
])

X = df.drop('target', axis=1)
y = df['target']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit ONLY on training data — critical for no leakage
X_tr_t = preprocessor.fit_transform(X_tr)
X_te_t = preprocessor.transform(X_te)

print(f"Input  shape: {X.shape}")
print(f"Output shape: {X_tr_t.shape}")
print(f"Features: {X_tr_t.shape[1]} (after OHE encoding)")"""
  },
  {
    "title": "Outlier Detection & Class Imbalance Handling",
    "code": """import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter

np.random.seed(42)

# Outlier detection comparison
X = np.random.randn(300, 2)
X_out = np.concatenate([X, np.random.uniform(-6, 6, (20, 2))])  # add outliers

# Method 1: Z-score
z_scores = np.abs((X_out - X_out.mean(0)) / X_out.std(0)).max(1)
outlier_z = z_scores > 3
print(f"Z-score method  : {outlier_z.sum()} outliers detected")

# Method 2: IQR
Q1, Q3 = np.percentile(X_out, 25, axis=0), np.percentile(X_out, 75, axis=0)
IQR = Q3 - Q1
outlier_iqr = ((X_out < Q1 - 1.5*IQR) | (X_out > Q3 + 1.5*IQR)).any(1)
print(f"IQR method      : {outlier_iqr.sum()} outliers detected")

# Method 3: Isolation Forest
iso = IsolationForest(contamination=0.06, random_state=42)
outlier_iso = iso.fit_predict(X_out) == -1
print(f"Isolation Forest: {outlier_iso.sum()} outliers detected")

# Class imbalance handling
print("\\n=== Class Imbalance Methods ===")
X_imb, y_imb = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
print(f"Original  : {Counter(y_imb)}")

for name, sampler in [
    ("SMOTE",      SMOTE(random_state=42)),
    ("ADASYN",     ADASYN(random_state=42)),
    ("SMOTETomek", SMOTETomek(random_state=42)),
]:
    X_r, y_r = sampler.fit_resample(X_imb, y_imb)
    print(f"{name:12s}: {Counter(y_r)}")"""
  },
],

# ══════════════════════════════════════════════════════
"statistical_tests": [
  {
    "title": "Hypothesis Testing: t-tests, Mann-Whitney, Chi-squared",
    "code": """import numpy as np
from scipy import stats

np.random.seed(42)

# ---- 1. One-sample t-test ----
data = np.random.normal(50.5, 10, 100)  # population mean = 50.5
t_stat, p_val = stats.ttest_1samp(data, popmean=50)
print(f"One-sample t-test (H0: mu=50): t={t_stat:.3f}, p={p_val:.4f}")

# ---- 2. Two-sample t-test (Welch's) ----
group_a = np.random.normal(100, 15, 80)
group_b = np.random.normal(105, 20, 90)
t2, p2 = stats.ttest_ind(group_a, group_b, equal_var=False)   # Welch's
print(f"Welch's t-test (H0: mu_A=mu_B): t={t2:.3f}, p={p2:.4f}")

# Cohen's d effect size
def cohens_d(a, b):
    pool_sd = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pool_sd
print(f"Cohen's d: {cohens_d(group_a, group_b):.4f}  (small=0.2, medium=0.5, large=0.8)")

# ---- 3. Mann-Whitney U (non-parametric) ----
skewed_a = np.random.exponential(2, 60)
skewed_b = np.random.exponential(3, 60)
u_stat, p_mw = stats.mannwhitneyu(skewed_a, skewed_b, alternative='two-sided')
print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")

# ---- 4. Chi-squared test of independence ----
contingency = np.array([[50, 30, 20],   # Feature A: low, med, high by class 0
                         [20, 40, 40]])  # Feature A: low, med, high by class 1
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-squared test: chi2={chi2:.3f}, p={p_chi:.4f}, dof={dof}")

# ---- 5. Bonferroni correction for multiple tests ----
p_values = [0.01, 0.03, 0.04, 0.05, 0.08, 0.12]
alpha = 0.05
alpha_bonf = alpha / len(p_values)
print(f"\\nBonferroni corrected alpha: {alpha_bonf:.4f}")
print(f"Significant after Bonferroni: {[p < alpha_bonf for p in p_values]}")"""
  },
  {
    "title": "A/B Testing & Power Analysis",
    "code": """import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(42)

# ---- A/B Test: Conversion Rate ----
n_a, n_b = 2000, 2000
conv_a   = np.random.binomial(1, 0.10, n_a)  # control: 10% conversion
conv_b   = np.random.binomial(1, 0.12, n_b)  # treatment: 12% conversion

print(f"Control rate    : {conv_a.mean():.4f}")
print(f"Treatment rate  : {conv_b.mean():.4f}")
print(f"Relative lift   : {(conv_b.mean() - conv_a.mean()) / conv_a.mean() * 100:.1f}%")

# Two-proportion z-test
count  = np.array([conv_a.sum(), conv_b.sum()])
nobs   = np.array([n_a, n_b])
z_stat, p_val = proportions_ztest(count, nobs, alternative='smaller')
print(f"Z-test: z={z_stat:.3f}, p={p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not significant'})")

# ---- Power Analysis: minimum sample size ----
print("\\n=== Power Analysis ===")
analysis = TTestIndPower()

for effect_size in [0.1, 0.2, 0.5, 0.8]:
    n = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
    print(f"Effect size={effect_size:.1f} | N per group = {n:.0f}")

# ---- ANOVA: compare 3 groups ----
print("\\n=== One-way ANOVA ===")
g1 = np.random.normal(20, 5, 30)
g2 = np.random.normal(22, 5, 30)
g3 = np.random.normal(25, 5, 30)
f_stat, p_anova = stats.f_oneway(g1, g2, g3)
print(f"F-stat: {f_stat:.3f}, p={p_anova:.4f}")
if p_anova < 0.05:
    print("Significant: at least one group mean differs")
    # Post-hoc Tukey HSD
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import pandas as pd
    data   = np.concatenate([g1, g2, g3])
    labels = ['G1']*30 + ['G2']*30 + ['G3']*30
    tukey  = pairwise_tukeyhsd(data, labels, alpha=0.05)
    print(tukey)"""
  },
],

# ══════════════════════════════════════════════════════
"model_compression": [
  {
    "title": "Knowledge Distillation: Teacher-Student Training",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
X, y = make_classification(n_samples=2000, n_features=20, n_informative=12, random_state=42)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    '''KL(student_soft || teacher_soft) + (1-alpha) * CE(student, true_labels).'''
    soft_target = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    kl  = F.kl_div(soft_target, teacher_soft, reduction='batchmean') * (T ** 2)
    ce  = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce

teacher = TeacherNet(); student = StudentNet()
t_params = sum(p.numel() for p in teacher.parameters())
s_params = sum(p.numel() for p in student.parameters())
print(f"Teacher params : {t_params:,}")
print(f"Student params : {s_params:,} ({100*s_params/t_params:.1f}% of teacher)")
print("\\nDistillation: student learns from teacher's soft probability distributions")
print("Temperature T=4 softens teacher output, revealing inter-class relationships")"""
  },
  {
    "title": "Magnitude-based Weight Pruning & Quantization",
    "code": """import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

model = SimpleNet()
total = sum(p.numel() for p in model.parameters())
print(f"Original params: {total:,}")

# Magnitude-based unstructured pruning
prune.l1_unstructured(model.fc1, name='weight', amount=0.5)  # prune 50%
prune.l1_unstructured(model.fc2, name='weight', amount=0.5)
prune.l1_unstructured(model.fc3, name='weight', amount=0.3)

# Make pruning permanent
prune.remove(model.fc1, 'weight')
prune.remove(model.fc2, 'weight')
prune.remove(model.fc3, 'weight')

nz = sum(p.count_nonzero().item() for p in model.parameters())
print(f"Non-zero params after pruning: {nz:,} ({100*nz/total:.1f}%)")

# Simulated INT8 quantization (Post-Training Quantization)
def quantize_tensor(tensor, bits=8):
    q_min, q_max = -(2**(bits-1)), (2**(bits-1) - 1)
    scale = tensor.abs().max() / q_max
    q_tensor = (tensor / scale).round().clamp(q_min, q_max)
    return q_tensor, scale

with torch.no_grad():
    w = model.fc1.weight
    w_q, scale = quantize_tensor(w)
    w_deq = w_q * scale
    quant_error = (w - w_deq).abs().mean().item()
    print(f"\\nINT8 quantization error (fc1.weight): {quant_error:.6f}")
    print(f"Compression: 32-bit -> 8-bit = 4x reduction")"""
  },
],

# ══════════════════════════════════════════════════════
"causal_inference": [
  {
    "title": "Propensity Score Matching & ATE Estimation",
    "code": """import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 1000

# Simulate observational data with confounders
X1 = np.random.randn(n)    # confounder: age (standardized)
X2 = np.random.randn(n)    # confounder: income
# Treatment propensity depends on confounders
p_treat = 1 / (1 + np.exp(-(0.5*X1 + 0.3*X2)))
T = np.random.binomial(1, p_treat)   # treatment assignment
# Outcome depends on treatment AND confounders
Y = 2.0 * T + 1.5*X1 + X2 + np.random.randn(n)
df = pd.DataFrame({'X1': X1, 'X2': X2, 'T': T, 'Y': Y})

# True ATE
Y1 = 2.0 * 1 + 1.5*X1 + X2  # potential outcome under treatment
Y0 = 2.0 * 0 + 1.5*X1 + X2  # potential outcome under control
print(f"True ATE: {(Y1 - Y0).mean():.4f}  (should be 2.0)")

# Naive estimate (ignores confounding)
ate_naive = df[df.T==1]['Y'].mean() - df[df.T==0]['Y'].mean()
print(f"Naive ATE: {ate_naive:.4f}  (biased due to confounding)")

# Propensity Score Matching
scaler = StandardScaler()
X_feat = scaler.fit_transform(df[['X1','X2']])
psm_model = LogisticRegression().fit(X_feat, df['T'])
df['ps'] = psm_model.predict_proba(X_feat)[:, 1]

# Nearest-neighbor matching (simplified)
treated = df[df.T==1].copy()
control = df[df.T==0].copy()
matched_control_outcomes = []
for _, row in treated.iterrows():
    dists = np.abs(control['ps'] - row['ps'])
    matched_idx = dists.idxmin()
    matched_control_outcomes.append(control.loc[matched_idx, 'Y'])
treated['Y0_matched'] = matched_control_outcomes
ate_psm = (treated['Y'] - treated['Y0_matched']).mean()
print(f"PSM ATE  : {ate_psm:.4f}  (after matching)")"""
  },
  {
    "title": "Difference-in-Differences & Regression Discontinuity",
    "code": """import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 400

# ---- Difference-in-Differences (DiD) ----
# Policy introduced in 2020 for treated cities only
df_did = pd.DataFrame({
    'city_id': np.repeat(np.arange(20), 20),
    'year'   : np.tile(np.arange(2018, 2022), 100)[:n],
    'treated': np.repeat(np.random.randint(0, 2, 20), 20)[:n],
})
df_did['post'] = (df_did['year'] >= 2020).astype(int)
# Parallel trends + treatment effect of 5
df_did['outcome'] = (10 + 2*df_did['year'] - 2018 +
                      5*df_did['treated']*df_did['post'] +
                      np.random.randn(n))

model_did = smf.ols('outcome ~ treated * post', data=df_did).fit()
print("=== Difference-in-Differences ===")
print(f"DiD estimator (interaction coef): {model_did.params['treated:post']:.4f}  (true=5.0)")
print(f"p-value: {model_did.pvalues['treated:post']:.4f}")

# ---- Regression Discontinuity Design (RDD) ----
# Running variable: test score. Cutoff at 50: above gets scholarship
print("\\n=== Regression Discontinuity ===")
scores  = np.random.uniform(30, 70, 500)
treat_r = (scores >= 50).astype(float)
# Outcome: continuous in score + jump of 8 at cutoff
outcomes_r = 2 * scores + 8 * treat_r + np.random.randn(500) * 5

df_rdd = pd.DataFrame({'score': scores, 'treat': treat_r, 'outcome': outcomes_r})
df_rdd['score_c'] = df_rdd['score'] - 50   # center at cutoff

# Local linear regression near cutoff (bandwidth=10)
bw = 10
local = df_rdd[df_rdd['score_c'].abs() <= bw]
rdd_model = smf.ols('outcome ~ score_c * treat', data=local).fit()
print(f"RDD estimated jump: {rdd_model.params['treat']:.4f}  (true=8.0)")
print(f"p-value: {rdd_model.pvalues['treat']:.4f}")"""
  },
],

# ══════════════════════════════════════════════════════
"multimodal": [
  {
    "title": "CLIP Zero-Shot Classification",
    "code": """import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import numpy as np

# Load CLIP
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Download a sample image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Zero-shot classification with text prompts
candidate_labels = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a person",
    "a photo of a car",
    "a photo of food"
]

with torch.no_grad():
    inputs = processor(text=candidate_labels, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # Cosine similarity between image and text embeddings
    logits = outputs.logits_per_image   # (1, num_labels)
    probs  = logits.softmax(dim=1).squeeze().numpy()

print("CLIP Zero-shot Classification:")
for label, prob in sorted(zip(candidate_labels, probs),
                           key=lambda x: x[1], reverse=True):
    print(f"  {prob:.4f}  {label}")

# Text-image similarity score
img_emb  = model.get_image_features(**processor(images=image, return_tensors='pt'))
txt_emb  = model.get_text_features(**processor(text=["a cat"], return_tensors='pt'))
img_emb  = img_emb / img_emb.norm(dim=-1, keepdim=True)
txt_emb  = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
sim = (img_emb @ txt_emb.T).item()
print(f"\\nCosine similarity (image, 'a cat'): {sim:.4f}")"""
  },
  {
    "title": "Vision-Language Model: Image + Text Embedding Fusion",
    "code": """import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal cross-modal fusion architecture
class CrossModalAttention(nn.Module):
    '''Cross-attention: query from one modality, key/value from another.'''
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):
        attn_out, attn_weights = self.attn(query, key_value, key_value)
        return self.norm(query + attn_out), attn_weights

class MultimodalFusion(nn.Module):
    '''Fuse image patch embeddings with text token embeddings.'''
    def __init__(self, img_dim=768, txt_dim=768, d_model=256, n_classes=5):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, d_model)
        self.txt_proj = nn.Linear(txt_dim, d_model)
        self.img_to_txt = CrossModalAttention(d_model)
        self.txt_to_img = CrossModalAttention(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, img_feats, txt_feats):
        # img_feats: (B, num_patches, img_dim)
        # txt_feats: (B, seq_len, txt_dim)
        img = self.img_proj(img_feats)
        txt = self.txt_proj(txt_feats)
        img_ctx, _ = self.img_to_txt(img, txt)  # image attends to text
        txt_ctx, _ = self.txt_to_img(txt, img)  # text attends to image
        # Pool and classify
        img_pooled = img_ctx.mean(1)
        txt_pooled = txt_ctx.mean(1)
        fused = torch.cat([img_pooled, txt_pooled], dim=-1)
        return self.classifier(fused)

# Demo
torch.manual_seed(42)
model = MultimodalFusion()
B = 4
img_feats = torch.randn(B, 196, 768)   # ViT: 14x14 = 196 patches
txt_feats = torch.randn(B, 32, 768)    # text: 32 tokens
logits = model(img_feats, txt_feats)
print(f"Image patches  : {img_feats.shape}")
print(f"Text tokens    : {txt_feats.shape}")
print(f"Output logits  : {logits.shape}")
print(f"Model params   : {sum(p.numel() for p in model.parameters()):,}")"""
  },
],

# ══════════════════════════════════════════════════════
"ethics_fairness": [
  {
    "title": "Fairness Metrics & Disparity Analysis",
    "code": """import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

np.random.seed(42)
n = 2000

# Simulate loan application data with historical bias
gender = np.random.choice([0, 1], n, p=[0.5, 0.5])  # 0=female, 1=male
merit  = np.random.randn(n)                           # creditworthiness
# Historical bias: male applicants historically approved more
true_label = ((merit + 0.5 * gender + np.random.randn(n) * 0.3) > 0).astype(int)
X = np.column_stack([merit, gender, np.random.randn(n, 3)])  # 3 extra features

X_s = StandardScaler().fit_transform(X)
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_s, true_label)
pred = model.predict(X_s)

for g_val, g_name in [(0, 'Female'), (1, 'Male')]:
    mask = gender == g_val
    y_g  = true_label[mask]
    p_g  = pred[mask]
    tn, fp, fn, tp = confusion_matrix(y_g, p_g).ravel()
    tpr = tp / (tp + fn)   # Sensitivity / Recall
    fpr = fp / (fp + tn)   # False Positive Rate
    ppv = tp / (tp + fp)   # Precision
    pos_rate = p_g.mean()  # Demographic parity numerator
    print(f"[{g_name:6s}]  Approval rate={pos_rate:.3f}  TPR={tpr:.3f}  FPR={fpr:.3f}  PPV={ppv:.3f}")

# Disparate Impact ratio (80% rule)
female_rate = pred[gender==0].mean()
male_rate   = pred[gender==1].mean()
di_ratio    = female_rate / male_rate
print(f"\\nDisparate Impact Ratio: {di_ratio:.4f}  (< 0.8 = discriminatory by EEOC rule)")
print(f"Compliant: {di_ratio >= 0.8}")"""
  },
  {
    "title": "SHAP Explainability & Differential Privacy",
    "code": """import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import shap

np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, random_state=42)
feature_names = [f'feature_{i}' for i in range(10)]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_tr, y_tr)

# SHAP values for feature importance and individual explanation
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te[:20])  # explain 20 test samples

print("SHAP global feature importances (mean |SHAP|):")
global_imp = np.abs(shap_values).mean(0)
for name, imp in sorted(zip(feature_names, global_imp), key=lambda x: x[1], reverse=True)[:5]:
    bar = '|' * int(imp / global_imp.max() * 20)
    print(f"  {name:12s}: {imp:.4f}  {bar}")

print(f"\\nSHAP for test sample 0 (predicted: {model.predict(X_te[:1])[0]}):")
for name, sv in zip(feature_names, shap_values[0]):
    direction = 'increases' if sv > 0 else 'decreases'
    print(f"  {name:12s}: {sv:+.4f}  {direction} P(positive)")

# Differential Privacy: Laplace mechanism
print("\\n=== Differential Privacy Demo ===")
real_mean = X[:, 0].mean()
for eps in [0.1, 1.0, 10.0]:
    sensitivity = (X[:, 0].max() - X[:, 0].min()) / len(X)  # global sensitivity
    noise = np.random.laplace(0, sensitivity / eps)
    dp_mean = real_mean + noise
    print(f"eps={eps:4.1f} | True mean={real_mean:.4f} | DP mean={dp_mean:.4f} | error={abs(dp_mean-real_mean):.4f}")"""
  },
  {
    "title": "Fairness-Aware Model Training & Threshold Optimization",
    "code": """import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

np.random.seed(0)
n = 3000

# Sensitive attribute: protected group (0 or 1)
group     = np.random.choice([0, 1], n, p=[0.6, 0.4])
X_feat    = np.column_stack([np.random.randn(n, 5), group])
y         = ((X_feat[:, 0] + X_feat[:, 1] + 0.3 * group + np.random.randn(n)*0.5) > 0).astype(int)

X_s    = StandardScaler().fit_transform(X_feat)
model  = LogisticRegression(C=1.0, max_iter=500).fit(X_s, y)
y_prob = model.predict_proba(X_s)[:, 1]

def fairness_report(y_true, y_prob, group, thresh_g0, thresh_g1):
    y_pred = np.where(group == 0,
                      (y_prob >= thresh_g0).astype(int),
                      (y_prob >= thresh_g1).astype(int))
    for g_val, thresh in [(0, thresh_g0), (1, thresh_g1)]:
        mask  = group == g_val
        rate  = y_pred[mask].mean()
        f1    = f1_score(y_true[mask], y_pred[mask])
        print(f"  Group {g_val} (thresh={thresh:.2f}): approval={rate:.3f}  F1={f1:.3f}")
    di = y_pred[group==0].mean() / (y_pred[group==1].mean() + 1e-9)
    print(f"  Disparate Impact Ratio: {di:.4f}")

print("Default threshold (0.5 for all):")
fairness_report(y, y_prob, group, 0.5, 0.5)

print("\\nPost-processing: group-specific thresholds for demographic parity:")
# Adjust group 0 threshold to match group 1 approval rate
target_rate = y_prob[group==1].mean()
best_thresh = 0.5
for t in np.linspace(0.1, 0.9, 100):
    if abs((y_prob[group==0] >= t).mean() - target_rate) < 0.01:
        best_thresh = t; break
fairness_report(y, y_prob, group, best_thresh, 0.5)"""
  },
],

}

if __name__ == "__main__":
    print(f"Code samples defined for {len(CODE_SAMPLES)} algorithms")
    for k, v in CODE_SAMPLES.items():
        print(f"  {k}: {len(v)} samples")
