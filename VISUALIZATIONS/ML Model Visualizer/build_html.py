import json, os, html as html_mod

with open('/home/user/workspace/ml-visualizer/charts.json') as f:
    charts = json.load(f)

from code_samples import CODE_SAMPLES

# ── Full tab definitions ──────────────────────────────────────────────────
ALL_TABS = [
    # ── SUPERVISED ─────────────────────────────────────────────────────
    {
        "id": "linear_regression", "label": "Linear Regression", "category": "supervised",
        "title": "Linear Regression", "subtitle": "Modeling continuous output as a linear function of inputs — the foundation of predictive modeling.",
        "theory": """<p><strong>Core Idea:</strong> Find the hyperplane β that minimizes the sum of squared residuals between predictions and actual values.</p>
<p><strong>Objective:</strong> Minimize <code>RSS = Σ(yᵢ − ŷᵢ)²</code> where <code>ŷ = Xβ</code></p>
<ul>
  <li><strong>OLS Solution:</strong> β = (XᵀX)⁻¹Xᵀy — closed-form, exact solution when XᵀX is invertible</li>
  <li><strong>Ridge (L2):</strong> β = (XᵀX + αI)⁻¹Xᵀy — shrinks all coefficients, handles multicollinearity</li>
  <li><strong>Lasso (L1):</strong> Adds |β| penalty — performs automatic feature selection by zeroing coefficients</li>
  <li><strong>Elastic Net:</strong> Convex combination of L1 and L2 penalties — best of both worlds</li>
  <li><strong>Assumptions:</strong> Linearity, Independence, Homoscedasticity (constant variance), Normality of residuals</li>
  <li><strong>Residual Diagnostics:</strong> Q-Q plots, residual vs fitted, Durbin-Watson test for autocorrelation</li>
</ul>
<p><strong>Key Metrics:</strong> R², Adjusted R², MSE, RMSE, MAE, AIC/BIC for model selection</p>""",
        "use_cases": ["House price prediction", "Sales forecasting", "Risk factor analysis", "Stock return modeling", "Medical dosage prediction"],
        "chart": "linear_regression"
    },
    {
        "id": "logistic_regression", "label": "Logistic Regression", "category": "supervised",
        "title": "Logistic Regression", "subtitle": "Binary and multi-class classification via sigmoid-transformed linear model with probabilistic output.",
        "theory": """<p><strong>Core Idea:</strong> Apply sigmoid activation to a linear combination to produce class probabilities.</p>
<p><strong>Model:</strong> <code>P(y=1|X) = σ(Xβ) = 1 / (1 + e⁻ᴮˣ)</code></p>
<ul>
  <li><strong>Decision Boundary:</strong> Where P = 0.5, i.e., Xβ = 0 — a linear hyperplane in feature space</li>
  <li><strong>Loss Function:</strong> Binary cross-entropy: <code>−Σ[y·log(ŷ) + (1−y)·log(1−ŷ)]</code></li>
  <li><strong>Optimization:</strong> Gradient Descent or Newton-Raphson (IRLS) — no closed form</li>
  <li><strong>Regularization:</strong> C = 1/λ — smaller C = stronger L2 regularization by default</li>
  <li><strong>Multiclass:</strong> One-vs-Rest (OvR) or Softmax/Multinomial logistic regression</li>
  <li><strong>Odds Ratio:</strong> exp(βᵢ) = multiplicative change in odds per unit increase in xᵢ</li>
  <li><strong>Calibration:</strong> Logistic regression is naturally well-calibrated for probabilities</li>
</ul>
<p><strong>Key Metrics:</strong> Accuracy, AUC-ROC, Log-loss, F1, Precision-Recall, Brier Score</p>""",
        "use_cases": ["Spam detection", "Credit default prediction", "Medical diagnosis", "Fraud detection", "Churn prediction"],
        "chart": "logistic_regression"
    },
    {
        "id": "decision_tree", "label": "Decision Tree", "category": "supervised",
        "title": "Decision Tree", "subtitle": "Hierarchical partitioning of feature space using recursive binary splitting on impurity criteria.",
        "theory": """<p><strong>Core Idea:</strong> Recursively split data into subsets that become increasingly pure in class labels.</p>
<ul>
  <li><strong>Gini Impurity:</strong> <code>G = 1 − Σpᵢ²</code> — lower is purer; used in CART algorithm</li>
  <li><strong>Information Gain:</strong> IG = H(parent) − Σ(wᵢ·H(childᵢ)) where H = entropy = −Σpᵢlog₂(pᵢ)</li>
  <li><strong>Regression Trees:</strong> Minimize MSE/MAE at each split instead of impurity</li>
  <li><strong>Pre-pruning:</strong> max_depth, min_samples_split, min_samples_leaf — stop early</li>
  <li><strong>Post-pruning:</strong> Cost-complexity pruning (ccp_alpha) — grow full tree, then prune</li>
  <li><strong>Feature Importance:</strong> Total weighted impurity decrease across all splits for a feature</li>
  <li><strong>Pros:</strong> Interpretable (can be visualized), handles mixed types, no feature scaling needed</li>
  <li><strong>Cons:</strong> High variance, prone to overfitting — single trees rarely used in production</li>
</ul>
<p><strong>Hyperparams:</strong> max_depth, min_samples_split, min_samples_leaf, max_features, ccp_alpha</p>""",
        "use_cases": ["Credit scoring rules", "Medical triage", "Fraud rule extraction", "Customer segmentation", "Feature selection"],
        "chart": "decision_tree"
    },
    {
        "id": "svm", "label": "SVM", "category": "supervised",
        "title": "Support Vector Machine", "subtitle": "Maximum-margin classifier finding the optimal separating hyperplane with kernel trick for non-linearity.",
        "theory": """<p><strong>Core Idea:</strong> Find the hyperplane that maximizes the margin (gap) between the two class boundaries.</p>
<ul>
  <li><strong>Hard Margin:</strong> Maximize 2/||w|| subject to yᵢ(wᵀxᵢ+b) ≥ 1 — only works if linearly separable</li>
  <li><strong>Soft Margin (C):</strong> Introduces slack variables ξᵢ ≥ 0 — trades off margin width vs misclassification</li>
  <li><strong>Kernel Trick:</strong> K(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ) — implicitly maps to high-dim space without computing φ</li>
  <li><strong>RBF Kernel:</strong> K(x,x') = exp(−γ||x−x'||²) — γ controls the radius of influence of each support vector</li>
  <li><strong>Polynomial:</strong> K(x,x') = (γxᵀx'+r)ᵈ — d controls degree of polynomial boundary</li>
  <li><strong>SVR:</strong> Regression version with ε-insensitive tube — ignore errors within ε, penalize those outside</li>
  <li><strong>Dual Problem:</strong> Solved via quadratic programming — only support vectors (on or inside margin) matter</li>
</ul>
<p><strong>Key Params:</strong> C (margin vs error), γ (RBF width), kernel, ε (SVR tube width)</p>""",
        "use_cases": ["Image classification", "Text categorization", "Bioinformatics", "Financial signal detection", "Face recognition"],
        "chart": "svm"
    },
    {
        "id": "svr", "label": "SVR", "category": "supervised",
        "title": "Support Vector Regression", "subtitle": "SVM adapted for regression — fits an ε-insensitive tube around the function, ignoring small errors.",
        "theory": """<p><strong>Core Idea:</strong> Fit a function f(x) such that most residuals fall within an ε-wide tube, penalizing only those outside.</p>
<ul>
  <li><strong>ε-Insensitive Loss:</strong> L(y, ŷ) = max(0, |y−ŷ| − ε) — zero loss for predictions within tube</li>
  <li><strong>Objective:</strong> Minimize ½||w||² + C·Σ(ξᵢ + ξᵢ*) where ξᵢ are slack variables for tube violations</li>
  <li><strong>C parameter:</strong> Controls trade-off between flatness (regularization) and tolerance for violations</li>
  <li><strong>ε parameter:</strong> Width of the tube — larger ε = fewer support vectors, smoother model</li>
  <li><strong>Support Vectors:</strong> Only the points lying on or outside the ε-tube influence the model</li>
  <li><strong>Kernels:</strong> Same as SVC — RBF is most common for non-linear regression</li>
  <li><strong>Advantage over LR:</strong> Robust to outliers (bounded loss), non-linear via kernels</li>
</ul>
<p><strong>Key Params:</strong> C, ε (epsilon), kernel, γ — tune with cross-validation</p>""",
        "use_cases": ["Time series forecasting", "Stock price prediction", "Energy consumption modeling", "Drug response prediction"],
        "chart": "svr"
    },
    {
        "id": "knn", "label": "KNN", "category": "supervised",
        "title": "K-Nearest Neighbors", "subtitle": "Non-parametric instance-based learning — classifies based on proximity of k nearest training samples.",
        "theory": """<p><strong>Core Idea:</strong> Classify or predict based on the k closest training samples in feature space (lazy learner — no explicit training).</p>
<ul>
  <li><strong>Distance Metrics:</strong> Euclidean (p=2), Manhattan (p=1), Minkowski (general p), Cosine (text/NLP)</li>
  <li><strong>Classification:</strong> Majority vote among k neighbors — ties broken by distance</li>
  <li><strong>Regression:</strong> Mean (or distance-weighted mean) of k neighbor values</li>
  <li><strong>k=1:</strong> Highly irregular boundary, high variance, overfits — training accuracy = 100%</li>
  <li><strong>k=N:</strong> Always predicts majority class — maximum bias, zero variance</li>
  <li><strong>Optimal k:</strong> Typically odd (avoid ties), chosen via CV — often √N as starting point</li>
  <li><strong>Critical:</strong> MUST normalize/standardize features — KNN is purely distance-based</li>
  <li><strong>Complexity:</strong> O(N·d) per prediction — Ball Tree or KD-Tree reduce to O(d·log N)</li>
</ul>
<p><strong>Variants:</strong> Weighted KNN (1/distance), Radius Neighbors, Ball Tree, KD-Tree indexing</p>""",
        "use_cases": ["Recommendation systems", "Anomaly detection", "Image recognition (simple)", "Imputation", "Drug similarity"],
        "chart": "knn"
    },
    {
        "id": "naive_bayes", "label": "Naive Bayes", "category": "supervised",
        "title": "Naive Bayes", "subtitle": "Probabilistic classifier applying Bayes' theorem with conditional feature independence assumption.",
        "theory": """<p><strong>Core Idea:</strong> Apply Bayes' theorem to compute the posterior probability of each class, assuming features are conditionally independent.</p>
<p><strong>Formula:</strong> <code>P(y|X) ∝ P(y) · Πᵢ P(xᵢ|y)</code></p>
<ul>
  <li><strong>Gaussian NB:</strong> Assumes P(xᵢ|y) = N(μ_{y,i}, σ²_{y,i}) — for continuous features</li>
  <li><strong>Multinomial NB:</strong> For count data (word frequencies in NLP) — P(xᵢ|y) ∝ θ_{y,i}^{xᵢ}</li>
  <li><strong>Bernoulli NB:</strong> Binary features (word presence/absence) — penalizes absent words</li>
  <li><strong>Complement NB:</strong> Better for imbalanced text data (estimates from complement class)</li>
  <li><strong>Laplace Smoothing:</strong> Adds α to all counts — avoids zero probabilities for unseen features</li>
  <li><strong>Why "Naive":</strong> Independence assumption is rarely true, but model is surprisingly robust</li>
  <li><strong>Log-space computation:</strong> Sum of log-probabilities to avoid numerical underflow</li>
</ul>
<p><strong>Pros:</strong> Fast O(N·d) training, works well with small data, excellent for text classification</p>""",
        "use_cases": ["Email spam filtering", "Sentiment analysis", "Document classification", "Medical diagnosis", "Real-time prediction"],
        "chart": "naive_bayes"
    },
    # ── ENSEMBLE ───────────────────────────────────────────────────────
    {
        "id": "bagging", "label": "Bagging", "category": "ensemble",
        "title": "Bagging (Bootstrap Aggregating)", "subtitle": "Reduce variance by training multiple models on bootstrap samples and aggregating predictions.",
        "theory": """<p><strong>Core Idea:</strong> Train B models on B bootstrap samples (sampling with replacement), then aggregate predictions to reduce variance.</p>
<ul>
  <li><strong>Bootstrap Sample:</strong> N samples drawn with replacement — ~63.2% unique rows, ~36.8% OOB</li>
  <li><strong>OOB Score:</strong> Out-of-bag samples (~37%) act as free validation — no separate hold-out needed</li>
  <li><strong>Aggregation:</strong> Classification = majority vote; Regression = mean across all models</li>
  <li><strong>Why it works:</strong> Averaging B uncorrelated models of variance σ² → variance σ²/B (ideal case)</li>
  <li><strong>Correlation problem:</strong> Trees trained on same features remain correlated → Random Forest adds feature randomization</li>
  <li><strong>Max Features:</strong> Subspace sampling (max_features) reduces correlation further</li>
  <li><strong>Best base learner:</strong> High-variance, low-bias models (deep unpruned trees) benefit most</li>
  <li><strong>Bagging vs Boosting:</strong> Bagging = parallel + variance reduction; Boosting = sequential + bias reduction</li>
</ul>
<p><strong>Key Params:</strong> n_estimators, max_samples, max_features, bootstrap, oob_score</p>""",
        "use_cases": ["Variance reduction in prediction", "Ensemble diversity", "Robust classification", "Medical imaging"],
        "chart": "bagging"
    },
    {
        "id": "random_forest", "label": "Random Forest", "category": "ensemble",
        "title": "Random Forest", "subtitle": "Bootstrap-aggregated ensemble of decorrelated decision trees with random feature subsampling.",
        "theory": """<p><strong>Core Idea:</strong> Combine many independently trained trees, each on a random data subset AND random feature subset to decorrelate them.</p>
<ul>
  <li><strong>Feature Randomness:</strong> At each split, only √p (classification) or p/3 (regression) features considered</li>
  <li><strong>Why decorrelate?</strong> Avg of correlated trees (corr ρ): Var = ρσ² + (1-ρ)σ²/B → must reduce ρ</li>
  <li><strong>OOB Score:</strong> Each tree evaluated on ~37% OOB samples — reliable CV estimate</li>
  <li><strong>Feature Importance:</strong> Mean decrease in Gini impurity (MDI) across all trees and splits</li>
  <li><strong>Permutation Importance:</strong> More reliable — measure accuracy drop when feature is shuffled</li>
  <li><strong>Proximity Matrix:</strong> Fraction of trees where two samples end in same leaf — useful for anomaly detection</li>
  <li><strong>Extremely Randomized Trees (ExtraTrees):</strong> Random split thresholds too — faster, more randomization</li>
</ul>
<p><strong>Key Params:</strong> n_estimators (↑ = better but slower), max_depth, max_features, min_samples_leaf</p>""",
        "use_cases": ["Credit risk scoring", "Medical diagnosis", "Feature selection pipeline", "Remote sensing", "Genomics"],
        "chart": "random_forest"
    },
    {
        "id": "gradient_boosting", "label": "Gradient Boosting", "category": "ensemble",
        "title": "Gradient Boosting (GBM)", "subtitle": "Sequential additive ensemble minimizing loss via gradient descent in function space — the competition winner.",
        "theory": """<p><strong>Core Idea:</strong> Build trees sequentially, each correcting the residual errors (pseudo-residuals) of the current ensemble.</p>
<ul>
  <li><strong>Algorithm:</strong> F_m(x) = F_{m−1}(x) + η·h_m(x) where h_m fits the negative gradient of L</li>
  <li><strong>Pseudo-residuals:</strong> rᵢ = −∂L/∂F(xᵢ) — for squared loss this is simply (yᵢ − ŷᵢ)</li>
  <li><strong>Shrinkage (η):</strong> Learning rate scales each tree's contribution — smaller η = better generalization (need more trees)</li>
  <li><strong>Subsample:</strong> Stochastic GBM — train each tree on a random row subsample — reduces variance, speeds up</li>
  <li><strong>XGBoost:</strong> Second-order (Newton) gradients + L1/L2 regularization + column subsampling + parallel split finding</li>
  <li><strong>LightGBM:</strong> Leaf-wise growth (vs level-wise) + GOSS + EFB — dramatically faster on large datasets</li>
  <li><strong>CatBoost:</strong> Ordered boosting + native categorical support + symmetric trees — reduces prediction shift</li>
  <li><strong>Early Stopping:</strong> Monitor validation loss, stop when it stops improving — prevents overfitting</li>
</ul>
<p><strong>Key Params:</strong> n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda</p>""",
        "use_cases": ["Kaggle competitions", "CTR/ad prediction", "Ranking problems", "Financial default modeling", "Recommendation"],
        "chart": "gradient_boosting"
    },
    {
        "id": "xgboost", "label": "XGBoost / LightGBM", "category": "ensemble",
        "title": "XGBoost & LightGBM", "subtitle": "Optimized gradient boosting frameworks with regularization, parallel processing, and advanced split strategies.",
        "theory": """<p><strong>XGBoost Innovations:</strong></p>
<ul>
  <li><strong>2nd-order gradients:</strong> Uses both gradient and Hessian for better step direction — Newton boosting</li>
  <li><strong>Regularization:</strong> Explicit L1 (α) and L2 (λ) on leaf weights in the objective function</li>
  <li><strong>Column subsampling:</strong> colsample_bytree, colsample_bylevel, colsample_bynode</li>
  <li><strong>Sparsity-aware:</strong> Default direction for missing values learned from data</li>
  <li><strong>Approximate split:</strong> Weighted quantile sketch — scalable to large datasets</li>
</ul>
<p><strong>LightGBM Innovations:</strong></p>
<ul>
  <li><strong>Leaf-wise growth:</strong> Grows the leaf with max loss reduction (vs level-wise) — fewer splits needed</li>
  <li><strong>GOSS:</strong> Gradient-based One-Side Sampling — keeps large-gradient samples, randomly drops small-gradient ones</li>
  <li><strong>EFB:</strong> Exclusive Feature Bundling — bundles mutually exclusive sparse features to reduce dimensions</li>
  <li><strong>Histogram-based:</strong> Buckets continuous features — much faster split finding O(#bins) vs O(N)</li>
</ul>
<p><strong>CatBoost:</strong> Ordered boosting (no target leakage), symmetric trees for fast inference, GPU support</p>""",
        "use_cases": ["Industrial ML pipelines", "High-cardinality categorical data", "Online ad click prediction", "Risk models", "Tabular data competitions"],
        "chart": "xgboost"
    },
    {
        "id": "adaboost", "label": "AdaBoost", "category": "ensemble",
        "title": "AdaBoost", "subtitle": "Adaptive boosting via iterative reweighting of misclassified samples to focus subsequent learners.",
        "theory": """<p><strong>Core Idea:</strong> Train weak learners sequentially, increasing the weight of misclassified samples each round so subsequent learners focus on hard cases.</p>
<ul>
  <li><strong>Weak Learner:</strong> Typically decision stump (depth=1) — just better than random (error &lt; 0.5)</li>
  <li><strong>Sample Weights:</strong> Initially uniform wᵢ = 1/N; misclassified samples receive higher weight</li>
  <li><strong>Estimator Weight:</strong> αₘ = 0.5·ln((1−errₘ)/errₘ) — accurate learners get higher vote weight</li>
  <li><strong>Weight Update:</strong> wᵢ ← wᵢ·exp(−αₘ·yᵢ·hₘ(xᵢ)) then normalize</li>
  <li><strong>Final Prediction:</strong> H(x) = sign(Σ αₘ·hₘ(x)) — weighted majority vote</li>
  <li><strong>Loss:</strong> Minimizes exponential loss exp(−y·F(x)) — sensitive to outliers/noise</li>
  <li><strong>SAMME / SAMME.R:</strong> Multi-class AdaBoost extensions</li>
  <li><strong>Connection to GBM:</strong> AdaBoost = GBM with exponential loss and no shrinkage</li>
</ul>
<p><strong>Theoretical guarantee:</strong> Training error decreases exponentially with number of rounds</p>""",
        "use_cases": ["Face detection (Viola-Jones)", "Medical imaging", "Text classification", "Anomaly detection"],
        "chart": "adaboost"
    },
    {
        "id": "ensemble_comparison", "label": "Ensemble Comparison", "category": "ensemble",
        "title": "Ensemble Methods Comparison", "subtitle": "Head-to-head comparison of all ensemble strategies including Voting and Stacking architectures.",
        "theory": """<p><strong>Voting Ensemble:</strong> Combine predictions from multiple diverse models without training a meta-learner.</p>
<ul>
  <li><strong>Hard Voting:</strong> Majority class vote — each model casts one vote</li>
  <li><strong>Soft Voting:</strong> Average predicted probabilities — requires predict_proba; generally better</li>
  <li><strong>Key:</strong> Works best when models are diverse (different algorithms/hyperparams)</li>
</ul>
<p><strong>Stacking (Stacked Generalization):</strong></p>
<ul>
  <li><strong>Level 0 (Base Learners):</strong> Diverse models trained on original features</li>
  <li><strong>Level 1 (Meta-Learner):</strong> Trained on out-of-fold predictions from base learners</li>
  <li><strong>Cross-val predictions:</strong> Base learners predict on held-out folds to avoid leakage</li>
  <li><strong>Meta-learner:</strong> Usually simple — Logistic Regression, Ridge, or XGBoost</li>
  <li><strong>Multi-level stacking:</strong> Additional layers possible but rarely beneficial</li>
</ul>
<p><strong>Rule:</strong> More diversity among base models = more benefit from combining them</p>""",
        "use_cases": ["Competition ML (Kaggle)", "Production model robustness", "Risk reduction", "Combining domain-specific models"],
        "chart": "ensemble_comparison"
    },
    # ── UNSUPERVISED ───────────────────────────────────────────────────
    {
        "id": "kmeans", "label": "K-Means", "category": "unsupervised",
        "title": "K-Means Clustering", "subtitle": "Partition-based clustering minimizing within-cluster sum of squared distances to centroids.",
        "theory": """<p><strong>Core Idea:</strong> Assign N points to K clusters by iteratively updating centroid assignments to minimize intra-cluster variance.</p>
<ul>
  <li><strong>Objective:</strong> Minimize <code>J = Σₖ Σ_{x∈Cₖ} ||x − μₖ||²</code></li>
  <li><strong>Lloyd's Algorithm:</strong> 1) Assign each point to nearest centroid 2) Update centroids to cluster mean 3) Repeat until convergence</li>
  <li><strong>K-Means++:</strong> Probabilistic initialization — selects centroids proportional to distance² — avoids poor starts</li>
  <li><strong>Convergence:</strong> Guaranteed to converge but may find local minimum — run multiple times (n_init)</li>
  <li><strong>Elbow Method:</strong> Plot inertia vs k — pick the "elbow" where diminishing returns begin</li>
  <li><strong>Silhouette Score:</strong> s = (b−a)/max(a,b) where a=intra-cluster dist, b=nearest-cluster dist ∈ [−1,+1]</li>
  <li><strong>Limitations:</strong> Assumes spherical, equal-size clusters; sensitive to outliers; must specify k</li>
</ul>
<p><strong>Variants:</strong> Mini-batch K-Means (faster), K-Medoids (PAM, robust to outliers), Fuzzy C-Means (soft assignments)</p>""",
        "use_cases": ["Customer segmentation", "Image color quantization", "Document clustering", "Anomaly detection baseline", "Vector quantization"],
        "chart": "kmeans"
    },
    {
        "id": "dbscan", "label": "DBSCAN", "category": "unsupervised",
        "title": "DBSCAN", "subtitle": "Density-based spatial clustering — discovers arbitrary shapes and automatically identifies outliers.",
        "theory": """<p><strong>Core Idea:</strong> Define clusters as dense regions of points (high local density) separated by sparser regions.</p>
<ul>
  <li><strong>Core Point:</strong> Has ≥ min_samples points within radius ε — dense enough to start a cluster</li>
  <li><strong>Border Point:</strong> Within ε of a core point but not itself a core point — belongs to cluster</li>
  <li><strong>Noise/Outlier:</strong> Neither core nor reachable from any core — labeled −1</li>
  <li><strong>eps (ε):</strong> Neighborhood radius — use k-distance plot (sorted k-NN distances) to find the "knee"</li>
  <li><strong>min_samples:</strong> Higher value → more conservative, fewer but denser clusters; rule of thumb ≥ dim+1</li>
  <li><strong>Density-reachability:</strong> Transitive closure of core-point neighborhoods defines clusters</li>
  <li><strong>Advantages over K-Means:</strong> No k needed, arbitrary shapes, built-in outlier detection</li>
  <li><strong>Weakness:</strong> Struggles with varying-density clusters, poor in high dimensions</li>
</ul>
<p><strong>Variants:</strong> HDBSCAN (hierarchical, variable density), OPTICS (density ordering), ST-DBSCAN (spatio-temporal)</p>""",
        "use_cases": ["Geospatial hotspot detection", "Network intrusion detection", "Image segmentation", "Trajectory clustering"],
        "chart": "dbscan"
    },
    {
        "id": "pca", "label": "PCA", "category": "unsupervised",
        "title": "Principal Component Analysis", "subtitle": "Linear dimensionality reduction finding orthogonal directions of maximum variance via eigendecomposition.",
        "theory": """<p><strong>Core Idea:</strong> Find orthogonal linear combinations of features that capture maximum variance in decreasing order.</p>
<ul>
  <li><strong>Covariance Matrix:</strong> C = XᵀX/(n−1) — captures all pairwise feature relationships</li>
  <li><strong>Eigendecomposition:</strong> Cv = λv — eigenvectors v are principal components, eigenvalues λ = variance explained</li>
  <li><strong>SVD approach:</strong> X = UΣVᵀ — numerically more stable; right singular vectors = PCs</li>
  <li><strong>Projection:</strong> Z = XV_k — project data onto top-k eigenvectors (dimensionality reduction)</li>
  <li><strong>Explained Variance Ratio:</strong> λᵢ/Σλⱼ — fraction of total variance captured by each PC</li>
  <li><strong>Reconstruction:</strong> X̂ = ZV_kᵀ — lossy reconstruction from reduced representation</li>
  <li><strong>Whitening:</strong> Scale PCs by 1/√λ — makes all components unit variance (useful for downstream models)</li>
  <li><strong>Kernel PCA:</strong> Non-linear extension using kernel trick (RBF, polynomial kernels)</li>
</ul>
<p><strong>Rule:</strong> Standardize features (zero mean, unit variance) before PCA. Keep components explaining ≥95% variance.</p>""",
        "use_cases": ["Dimensionality reduction before ML", "Visualization (2D/3D)", "Noise removal", "Multicollinearity removal", "Compression"],
        "chart": "pca"
    },
    {
        "id": "tsne", "label": "t-SNE", "category": "unsupervised",
        "title": "t-SNE", "subtitle": "Non-linear dimensionality reduction preserving local neighborhood structure — designed purely for visualization.",
        "theory": """<p><strong>Core Idea:</strong> Map high-dim data to 2D/3D by preserving pairwise similarity structure — nearby points stay nearby.</p>
<ul>
  <li><strong>High-dim similarities:</strong> Pᵢⱼ = (Pⱼ|ᵢ + Pᵢ|ⱼ) / 2N where Pⱼ|ᵢ ∝ exp(−||xᵢ−xⱼ||²/2σᵢ²)</li>
  <li><strong>Low-dim similarities:</strong> Qᵢⱼ ∝ (1 + ||yᵢ−yⱼ||²)⁻¹ — Student t-distribution (1 DoF)</li>
  <li><strong>Objective:</strong> Minimize KL(P||Q) via gradient descent — makes Qᵢⱼ ≈ Pᵢⱼ</li>
  <li><strong>t-distribution:</strong> Heavy tails push dissimilar points far apart — solves crowding problem of Gaussian kernel</li>
  <li><strong>Perplexity:</strong> Effective number of neighbors (5–50) — controls σᵢ adaptively per point</li>
  <li><strong>Non-deterministic:</strong> Results depend on random seed, multiple runs recommended</li>
  <li><strong>NOT for features:</strong> t-SNE embeddings cannot be used as input features for ML — only visualization</li>
  <li><strong>Global structure:</strong> t-SNE does NOT preserve global distances — cluster positions are arbitrary</li>
</ul>
<p><strong>Modern Alternative:</strong> UMAP — faster, better global structure, can be used as features, parametric version</p>""",
        "use_cases": ["High-dim visualization", "Cluster exploration", "Embedding quality inspection", "Single-cell RNA-seq", "Anomaly visualization"],
        "chart": "tsne"
    },
    {
        "id": "dimensionality_methods", "label": "PCA vs t-SNE vs UMAP", "category": "unsupervised",
        "title": "Dimensionality Reduction Comparison", "subtitle": "Side-by-side comparison of linear (PCA), neighborhood-preserving (t-SNE) and topology-preserving (UMAP) methods.",
        "theory": """<p><strong>When to use each method:</strong></p>
<ul>
  <li><strong>PCA:</strong> Linear, fast, deterministic, preserves global variance — best first step, interpretable components</li>
  <li><strong>t-SNE:</strong> Non-linear, slow O(N²), non-deterministic, preserves local structure only — visualization only</li>
  <li><strong>UMAP:</strong> Non-linear, fast O(N·log N), preserves both local AND global structure, can be used as features</li>
  <li><strong>Kernel PCA:</strong> Non-linear extension of PCA via kernel trick — good middle ground</li>
  <li><strong>Autoencoders:</strong> Learned non-linear compression — flexible but requires training</li>
  <li><strong>ICA:</strong> Independent Component Analysis — finds statistically independent sources (blind source separation)</li>
  <li><strong>NMF:</strong> Non-negative Matrix Factorization — interpretable parts-based decomposition for non-negative data</li>
  <li><strong>MDS:</strong> Multi-Dimensional Scaling — preserves pairwise distances (not similarities)</li>
</ul>
<p><strong>Choosing:</strong> Start with PCA. If non-linear structure, try UMAP (for features) or t-SNE (for visualization only).</p>""",
        "use_cases": ["Exploratory data analysis", "Feature engineering", "Visualization pipeline", "Clustering preprocessing", "Genomics"],
        "chart": "dimensionality_methods"
    },
    {
        "id": "clustering_advanced", "label": "Hierarchical & GMM", "category": "unsupervised",
        "title": "Hierarchical Clustering & GMM", "subtitle": "Tree-based merging via linkage criteria and soft probabilistic cluster assignment via Gaussian Mixtures.",
        "theory": """<p><strong>Hierarchical Clustering:</strong> Build a dendrogram by iteratively merging (agglomerative) or splitting (divisive) clusters.</p>
<ul>
  <li><strong>Single Linkage:</strong> Min distance between clusters — tends to create elongated chains</li>
  <li><strong>Complete Linkage:</strong> Max distance — creates compact clusters, sensitive to outliers</li>
  <li><strong>Average Linkage (UPGMA):</strong> Mean pairwise distance — good balance</li>
  <li><strong>Ward Linkage:</strong> Minimize total within-cluster variance increase — produces compact, equal-size clusters (most used)</li>
  <li><strong>Dendrogram cut:</strong> Cut tree at desired height to get k clusters — k need not be chosen upfront</li>
</ul>
<p><strong>Gaussian Mixture Models (GMM):</strong></p>
<ul>
  <li><strong>Model:</strong> p(x) = Σₖ πₖ·N(x|μₖ,Σₖ) where πₖ are mixing weights (Σπₖ=1)</li>
  <li><strong>EM Algorithm:</strong> E-step: compute γᵢₖ = P(z=k|xᵢ); M-step: update μₖ, Σₖ, πₖ</li>
  <li><strong>Soft clustering:</strong> Each point has a probability of belonging to each cluster</li>
  <li><strong>Covariance types:</strong> Full, Tied (shared Σ), Diagonal, Spherical — control-off complexity</li>
  <li><strong>Model selection:</strong> BIC = log(N)·k − 2·log(L) or AIC = 2k − 2·log(L) to choose K</li>
</ul>""",
        "use_cases": ["Phylogenetics / evolutionary biology", "Topic modeling (PLSA)", "Image segmentation", "Density estimation", "Speaker diarization"],
        "chart": "clustering_advanced"
    },
    {
        "id": "autoencoder", "label": "Autoencoder", "category": "unsupervised",
        "title": "Autoencoder", "subtitle": "Unsupervised neural network learning compressed latent representations by reconstructing its own input.",
        "theory": """<p><strong>Core Idea:</strong> Train a network to compress input into a bottleneck (latent code) and reconstruct it — forces learning of essential structure.</p>
<ul>
  <li><strong>Encoder:</strong> f_θ: X → Z — maps input to latent space (dimensionality reduction)</li>
  <li><strong>Decoder:</strong> g_φ: Z → X̂ — maps latent code back to original space (reconstruction)</li>
  <li><strong>Loss:</strong> Reconstruction error — MSE for continuous, BCE for binary data</li>
  <li><strong>Bottleneck:</strong> Forces the network to learn a compressed representation — only keep important features</li>
  <li><strong>Undercomplete:</strong> dim(Z) &lt; dim(X) — compression forces feature learning</li>
  <li><strong>Denoising AE:</strong> Corrupt input with noise, reconstruct clean — learns robust features</li>
  <li><strong>Variational AE (VAE):</strong> Encoder outputs μ and σ — latent space is continuous and structured; enables generation</li>
  <li><strong>Contractive AE:</strong> Adds Frobenius norm of Jacobian — makes representation robust to small input changes</li>
  <li><strong>Sparse AE:</strong> L1 penalty on activations — forces few active neurons at a time</li>
</ul>
<p><strong>Applications:</strong> Anomaly detection (high reconstruction error = anomaly), compression, representation learning, generation (VAE)</p>""",
        "use_cases": ["Anomaly detection", "Image denoising", "Recommendation systems", "Data compression", "Generative modeling (VAE)"],
        "chart": "autoencoder"
    },
    {
        "id": "anomaly_detection", "label": "Anomaly Detection", "category": "unsupervised",
        "title": "Anomaly Detection", "subtitle": "Identifying rare, abnormal observations that deviate significantly from the majority of data.",
        "theory": """<p><strong>Core Idea:</strong> Learn the distribution of normal data; flag points that are unlikely under that distribution.</p>
<ul>
  <li><strong>Isolation Forest:</strong> Anomalies isolated in fewer splits — builds random trees, short path length = anomaly score</li>
  <li><strong>Local Outlier Factor (LOF):</strong> Compares local density of a point to neighbors — LOF &gt; 1 = anomaly</li>
  <li><strong>Elliptic Envelope:</strong> Fits a multivariate Gaussian to the data — Mahalanobis distance threshold</li>
  <li><strong>One-Class SVM:</strong> Finds a decision boundary around normal data in feature space</li>
  <li><strong>Autoencoder:</strong> High reconstruction error = anomaly (the model can't reconstruct unseen patterns)</li>
  <li><strong>Statistical:</strong> Z-score, IQR, Grubbs test — simple but univariate only</li>
  <li><strong>Contamination:</strong> Expected fraction of outliers — crucial hyperparameter for all methods</li>
  <li><strong>Evaluation:</strong> Hard because true labels rare — use AUC, precision@k, or domain expert validation</li>
</ul>
<p><strong>Types:</strong> Point anomalies (single unusual point), Contextual (unusual in context), Collective (group of unusual points)</p>""",
        "use_cases": ["Fraud detection", "Network intrusion detection", "Manufacturing defect detection", "Medical outlier flagging", "Log anomaly detection"],
        "chart": "anomaly_detection"
    },
    {
        "id": "gaussian_process", "label": "Gaussian Processes", "category": "unsupervised",
        "title": "Gaussian Processes", "subtitle": "Non-parametric Bayesian approach providing predictions with calibrated uncertainty estimates via kernel functions.",
        "theory": """<p><strong>Core Idea:</strong> Define a distribution over functions — any finite set of function values follows a joint Gaussian distribution.</p>
<ul>
  <li><strong>Prior:</strong> f(x) ~ GP(m(x), k(x,x')) — mean function m and covariance kernel k define the GP</li>
  <li><strong>Posterior:</strong> After observing data, update via Bayes' rule — still a GP with closed-form update</li>
  <li><strong>Prediction:</strong> Returns both mean μ* and variance σ² — natural uncertainty quantification</li>
  <li><strong>RBF Kernel:</strong> k(x,x') = σ²·exp(−||x−x'||²/2l²) — l = length scale, σ² = signal variance</li>
  <li><strong>Matérn Kernel:</strong> Controls smoothness via ν parameter — ν=1.5 (once differentiable), ν=2.5 (twice)</li>
  <li><strong>Noise:</strong> Add WhiteKernel or set alpha parameter — handles observation noise</li>
  <li><strong>Marginal Likelihood:</strong> log p(y|X) used to optimize hyperparameters — automatic relevance determination</li>
  <li><strong>Complexity:</strong> O(N³) for exact inference — approximations (sparse GP, inducing points) for large N</li>
</ul>
<p><strong>Use cases:</strong> Small data regime, safety-critical systems needing calibrated uncertainty, Bayesian optimization acquisition function</p>""",
        "use_cases": ["Bayesian optimization (AutoML)", "Geostatistics (Kriging)", "Small dataset regression", "Robotics motion planning", "Drug discovery"],
        "chart": "gaussian_process"
    },
    # ── NEURAL ─────────────────────────────────────────────────────────
    {
        "id": "neural_network", "label": "Neural Networks (MLP)", "category": "neural",
        "title": "Neural Networks (MLP)", "subtitle": "Universal function approximators via layered learned representations with backpropagation training.",
        "theory": """<p><strong>Core Idea:</strong> Stack layers of neurons, each computing a weighted sum followed by a non-linear activation function.</p>
<ul>
  <li><strong>Forward Pass:</strong> a⁽ˡ⁾ = f(W⁽ˡ⁾·a⁽ˡ⁻¹⁾ + b⁽ˡ⁾) — repeated for each layer</li>
  <li><strong>Backpropagation:</strong> Compute ∂L/∂W via chain rule — propagate gradient backwards through layers</li>
  <li><strong>Universal Approximation:</strong> 1 hidden layer with enough neurons approximates any continuous function (Hornik 1989)</li>
  <li><strong>Width vs Depth:</strong> Deeper networks learn hierarchical representations more efficiently than wide shallow ones</li>
  <li><strong>Optimizers:</strong> SGD (+ momentum), Adam (adaptive moments), RMSprop, AdaGrad, LAMB</li>
  <li><strong>Regularization:</strong> Dropout (randomly zero neurons), L2 weight decay, Batch Normalization, Early Stopping</li>
  <li><strong>Weight Init:</strong> He/Kaiming init for ReLU: N(0, 2/n_in); Xavier/Glorot for Sigmoid/Tanh: N(0, 2/(n_in+n_out))</li>
  <li><strong>Vanishing Gradient:</strong> Deep nets with Sigmoid/Tanh — gradients shrink → lower layers don't learn; ReLU + BatchNorm fixes this</li>
</ul>
<p><strong>Key Hyperparams:</strong> #layers, hidden sizes, activation, lr, batch size, epochs, dropout rate</p>""",
        "use_cases": ["Tabular data with complex interactions", "Image recognition (CNN)", "NLP (Transformers)", "Time series (LSTM)", "Game playing (RL)"],
        "chart": "neural_network"
    },
    {
        "id": "activations", "label": "Activation Functions", "category": "neural",
        "title": "Activation Functions", "subtitle": "Non-linearities enabling neural networks to learn complex representations — the heart of deep learning.",
        "theory": """<p><strong>Why non-linearity matters:</strong> Without activations, any stack of linear layers collapses to a single linear transformation — no representational power gained.</p>
<ul>
  <li><strong>Sigmoid:</strong> σ(z) = 1/(1+e⁻ᶻ) ∈ (0,1) — saturates at extremes (vanishing gradient), not zero-centered; use for binary output only</li>
  <li><strong>Tanh:</strong> (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) ∈ (−1,1) — zero-centered improvement over Sigmoid, still saturates; good for RNNs</li>
  <li><strong>ReLU:</strong> max(0,z) — sparse (50% neurons zero), fast gradient, no saturation for z&gt;0; dying ReLU for z&lt;0</li>
  <li><strong>Leaky ReLU:</strong> z if z&gt;0 else αz (α≈0.01) — fixes dying ReLU, small negative gradient kept</li>
  <li><strong>ELU:</strong> z if z&gt;0 else α(eᶻ−1) — smooth negative region, mean activations near zero, faster convergence</li>
  <li><strong>Swish:</strong> z·σ(z) — self-gated, non-monotonic, empirically outperforms ReLU in deep nets (Google Brain)</li>
  <li><strong>GELU:</strong> z·Φ(z) ≈ 0.5z(1+tanh(√(2/π)(z+0.044715z³))) — used in GPT, BERT, ViT; stochastic-like regularization</li>
  <li><strong>Softmax:</strong> exp(zᵢ)/Σexp(zⱼ) — multi-class output; numerical stability: subtract max(z) before exp</li>
</ul>""",
        "use_cases": ["Hidden layers → ReLU/GELU/Swish", "Binary output → Sigmoid", "Multi-class output → Softmax", "RNN gates → Tanh+Sigmoid", "Regression output → Linear"],
        "chart": "activations"
    },
    {
        "id": "attention_transformer", "label": "Attention & Transformers", "category": "neural",
        "title": "Attention Mechanisms & Transformers", "subtitle": "Self-attention computes pairwise relationships between all sequence positions — the foundation of modern LLMs.",
        "theory": """<p><strong>Scaled Dot-Product Attention:</strong> Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V</p>
<ul>
  <li><strong>Query, Key, Value:</strong> Q = XWᵍ, K = XWᵏ, V = XWᵛ — three learned projections of the same input</li>
  <li><strong>Scaling by √dₖ:</strong> Prevents dot products from growing too large → softmax saturation → vanishing gradient</li>
  <li><strong>Self-Attention:</strong> Q, K, V all come from same sequence — each position attends to all others simultaneously</li>
  <li><strong>Multi-Head Attention:</strong> Run h parallel attention heads with different projections — each learns different relationship types</li>
  <li><strong>Positional Encoding:</strong> sin/cos embeddings or learned positions — inject sequence order information (attention is permutation-invariant)</li>
  <li><strong>Transformer Block:</strong> Multi-Head Attn → Add&Norm → FFN (2-layer MLP, 4x dim) → Add&Norm</li>
  <li><strong>Residual Connections:</strong> x + Sublayer(LayerNorm(x)) — enable gradient flow in very deep networks</li>
  <li><strong>Complexity:</strong> O(N²·d) in sequence length N — quadratic bottleneck (Flash Attention, Sparse Attention address this)</li>
  <li><strong>BERT:</strong> Bidirectional encoder; <strong>GPT:</strong> Causal decoder (masked); <strong>T5:</strong> Encoder-decoder</li>
</ul>""",
        "use_cases": ["Large Language Models (GPT, LLaMA)", "Vision Transformers (ViT)", "Machine translation", "Document summarization", "Code generation"],
        "chart": "attention_transformer"
    },
    # ── CONCEPTS ───────────────────────────────────────────────────────
    {
        "id": "feature_engineering", "label": "Feature Engineering", "category": "concepts",
        "title": "Feature Engineering", "subtitle": "The art of transforming raw data into informative representations that make models learn better.",
        "theory": """<p><strong>Core Idea:</strong> Better features = better models, regardless of algorithm complexity.</p>
<ul>
  <li><strong>Scaling:</strong> StandardScaler (z-score) for distance-based models (SVM, KNN, PCA); MinMaxScaler for neural nets; RobustScaler for outlier-heavy data</li>
  <li><strong>Polynomial Features:</strong> Add xᵢ², xᵢxⱼ — captures non-linear relationships for linear models</li>
  <li><strong>Log Transform:</strong> Log(x) — compresses right-skewed distributions, stabilizes variance</li>
  <li><strong>Binning:</strong> Discretize continuous features — can reduce overfitting, capture non-linear relationships</li>
  <li><strong>Target Encoding:</strong> Replace categorical with mean of target — powerful but prone to leakage (use CV)</li>
  <li><strong>Date/Time:</strong> Extract hour, day-of-week, is_weekend, cyclic encoding (sin/cos of day/hour)</li>
  <li><strong>Interaction Features:</strong> Multiply or concatenate pairs — captures joint effects</li>
  <li><strong>Missing Values:</strong> Mean/median imputation, KNN imputation, model-based (iterative), indicator columns for MCAR</li>
  <li><strong>Imbalanced Classes:</strong> SMOTE (synthetic oversampling), ADASYN, class_weight='balanced', undersampling</li>
  <li><strong>Feature Selection:</strong> Filter (correlation), Wrapper (RFE), Embedded (Lasso, tree importance)</li>
</ul>""",
        "use_cases": ["Tabular data preprocessing", "NLP feature extraction", "Time series features", "Image feature engineering", "Competition data science"],
        "chart": "feature_engineering"
    },
    {
        "id": "regularization", "label": "Regularization", "category": "concepts",
        "title": "Regularization", "subtitle": "Techniques to reduce overfitting by constraining model complexity and improving generalization.",
        "theory": """<p><strong>Core Idea:</strong> Penalize model complexity in the loss function to discourage overfitting on training data.</p>
<ul>
  <li><strong>L2 (Ridge/Weight Decay):</strong> Loss + λΣβᵢ² — shrinks all coefficients towards zero equally; differentiable everywhere</li>
  <li><strong>L1 (Lasso):</strong> Loss + λΣ|βᵢ| — creates sparse solutions (exact zeros); automatic feature selection; not differentiable at 0</li>
  <li><strong>Elastic Net:</strong> ρ·L1 + (1−ρ)·L2 — groups correlated features (unlike Lasso which picks one)</li>
  <li><strong>Dropout:</strong> Randomly zero neurons with probability p — reduces co-adaptation; equivalent to ensemble of 2^N networks at inference use (1−p) scaling</li>
  <li><strong>Batch Normalization:</strong> Normalize layer inputs per mini-batch — stabilizes training, mild regularization, allows higher LR</li>
  <li><strong>Layer Normalization:</strong> Normalize across features (not batch) — standard in Transformers, works for variable batch sizes</li>
  <li><strong>Data Augmentation:</strong> Artificially expand training data (rotations, flips, crops, mixup, cutout)</li>
  <li><strong>Early Stopping:</strong> Monitor validation loss, stop when no improvement for N epochs — simple and very effective</li>
  <li><strong>Label Smoothing:</strong> Target = 1−ε for positive class, ε/K for others — prevents overconfidence</li>
  <li><strong>λ / Dropout rate tuning:</strong> Cross-validation or Bayesian optimization; often λ∈[1e-5, 1e-1]</li>
</ul>""",
        "use_cases": ["Polynomial regression", "Neural network training", "High-dimensional sparse data", "Small dataset regime", "NLP fine-tuning"],
        "chart": "regularization"
    },
    {
        "id": "bias_variance", "label": "Bias-Variance", "category": "concepts",
        "title": "Bias-Variance Tradeoff", "subtitle": "Fundamental decomposition of generalization error guiding model selection and complexity tuning.",
        "theory": """<p><strong>Expected Test Error = Bias² + Variance + Irreducible Noise</strong></p>
<ul>
  <li><strong>Bias:</strong> Error from wrong assumptions about the data-generating process — systematic, consistent misprediction</li>
  <li><strong>Variance:</strong> Error from sensitivity to fluctuations in the specific training set — model changes drastically with different data</li>
  <li><strong>Irreducible Noise:</strong> ε = inherent randomness/noise in y — fundamental lower bound; cannot be reduced by any model</li>
  <li><strong>High Bias (Underfitting):</strong> Simple model on complex data — train error ≈ val error, both high; learning curves plateau at high error</li>
  <li><strong>High Variance (Overfitting):</strong> Complex model memorizes training data — low train error, high val error (large gap); learning curves have wide gap</li>
  <li><strong>Optimal complexity:</strong> Balance at the "sweet spot" — use validation curve + cross-validation</li>
  <li><strong>Bagging → reduces Variance</strong> (averaging uncorrelated models)</li>
  <li><strong>Boosting → reduces Bias</strong> (sequential correction of errors)</li>
  <li><strong>More data:</strong> Reduces variance but not bias — won't help underfitting</li>
  <li><strong>Regularization:</strong> Increases bias, decreases variance — trades off the two</li>
</ul>""",
        "use_cases": ["Model selection", "Hyperparameter tuning", "Ensemble design", "Architecture search", "Training data size planning"],
        "chart": "bias_variance"
    },
    {
        "id": "model_evaluation", "label": "Model Evaluation", "category": "concepts",
        "title": "Model Evaluation", "subtitle": "Metrics, curves, and diagnostic tools for rigorously assessing classifier and regressor performance.",
        "theory": """<p><strong>Classification Metrics (from Confusion Matrix TP/FP/TN/FN):</strong></p>
<ul>
  <li><strong>Accuracy:</strong> (TP+TN)/N — misleading for imbalanced classes (99% baseline on 1% anomaly data)</li>
  <li><strong>Precision:</strong> TP/(TP+FP) — of predicted positives, what fraction are real? (minimize FP)</li>
  <li><strong>Recall (Sensitivity):</strong> TP/(TP+FN) — of real positives, what fraction detected? (minimize FN)</li>
  <li><strong>F1:</strong> 2·P·R/(P+R) — harmonic mean; balances precision and recall; good for imbalanced</li>
  <li><strong>F-beta:</strong> (1+β²)·P·R/(β²·P+R) — β&gt;1 weights recall more (medical diagnosis); β&lt;1 weights precision more</li>
  <li><strong>AUC-ROC:</strong> Area under TPR vs FPR curve across all thresholds — threshold-independent; 0.5 = random, 1.0 = perfect</li>
  <li><strong>AUC-PR:</strong> More informative than ROC for heavily imbalanced datasets — focuses on positive class</li>
  <li><strong>MCC:</strong> Matthews Correlation Coefficient — balanced metric even when classes are very imbalanced</li>
</ul>
<p><strong>Regression Metrics:</strong> MSE, RMSE, MAE, MAPE, R², Adjusted R², Huber loss</p>
<p><strong>Learning Curves:</strong> Plot train/val score vs training set size — diagnose bias/variance source</p>""",
        "use_cases": ["Model comparison", "Threshold selection for deployment", "Imbalanced class handling", "Production monitoring", "A/B testing models"],
        "chart": "model_evaluation"
    },
    {
        "id": "cross_validation", "label": "Cross Validation", "category": "concepts",
        "title": "Cross Validation", "subtitle": "Robust model evaluation using multiple train/test splits to estimate out-of-sample generalization performance.",
        "theory": """<p><strong>Core Idea:</strong> Estimate model performance on unseen data by rotating which portion of data is held out — unbiased estimate.</p>
<ul>
  <li><strong>K-Fold:</strong> Split into k equal folds; train on k−1, test on 1; rotate k times; final score = mean ± std</li>
  <li><strong>Stratified K-Fold:</strong> Maintains class proportions in each fold — ALWAYS use for classification</li>
  <li><strong>Leave-One-Out (LOOCV):</strong> k=N — minimum bias, maximum variance, computationally expensive; good for tiny datasets</li>
  <li><strong>Repeated K-Fold:</strong> Run K-fold multiple times with different splits — more reliable estimate, reduces randomness</li>
  <li><strong>Time Series CV (Walk-forward):</strong> Training set grows; never use future data to predict past — essential for temporal data</li>
  <li><strong>Group K-Fold:</strong> Ensures same group (e.g., patient) never appears in both train and test — prevents data leakage</li>
  <li><strong>Validation Curve:</strong> Plot train/val score vs hyperparameter value — find optimal without bias</li>
  <li><strong>Nested CV:</strong> Inner loop for hyperparameter tuning, outer for unbiased model evaluation — gold standard</li>
  <li><strong>Critical:</strong> Always apply CV to full pipeline including preprocessing (fit preprocessor on train only!)</li>
</ul>""",
        "use_cases": ["Hyperparameter tuning", "Model selection", "Feature selection", "Pipeline validation", "Performance reporting"],
        "chart": "cross_validation"
    },
    {
        "id": "hyperparameter_tuning", "label": "Hyperparameter Tuning", "category": "concepts",
        "title": "Hyperparameter Tuning", "subtitle": "Systematic search strategies to find optimal model configuration — from grid search to Bayesian optimization.",
        "theory": """<p><strong>Core Idea:</strong> Find the hyperparameter combination that maximizes validation performance without touching the test set.</p>
<ul>
  <li><strong>Grid Search:</strong> Exhaustive search over all combinations — guaranteed to find best in grid but exponential in #params</li>
  <li><strong>Random Search:</strong> Sample random combinations — often 60x more efficient than grid (Bergstra & Bengio 2012); better when few params matter</li>
  <li><strong>Bayesian Optimization:</strong> Build a probabilistic surrogate model (GP or TPE) of the objective; use acquisition function (EI, UCB) to select next point — converges faster than random</li>
  <li><strong>Successive Halving / Hyperband:</strong> Allocate resources (epochs/data) progressively — prune bad configs early; very efficient for NNs</li>
  <li><strong>ASHA:</strong> Asynchronous version of Hyperband — parallel workers, no synchronization barrier</li>
  <li><strong>Population-Based Training (PBT):</strong> Evolve hyperparameters during training — DeepMind's approach</li>
  <li><strong>NAS (Neural Architecture Search):</strong> Search over architecture choices (layers, connections) — DARTS, ENAS, one-shot methods</li>
  <li><strong>Tools:</strong> Optuna (TPE + pruning), Ray Tune (Hyperband/ASHA), Weights & Biases Sweeps, Google Vizier</li>
</ul>
<p><strong>Best Practice:</strong> Use Optuna with TPE sampler + MedianPruner. Log all trials. Set seed for reproducibility.</p>""",
        "use_cases": ["AutoML pipelines", "Neural network architecture search", "Production model optimization", "Kaggle competitions"],
        "chart": "hyperparameter_tuning"
    },
    {
        "id": "optimization", "label": "Optimization", "category": "concepts",
        "title": "Gradient Descent & Optimizers", "subtitle": "First and second-order optimization algorithms for minimizing loss functions in machine learning models.",
        "theory": """<p><strong>Core Idea:</strong> Iteratively update parameters in the direction of steepest descent of the loss function.</p>
<ul>
  <li><strong>BGD:</strong> Batch Gradient Descent — exact gradient over full dataset; slow, no noise; not used in practice for large data</li>
  <li><strong>SGD:</strong> Stochastic GD — gradient over 1 sample; noisy but fast; can escape local minima; learning rate crucial</li>
  <li><strong>Mini-Batch SGD:</strong> Gradient over B samples (B=32–256); balance of noise and stability; GPU-parallelizable</li>
  <li><strong>Momentum:</strong> v ← βv − η∇L; θ ← θ + v — accelerates in consistent direction, dampens oscillations (β≈0.9)</li>
  <li><strong>RMSprop:</strong> Divide lr by running avg of squared gradients — adapts lr per-parameter; good for RNNs</li>
  <li><strong>Adam:</strong> First moment (momentum) + second moment (RMSprop) estimates; m₁/(1−β₁ᵗ) bias correction; de facto default</li>
  <li><strong>AdaGrad:</strong> Accumulates squared gradients — learning rate decays, good for sparse data; dies out in deep nets</li>
  <li><strong>AdamW:</strong> Adam + decoupled L2 weight decay — better generalization; standard for Transformers</li>
  <li><strong>Learning Rate Schedules:</strong> Step decay, Cosine annealing, Warmup + Cosine, OneCycleLR, ReduceLROnPlateau</li>
  <li><strong>Gradient Clipping:</strong> Clip grad norm to max value — prevents exploding gradients in RNNs</li>
</ul>""",
        "use_cases": ["Neural network training", "Logistic regression", "Matrix factorization", "Reinforcement learning policy gradient", "GAN training"],
        "chart": "optimization"
    },
    {
        "id": "interpretability", "label": "Interpretability (XAI)", "category": "concepts",
        "title": "Model Interpretability (XAI)", "subtitle": "Techniques to explain, understand and trust black-box ML model predictions — essential for production AI.",
        "theory": """<p><strong>Why interpretability matters:</strong> Regulatory compliance (GDPR right to explanation), debugging, trust, and bias detection.</p>
<ul>
  <li><strong>Intrinsically Interpretable:</strong> Linear models, Logistic Regression, Decision Trees, Rule-based systems — directly inspectable</li>
  <li><strong>Feature Importance (MDI):</strong> Mean decrease in impurity across splits — fast but biased towards high-cardinality features</li>
  <li><strong>Permutation Importance:</strong> Measure accuracy drop when feature is randomly shuffled — model-agnostic, unbiased</li>
  <li><strong>Partial Dependence Plot (PDP):</strong> Marginal effect of one/two features on prediction — averaged over other features</li>
  <li><strong>Individual Conditional Expectation (ICE):</strong> PDP for individual samples — reveals heterogeneous effects</li>
  <li><strong>SHAP (SHapley Additive exPlanations):</strong> Game-theoretic attribution — fairly distributes prediction among features; satisfies Shapley axioms (efficiency, symmetry, linearity)</li>
  <li><strong>LIME:</strong> Local Interpretable Model-agnostic Explanations — fits interpretable linear model locally around prediction</li>
  <li><strong>Attention weights:</strong> Visualize which tokens the model attends to (note: not always faithful)</li>
  <li><strong>Grad-CAM:</strong> Gradient-weighted Class Activation Maps — visual explanations for CNNs</li>
</ul>
<p><strong>Tools:</strong> SHAP library, LIME, Alibi, InterpretML, Captum (PyTorch)</p>""",
        "use_cases": ["Regulatory compliance (finance, healthcare)", "Model debugging", "Bias and fairness auditing", "Customer-facing explanations", "Feature selection"],
        "chart": "interpretability"
    },
    {
        "id": "time_series", "label": "Time Series", "category": "concepts",
        "title": "Time Series Analysis & Forecasting", "subtitle": "Modeling temporal dependencies in sequential data — trend, seasonality, autocorrelation, and forecasting.",
        "theory": """<p><strong>Core components:</strong> Trend (long-term direction), Seasonality (periodic patterns), Cyclicality (irregular multi-year cycles), Residual (noise).</p>
<ul>
  <li><strong>Stationarity:</strong> Mean and variance constant over time — required by many models; test with ADF or KPSS test</li>
  <li><strong>Differencing:</strong> Δyₜ = yₜ − yₜ₋₁ — removes trend; seasonal differencing removes seasonality</li>
  <li><strong>ACF:</strong> Autocorrelation Function — correlation of series with its own lags; decays slowly for non-stationary data</li>
  <li><strong>PACF:</strong> Partial ACF — direct correlation at lag k, removing effect of shorter lags — used to identify AR order</li>
  <li><strong>ARIMA(p,d,q):</strong> AutoRegressive Integrated Moving Average — AR(p) + differencing d + MA(q)</li>
  <li><strong>SARIMA:</strong> Seasonal ARIMA with seasonal (P,D,Q,m) parameters — handles seasonal patterns</li>
  <li><strong>Prophet:</strong> Facebook's decomposable model: y(t) = g(t) + s(t) + h(t) + ε — handles holidays, changepoints</li>
  <li><strong>LSTM/GRU:</strong> Recurrent networks with gated memory — learn long-range dependencies</li>
  <li><strong>Temporal Fusion Transformer:</strong> State-of-art for multi-horizon, multi-variate forecasting</li>
  <li><strong>Walk-forward validation:</strong> Never use future data in training split — expanding or sliding window</li>
</ul>""",
        "use_cases": ["Stock price forecasting", "Demand forecasting", "Anomaly detection in metrics", "Weather prediction", "IoT sensor monitoring"],
        "chart": "time_series"
    },
    {
        "id": "feature_engineering2", "label": "Pipeline & MLOps", "category": "concepts",
        "title": "ML Pipeline & MLOps", "subtitle": "End-to-end production ML system design — from data ingestion to model monitoring and retraining.",
        "theory": """<p><strong>MLOps = DevOps + DataOps + ModelOps</strong> — practices for reliable, scalable production ML systems.</p>
<ul>
  <li><strong>Data Versioning:</strong> DVC, Delta Lake, LakeFS — track dataset versions like code versions (Git for data)</li>
  <li><strong>Feature Stores:</strong> Feast, Tecton, Hopsworks — centralized feature registry, eliminate train-serve skew</li>
  <li><strong>Experiment Tracking:</strong> MLflow, W&B, Comet — log params, metrics, artifacts, model versions</li>
  <li><strong>Model Registry:</strong> Stage transitions (Staging → Production) with metadata and lineage tracking</li>
  <li><strong>CI/CD for ML:</strong> Automated testing (data validation, model performance gates) before deployment</li>
  <li><strong>Serving:</strong> REST API (FastAPI/Flask), gRPC (TF Serving), batch inference (Spark), streaming (Kafka + model)</li>
  <li><strong>Model Monitoring:</strong> Track data drift (PSI, KS test), concept drift (performance degradation), prediction distribution</li>
  <li><strong>Retraining Triggers:</strong> Scheduled, performance-based, or drift-detected retraining pipelines</li>
  <li><strong>A/B Testing:</strong> Shadow deployment, canary release, multi-armed bandit for online model comparison</li>
  <li><strong>Tools:</strong> Vertex AI, SageMaker, Databricks, Kubeflow, Seldon, BentoML, Ray Serve</li>
</ul>""",
        "use_cases": ["Production model deployment", "Continuous training pipelines", "Model drift monitoring", "Multi-model management", "Real-time inference serving"],
        "chart": "pipeline_mlops"
    },
    # ── EVALUATION & METRICS ───────────────────────────────────────────────
    {
        "id": "eval_classification", "label": "Classification Metrics", "category": "evaluation",
        "title": "Classification Evaluation Metrics",
        "subtitle": "Comprehensive metrics for evaluating classifiers — from accuracy to AUC-ROC, F1, calibration, and beyond.",
        "theory": """<p><strong>Core Metrics:</strong> Accuracy = (TP+TN)/(TP+TN+FP+FN), Precision = TP/(TP+FP), Recall/Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP)</p>
<ul>
  <li><strong>F1 Score:</strong> 2·(P·R)/(P+R) — harmonic mean; F-beta generalizes with β weighting recall vs precision</li>
  <li><strong>ROC Curve:</strong> TPR vs FPR at varying thresholds; AUC = probability that model ranks random positive above random negative</li>
  <li><strong>Precision-Recall Curve:</strong> Better for imbalanced datasets — AUC-PR focuses on minority class performance</li>
  <li><strong>Log-Loss:</strong> −(1/N)Σ[y·log(p) + (1−y)·log(1−p)] — penalizes confident wrong predictions heavily</li>
  <li><strong>Matthews Correlation Coefficient (MCC):</strong> √[(TP·TN − FP·FN) / ((TP+FP)(TP+FN)(TN+FP)(TN+FN))] — balanced metric for imbalanced classes</li>
  <li><strong>Cohen's Kappa:</strong> κ = (p_o − p_e)/(1 − p_e) — measures agreement beyond chance</li>
  <li><strong>Calibration:</strong> ECE (Expected Calibration Error) — measures alignment between predicted probabilities and observed frequencies</li>
  <li><strong>Brier Score:</strong> mean squared difference between predicted probability and actual outcome</li>
  <li><strong>Class Imbalance Strategies:</strong> SMOTE, class weights, threshold tuning, resampling</li>
</ul>""",
        "use_cases": ["Medical diagnosis evaluation", "Fraud detection scoring", "Model selection", "Imbalanced classification", "Probability calibration"],
        "chart": "eval_classification"
    },
    {
        "id": "eval_regression", "label": "Regression Metrics", "category": "evaluation",
        "title": "Regression Evaluation Metrics",
        "subtitle": "Metrics for measuring continuous prediction quality — from simple error measures to information criteria.",
        "theory": """<p><strong>Core Error Metrics:</strong></p>
<ul>
  <li><strong>MAE:</strong> (1/N)Σ|yᵢ−ŷᵢ| — robust to outliers, same units as target</li>
  <li><strong>MSE:</strong> (1/N)Σ(yᵢ−ŷᵢ)² — penalizes large errors quadratically</li>
  <li><strong>RMSE:</strong> √MSE — interpretable in target units, sensitive to outliers</li>
  <li><strong>R² (Coefficient of Determination):</strong> 1 − SS_res/SS_tot — proportion of variance explained; 1=perfect, 0=mean baseline</li>
  <li><strong>Adjusted R²:</strong> 1 − (1−R²)(N−1)/(N−p−1) — penalizes for adding irrelevant features</li>
  <li><strong>MAPE:</strong> (100/N)Σ|yᵢ−ŷᵢ|/|yᵢ| — percentage error, undefined if yᵢ=0</li>
  <li><strong>SMAPE:</strong> (200/N)Σ|yᵢ−ŷᵢ|/(|yᵢ|+|ŷᵢ|) — symmetric, bounded 0–200%</li>
  <li><strong>Huber Loss:</strong> quadratic for small errors, linear for large — best of MSE and MAE</li>
  <li><strong>Quantile Loss:</strong> useful for prediction intervals and asymmetric costs</li>
  <li><strong>AIC/BIC:</strong> model selection criteria penalizing complexity — AIC=2k−2ln(L), BIC=k·ln(N)−2ln(L)</li>
  <li><strong>Residual Diagnostics:</strong> normality (Shapiro-Wilk), homoscedasticity (Breusch-Pagan), autocorrelation (Durbin-Watson)</li>
</ul>""",
        "use_cases": ["House price prediction", "Sales forecasting", "Model comparison", "Prediction intervals", "Feature selection"],
        "chart": "eval_regression"
    },
    {
        "id": "cluster_evaluation", "label": "Clustering Metrics", "category": "evaluation",
        "title": "Clustering Evaluation Metrics",
        "subtitle": "Internal and external metrics for evaluating unsupervised clustering quality without or with ground truth.",
        "theory": """<p><strong>Internal Metrics (no ground truth):</strong></p>
<ul>
  <li><strong>Silhouette Score:</strong> (b−a)/max(a,b) where a=intra-cluster dist, b=nearest-cluster dist; range [−1,1]</li>
  <li><strong>Davies-Bouldin Index:</strong> ratio of within-cluster scatter to between-cluster separation — lower is better</li>
  <li><strong>Calinski-Harabasz Index:</strong> ratio of between/within cluster variance — higher is better</li>
  <li><strong>Inertia (WCSS):</strong> sum of squared distances to cluster centers — elbow method for K selection</li>
</ul>
<p><strong>External Metrics (with ground truth):</strong></p>
<ul>
  <li><strong>Adjusted Rand Index (ARI):</strong> similarity between two clusterings corrected for chance; 1=perfect, 0=random, can be negative</li>
  <li><strong>Normalized Mutual Information (NMI):</strong> I(Y;C)/√(H(Y)·H(C)) — information-theoretic measure, [0,1]</li>
  <li><strong>V-measure, Fowlkes-Mallows:</strong> additional external validation metrics</li>
  <li><strong>Hopkins Statistic:</strong> tests whether data has cluster tendency vs random distribution</li>
  <li><strong>Dunn Index:</strong> ratio of minimum inter-cluster distance to maximum intra-cluster diameter</li>
</ul>""",
        "use_cases": ["Customer segmentation validation", "Document clustering", "Image segmentation", "Anomaly cluster quality", "K selection"],
        "chart": "cluster_evaluation"
    },
    # ── DEEP LEARNING ──────────────────────────────────────────────────────
    {
        "id": "cnn", "label": "CNN", "category": "deep_learning",
        "title": "Convolutional Neural Networks",
        "subtitle": "Hierarchical feature learning through convolution, pooling, and deep stacking — the backbone of computer vision.",
        "theory": """<p><strong>Core Idea:</strong> Local kernel detects spatial patterns; weight sharing drastically reduces parameters vs fully-connected layers.</p>
<ul>
  <li><strong>Convolution:</strong> (f★g)(i,j) = ΣΣ f(m,n)·g(i−m,j−n) — sliding kernel extracts local features</li>
  <li><strong>Output size:</strong> (W−K+2P)/S + 1 where W=input width, K=kernel size, P=padding, S=stride</li>
  <li><strong>Parameters:</strong> K×K×C_in×C_out + C_out (bias) per conv layer — weight sharing drastically reduces params</li>
  <li><strong>Pooling:</strong> Max pooling (translation invariance), Average pooling (smooth downsampling), Global Average Pooling (GAP for classification head)</li>
  <li><strong>Receptive Field:</strong> grows with depth — layer l has RF = l×(K−1) + 1 for stride-1 networks</li>
  <li><strong>Architectures:</strong> AlexNet (2012) → VGG (depth) → ResNet (skip connections) → EfficientNet (compound scaling) → ViT (patches)</li>
  <li><strong>Skip Connections (ResNet):</strong> F(x) + x — gradient highway preventing vanishing gradients; enables 100+ layer training</li>
  <li><strong>Depthwise Separable Convolutions (MobileNet):</strong> spatial + pointwise — 8-9x parameter reduction</li>
  <li><strong>Batch Normalization:</strong> normalize activations per batch → stabilizes training, acts as regularizer</li>
  <li><strong>Data Augmentation:</strong> random crop, flip, rotation, color jitter, cutout, mixup — key for generalization</li>
</ul>""",
        "use_cases": ["Image classification", "Object detection", "Medical imaging", "Video understanding", "Face recognition"],
        "chart": "cnn"
    },
    {
        "id": "rnn_lstm", "label": "RNN / LSTM", "category": "deep_learning",
        "title": "Recurrent Networks — RNN, LSTM, GRU",
        "subtitle": "Sequential data modeling via recurrent connections — with gated memory cells to handle long-range dependencies.",
        "theory": """<p><strong>Core Idea:</strong> Hidden state carries memory through time; gates control information flow to address vanishing gradients.</p>
<ul>
  <li><strong>Vanilla RNN:</strong> hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b) — hidden state carries memory through time</li>
  <li><strong>Vanishing Gradient:</strong> gradients multiply through T timesteps → ∂L/∂h₀ involves W^T → vanishes or explodes for large T</li>
  <li><strong>LSTM Gates:</strong> Forget (fₜ), Input (iₜ), Cell update (C̃ₜ), Output (oₜ) — sigmoid gates control information flow</li>
  <li><strong>LSTM Cell:</strong> Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ — additive cell update prevents vanishing gradient</li>
  <li><strong>GRU:</strong> Simplified 2-gate version (reset, update) — comparable performance, fewer parameters than LSTM</li>
  <li><strong>Bidirectional RNN:</strong> process sequence forward and backward — captures both past and future context</li>
  <li><strong>Stacked RNNs:</strong> multiple layers for hierarchical feature extraction</li>
  <li><strong>Sequence-to-Sequence:</strong> encoder LSTM encodes input → decoder LSTM generates output (machine translation)</li>
  <li><strong>Attention in RNNs:</strong> context vector = Σαᵢhᵢ where αᵢ are learned attention weights — selectively focus on input positions</li>
  <li><strong>BPTT (Backprop Through Time):</strong> unroll T steps and apply standard backprop — truncated BPTT for long sequences</li>
</ul>""",
        "use_cases": ["Text generation", "Machine translation", "Speech recognition", "Time series prediction", "Sentiment analysis"],
        "chart": "rnn_lstm"
    },
    {
        "id": "gan", "label": "GANs", "category": "deep_learning",
        "title": "Generative Adversarial Networks",
        "subtitle": "Two-player minimax game between generator and discriminator — learning to synthesize realistic data distributions.",
        "theory": """<p><strong>Objective:</strong> min_G max_D V(D,G) = 𝔼[log D(x)] + 𝔼[log(1 − D(G(z)))]</p>
<ul>
  <li><strong>Generator G:</strong> maps noise z~p_z(z) to data space; tries to fool D into outputting high probability</li>
  <li><strong>Discriminator D:</strong> binary classifier distinguishing real from generated; provides gradient signal to G</li>
  <li><strong>Training:</strong> alternate D and G updates — D maximizes log(D(x))+log(1−D(G(z))), G minimizes log(1−D(G(z)))</li>
  <li><strong>Mode Collapse:</strong> G produces limited variety — addressed by minibatch discrimination, unrolled GANs</li>
  <li><strong>Training Instability:</strong> D too strong → vanishing gradients for G; D too weak → meaningless gradient</li>
  <li><strong>DCGAN:</strong> convolutional GAN with batch norm, LeakyReLU, transposed convolutions — stable training</li>
  <li><strong>Wasserstein GAN (WGAN):</strong> Earth Mover distance as loss — removes log saturating, uses critic (no sigmoid), 1-Lipschitz via gradient penalty (WGAN-GP)</li>
  <li><strong>Conditional GAN (cGAN):</strong> condition both G and D on class label y — control generation</li>
  <li><strong>Progressive GAN / StyleGAN:</strong> grow resolution progressively; style-based generator with AdaIN for high-quality faces</li>
</ul>""",
        "use_cases": ["Image synthesis", "Data augmentation", "Style transfer", "Super-resolution", "Anomaly detection"],
        "chart": "gan"
    },
    {
        "id": "vae", "label": "VAEs", "category": "deep_learning",
        "title": "Variational Autoencoders",
        "subtitle": "Probabilistic generative model learning a structured latent space via variational inference and reparameterization.",
        "theory": """<p><strong>Generative Model:</strong> p(x,z) = p(x|z)·p(z) — latent z~N(0,I), decoder p(x|z) generates data</p>
<ul>
  <li><strong>Encoder:</strong> q_φ(z|x) ≈ p(z|x) — approximate posterior (inference network) outputs μ and σ</li>
  <li><strong>ELBO:</strong> L = 𝔼_q[log p(x|z)] − KL(q(z|x) || p(z)) — reconstruction − KL regularization</li>
  <li><strong>Reparameterization Trick:</strong> z = μ + σ⊙ε, ε~N(0,I) — enables backprop through sampling</li>
  <li><strong>KL Divergence:</strong> KL(N(μ,σ²) || N(0,1)) = ½Σ(μ² + σ² − 1 − log σ²) — closed form for Gaussians</li>
  <li><strong>β-VAE:</strong> L = reconstruction − β·KL — larger β encourages disentangled representations</li>
  <li><strong>VQ-VAE:</strong> discrete latent codes via vector quantization — no KL, enables DALL-E style generation</li>
  <li><strong>Posterior Collapse:</strong> decoder ignores z if too powerful — KL annealing, free bits, or LSTM decoder prevention</li>
  <li><strong>Latent Space Interpolation:</strong> lerp between zᵢ and zⱼ in latent space → smooth image morphing</li>
  <li><strong>Applications:</strong> generation, compression, anomaly detection (high reconstruction error = anomaly)</li>
</ul>""",
        "use_cases": ["Image generation", "Anomaly detection", "Drug discovery", "Disentangled representation", "Data imputation"],
        "chart": "vae"
    },
    {
        "id": "nlp_fundamentals", "label": "NLP Fundamentals", "category": "deep_learning",
        "title": "NLP Fundamentals",
        "subtitle": "Core text processing techniques — from tokenization and embeddings to sequence labeling and language modeling.",
        "theory": """<p><strong>Core Concepts:</strong> Tokenization, embeddings, and language models form the backbone of NLP pipelines.</p>
<ul>
  <li><strong>Tokenization:</strong> word-level, sub-word (BPE, WordPiece, SentencePiece), character-level — BPE builds vocabulary greedily by merging frequent pairs</li>
  <li><strong>TF-IDF:</strong> TF(t,d)·IDF(t) = (count/doc_len)·log(N/df_t) — sparse bag-of-words representation</li>
  <li><strong>Word2Vec:</strong> Skip-gram (predict context from word) or CBOW (predict word from context); learns dense d-dim vectors</li>
  <li><strong>Negative Sampling:</strong> approximate softmax by sampling k negatives — makes training tractable</li>
  <li><strong>GloVe:</strong> global co-occurrence matrix factorization — combines local (Word2Vec) and global (LSA) statistics</li>
  <li><strong>FastText:</strong> sub-word embeddings via character n-grams — handles OOV words, morphologically rich languages</li>
  <li><strong>NER:</strong> sequence labeling with BIO tagging — BERT-based NER is standard</li>
  <li><strong>Language Modeling:</strong> P(wₜ|w₁...wₜ₋₁) — perplexity = exp(−(1/N)Σlog P(wᵢ)) measures model quality</li>
  <li><strong>BERT:</strong> bidirectional transformer pre-trained with MLM + NSP; fine-tuned with CLS token for classification</li>
  <li><strong>Sentence Embeddings:</strong> SBERT, Universal Sentence Encoder — semantic similarity via cosine distance</li>
</ul>""",
        "use_cases": ["Text classification", "Named entity recognition", "Semantic search", "Machine translation", "Summarization"],
        "chart": "nlp_fundamentals"
    },
    {
        "id": "transfer_learning", "label": "Transfer Learning", "category": "deep_learning",
        "title": "Transfer Learning & Fine-Tuning",
        "subtitle": "Leveraging pre-trained model knowledge for new tasks — from frozen features to full fine-tuning and prompt tuning.",
        "theory": """<p><strong>Core Idea:</strong> Weights learned on a large source task encode general features useful for target tasks.</p>
<ul>
  <li><strong>Feature Extraction:</strong> freeze pre-trained layers, train only new head — fast, few parameters, good for small datasets</li>
  <li><strong>Fine-tuning:</strong> unfreeze some or all layers, train with small LR — better accuracy, risk of catastrophic forgetting</li>
  <li><strong>Domain Gap:</strong> source and target domain similarity matters — ImageNet→medical: fine-tune more; text→code: domain-specific pre-training</li>
  <li><strong>Discriminative Learning Rates (ULMFiT):</strong> different LR per layer group — key insight for NLP transfer learning</li>
  <li><strong>Catastrophic Forgetting:</strong> sequential training erases previous task knowledge — addressed by Elastic Weight Consolidation (EWC)</li>
  <li><strong>PEFT (Parameter-Efficient Fine-Tuning):</strong> LoRA (low-rank adapters), Prefix Tuning, Prompt Tuning, Adapter layers — fine-tune &lt;1% of params</li>
  <li><strong>LoRA:</strong> ΔW = BA where B∈ℝ^(d×r), A∈ℝ^(r×k), rank r&lt;&lt;d — efficient adaptation of LLMs</li>
  <li><strong>Zero-shot / Few-shot:</strong> GPT-style models generalize from task description alone without gradient updates</li>
</ul>""",
        "use_cases": ["Medical image classification", "Domain-specific NLP", "Low-resource learning", "LLM fine-tuning", "Satellite imagery analysis"],
        "chart": "transfer_learning"
    },
    {
        "id": "reinforcement_learning", "label": "Reinforcement Learning", "category": "deep_learning",
        "title": "Reinforcement Learning",
        "subtitle": "Learning optimal policies through trial-and-error interaction with environments — from Q-learning to PPO and AlphaZero.",
        "theory": """<p><strong>MDP:</strong> (S, A, P, R, γ) — State space, Action space, Transition probabilities, Reward function, discount factor</p>
<ul>
  <li><strong>Return:</strong> Gₜ = Σₖ₌₀ γᵏRₜ₊ₖ₊₁ — discounted sum of future rewards; γ∈[0,1] trades off immediate vs future</li>
  <li><strong>Bellman Equation:</strong> V(s) = maxₐ[R(s,a) + γΣₛ' P(s'|s,a)V(s')] — recursive value decomposition</li>
  <li><strong>Q-Learning:</strong> Q(s,a) ← Q(s,a) + α[r + γ maxₐ' Q(s',a') − Q(s,a)] — off-policy TD learning</li>
  <li><strong>DQN:</strong> neural network approximates Q-function; experience replay (decorrelate transitions) + target network (stabilize updates)</li>
  <li><strong>Policy Gradient:</strong> ∇J(θ) = 𝔼[∇log π_θ(a|s)·Gₜ] — REINFORCE algorithm; high variance, baseline subtraction helps</li>
  <li><strong>Actor-Critic:</strong> actor (policy) + critic (value function) — critic provides low-variance baseline for actor updates</li>
  <li><strong>PPO:</strong> clip objective L^CLIP = min(r·Â, clip(r, 1-ε, 1+ε)·Â) — prevents large policy updates</li>
  <li><strong>MCTS (AlphaZero):</strong> tree search guided by neural policy + value network — self-play improves both networks</li>
  <li><strong>Multi-Armed Bandit:</strong> simplified RL with no state transitions — UCB, Thompson Sampling, ε-greedy exploration</li>
</ul>""",
        "use_cases": ["Game playing (Chess, Go, Atari)", "Robot control", "Recommendation systems", "Trading algorithms", "RLHF for LLMs"],
        "chart": "reinforcement_learning"
    },
    {
        "id": "bandits_active", "label": "Bandits & Active Learning", "category": "deep_learning",
        "title": "Bandits, Active Learning & Exploration",
        "subtitle": "Sequential decision-making under uncertainty — balancing exploration vs exploitation and querying labels efficiently.",
        "theory": """<p><strong>Multi-Armed Bandit:</strong> K arms with unknown reward distributions μ₁...μₖ; goal = maximize cumulative reward</p>
<ul>
  <li><strong>Regret:</strong> Rₙ = n·μ* − Σᵢ μᵢ·nᵢ — gap between optimal and achieved reward</li>
  <li><strong>ε-Greedy:</strong> with prob ε explore random arm, else exploit best known — simple but fixed exploration rate</li>
  <li><strong>UCB1:</strong> select arm maximizing μ̂ᵢ + √(2ln(t)/nᵢ) — confidence interval exploration; O(log n) regret</li>
  <li><strong>Thompson Sampling:</strong> sample θᵢ from posterior Beta(αᵢ,βᵢ), pick argmax θᵢ — Bayesian, empirically excellent</li>
  <li><strong>Contextual Bandits:</strong> action depends on context x — LinUCB, NeuralUCB, reward = fθ(x,a)</li>
  <li><strong>Active Learning:</strong> query oracle for labels of most informative samples — reduces labeling cost</li>
  <li><strong>Uncertainty Sampling:</strong> pick sample x* = argmax 1−max_c P(y=c|x) — most uncertain prediction</li>
  <li><strong>Query Strategies:</strong> uncertainty sampling (least confident), query by committee, expected model change, core-sets</li>
  <li><strong>Pool-based vs stream-based active learning; stopping criteria</strong></li>
</ul>""",
        "use_cases": ["A/B testing", "Clinical trials", "Recommendation exploration", "Label-efficient learning", "Online advertising"],
        "chart": "bandits_active"
    },
    {
        "id": "semi_self_supervised", "label": "Self-Supervised Learning", "category": "deep_learning",
        "title": "Self-Supervised & Semi-Supervised Learning",
        "subtitle": "Learning rich representations from unlabeled data via pretext tasks and contrastive objectives.",
        "theory": """<p><strong>Core Idea:</strong> Labels are expensive; self-supervised pretext tasks create supervision from the data structure itself.</p>
<ul>
  <li><strong>Semi-supervised:</strong> leverage large unlabeled + small labeled data — pseudolabels, consistency regularization, graph-based</li>
  <li><strong>Self-training:</strong> train on labeled, predict unlabeled, add high-confidence predictions as pseudo-labels — iterative</li>
  <li><strong>Consistency Regularization:</strong> model predictions should be invariant to perturbations — MixMatch, FixMatch, UDA</li>
  <li><strong>FixMatch:</strong> only unlabeled samples where strong-aug prediction matches weak-aug (above threshold) used for pseudo-labeling</li>
  <li><strong>Contrastive Learning (SimCLR):</strong> pull together two augmentations of same image (positive pair), push apart others (negatives)</li>
  <li><strong>NT-Xent Loss:</strong> L = −log[exp(sim(zᵢ,zⱼ)/τ) / Σₖ≠ᵢ exp(sim(zᵢ,zₖ)/τ)] — temperature τ controls concentration</li>
  <li><strong>MoCo:</strong> momentum encoder maintains large queue of negatives — efficient contrastive learning</li>
  <li><strong>BYOL/SimSiam:</strong> self-distillation without negatives — online network + target network (EMA or stop-gradient)</li>
  <li><strong>DINO/DINOv2:</strong> self-distillation with ViT — emerges semantic segmentation without any supervision</li>
  <li><strong>Masked Autoencoders (MAE):</strong> mask 75% of image patches, reconstruct — scalable self-supervised ViT pre-training</li>
</ul>""",
        "use_cases": ["Limited label scenarios", "Medical imaging", "NLP pre-training", "Representation learning", "Transfer learning foundation"],
        "chart": "semi_self_supervised"
    },
    {
        "id": "pgm", "label": "Probabilistic Models", "category": "deep_learning",
        "title": "Probabilistic Graphical Models",
        "subtitle": "Structured probabilistic representations using graphs — Bayesian Networks, MRFs, HMMs, and CRFs.",
        "theory": """<p><strong>Core Idea:</strong> Encode conditional independence structure in a graph to factorize high-dimensional distributions.</p>
<ul>
  <li><strong>Bayesian Network:</strong> DAG where nodes = variables, edges = conditional dependencies; P(X) = ΠP(Xᵢ|Parents(Xᵢ))</li>
  <li><strong>d-separation:</strong> determines conditional independence from graph structure — key for causal reasoning</li>
  <li><strong>Markov Random Field (MRF):</strong> undirected graphical model; P(X) ∝ Πᵢ φᵢ(Cᵢ) — product of potential functions over cliques</li>
  <li><strong>Hidden Markov Model (HMM):</strong> latent Markov chain → observation model; forward-backward, Viterbi, Baum-Welch</li>
  <li><strong>Viterbi Algorithm:</strong> dynamic programming for MAP sequence decoding — O(T·K²) where K=states, T=sequence length</li>
  <li><strong>CRF:</strong> discriminative MRF conditioned on input; P(Y|X) ∝ exp(Σλₖfₖ(Y,X)) — better than HMM for NLP</li>
  <li><strong>Variational Inference:</strong> approximate intractable posterior p(z|x) with q(z) by minimizing KL(q||p) — ELBO objective</li>
  <li><strong>MCMC (Gibbs, MH):</strong> sample from posterior when VI is too restrictive — slower but exact in the limit</li>
  <li><strong>LDA:</strong> topic model; doc ~ mixture of topics, topic ~ distribution over words</li>
  <li><strong>EM Algorithm:</strong> alternate E-step (estimate latents) and M-step (maximize parameters) — for GMMs, HMMs, LDA</li>
</ul>""",
        "use_cases": ["Medical diagnosis", "Speech recognition", "Topic modeling", "Sequence labeling", "Causal discovery"],
        "chart": "pgm"
    },
    {
        "id": "object_detection", "label": "Object Detection", "category": "deep_learning",
        "title": "Object Detection & Segmentation",
        "subtitle": "Localizing and classifying objects in images — from anchor-based R-CNN to anchor-free DETR and instance segmentation.",
        "theory": """<p><strong>Core Pipeline:</strong> Backbone (feature extraction) → Neck (multi-scale fusion) → Head (detect/segment).</p>
<ul>
  <li><strong>Two-stage:</strong> Region Proposal Network (RPN) → RoI features → classify + regress (Faster R-CNN)</li>
  <li><strong>One-stage:</strong> direct prediction without proposals — YOLO, SSD, RetinaNet; faster but historically less accurate</li>
  <li><strong>Anchors:</strong> predefined boxes at multiple scales/ratios; predict offset (Δx,Δy,Δw,Δh) + objectness</li>
  <li><strong>IoU:</strong> |A∩B|/|A∪B| — standard localization metric; threshold 0.5 for detection</li>
  <li><strong>NMS:</strong> suppress overlapping predictions, keep highest-confidence box</li>
  <li><strong>mAP:</strong> mean AP across IoU thresholds and classes — COCO mAP uses 0.5:0.95 range (10 thresholds)</li>
  <li><strong>Focal Loss:</strong> FL = −αₜ(1−pₜ)ᵞlog(pₜ) — down-weights easy negatives (class imbalance in detection)</li>
  <li><strong>FPN:</strong> multi-scale features from single backbone — detects objects at all scales</li>
  <li><strong>DETR:</strong> end-to-end detection with transformer + bipartite matching loss — no anchors or NMS</li>
  <li><strong>Instance Segmentation (Mask R-CNN):</strong> adds mask head to Faster R-CNN — pixel-level instance masks</li>
  <li><strong>Semantic Segmentation:</strong> FCN, U-Net, DeepLab (atrous convolution + ASPP) — pixel-wise class labels</li>
</ul>""",
        "use_cases": ["Autonomous driving", "Medical imaging", "Retail shelf analytics", "Surveillance", "Robotics"],
        "chart": "object_detection"
    },
    {
        "id": "gnn", "label": "Graph Neural Networks", "category": "deep_learning",
        "title": "Graph Neural Networks (GNNs)",
        "subtitle": "Deep learning on graph-structured data — message passing, spectral and spatial methods for node, edge, and graph tasks.",
        "theory": """<p><strong>Core Framework:</strong> Message passing aggregates neighbor information to update node representations.</p>
<ul>
  <li><strong>Message Passing:</strong> hᵥ⁽ˡ⁺¹⁾ = UPDATE(hᵥ⁽ˡ⁾, AGGREGATE({hᵤ⁽ˡ⁾: u∈N(v)})) — general GNN framework</li>
  <li><strong>GCN:</strong> hᵥ⁽ˡ⁺¹⁾ = σ(D̃⁻¹/²ÃD̃⁻¹/²H⁽ˡ⁾W⁽ˡ⁾) — spectral approach with normalized Laplacian</li>
  <li><strong>GraphSAGE:</strong> sample and aggregate neighbors — inductive, scales to large graphs</li>
  <li><strong>GAT:</strong> αᵤᵥ = softmax(LeakyReLU(aᵀ[Whᵤ||Whᵥ])) — weighted aggregation via learned attention</li>
  <li><strong>GIN:</strong> hᵥ⁽ˡ⁾ = MLP((1+ε)·hᵥ⁽ˡ⁻¹⁾ + Σhᵤ⁽ˡ⁻¹⁾) — maximally expressive (as powerful as WL test)</li>
  <li><strong>Over-smoothing:</strong> deep GNNs converge to same representation — residual connections, jumping knowledge networks</li>
  <li><strong>Graph Transformer:</strong> self-attention on nodes with edge features and positional encoding</li>
  <li><strong>Task Types:</strong> Node Classification, Link Prediction, Graph Classification</li>
  <li><strong>Heterogeneous Graphs:</strong> multiple node/edge types — HAN, RGCN for knowledge graphs</li>
  <li><strong>Point Cloud (3D):</strong> PointNet, PointNet++ treat 3D points as sets/graphs</li>
</ul>""",
        "use_cases": ["Molecular property prediction", "Social network analysis", "Knowledge graphs", "Recommendation systems", "Traffic forecasting"],
        "chart": "gnn"
    },
    # ── ADVANCED AI ────────────────────────────────────────────────────────
    {
        "id": "loss_functions", "label": "Loss Functions", "category": "ai_advanced",
        "title": "Loss Functions in Depth",
        "subtitle": "Complete reference for regression, classification, ranking, contrastive, and generative loss functions.",
        "theory": """<p><strong>Core Idea:</strong> Loss functions define what the model optimizes — choosing the right loss shapes the solution space.</p>
<ul>
  <li><strong>Regression:</strong> MSE (L2), MAE (L1), Huber (smooth L1), Log-Cosh, Quantile loss, Tweedie loss</li>
  <li><strong>Classification:</strong> BCE (binary cross-entropy), CCE (categorical CE), Focal Loss, Label Smoothing</li>
  <li><strong>Label Smoothing:</strong> ỹ = (1−ε)·y + ε/K — soft targets reduce overconfidence, improve calibration</li>
  <li><strong>Triplet Loss:</strong> max(0, d(a,p) − d(a,n) + margin) — learns metric space; anchor, positive, negative</li>
  <li><strong>Contrastive Loss:</strong> (1−Y)·½d² + Y·½max(0,m−d)² — pulls similar pairs together, pushes different apart</li>
  <li><strong>InfoNCE:</strong> L = −log[exp(sim(q,k+)/τ) / Σexp(sim(q,kᵢ)/τ)] — self-supervised contrastive; basis for CLIP</li>
  <li><strong>Hinge Loss (SVM):</strong> max(0, 1−yf(x)) — convex surrogate for 0-1 loss</li>
  <li><strong>KL Divergence:</strong> DKL(P||Q) = ΣP(x)log(P(x)/Q(x)) — asymmetric; basis for VAE loss</li>
  <li><strong>Dice Loss:</strong> 1 − 2|A∩B|/(|A|+|B|) — for segmentation with class imbalance; directly optimizes IoU-like metric</li>
  <li><strong>IoU Loss:</strong> 1 − IoU — differentiable, for object detection regression</li>
</ul>""",
        "use_cases": ["Model design", "Imbalanced learning", "Metric learning", "Generative models", "Dense prediction tasks"],
        "chart": "loss_functions"
    },
    {
        "id": "data_preprocessing", "label": "Data Preprocessing", "category": "ai_advanced",
        "title": "Data Preprocessing & Cleaning",
        "subtitle": "Transforming raw data into ML-ready format — scaling, encoding, imputation, outlier handling, and splits.",
        "theory": """<p><strong>Core Principle:</strong> Garbage in, garbage out — preprocessing quality directly determines model quality.</p>
<ul>
  <li><strong>Scaling:</strong> StandardScaler (z = (x−μ)/σ), MinMaxScaler (0-1), RobustScaler (median/IQR — outlier resistant), L2 Normalize</li>
  <li><strong>Encoding:</strong> OrdinalEncoder (ordered categories), OneHotEncoder (nominal), TargetEncoder (replace with mean target), HashingEncoder</li>
  <li><strong>Missing Values:</strong> mean/median/mode imputation, MICE (multiple imputation), KNN imputation, indicator features for MAR patterns</li>
  <li><strong>MCAR/MAR/MNAR:</strong> Missing Completely At Random, At Random (depends on observed), Not At Random (depends on missing value itself)</li>
  <li><strong>Outliers:</strong> IQR method (1.5×IQR), Z-score (|z|&gt;3), Isolation Forest, DBSCAN — winsorize or remove</li>
  <li><strong>Skewness:</strong> log(x+1), Box-Cox λ: y^λ, Yeo-Johnson (handles negatives), sqrt — for approximately normal distributions</li>
  <li><strong>Train/Val/Test Split:</strong> temporal aware split for time series, stratified split for classification, group split for leakage prevention</li>
  <li><strong>Data Leakage:</strong> fit scalers/encoders on train only, apply to test — never use test statistics in preprocessing</li>
  <li><strong>Class Imbalance:</strong> oversample minority (SMOTE, ADASYN), undersample majority (Tomek links), class weights, ensemble methods</li>
  <li><strong>Feature Crossing:</strong> polynomial features, manual domain crosses, embedding tables for high-cardinality categoricals</li>
</ul>""",
        "use_cases": ["Feature pipelines", "Data quality improvement", "Preprocessing for neural nets", "Categorical encoding", "Handling real-world messy data"],
        "chart": "data_preprocessing"
    },
    {
        "id": "statistical_tests", "label": "Statistical Tests", "category": "ai_advanced",
        "title": "Statistical Tests & Hypothesis Testing",
        "subtitle": "Rigorous framework for making data-driven decisions — from t-tests to chi-squared, ANOVA, and A/B testing.",
        "theory": """<p><strong>Framework:</strong> H₀ (null hypothesis) vs H₁ (alternative); reject H₀ if p &lt; α (typically 0.05)</p>
<ul>
  <li><strong>p-value:</strong> P(data this extreme | H₀ is true) — NOT P(H₀ is true | data)</li>
  <li><strong>Type I Error (α):</strong> false positive — rejecting true H₀; Type II Error (β): false negative — failing to reject false H₀</li>
  <li><strong>Power (1−β):</strong> probability of detecting effect when it exists; increases with sample size and effect size</li>
  <li><strong>t-test:</strong> one-sample (μ vs constant), two-sample (independent groups), paired (within-subject); assumes normality</li>
  <li><strong>Welch's t-test:</strong> unequal variances — preferred over Student's when σ₁≠σ₂</li>
  <li><strong>Mann-Whitney U:</strong> non-parametric alternative to t-test — compares distributions without normality assumption</li>
  <li><strong>Chi-squared Test:</strong> independence between categorical variables; χ² = Σ(O−E)²/E</li>
  <li><strong>ANOVA:</strong> F-test comparing means across 3+ groups; one-way, two-way, repeated measures</li>
  <li><strong>Bonferroni Correction:</strong> α' = α/m for m tests — controls familywise error rate (conservative)</li>
  <li><strong>Cohen's d:</strong> effect size = (μ₁−μ₂)/σ_pooled; small=0.2, medium=0.5, large=0.8</li>
  <li><strong>A/B Testing:</strong> two-sample test of proportions/means; sequential testing with SPRT for early stopping</li>
</ul>""",
        "use_cases": ["A/B testing", "Feature significance", "Model comparison", "Clinical trials", "Survey analysis"],
        "chart": "statistical_tests"
    },
    {
        "id": "model_compression", "label": "Model Compression", "category": "ai_advanced",
        "title": "Model Compression & Efficient AI",
        "subtitle": "Making models faster and smaller — pruning, quantization, knowledge distillation, and neural architecture search.",
        "theory": """<p><strong>Core Idea:</strong> Deploy powerful models on constrained hardware by reducing size and latency without sacrificing accuracy.</p>
<ul>
  <li><strong>Pruning:</strong> remove unimportant weights/neurons; magnitude pruning (|w|&lt;threshold), structured (filter/channel), lottery ticket hypothesis</li>
  <li><strong>Lottery Ticket Hypothesis:</strong> dense networks contain sparse subnetworks ('winning tickets') trainable to same accuracy</li>
  <li><strong>Quantization:</strong> reduce weight precision — INT8 (4x compression, ~1% accuracy loss), INT4, binary weights, quantization-aware training (QAT)</li>
  <li><strong>Post-Training Quantization (PTQ):</strong> quantize after training with calibration data — fast but less accurate than QAT</li>
  <li><strong>Knowledge Distillation:</strong> student mimics soft logits of teacher; T = temperature softens distribution, reveals dark knowledge</li>
  <li><strong>Soft targets:</strong> KL(student||teacher) + cross-entropy(student, true) — teacher provides richer signal than one-hot labels</li>
  <li><strong>NAS (Neural Architecture Search):</strong> DARTS (differentiable), evolutionary, Bayesian — automate architecture design</li>
  <li><strong>Efficient Architectures:</strong> MobileNet, EfficientNet, DistilBERT, TinyBERT — purpose-built for edge deployment</li>
  <li><strong>Speculative Decoding:</strong> draft model generates tokens, large model verifies — 2-3x LLM speedup</li>
  <li><strong>ONNX/TensorRT:</strong> inference optimization frameworks — operator fusion, memory layout optimization, hardware-specific kernels</li>
</ul>""",
        "use_cases": ["Mobile deployment", "Edge devices", "Real-time inference", "Cost reduction", "Embedded systems"],
        "chart": "model_compression"
    },
    {
        "id": "causal_inference", "label": "Causal Inference", "category": "ai_advanced",
        "title": "Causal Inference & Experimental Design",
        "subtitle": "Moving beyond correlation to causation — DAGs, potential outcomes, natural experiments, and treatment effect estimation.",
        "theory": """<p><strong>Fundamental Problem:</strong> For unit i, can only observe Yᵢ(1) or Yᵢ(0) — never both (fundamental problem of causal inference).</p>
<ul>
  <li><strong>Rubin Potential Outcomes:</strong> Yᵢ(1) treated, Yᵢ(0) control; ATE = E[Y(1)−Y(0)]</li>
  <li><strong>ATT:</strong> E[Y(1)−Y(0)|T=1] — Average Treatment on Treated</li>
  <li><strong>Confounding:</strong> variable U affects both treatment T and outcome Y → spurious correlation; requires adjustment</li>
  <li><strong>DAGs:</strong> encode causal structure; d-separation for conditional independence; do-calculus for intervention</li>
  <li><strong>Backdoor Criterion:</strong> block all backdoor paths from T to Y by conditioning on sufficient adjustment set</li>
  <li><strong>RCT:</strong> gold standard — random assignment breaks T-U correlation</li>
  <li><strong>Propensity Score Matching (PSM):</strong> match treated/control on P(T=1|X) — reduces confounding in observational data</li>
  <li><strong>Difference-in-Differences (DiD):</strong> (treated post − treated pre) − (control post − control pre) — natural experiment</li>
  <li><strong>Regression Discontinuity (RDD):</strong> exploit threshold assignment for causal identification</li>
  <li><strong>Instrumental Variables (IV):</strong> Z→X→Y, Z⊥U — 2SLS estimator for endogenous regressors</li>
</ul>""",
        "use_cases": ["Policy evaluation", "Medical treatment effects", "Marketing attribution", "A/B test analysis", "Fairness analysis"],
        "chart": "causal_inference"
    },
    {
        "id": "multimodal", "label": "Multimodal AI", "category": "ai_advanced",
        "title": "Multimodal AI — Vision-Language Models",
        "subtitle": "AI systems that process and align multiple modalities — text, images, audio — enabling cross-modal understanding and generation.",
        "theory": """<p><strong>Core Challenge:</strong> Different modalities have different statistical structures — bridging them requires learned alignment.</p>
<ul>
  <li><strong>Modalities:</strong> text, images, audio, video, structured data, 3D point clouds — each requires different encoders</li>
  <li><strong>Fusion Strategies:</strong> early fusion (concat inputs), late fusion (combine predictions), cross-modal attention (transformer)</li>
  <li><strong>CLIP:</strong> dual encoder — visual transformer + text transformer; contrastive training on 400M image-text pairs; zero-shot transfer</li>
  <li><strong>CLIP Training:</strong> minimize −Σdiag(log softmax(V·Tᵀ/τ)) — image and text embeddings aligned for matching pairs</li>
  <li><strong>VQA:</strong> given image + question → answer; tests cross-modal reasoning</li>
  <li><strong>BLIP/BLIP-2:</strong> bootstrapped language-image pretraining; Q-Former bridges vision encoder and frozen LLM</li>
  <li><strong>LLaVA / GPT-4V:</strong> project image patch embeddings into LLM token space via linear projection or MLP</li>
  <li><strong>DALL-E / Stable Diffusion:</strong> text-to-image generation; SD uses latent diffusion in compressed latent space</li>
  <li><strong>Audio-Language:</strong> Whisper (speech→text), SpeechT5, SeamlessM4T (speech translation)</li>
  <li><strong>Emerging:</strong> video understanding (Video-LLaMA), 3D-language (CLIP3D), EHR+imaging for medical AI</li>
</ul>""",
        "use_cases": ["Image captioning", "Visual QA", "Text-to-image generation", "Medical report generation", "Video understanding"],
        "chart": "multimodal"
    },
    {
        "id": "ethics_fairness", "label": "AI Ethics & Fairness", "category": "ai_advanced",
        "title": "AI Ethics, Fairness & Privacy",
        "subtitle": "Responsible AI development — bias detection and mitigation, fairness metrics, privacy-preserving ML, and governance.",
        "theory": """<p><strong>Core Challenge:</strong> Fairness is not one thing — different definitions are often mathematically incompatible.</p>
<ul>
  <li><strong>Demographic Parity:</strong> equal positive prediction rates across groups</li>
  <li><strong>Equal Opportunity:</strong> equal TPR across groups</li>
  <li><strong>Equalized Odds:</strong> equal TPR + FPR across groups</li>
  <li><strong>Individual Fairness:</strong> similar inputs → similar outputs</li>
  <li><strong>Impossibility Theorem:</strong> can't simultaneously satisfy demographic parity + equalized odds + calibration (except trivial cases)</li>
  <li><strong>Disparate Impact:</strong> P(Ŷ=1|A=0)/P(Ŷ=1|A=1) &lt; 0.8 = discriminatory by 80% rule (US EEOC)</li>
  <li><strong>Sources of Bias:</strong> historical bias, representation bias, measurement bias (proxy variables), aggregation bias, deployment shift</li>
  <li><strong>Debiasing:</strong> Pre-processing (reweighting/resampling), In-processing (adversarial fairness), Post-processing (threshold calibration)</li>
  <li><strong>Differential Privacy:</strong> M(x) = f(x) + Lap(Δf/ε) — ε-DP guarantees plausible deniability; smaller ε = stronger privacy</li>
  <li><strong>Federated Learning:</strong> train on distributed local data, aggregate gradients — no raw data leaves devices (FedAvg)</li>
  <li><strong>Explainability:</strong> SHAP, LIME, Integrated Gradients — right to explanation (GDPR Article 22)</li>
  <li><strong>AI Governance:</strong> EU AI Act risk tiers, model cards, datasheets for datasets, algorithmic auditing</li>
</ul>""",
        "use_cases": ["Hiring model audits", "Credit scoring compliance", "Healthcare AI equity", "GDPR compliance", "Federated learning deployment"],
        "chart": "ethics_fairness"
    },
]

# ── Category metadata ─────────────────────────────────────────────────────
categories = {
    "supervised":   {"name": "Supervised Learning",    "color": "#20808D", "icon": "S"},
    "ensemble":     {"name": "Ensemble Methods",       "color": "#A84B2F", "icon": "E"},
    "unsupervised": {"name": "Unsupervised Learning",  "color": "#A86FDF", "icon": "U"},
    "neural":       {"name": "Neural Networks",        "color": "#FFC553", "icon": "N"},
    "concepts":     {"name": "ML Concepts",            "color": "#6DAA45", "icon": "C"},
    "evaluation":   {"name": "Evaluation & Metrics",    "color": "#20A05A", "icon": "V"},
    "deep_learning":{"name": "Deep Learning",           "color": "#E06C3A", "icon": "D"},
    "ai_advanced":  {"name": "Advanced AI",             "color": "#7B5EA7", "icon": "A"},
}
cat_order = ["supervised", "ensemble", "unsupervised", "neural", "concepts", "evaluation", "deep_learning", "ai_advanced"]

# ── Build sidebar HTML (fully static) ────────────────────────────────────
def build_sidebar(tabs):
    html = ""
    by_cat = {c: [] for c in cat_order}
    for t in tabs:
        by_cat[t["category"]].append(t)
    for cat in cat_order:
        if not by_cat[cat]: continue
        meta = categories[cat]
        html += f'<div class="sidebar-group" id="sg-{cat}">\n'
        html += f'<div class="sidebar-group-label">\n'
        html += f'  <span class="sg-dot" style="background:{meta["color"]}"></span>{meta["name"]}\n'
        html += f'</div>\n'
        for t in by_cat[cat]:
            html += f'<button class="tab-btn" id="btn-{t["id"]}" data-tab="{t["id"]}" data-cat="{t["category"]}" onclick="switchTab(\'{t["id"]}\')">{t["label"]}</button>\n'
        html += '</div>\n'
    return html

# ── Build code samples HTML ───────────────────────────────────────────────
def build_code_section(tab_id):
    samples = CODE_SAMPLES.get(tab_id, [])
    if not samples:
        return '<div class="code-empty">No code samples available for this topic.</div>'
    parts = []
    for i, s in enumerate(samples):
        escaped = html_mod.escape(s["code"])
        parts.append(f'''<div class="code-sample">
  <div class="code-sample-header">
    <span class="code-sample-num">{i+1:02d}</span>
    <span class="code-sample-title">{html_mod.escape(s["title"])}</span>
    <button class="copy-btn" onclick="copyCode(this)" title="Copy"><svg viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h-1v1H2V6h1V5H2z"/></svg> Copy</button>
  </div>
  <pre class="code-pre"><code class="language-python">{escaped}</code></pre>
</div>''')
    return "\n".join(parts)

# ── Build panel HTML ──────────────────────────────────────────────────────
def build_panels(tabs, charts):
    html = ""
    for t in tabs:
        cat_meta = categories[t["category"]]
        use_cases_html = "".join(f'<span class="uc-tag">{uc}</span>' for uc in t["use_cases"])
        img_b64 = charts.get(t["chart"], "")
        code_html = build_code_section(t["id"])
        pid = t["id"]
        html += f'''<div class="tab-panel" id="panel-{pid}">
  <div class="panel-header">
    <span class="panel-cat-badge" style="background:{cat_meta["color"]}22;color:{cat_meta["color"]};border:1px solid {cat_meta["color"]}44">{cat_meta["name"]}</span>
    <h2 class="panel-title">{t["title"]}</h2>
    <p class="panel-sub">{t["subtitle"]}</p>
  </div>
  <!-- View switcher -->
  <div class="view-switcher" id="vs-{pid}">
    <button class="view-btn active" onclick="switchView(\'{pid}\', \'theory\', this)">&#9096; Theory &amp; Charts</button>
    <button class="view-btn" onclick="switchView(\'{pid}\', \'code\', this)">&#9654; Python Code</button>
  </div>
  <!-- Theory + Charts view -->
  <div class="view-pane" id="view-theory-{pid}">
    <div class="info-row">
      <div class="theory-card">
        <div class="card-label">Theory &amp; Mathematics</div>
        {t["theory"]}
      </div>
      <div class="uc-card">
        <div class="card-label">Common Use Cases</div>
        <div class="uc-tags">{use_cases_html}</div>
      </div>
    </div>
    <div class="chart-card">
      <div class="chart-label">Visualizations — <span class="chart-hint">Click to zoom</span></div>
      <img src="data:image/png;base64,{img_b64}" class="chart-img" alt="{t["title"]} chart" loading="lazy"/>
    </div>
  </div>
  <!-- Code view -->
  <div class="view-pane hidden" id="view-code-{pid}">
    <div class="code-section">
      <div class="code-section-header">
        <div class="card-label">Python Code Samples</div>
        <span class="code-count">{len(CODE_SAMPLES.get(t["id"], []))} examples</span>
      </div>
{code_html}
    </div>
  </div>
</div>\n'''
    return html

# ── Build filter buttons ──────────────────────────────────────────────────
def build_filters():
    html = '<button class="filter-btn active" data-filter="all" onclick="setFilter(\'all\')">All Models</button>\n'
    for cat in cat_order:
        m = categories[cat]
        html += f'<button class="filter-btn" data-filter="{cat}" style="--cc:{m["color"]}" onclick="setFilter(\'{cat}\')">{m["name"]}</button>\n'
    return html

# ── Tab JS data ───────────────────────────────────────────────────────────
tabs_js = json.dumps([{"id": t["id"], "label": t["label"], "cat": t["category"]} for t in ALL_TABS])

sidebar_html = build_sidebar(ALL_TABS)
panels_html  = build_panels(ALL_TABS, charts)
filters_html = build_filters()
first_tab    = ALL_TABS[0]["id"]
total_models = len(ALL_TABS)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ML Models — Visual Reference</title>
<link id="hljs-theme-link" rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
/* ── Theme definitions ── */
:root{{
  --bg:#0F0E0C;--surf:#1A1916;--surf2:#222120;--surf3:#282624;
  --bd:#2E2D2A;--bd2:#3A3835;
  --tx:#CDCCCA;--tx2:#9A9896;--tx3:#616060;
  --teal:#20808D;--rust:#A84B2F;--gold:#FFC553;--purple:#A86FDF;--green:#6DAA45;
  --sidebar:210px;--header:54px;--filter:46px;
  --hljs-theme:url('https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css');
}}
/* Light theme */
[data-theme="light"]{{
  --bg:#F5F4F0;--surf:#FFFFFF;--surf2:#F0EFEc;--surf3:#E8E7E3;
  --bd:#D8D7D3;--bd2:#C8C7C3;
  --tx:#1A1916;--tx2:#4A4845;--tx3:#8A8885;
  --teal:#1A707D;--rust:#B55533;--gold:#C8940D;--purple:#8B52CC;--green:#4A8A28;
}}
/* Ocean theme */
[data-theme="ocean"]{{
  --bg:#0A1628;--surf:#0E1F38;--surf2:#152848;--surf3:#1A3058;
  --bd:#1E3A5F;--bd2:#264878;
  --tx:#B8D4F0;--tx2:#7AA8D8;--tx3:#4A7AAA;
  --teal:#3AAFCC;--rust:#E07050;--gold:#F0C040;--purple:#9A7AE0;--green:#50C090;
}}
/* Forest theme */
[data-theme="forest"]{{
  --bg:#0D1A0F;--surf:#122016;--surf2:#18291C;--surf3:#1E3222;
  --bd:#244028;--bd2:#2E5035;
  --tx:#C0D8B8;--tx2:#7AAA6A;--tx3:#4A7248;
  --teal:#3AB870;--rust:#D06840;--gold:#E0B840;--purple:#9870D0;--green:#60D080;
}}
/* Midnight theme */
[data-theme="midnight"]{{
  --bg:#0C0818;--surf:#130E24;--surf2:#1A1230;--surf3:#21163C;
  --bd:#2A1E4A;--bd2:#352860;
  --tx:#C8B8E8;--tx2:#9880C8;--tx3:#604878;
  --teal:#7060E0;--rust:#E05080;--gold:#F0C050;--purple:#C080F0;--green:#60D0A0;
}}
/* Solarized theme */
[data-theme="solarized"]{{
  --bg:#FDF6E3;--surf:#EEE8D5;--surf2:#E5DFCC;--surf3:#DDD8C4;
  --bd:#CBC5AE;--bd2:#BAB4A0;
  --tx:#586E75;--tx2:#839496;--tx3:#93A1A1;
  --teal:#2AA198;--rust:#CB4B16;--gold:#B58900;--purple:#6C71C4;--green:#859900;
}}
html,body{{height:100%;background:var(--bg);color:var(--tx);font-family:-apple-system,'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6;overflow:hidden}}

/* ── App shell ── */
.app{{display:flex;flex-direction:column;height:100vh}}
.header{{
  height:var(--header);flex-shrink:0;
  background:var(--surf);border-bottom:1px solid var(--bd);
  display:flex;align-items:center;padding:0 20px;gap:14px
}}
.logo{{display:flex;align-items:center;gap:9px;flex-shrink:0}}
.logo-icon{{width:32px;height:32px}}
.logo-text{{font-size:16px;font-weight:700;letter-spacing:-.3px}}
.logo-sub{{font-size:11px;color:var(--tx3);margin-left:1px}}
.search-wrap{{flex:1;max-width:240px;margin-left:auto}}
.search{{
  width:100%;padding:6px 12px;background:var(--surf2);border:1px solid var(--bd);
  border-radius:7px;color:var(--tx);font-size:13px;outline:none;transition:.15s border-color
}}
.search::placeholder{{color:var(--tx3)}}
.search:focus{{border-color:var(--teal)}}
.stats{{display:flex;gap:8px;flex-shrink:0}}
.chip{{
  padding:3px 10px;border-radius:20px;background:var(--surf2);
  border:1px solid var(--bd);font-size:11px;color:var(--tx2)
}}
.chip b{{color:var(--teal)}}

/* ── Filter bar ── */
.filter-bar{{
  height:var(--filter);flex-shrink:0;
  background:var(--bg);border-bottom:1px solid var(--bd);
  display:flex;align-items:center;padding:0 16px;gap:7px;overflow-x:auto
}}
.filter-bar::-webkit-scrollbar{{height:0}}
.filter-lbl{{font-size:10px;color:var(--tx3);text-transform:uppercase;letter-spacing:.8px;white-space:nowrap;margin-right:2px}}
.filter-btn{{
  padding:4px 13px;border-radius:20px;font-size:12px;font-weight:500;
  border:1px solid var(--bd);background:transparent;color:var(--tx2);
  cursor:pointer;transition:.15s all;white-space:nowrap;flex-shrink:0
}}
.filter-btn:hover{{border-color:var(--cc,var(--teal));color:var(--tx)}}
.filter-btn.active{{background:var(--cc,var(--teal));color:#fff;border-color:var(--cc,var(--teal))}}

/* ── Body: sidebar + content ── */
.body{{display:flex;flex:1;overflow:hidden}}

/* ── Sidebar ── */
.sidebar{{
  width:var(--sidebar);flex-shrink:0;
  background:var(--surf);border-right:1px solid var(--bd);
  overflow-y:auto;overflow-x:hidden;padding:6px 0 16px
}}
.sidebar::-webkit-scrollbar{{width:3px}}
.sidebar::-webkit-scrollbar-thumb{{background:var(--bd2);border-radius:2px}}

.sidebar-group{{margin-bottom:2px}}
.sidebar-group.hidden{{display:none}}
.sidebar-group-label{{
  display:flex;align-items:center;gap:7px;
  padding:8px 14px 3px;font-size:10px;text-transform:uppercase;
  letter-spacing:.9px;color:var(--tx3);font-weight:600
}}
.sg-dot{{width:6px;height:6px;border-radius:50%;flex-shrink:0}}
.tab-btn{{
  display:block;width:100%;padding:7px 14px 7px 22px;
  background:transparent;border:none;border-left:2px solid transparent;
  color:var(--tx2);font-size:12.5px;cursor:pointer;text-align:left;
  transition:.13s all;white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}}
.tab-btn:hover{{background:var(--surf2);color:var(--tx)}}
.tab-btn.active{{background:var(--surf2);color:var(--tx);border-left-color:var(--teal);font-weight:600}}
.tab-btn.hidden{{display:none}}

/* ── Content area ── */
.content{{
  flex:1;overflow-y:auto;overflow-x:hidden;
  padding:24px 28px 40px
}}
.content::-webkit-scrollbar{{width:5px}}
.content::-webkit-scrollbar-thumb{{background:var(--bd2);border-radius:3px}}

/* ── Panels ── */
.tab-panel{{display:none;animation:fadeIn .18s ease}}
.tab-panel.active{{display:block}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:none}}}}

.panel-header{{margin-bottom:18px}}
.panel-cat-badge{{
  display:inline-block;padding:2px 10px;border-radius:20px;
  font-size:11px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;margin-bottom:8px
}}
.panel-title{{font-size:24px;font-weight:700;letter-spacing:-.4px;margin-bottom:5px}}
.panel-sub{{font-size:13.5px;color:var(--tx2);max-width:750px}}

/* ── Info row ── */
.info-row{{display:grid;grid-template-columns:1fr 280px;gap:14px;margin-bottom:14px}}
@media(max-width:900px){{.info-row{{grid-template-columns:1fr}}}}

.theory-card,.uc-card,.chart-card{{
  background:var(--surf);border:1px solid var(--bd);border-radius:10px;padding:16px 18px
}}
.card-label{{
  font-size:10px;text-transform:uppercase;letter-spacing:.9px;
  color:var(--teal);font-weight:600;margin-bottom:10px
}}
.theory-card p{{margin-bottom:8px;font-size:13px;color:var(--tx)}}
.theory-card ul{{padding-left:16px}}
.theory-card li{{margin-bottom:4px;font-size:12.5px;color:var(--tx)}}
.theory-card strong{{color:var(--gold)}}
.theory-card code{{
  background:var(--surf2);border:1px solid var(--bd);border-radius:4px;
  padding:1px 5px;font-family:'SF Mono','Fira Code',monospace;font-size:11.5px;color:var(--teal)
}}
.uc-tags{{display:flex;flex-wrap:wrap;gap:7px}}
.uc-tag{{
  padding:4px 11px;border-radius:20px;background:var(--surf2);
  border:1px solid var(--bd);font-size:11.5px;color:var(--tx)
}}

/* ── Chart ── */
.chart-card{{padding:14px}}
.chart-label{{font-size:10px;text-transform:uppercase;letter-spacing:.9px;color:var(--teal);font-weight:600;margin-bottom:10px}}
.chart-hint{{color:var(--tx3);font-size:10px;letter-spacing:0;text-transform:none;font-weight:400}}
.chart-img{{width:100%;height:auto;display:block;border-radius:6px;cursor:zoom-in;transition:.2s transform}}

/* ── Zoom overlay ── */
.zoom-overlay{{
  display:none;position:fixed;inset:0;z-index:1000;
  background:rgba(15,14,12,.97);cursor:zoom-out;
  align-items:center;justify-content:center;padding:24px
}}
.zoom-overlay.open{{display:flex}}
.zoom-overlay img{{max-width:100%;max-height:100%;border-radius:8px;object-fit:contain}}

/* ── Theme switcher ── */
.theme-switcher{{
  display:flex;align-items:center;gap:6px;flex-shrink:0
}}
.theme-label{{
  font-size:10px;color:var(--tx3);text-transform:uppercase;
  letter-spacing:.7px;white-space:nowrap
}}
.theme-swatches{{
  display:flex;gap:4px;align-items:center
}}
.theme-swatch{{
  width:18px;height:18px;border-radius:50%;cursor:pointer;
  border:2px solid transparent;transition:.15s all;
  display:flex;align-items:center;justify-content:center;
  position:relative
}}
.theme-swatch:hover{{transform:scale(1.15)}}
.theme-swatch.active{{border-color:var(--tx);box-shadow:0 0 0 2px var(--bg)}}
.theme-swatch[data-t="dark"]{{background:radial-gradient(circle at 40% 40%,#2E2D2A,#0F0E0C)}}
.theme-swatch[data-t="light"]{{background:radial-gradient(circle at 40% 40%,#FFFFFF,#D8D7D3)}}
.theme-swatch[data-t="ocean"]{{background:radial-gradient(circle at 40% 40%,#1E3A5F,#0A1628)}}
.theme-swatch[data-t="forest"]{{background:radial-gradient(circle at 40% 40%,#244028,#0D1A0F)}}
.theme-swatch[data-t="midnight"]{{background:radial-gradient(circle at 40% 40%,#2A1E4A,#0C0818)}}
.theme-swatch[data-t="solarized"]{{background:radial-gradient(circle at 40% 40%,#EEE8D5,#FDF6E3)}}
/* Light + Solarized: dark hljs swap */
[data-theme="light"] .hljs,
[data-theme="solarized"] .hljs{{background:#F5F5F5 !important;color:#383A42 !important}}
[data-theme="light"] .code-pre,
[data-theme="solarized"] .code-pre{{background:#F5F5F5 !important}}

/* ── Empty state ── */
.empty{{
  display:none;flex-direction:column;align-items:center;
  justify-content:center;height:300px;gap:12px;color:var(--tx3)
}}
.empty.show{{display:flex}}
.empty-icon{{font-size:40px;opacity:.4}}

/* ── View switcher (Theory / Code) ── */
.view-switcher{{
  display:flex;gap:6px;margin-bottom:16px;
  border-bottom:1px solid var(--bd);padding-bottom:12px
}}
.view-btn{{
  padding:6px 16px;border-radius:7px;font-size:12.5px;font-weight:500;
  border:1px solid var(--bd);background:transparent;color:var(--tx2);
  cursor:pointer;transition:.15s all
}}
.view-btn:hover{{background:var(--surf2);color:var(--tx);border-color:var(--bd2)}}
.view-btn.active{{background:var(--teal);color:#fff;border-color:var(--teal)}}
.view-pane{{animation:fadeIn .18s ease}}
.view-pane.hidden{{display:none}}

/* ── Code section ── */
.code-section{{
  display:flex;flex-direction:column;gap:16px
}}
.code-section-header{{
  display:flex;align-items:center;gap:10px;margin-bottom:4px
}}
.code-count{{
  font-size:10px;color:var(--tx3);background:var(--surf2);
  border:1px solid var(--bd);border-radius:20px;padding:1px 9px
}}
.code-sample{{
  background:var(--surf);border:1px solid var(--bd);border-radius:10px;overflow:hidden
}}
.code-sample-header{{
  display:flex;align-items:center;gap:10px;
  padding:10px 14px;background:var(--surf2);border-bottom:1px solid var(--bd)
}}
.code-sample-num{{
  font-family:'SF Mono','Fira Code',monospace;font-size:10.5px;
  color:var(--teal);font-weight:700;letter-spacing:.5px
}}
.code-sample-title{{
  flex:1;font-size:13px;font-weight:600;color:var(--tx)
}}
.copy-btn{{
  display:flex;align-items:center;gap:5px;
  padding:3px 10px;border-radius:5px;font-size:11.5px;
  background:var(--surf3);border:1px solid var(--bd);color:var(--tx2);
  cursor:pointer;transition:.13s all
}}
.copy-btn:hover{{background:var(--bd);color:var(--tx)}}
.copy-btn.copied{{color:var(--green);border-color:var(--green)}}
.code-pre{{
  margin:0;padding:0;font-size:12.5px;line-height:1.65;
  overflow-x:auto;background:#1e1e2e !important
}}
.code-pre::-webkit-scrollbar{{height:4px}}
.code-pre::-webkit-scrollbar-thumb{{background:var(--bd2);border-radius:2px}}
/* Override hljs background to match our theme */
.hljs{{background:#1e1e2e !important;padding:14px 18px !important}}
.code-empty{{
  color:var(--tx3);font-size:13px;padding:20px 0
}}
</style>
</head>
<body>
<div class="app">

<!-- HEADER -->
<header class="header">
  <div class="logo">
    <svg class="logo-icon" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="32" height="32" rx="7" fill="#1A1916"/>
      <circle cx="16" cy="16" r="9" stroke="#20808D" stroke-width="1.8"/>
      <circle cx="16" cy="16" r="3.5" fill="#20808D"/>
      <line x1="16" y1="7" x2="16" y2="11" stroke="#FFC553" stroke-width="1.4" stroke-linecap="round"/>
      <line x1="16" y1="21" x2="16" y2="25" stroke="#FFC553" stroke-width="1.4" stroke-linecap="round"/>
      <line x1="7" y1="16" x2="11" y2="16" stroke="#FFC553" stroke-width="1.4" stroke-linecap="round"/>
      <line x1="21" y1="16" x2="25" y2="16" stroke="#FFC553" stroke-width="1.4" stroke-linecap="round"/>
      <circle cx="10.5" cy="10.5" r="1.4" fill="#A84B2F"/>
      <circle cx="21.5" cy="10.5" r="1.4" fill="#A84B2F"/>
      <circle cx="10.5" cy="21.5" r="1.4" fill="#A84B2F"/>
      <circle cx="21.5" cy="21.5" r="1.4" fill="#A84B2F"/>
    </svg>
    <div>
      <div class="logo-text">ML Model Visualizer</div>
      <div class="logo-sub">Comprehensive Interactive Reference</div>
    </div>
  </div>
  <div class="search-wrap">
    <input type="text" class="search" placeholder="Search models..." oninput="onSearch(this.value)"/>
  </div>
  <div class="theme-switcher">
    <span class="theme-label">Theme</span>
    <div class="theme-swatches">
      <div class="theme-swatch active" data-t="dark"    onclick="setTheme(\'dark\')"></div>
      <div class="theme-swatch"        data-t="light"   onclick="setTheme(\'light\')"></div>
      <div class="theme-swatch"        data-t="ocean"   onclick="setTheme(\'ocean\')"></div>
      <div class="theme-swatch"        data-t="forest"  onclick="setTheme(\'forest\')"></div>
      <div class="theme-swatch"        data-t="midnight" onclick="setTheme(\'midnight\')"></div>
      <div class="theme-swatch"        data-t="solarized" onclick="setTheme(\'solarized\')"></div>
    </div>
  </div>
  <div class="stats">
    <span class="chip"><b>{total_models}</b> Models</span>
    <span class="chip"><b>8</b> Categories</span>
    <span class="chip"><b>57</b> Code Sets</span>
  </div>
</header>

<!-- FILTER BAR -->
<div class="filter-bar">
  <span class="filter-lbl">Filter</span>
  {filters_html}
</div>

<!-- BODY -->
<div class="body">

  <!-- SIDEBAR -->
  <nav class="sidebar" id="sidebar">
{sidebar_html}
  </nav>

  <!-- CONTENT -->
  <div class="content" id="content">
{panels_html}
    <div class="empty" id="empty">
      <div class="empty-icon">🔍</div>
      <div>No models match your search</div>
    </div>
  </div>

</div><!-- /body -->
</div><!-- /app -->

<!-- ZOOM OVERLAY -->
<div class="zoom-overlay" id="zoomOverlay" onclick="closeZoom()">
  <img id="zoomImg" src="" alt="Zoomed chart"/>
</div>

<script>
const TABS = {tabs_js};
let activeCat = 'all';
let searchQ = '';
let activeId = '{first_tab}';

function switchTab(id) {{
  // deactivate old
  const oldBtn = document.getElementById('btn-' + activeId);
  const oldPanel = document.getElementById('panel-' + activeId);
  if (oldBtn) oldBtn.classList.remove('active');
  if (oldPanel) oldPanel.classList.remove('active');
  // activate new
  activeId = id;
  const newBtn = document.getElementById('btn-' + id);
  const newPanel = document.getElementById('panel-' + id);
  if (newBtn) newBtn.classList.add('active');
  if (newPanel) newPanel.classList.add('active');
  document.getElementById('content').scrollTop = 0;
  // scroll sidebar btn into view
  if (newBtn) newBtn.scrollIntoView({{block:'nearest'}});
}}

function setFilter(cat) {{
  activeCat = cat;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  const fb = document.querySelector('.filter-btn[data-filter="'+cat+'"]');
  if (fb) fb.classList.add('active');
  applyFilters();
}}

function onSearch(q) {{
  searchQ = q.toLowerCase().trim();
  applyFilters();
}}

function applyFilters() {{
  let anyVisible = false;
  const groupVisible = {{}};

  TABS.forEach(t => {{
    const btn = document.getElementById('btn-' + t.id);
    if (!btn) return;
    const catOk = activeCat === 'all' || t.cat === activeCat;
    const searchOk = !searchQ || t.label.toLowerCase().includes(searchQ) || t.cat.toLowerCase().includes(searchQ);
    const visible = catOk && searchOk;
    btn.classList.toggle('hidden', !visible);
    if (visible) {{ anyVisible = true; groupVisible[t.cat] = true; }}
  }});

  // show/hide groups
  Object.keys({json.dumps({c: True for c in cat_order})}).forEach(cat => {{
    const g = document.getElementById('sg-' + cat);
    if (g) g.classList.toggle('hidden', !groupVisible[cat]);
  }});

  // if active tab is now hidden, pick first visible
  const activeBtn = document.getElementById('btn-' + activeId);
  if (activeBtn && activeBtn.classList.contains('hidden')) {{
    const first = TABS.find(t => {{
      const b = document.getElementById('btn-' + t.id);
      return b && !b.classList.contains('hidden');
    }});
    if (first) switchTab(first.id);
  }}

  document.getElementById('empty').classList.toggle('show', !anyVisible);
}}

// zoom
document.querySelectorAll('.chart-img').forEach(img => {{
  img.addEventListener('click', () => {{
    document.getElementById('zoomImg').src = img.src;
    document.getElementById('zoomOverlay').classList.add('open');
  }});
}});
function closeZoom() {{ document.getElementById('zoomOverlay').classList.remove('open'); }}

// keyboard
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') closeZoom();
  if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && !e.target.matches('input')) {{
    const visible = TABS.filter(t => {{
      const b = document.getElementById('btn-' + t.id);
      return b && !b.classList.contains('hidden');
    }});
    const idx = visible.findIndex(t => t.id === activeId);
    const next = e.key === 'ArrowDown' ? visible[idx+1] : visible[idx-1];
    if (next) switchTab(next.id);
    e.preventDefault();
  }}
}});

// init
switchTab('{first_tab}');

// ── Theme switcher ──
const HLJS_THEMES = {{
  dark:      'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css',
  light:     'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-light.min.css',
  ocean:     'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/nord.min.css',
  forest:    'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/base16/green-screen.min.css',
  midnight:  'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/night-owl.min.css',
  solarized: 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/base16/solarized-light.min.css',
}};
function setTheme(name) {{
  // Apply data-theme attribute
  document.documentElement.setAttribute('data-theme', name === 'dark' ? '' : name);
  // Update swatch active state
  document.querySelectorAll('.theme-swatch').forEach(s => {{
    s.classList.toggle('active', s.dataset.t === name);
  }});
  // Swap highlight.js stylesheet
  const hljsLink = document.getElementById('hljs-theme-link');
  if (hljsLink) hljsLink.href = HLJS_THEMES[name] || HLJS_THEMES.dark;
  // Re-highlight any already-highlighted code panes (reset and re-run)
  document.querySelectorAll('.view-pane[data-highlighted]').forEach(pane => {{
    pane.removeAttribute('data-highlighted');
    pane.querySelectorAll('pre code').forEach(block => {{
      block.removeAttribute('data-highlighted');
      block.className = 'language-python';
    }});
  }});
  // Persist preference in memory
  window._mlVizTheme = name;
}}
// No persistence needed — theme resets on reload (localStorage not available in iframe)

// ── View switcher (Theory / Code) ──
function switchView(tabId, view, btn) {{
  // toggle buttons
  const switcher = document.getElementById('vs-' + tabId);
  switcher.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  // toggle panes
  const theoryPane = document.getElementById('view-theory-' + tabId);
  const codePane   = document.getElementById('view-code-'   + tabId);
  if (view === 'theory') {{
    theoryPane.classList.remove('hidden');
    codePane.classList.add('hidden');
  }} else {{
    codePane.classList.remove('hidden');
    theoryPane.classList.add('hidden');
    // trigger hljs on first reveal
    if (!codePane.dataset.highlighted) {{
      codePane.querySelectorAll('pre code').forEach(block => {{
        if (typeof hljs !== 'undefined') hljs.highlightElement(block);
      }});
      codePane.dataset.highlighted = '1';
    }}
  }}
}}

// ── Copy code button ──
function copyCode(btn) {{
  const code = btn.closest('.code-sample').querySelector('code');
  navigator.clipboard.writeText(code.innerText).then(() => {{
    btn.classList.add('copied');
    btn.innerHTML = '<svg viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0z"/></svg> Copied!';
    setTimeout(() => {{
      btn.classList.remove('copied');
      btn.innerHTML = '<svg viewBox="0 0 16 16" fill="currentColor" width="13" height="13"><path d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h-1v1H2V6h1V5H2z"/></svg> Copy';
    }}, 1800);
  }}).catch(() => {{
    // fallback
    const ta = document.createElement('textarea');
    ta.value = code.innerText;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }});
}}
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
</body>
</html>"""

out = '/home/user/workspace/ml-visualizer/index.html'
with open(out, 'w') as f:
    f.write(HTML)

size = os.path.getsize(out) / 1024 / 1024
print(f"Written: {out} ({size:.1f} MB)  |  Models: {total_models}")
