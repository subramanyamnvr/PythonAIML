import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,
                              mean_squared_error, r2_score, classification_report)
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import norm
import io, base64, json, warnings
warnings.filterwarnings('ignore')

# ── palette ──────────────────────────────────────────────────────────────
BG      = '#0F0E0C'
SURFACE = '#1A1916'
BORDER  = '#2E2D2A'
TEXT    = '#CDCCCA'
MUTED   = '#797876'
C1='#20808D'; C2='#A84B2F'; C3='#BCE2E7'; C4='#FFC553'; C5='#944454'; C6='#6DAA45'; C7='#A86FDF'; C8='#5591C7'
PALETTE = [C1,C2,C3,C4,C5,C6,C7,C8]

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    if title:  ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.7)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG, edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def make_fig(nrows=1, ncols=1, w=14, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig, axes

charts = {}

# ══════════════════════════════════════════════════════════════════════════
# 1. LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════
np.random.seed(42)
X_lr = np.linspace(0,10,120).reshape(-1,1)
y_lr = 2.5*X_lr.ravel() + 5 + np.random.normal(0, 4, 120)
model_lr = LinearRegression().fit(X_lr, y_lr)
y_pred_lr = model_lr.predict(X_lr)

fig, axes = make_fig(1, 3, 16, 5)
# scatter + fit
ax = axes[0]
ax.scatter(X_lr, y_lr, color=C1, alpha=0.5, s=20, label='Data')
ax.plot(X_lr, y_pred_lr, color=C4, lw=2.5, label=f'Fit  y={model_lr.coef_[0]:.2f}x+{model_lr.intercept_:.2f}')
ax.fill_between(X_lr.ravel(), y_pred_lr-8, y_pred_lr+8, alpha=0.15, color=C4)
style_ax(ax,'Linear Regression – Fit','X','y')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# residuals
ax = axes[1]
res = y_lr - y_pred_lr
ax.scatter(y_pred_lr, res, color=C2, alpha=0.6, s=20)
ax.axhline(0, color=C4, lw=1.5, ls='--')
style_ax(ax,'Residuals vs Fitted','Fitted','Residual')

# Ridge vs Lasso coefficient paths
alphas = np.logspace(-3, 3, 80)
ridge_coefs = [Ridge(alpha=a).fit(X_lr, y_lr).coef_[0] for a in alphas]
lasso_coefs = [Lasso(alpha=a, max_iter=5000).fit(X_lr, y_lr).coef_[0] for a in alphas]
ax = axes[2]
ax.plot(alphas, ridge_coefs, color=C1, lw=2, label='Ridge')
ax.plot(alphas, lasso_coefs, color=C2, lw=2, ls='--', label='Lasso')
ax.set_xscale('log'); style_ax(ax,'Ridge vs Lasso – Coef Path','α (log)','Coefficient')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

charts['linear_regression'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 2. LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════
X_log, y_log = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0,
                                             n_clusters_per_class=1, random_state=42)
clf_log = LogisticRegression().fit(X_log, y_log)

fig, axes = make_fig(1, 3, 16, 5)
# decision boundary
ax = axes[0]
h=0.04; xmn,xmx=X_log[:,0].min()-1,X_log[:,0].max()+1; ymn,ymx=X_log[:,1].min()-1,X_log[:,1].max()+1
xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
Z=clf_log.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
ax.contourf(xx,yy,Z, alpha=0.3, cmap=ListedColormap([C2,C1]))
ax.scatter(X_log[:,0],X_log[:,1],c=[C1 if y==1 else C2 for y in y_log],s=25,edgecolors='none',alpha=0.8)
style_ax(ax,'Logistic Regression – Decision Boundary','Feature 1','Feature 2')

# sigmoid
ax = axes[1]
z=np.linspace(-6,6,200); sig=1/(1+np.exp(-z))
ax.plot(z,sig,color=C1,lw=2.5)
ax.axhline(0.5,color=C4,ls='--',lw=1); ax.axvline(0,color=BORDER,lw=1)
ax.fill_between(z,0,sig,alpha=0.15,color=C1)
style_ax(ax,'Sigmoid Function','z = Xβ','P(y=1|X)')

# probability dist
ax = axes[2]
probs = clf_log.predict_proba(X_log)[:,1]
ax.hist(probs[y_log==0],bins=20,color=C2,alpha=0.7,label='Class 0')
ax.hist(probs[y_log==1],bins=20,color=C1,alpha=0.7,label='Class 1')
ax.axvline(0.5,color=C4,ls='--',lw=1.5)
style_ax(ax,'Predicted Probability Distribution','P(y=1|X)','Count')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

charts['logistic_regression'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 3. DECISION TREE
# ══════════════════════════════════════════════════════════════════════════
iris = datasets.load_iris()
X_iris, y_iris = iris.data[:, :2], iris.target
dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_iris, y_iris)

fig, axes = make_fig(1, 3, 16, 5)
# boundary
ax = axes[0]
h=0.03; x0min,x0max=X_iris[:,0].min()-0.5,X_iris[:,0].max()+0.5
x1min,x1max=X_iris[:,1].min()-0.5,X_iris[:,1].max()+0.5
xx,yy=np.meshgrid(np.arange(x0min,x0max,h),np.arange(x1min,x1max,h))
Z=dt.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
ax.contourf(xx,yy,Z,alpha=0.25,cmap=ListedColormap([C1,C2,C4]))
colors_iris=[C1,C2,C4]
for i,name in enumerate(iris.target_names):
    mask=y_iris==i
    ax.scatter(X_iris[mask,0],X_iris[mask,1],color=colors_iris[i],s=25,label=name,alpha=0.9)
style_ax(ax,'Decision Tree – Decision Boundary','Sepal length','Sepal width')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# depth vs accuracy
from sklearn.model_selection import cross_val_score
depths=range(1,15)
scores=[cross_val_score(DecisionTreeClassifier(max_depth=d,random_state=42),X_iris,y_iris,cv=5).mean() for d in depths]
ax = axes[1]
ax.plot(depths, scores, color=C1, lw=2, marker='o', markersize=5, markerfacecolor=C4)
ax.axvline(4, color=C2, ls='--', lw=1.5, label='Chosen depth=4')
style_ax(ax,'Depth vs CV Accuracy','Max Depth','CV Accuracy')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# feature importance
ax = axes[2]
feats=['Sepal Len','Sepal Wid','Petal Len','Petal Wid']
dt_full=DecisionTreeClassifier(max_depth=4,random_state=42).fit(iris.data,y_iris)
imp=dt_full.feature_importances_
bars=ax.barh(feats, imp, color=[C1,C3,C4,C2])
for bar,val in zip(bars,imp): ax.text(val+0.01,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',color=TEXT,fontsize=9)
style_ax(ax,'Feature Importances','Importance','')

charts['decision_tree'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 4. SVM
# ══════════════════════════════════════════════════════════════════════════
X_svm, y_svm = datasets.make_classification(n_samples=200, n_features=2, n_redundant=0,
                                             n_clusters_per_class=1, random_state=5)
svm_lin = SVC(kernel='linear', C=1).fit(X_svm, y_svm)
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.5).fit(X_svm, y_svm)

fig, axes = make_fig(1, 3, 16, 5)
for ax, clf, title in zip(axes[:2], [svm_lin, svm_rbf], ['SVM – Linear Kernel','SVM – RBF Kernel']):
    h=0.04; xmn,xmx=X_svm[:,0].min()-1,X_svm[:,0].max()+1; ymn,ymx=X_svm[:,1].min()-1,X_svm[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx,yy,Z,alpha=0.2,cmap=ListedColormap([C2,C1]))
    ax.scatter(X_svm[:,0],X_svm[:,1],c=[C1 if y==1 else C2 for y in y_svm],s=22,alpha=0.85)
    # support vectors
    sv=clf.support_vectors_
    ax.scatter(sv[:,0],sv[:,1],s=120,linewidths=2,facecolors='none',edgecolors=C4,label='Support Vectors')
    if hasattr(clf,'coef_'):
        w=clf.coef_[0]; b=clf.intercept_[0]
        xs=np.linspace(xmn,xmx,200); ys=-(w[0]*xs+b)/w[1]
        ax.plot(xs,ys,color=C4,lw=2,label='Hyperplane')
        ax.plot(xs,ys+1/np.sqrt(np.sum(w**2)),color=C4,lw=1,ls='--')
        ax.plot(xs,ys-1/np.sqrt(np.sum(w**2)),color=C4,lw=1,ls='--')
    style_ax(ax,title,'F1','F2')
    ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# C-param effect on margin
ax = axes[2]
Cs=np.logspace(-2,3,50)
n_sv=[SVC(kernel='linear',C=c).fit(X_svm,y_svm).n_support_.sum() for c in Cs]
ax2=ax.twinx()
scores=[SVC(kernel='rbf',C=c).fit(X_svm,y_svm).score(X_svm,y_svm) for c in Cs]
ax.plot(Cs,n_sv,color=C1,lw=2,label='#Support Vectors')
ax2.plot(Cs,scores,color=C4,lw=2,ls='--',label='Train Acc (RBF)')
ax.set_xscale('log'); ax.set_facecolor(SURFACE); ax.tick_params(colors=MUTED)
ax2.tick_params(colors=MUTED); ax2.yaxis.label.set_color(MUTED)
for sp in ax.spines.values(): sp.set_color(BORDER)
ax.set_title('C vs Support Vectors & Accuracy', color=TEXT, fontsize=11, fontweight='bold', pad=8)
ax.set_xlabel('C (log)', color=MUTED); ax.set_ylabel('#SVs', color=C1)
ax2.set_ylabel('Accuracy', color=C4)
ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.7)
lines1,labels1=ax.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,labels1+labels2,fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

charts['svm'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 5. KNN
# ══════════════════════════════════════════════════════════════════════════
X_knn, y_knn = datasets.make_blobs(n_samples=250, centers=3, cluster_std=1.3, random_state=42)

fig, axes = make_fig(1, 3, 16, 5)
for idx, k in enumerate([1, 5, 15]):
    ax = axes[idx]
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_knn, y_knn)
    h=0.08; xmn,xmx=X_knn[:,0].min()-1,X_knn[:,0].max()+1; ymn,ymx=X_knn[:,1].min()-1,X_knn[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
    Z=knn.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx,yy,Z,alpha=0.2,cmap=ListedColormap([C1,C2,C4]))
    for c,col in enumerate([C1,C2,C4]):
        ax.scatter(X_knn[y_knn==c,0],X_knn[y_knn==c,1],color=col,s=22,alpha=0.85)
    acc=knn.score(X_knn,y_knn)
    style_ax(ax,f'KNN k={k}  (Acc={acc:.2f})','F1','F2')

charts['knn'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 6. NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════
X_nb, y_nb = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0,
                                           n_clusters_per_class=1, random_state=7)
nb = GaussianNB().fit(X_nb, y_nb)

fig, axes = make_fig(1, 3, 16, 5)
# boundary
ax = axes[0]
h=0.05; xmn,xmx=X_nb[:,0].min()-1,X_nb[:,0].max()+1; ymn,ymx=X_nb[:,1].min()-1,X_nb[:,1].max()+1
xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
Z=nb.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
ax.contourf(xx,yy,Z,alpha=0.2,cmap=ListedColormap([C2,C1]))
ax.scatter(X_nb[:,0],X_nb[:,1],c=[C1 if y==1 else C2 for y in y_nb],s=22,alpha=0.8)
style_ax(ax,'Naive Bayes – Decision Boundary','F1','F2')

# class-conditional gaussians f1
ax = axes[1]
for cls, col, label in zip([0,1],[C2,C1],['Class 0','Class 1']):
    mu=nb.theta_[cls,0]; sigma=np.sqrt(nb.var_[cls,0])
    x=np.linspace(X_nb[:,0].min()-1, X_nb[:,0].max()+1, 300)
    ax.plot(x, norm.pdf(x,mu,sigma), color=col, lw=2.5, label=label)
    ax.fill_between(x, norm.pdf(x,mu,sigma), alpha=0.15, color=col)
    ax.axvline(mu, color=col, ls='--', lw=1)
style_ax(ax,'Class-Conditional Gaussians (F1)','Feature 1','Density')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# posterior probabilities
ax = axes[2]
probs_nb = nb.predict_proba(X_nb)[:,1]
ax.hist(probs_nb[y_nb==0],bins=20,color=C2,alpha=0.7,label='Class 0')
ax.hist(probs_nb[y_nb==1],bins=20,color=C1,alpha=0.7,label='Class 1')
ax.axvline(0.5,color=C4,ls='--',lw=1.5)
style_ax(ax,'Posterior P(y=1|X)','Probability','Count')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

charts['naive_bayes'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 7. RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════
X_rf, y_rf = datasets.make_classification(n_samples=400, n_features=2, n_redundant=0,
                                           n_clusters_per_class=1, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X_rf, y_rf)

fig, axes = make_fig(1, 3, 16, 5)
# ensemble boundary
ax = axes[0]
h=0.04; xmn,xmx=X_rf[:,0].min()-1,X_rf[:,0].max()+1; ymn,ymx=X_rf[:,1].min()-1,X_rf[:,1].max()+1
xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
Z=rf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
ax.contourf(xx,yy,Z,alpha=0.2,cmap=ListedColormap([C2,C1]))
ax.scatter(X_rf[:,0],X_rf[:,1],c=[C1 if y==1 else C2 for y in y_rf],s=20,alpha=0.7)
style_ax(ax,'Random Forest – Decision Boundary','F1','F2')

# n_estimators vs oob
oob_scores=[]
ns=range(5,201,10)
for n in ns:
    m=RandomForestClassifier(n_estimators=n,oob_score=True,random_state=42).fit(X_rf,y_rf)
    oob_scores.append(m.oob_score_)
ax = axes[1]
ax.plot(ns,oob_scores,color=C1,lw=2,marker='o',markersize=3,markerfacecolor=C4)
ax.axhline(max(oob_scores),color=C2,ls='--',lw=1,label=f'Max OOB={max(oob_scores):.3f}')
style_ax(ax,'Trees vs OOB Score','# Estimators','OOB Score')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# feature importance
ax = axes[2]
fi=rf.feature_importances_
ax.bar(['F1','F2'],fi,color=[C1,C2])
ax.set_ylim(0,1); style_ax(ax,'Feature Importances','Feature','Gini Importance')
for i,(f,v) in enumerate(zip(['F1','F2'],fi)): ax.text(i,v+0.01,f'{v:.3f}',ha='center',color=TEXT,fontsize=10)

charts['random_forest'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 8. GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════════
X_gb, y_gb = datasets.make_classification(n_samples=400, n_features=2, n_redundant=0, random_state=42)
Xtr_gb,Xte_gb,ytr_gb,yte_gb = train_test_split(X_gb,y_gb,test_size=0.2,random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(Xtr_gb,ytr_gb)

fig, axes = make_fig(1, 3, 16, 5)
# staged scores
train_scores=[gb.train_score_[i] for i in range(200)]
ax = axes[0]
ax.plot(range(1,201),train_scores,color=C1,lw=2,label='Train Loss')
style_ax(ax,'GBM – Training Loss Curve','Boosting Iteration','Log Loss')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# boundary
ax = axes[1]
h=0.05; xmn,xmx=X_gb[:,0].min()-1,X_gb[:,0].max()+1; ymn,ymx=X_gb[:,1].min()-1,X_gb[:,1].max()+1
xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
Z=gb.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
ax.contourf(xx,yy,Z,alpha=0.2,cmap=ListedColormap([C2,C1]))
ax.scatter(X_gb[:,0],X_gb[:,1],c=[C1 if y==1 else C2 for y in y_gb],s=20,alpha=0.7)
style_ax(ax,'GBM – Decision Boundary','F1','F2')

# learning rate comparison
ax = axes[2]
for lr,col in zip([0.01,0.1,0.5,1.0],[C3,C1,C4,C2]):
    m=GradientBoostingClassifier(n_estimators=150,learning_rate=lr,max_depth=3,random_state=42).fit(Xtr_gb,ytr_gb)
    ax.plot(range(1,151),[m.train_score_[i] for i in range(150)],color=col,lw=1.8,label=f'lr={lr}')
style_ax(ax,'Learning Rate Comparison','Iteration','Train Log Loss')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

charts['gradient_boosting'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 9. ADABOOST
# ══════════════════════════════════════════════════════════════════════════
X_ada, y_ada = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, random_state=42)
ada = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X_ada, y_ada)

fig, axes = make_fig(1, 3, 16, 5)
# boundary
ax = axes[0]
h=0.05; xmn,xmx=X_ada[:,0].min()-1,X_ada[:,0].max()+1; ymn,ymx=X_ada[:,1].min()-1,X_ada[:,1].max()+1
xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
Z=ada.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
ax.contourf(xx,yy,Z,alpha=0.2,cmap=ListedColormap([C2,C1]))
ax.scatter(X_ada[:,0],X_ada[:,1],c=[C1 if y==1 else C2 for y in y_ada],s=22,alpha=0.8)
style_ax(ax,'AdaBoost – Decision Boundary','F1','F2')

# estimator weights
ax = axes[1]
ax.bar(range(len(ada.estimator_weights_)),ada.estimator_weights_,color=C1,alpha=0.8,width=1)
style_ax(ax,'Estimator Weights','Estimator Index','Alpha Weight')

# staged accuracy
staged=[ada.staged_score(X_ada,y_ada).__next__() for _ in range(1)]
staged_acc=list(ada.staged_score(X_ada,y_ada))
ax = axes[2]
ax.plot(range(1,len(staged_acc)+1),staged_acc,color=C1,lw=2)
ax.axhline(max(staged_acc),color=C4,ls='--',lw=1.5,label=f'Max={max(staged_acc):.3f}')
style_ax(ax,'AdaBoost Staged Accuracy','Estimators','Accuracy')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

charts['adaboost'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 10. K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
X_km, _ = datasets.make_blobs(n_samples=400, centers=4, cluster_std=1.1, random_state=42)
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_km)

fig, axes = make_fig(1, 3, 16, 5)
# clusters
ax = axes[0]
colors_km = [C1,C2,C4,C5]
for c in range(4):
    m=km.labels_==c
    ax.scatter(X_km[m,0],X_km[m,1],color=colors_km[c],s=22,alpha=0.7,label=f'Cluster {c}')
ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='X',s=200,color=TEXT,zorder=5,edgecolors=BG,lw=1,label='Centroids')
style_ax(ax,'K-Means – Clusters','F1','F2')
ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# elbow
ax = axes[1]
inertias=[KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_km).inertia_ for k in range(1,11)]
ax.plot(range(1,11),inertias,color=C1,lw=2,marker='o',markersize=6,markerfacecolor=C4)
ax.axvline(4,color=C2,ls='--',lw=1.5,label='Optimal k=4')
style_ax(ax,'Elbow Curve – Inertia','k','Inertia')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# silhouette
from sklearn.metrics import silhouette_score
sil=[silhouette_score(X_km,KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_km).labels_) for k in range(2,11)]
ax = axes[2]
ax.plot(range(2,11),sil,color=C1,lw=2,marker='o',markersize=6,markerfacecolor=C4)
ax.axvline(4,color=C2,ls='--',lw=1.5,label='k=4')
style_ax(ax,'Silhouette Score vs k','k','Silhouette Score')
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

charts['kmeans'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 11. DBSCAN
# ══════════════════════════════════════════════════════════════════════════
X_db, _ = datasets.make_moons(n_samples=400, noise=0.07, random_state=42)
X_db2, _ = datasets.make_circles(n_samples=400, noise=0.05, factor=0.5, random_state=42)

fig, axes = make_fig(1, 3, 16, 5)
for ax, X_d, eps_val, title in zip(axes[:2],
    [X_db, X_db2], [0.15, 0.18], ['DBSCAN – Moons','DBSCAN – Circles']):
    db_m = DBSCAN(eps=eps_val, min_samples=5).fit(X_d)
    labels=db_m.labels_; unique=set(labels)
    col_map={-1:MUTED,0:C1,1:C2,2:C4,3:C5}
    for lbl in unique:
        m=labels==lbl; col=col_map.get(lbl,C8)
        ax.scatter(X_d[m,0],X_d[m,1],color=col,s=22,alpha=0.8,
                   label='Noise' if lbl==-1 else f'Cluster {lbl}')
    style_ax(ax,title,'F1','F2')
    ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)

# eps vs n_clusters
ax = axes[2]
eps_vals=np.linspace(0.05,0.5,40)
n_clusters=[len(set(DBSCAN(eps=e,min_samples=5).fit(X_db).labels_)-{-1}) for e in eps_vals]
noise_pcts=[np.sum(DBSCAN(eps=e,min_samples=5).fit(X_db).labels_==-1)/len(X_db) for e in eps_vals]
ax2=ax.twinx()
ax.plot(eps_vals,n_clusters,color=C1,lw=2,label='# Clusters')
ax2.plot(eps_vals,noise_pcts,color=C2,lw=2,ls='--',label='Noise %')
ax.set_facecolor(SURFACE); ax.tick_params(colors=MUTED); ax2.tick_params(colors=MUTED)
for sp in ax.spines.values(): sp.set_color(BORDER)
ax.set_title('DBSCAN: eps vs Clusters & Noise',color=TEXT,fontsize=11,fontweight='bold',pad=8)
ax.set_xlabel('eps',color=MUTED); ax.set_ylabel('#Clusters',color=C1); ax2.set_ylabel('Noise %',color=C2)
ax.grid(True,color=BORDER,linewidth=0.5,alpha=0.7)
lines1,l1=ax.get_legend_handles_labels(); lines2,l2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,l1+l2,fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

charts['dbscan'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 12. PCA
# ══════════════════════════════════════════════════════════════════════════
digits = datasets.load_digits()
Xd = StandardScaler().fit_transform(digits.data)
pca_full = PCA().fit(Xd)
pca2 = PCA(n_components=2).fit_transform(Xd)

fig, axes = make_fig(1, 3, 16, 5)
# 2D projection
ax = axes[0]
for c in range(10):
    m=digits.target==c
    ax.scatter(pca2[m,0],pca2[m,1],s=12,alpha=0.7,label=str(c))
style_ax(ax,'PCA – 2D Projection (Digits)','PC1','PC2')
ax.legend(fontsize=6,ncol=2,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# explained variance
ax = axes[1]
ev=pca_full.explained_variance_ratio_
ax.bar(range(1,21),ev[:20],color=C1,alpha=0.8)
ax2=ax.twinx()
ax2.plot(range(1,21),np.cumsum(ev[:20]),color=C4,lw=2,marker='o',markersize=4)
ax2.axhline(0.95,color=C2,ls='--',lw=1,label='95% variance')
ax.set_facecolor(SURFACE); ax.tick_params(colors=MUTED); ax2.tick_params(colors=MUTED)
for sp in ax.spines.values(): sp.set_color(BORDER)
ax.set_title('Explained Variance Ratio',color=TEXT,fontsize=11,fontweight='bold',pad=8)
ax.set_xlabel('Component',color=MUTED); ax.set_ylabel('Explained Var',color=C1); ax2.set_ylabel('Cumulative',color=C4)
ax.grid(True,color=BORDER,linewidth=0.5,alpha=0.7)
ax2.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# biplot (first 2 PCs, first 8 features)
ax = axes[2]
pca4 = PCA(n_components=2).fit(Xd)
coeff=pca4.components_.T
scale=1/(coeff[:8,:].max()-coeff[:8,:].min())
ax.scatter(pca2[:,0],pca2[:,1],s=6,alpha=0.2,color=C3)
for i in range(8):
    ax.annotate('',xy=(coeff[i,0]*scale*30, coeff[i,1]*scale*30),xytext=(0,0),
                arrowprops=dict(arrowstyle='->',color=C4,lw=1.5))
    ax.text(coeff[i,0]*scale*33,coeff[i,1]*scale*33,f'F{i+1}',color=C4,fontsize=7)
style_ax(ax,'PCA Biplot (8 features)','PC1','PC2')

charts['pca'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 13. t-SNE
# ══════════════════════════════════════════════════════════════════════════
from sklearn.datasets import load_digits
dg=load_digits(); X_tsne=dg.data[:500]; y_tsne=dg.target[:500]
X_sc=StandardScaler().fit_transform(X_tsne)

fig, axes = make_fig(1, 3, 16, 5)
for ax, perp in zip(axes, [5,30,50]):
    emb=TSNE(n_components=2,perplexity=perp,random_state=42,max_iter=500).fit_transform(X_sc)
    for c in range(10):
        m=y_tsne==c
        ax.scatter(emb[m,0],emb[m,1],s=14,alpha=0.8,label=str(c))
    style_ax(ax,f't-SNE perplexity={perp}','Dim 1','Dim 2')
    if perp==5: ax.legend(fontsize=6,ncol=2,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

charts['tsne'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 14. MODEL EVALUATION – ROC, PR, Confusion Matrix, Calibration
# ══════════════════════════════════════════════════════════════════════════
X_ev, y_ev = datasets.make_classification(n_samples=600, n_features=10, random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X_ev,y_ev,test_size=0.3,random_state=42)

models_ev = {
    'Logistic Reg': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=50,random_state=42),
    'GBM': GradientBoostingClassifier(n_estimators=50,random_state=42),
    'SVM (RBF)': SVC(probability=True,kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=7),
}
cols_ev=[C1,C2,C4,C5,C6]

fig, axes = make_fig(2, 2, 14, 10)
axes=axes.ravel()

# ROC
ax=axes[0]
for (name,m),col in zip(models_ev.items(),cols_ev):
    m.fit(Xtr,ytr); probs=m.predict_proba(Xte)[:,1]
    fpr,tpr,_=roc_curve(yte,probs); roc_auc=auc(fpr,tpr)
    ax.plot(fpr,tpr,color=col,lw=2,label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],color=MUTED,ls='--',lw=1)
ax.fill_between([0,1],[0,1],alpha=0.05,color=MUTED)
style_ax(ax,'ROC Curves','FPR','TPR'); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
ax.legend(fontsize=7.5,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Precision-Recall
ax=axes[1]
for (name,m),col in zip(models_ev.items(),cols_ev):
    m.fit(Xtr,ytr); probs=m.predict_proba(Xte)[:,1]
    prec,rec,_=precision_recall_curve(yte,probs); pr_auc=auc(rec,prec)
    ax.plot(rec,prec,color=col,lw=2,label=f'{name} (AUC={pr_auc:.3f})')
style_ax(ax,'Precision-Recall Curves','Recall','Precision')
ax.legend(fontsize=7.5,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Confusion Matrix (best model = RF)
ax=axes[2]
rf_ev=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr)
cm=confusion_matrix(yte,rf_ev.predict(Xte))
im=ax.imshow(cm,cmap='YlOrRd',aspect='auto')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred 0','Pred 1'],color=TEXT); ax.set_yticklabels(['True 0','True 1'],color=TEXT)
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=22,fontweight='bold',color=TEXT)
ax.set_facecolor(SURFACE); ax.set_title('Confusion Matrix (Random Forest)',color=TEXT,fontsize=11,fontweight='bold',pad=8)
fig.colorbar(im,ax=ax)

# Learning Curves
ax=axes[3]
for (name,m),col in zip(list(models_ev.items())[:3],cols_ev[:3]):
    train_sz,tr_sc,val_sc=learning_curve(m,X_ev,y_ev,cv=5,n_jobs=-1,
                                          train_sizes=np.linspace(0.1,1,8))
    ax.plot(train_sz,val_sc.mean(axis=1),color=col,lw=2,label=name)
    ax.fill_between(train_sz,val_sc.mean(1)-val_sc.std(1),val_sc.mean(1)+val_sc.std(1),alpha=0.1,color=col)
style_ax(ax,'Learning Curves (Validation)','Training Size','CV Accuracy')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

charts['model_evaluation'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 15. NEURAL NETWORK – Architecture + Training Visualization
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16,5)); fig.patch.set_facecolor(BG)
# --- panel 1: MLP architecture diagram ---
ax1 = fig.add_subplot(1,3,1); ax1.set_facecolor(SURFACE); ax1.set_aspect('equal')
layer_sizes=[3,5,4,2]; layer_names=['Input','Hidden 1','Hidden 2','Output']
layer_x=[0.1,0.35,0.65,0.9]
node_positions=[]
for li,(sz,lx) in enumerate(zip(layer_sizes,layer_x)):
    ys=np.linspace(0.15,0.85,sz); node_positions.append(list(zip([lx]*sz,ys)))
    for y in ys:
        col=C1 if li==0 else (C2 if li==len(layer_sizes)-1 else C4)
        circ=plt.Circle((lx,y),0.035,color=col,zorder=4); ax1.add_patch(circ)
    ax1.text(lx,0.05,layer_names[li],ha='center',va='center',color=MUTED,fontsize=8)
for li in range(len(layer_sizes)-1):
    for (x1,y1) in node_positions[li]:
        for (x2,y2) in node_positions[li+1]:
            ax1.plot([x1,x2],[y1,y2],color=BORDER,lw=0.8,zorder=1,alpha=0.8)
ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.axis('off')
ax1.set_title('MLP Architecture',color=TEXT,fontsize=11,fontweight='bold',pad=8)
for sp in ax1.spines.values(): sp.set_color(BORDER)

# --- panel 2: simulated training curves ---
ax2=fig.add_subplot(1,3,2); ax2.set_facecolor(SURFACE)
epochs=np.arange(1,101)
tr_loss=0.8*np.exp(-0.05*epochs)+0.08+np.random.normal(0,0.01,100)
va_loss=0.85*np.exp(-0.04*epochs)+0.12+np.random.normal(0,0.015,100)
tr_acc=1-(0.6*np.exp(-0.06*epochs)+0.05)+np.random.normal(0,0.008,100)
va_acc=1-(0.62*np.exp(-0.05*epochs)+0.08)+np.random.normal(0,0.012,100)
ax2.plot(epochs,tr_loss,color=C1,lw=2,label='Train Loss')
ax2.plot(epochs,va_loss,color=C2,lw=2,ls='--',label='Val Loss')
ax2.set_title('Training & Validation Loss',color=TEXT,fontsize=11,fontweight='bold',pad=8)
ax2.tick_params(colors=MUTED,labelsize=9); ax2.set_xlabel('Epoch',color=MUTED); ax2.set_ylabel('Loss',color=MUTED)
for sp in ax2.spines.values(): sp.set_color(BORDER)
ax2.grid(True,color=BORDER,linewidth=0.5,alpha=0.7)
ax2.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# --- panel 3: accuracy curves ---
ax3=fig.add_subplot(1,3,3); ax3.set_facecolor(SURFACE)
ax3.plot(epochs,np.clip(tr_acc,0,1),color=C4,lw=2,label='Train Acc')
ax3.plot(epochs,np.clip(va_acc,0,1),color=C6,lw=2,ls='--',label='Val Acc')
ax3.set_title('Training & Validation Accuracy',color=TEXT,fontsize=11,fontweight='bold',pad=8)
ax3.tick_params(colors=MUTED,labelsize=9); ax3.set_xlabel('Epoch',color=MUTED); ax3.set_ylabel('Accuracy',color=MUTED)
for sp in ax3.spines.values(): sp.set_color(BORDER)
ax3.grid(True,color=BORDER,linewidth=0.5,alpha=0.7)
ax3.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

plt.tight_layout(pad=1.5)
charts['neural_network'] = fig_to_b64(fig)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# 16. ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
x_act = np.linspace(-4,4,300)
activations = {
    'Sigmoid': (1/(1+np.exp(-x_act)), C1),
    'Tanh': (np.tanh(x_act), C2),
    'ReLU': (np.maximum(0,x_act), C4),
    'Leaky ReLU': (np.where(x_act>0,x_act,0.1*x_act), C5),
    'ELU': (np.where(x_act>0,x_act,np.exp(x_act)-1), C6),
    'Swish': (x_act/(1+np.exp(-x_act)), C7),
}

fig, axes = make_fig(2,3,16,8)
axes=axes.ravel()
for ax,(name,(vals,col)) in zip(axes,activations.items()):
    ax.plot(x_act,vals,color=col,lw=2.5)
    ax.axhline(0,color=BORDER,lw=0.8); ax.axvline(0,color=BORDER,lw=0.8)
    ax.fill_between(x_act,0,vals,alpha=0.1,color=col)
    style_ax(ax,name,'z','f(z)')
plt.tight_layout(pad=1.5)
charts['activations'] = fig_to_b64(fig)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# 17. REGULARIZATION: L1/L2/Dropout effect
# ══════════════════════════════════════════════════════════════════════════
fig, axes = make_fig(1,3,16,5)
np.random.seed(0)
X_reg=np.sort(np.random.uniform(-3,3,60))
y_reg=np.sin(X_reg)+np.random.normal(0,0.4,60)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

ax=axes[0]
ax.scatter(X_reg,y_reg,color=C3,s=25,alpha=0.7,zorder=3)
Xr=X_reg.reshape(-1,1)
colors_reg=[C4,C1,C2,C5]
for d,col,lbl in zip([1,5,10,15],colors_reg,['degree=1','degree=5','degree=10','degree=15']):
    m=make_pipeline(PolynomialFeatures(d),LinearRegression()).fit(Xr,y_reg)
    xs=np.linspace(-3,3,300).reshape(-1,1)
    ax.plot(xs,m.predict(xs),color=col,lw=2,label=lbl)
ax.set_ylim(-3,3)
style_ax(ax,'Overfitting vs Degree','X','y')
ax.legend(fontsize=7,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

ax=axes[1]
for alpha,col,lbl in zip([0.001,0.01,0.1,1.0],colors_reg,['α=0.001','α=0.01','α=0.1','α=1.0']):
    m=make_pipeline(PolynomialFeatures(15),Ridge(alpha=alpha)).fit(Xr,y_reg)
    xs=np.linspace(-3,3,300).reshape(-1,1)
    ax.plot(xs,m.predict(xs),color=col,lw=2,label=lbl)
ax.scatter(X_reg,y_reg,color=C3,s=15,alpha=0.5,zorder=3)
ax.set_ylim(-3,3)
style_ax(ax,'Ridge (L2) Regularization','X','y')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

ax=axes[2]
for alpha,col,lbl in zip([0.001,0.01,0.05,0.1],colors_reg,['α=0.001','α=0.01','α=0.05','α=0.1']):
    m=make_pipeline(PolynomialFeatures(15),Lasso(alpha=alpha,max_iter=10000)).fit(Xr,y_reg)
    xs=np.linspace(-3,3,300).reshape(-1,1)
    ax.plot(xs,m.predict(xs),color=col,lw=2,label=lbl)
ax.scatter(X_reg,y_reg,color=C3,s=15,alpha=0.5,zorder=3)
ax.set_ylim(-3,3)
style_ax(ax,'Lasso (L1) Regularization','X','y')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

charts['regularization'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 18. BIAS-VARIANCE TRADEOFF
# ══════════════════════════════════════════════════════════════════════════
fig, axes = make_fig(1,2,14,5)
complexity=np.linspace(0,10,200)
bias2=np.exp(-0.5*complexity)+0.05
variance=0.05+0.003*np.exp(0.6*complexity)
noise=np.ones_like(complexity)*0.1
total=bias2+variance+noise

ax=axes[0]
ax.plot(complexity,bias2,color=C2,lw=2.5,label='Bias²')
ax.plot(complexity,variance,color=C4,lw=2.5,label='Variance')
ax.plot(complexity,noise,color=MUTED,lw=1.5,ls=':',label='Irreducible Noise')
ax.plot(complexity,total,color=C1,lw=2.5,ls='--',label='Total Error')
opt=np.argmin(total)
ax.axvline(complexity[opt],color=C6,ls='--',lw=1.5,alpha=0.9,label='Optimal Complexity')
ax.fill_between(complexity,0,bias2,alpha=0.08,color=C2)
ax.fill_between(complexity,0,variance,alpha=0.08,color=C4)
style_ax(ax,'Bias-Variance Tradeoff','Model Complexity','Error')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
ax.set_ylim(0,1.5)

# underfitting vs overfitting diagram
ax=axes[1]
np.random.seed(1)
X_bv=np.sort(np.random.uniform(0,6,40))
y_bv=np.sin(X_bv)+np.random.normal(0,0.3,40)
xs=np.linspace(0,6,300).reshape(-1,1)
Xbv=X_bv.reshape(-1,1)
ax.scatter(X_bv,y_bv,color=C3,s=30,zorder=3,alpha=0.9,label='Data')
m1=make_pipeline(PolynomialFeatures(1),LinearRegression()).fit(Xbv,y_bv)
m2=make_pipeline(PolynomialFeatures(4),LinearRegression()).fit(Xbv,y_bv)
m3=make_pipeline(PolynomialFeatures(18),LinearRegression()).fit(Xbv,y_bv)
ax.plot(xs,m1.predict(xs),color=C2,lw=2,label='Underfit (degree=1)')
ax.plot(xs,m2.predict(xs),color=C6,lw=2,label='Good fit (degree=4)')
ax.plot(xs,np.clip(m3.predict(xs),-3,3),color=C4,lw=2,ls='--',label='Overfit (degree=18)')
style_ax(ax,'Underfitting vs Overfitting','X','y')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
ax.set_ylim(-3,3)

charts['bias_variance'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 19. CROSS VALIDATION
# ══════════════════════════════════════════════════════════════════════════
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
X_cv, y_cv = datasets.make_classification(n_samples=300, n_features=5, random_state=42)

fig, axes = make_fig(1,3,16,5)
# KFold illustration
ax=axes[0]; ax.set_facecolor(SURFACE); ax.axis('off')
n_splits=5; n_samples_show=20
kf=KFold(n_splits=n_splits)
fold_colors=[C1,C2,C4,C5,C6]
idx=np.arange(n_samples_show)
for fold_idx,(tr_idx,te_idx) in enumerate(kf.split(idx)):
    for i in idx:
        col=fold_colors[fold_idx] if i in te_idx else SURFACE
        rect=mpatches.FancyBboxPatch((i*0.047,1-(fold_idx+1)*0.17),0.04,0.13,
                                      boxstyle='round,pad=0.005',
                                      facecolor=col,edgecolor=BORDER,lw=0.8)
        ax.add_patch(rect)
    ax.text(-0.05,1-(fold_idx+0.5)*0.17,f'Fold {fold_idx+1}',va='center',ha='right',color=TEXT,fontsize=8)
ax.set_xlim(-0.15,1.05); ax.set_ylim(0.1,1.1)
ax.set_title('K-Fold Cross Validation (k=5)',color=TEXT,fontsize=11,fontweight='bold',pad=8)
train_p=mpatches.Patch(color=SURFACE,edgecolor=BORDER,label='Train')
test_p=mpatches.Patch(color=C1,label='Test')
ax.legend(handles=[train_p,test_p],fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,loc='lower right')

# CV scores comparison
ax=axes[1]
model_names=['LR','DT','RF','GBM','SVM','KNN','NB']
cv_models=[LogisticRegression(),DecisionTreeClassifier(max_depth=5,random_state=42),
           RandomForestClassifier(n_estimators=50,random_state=42),
           GradientBoostingClassifier(n_estimators=50,random_state=42),
           SVC(probability=True),KNeighborsClassifier(),GaussianNB()]
means=[]; stds=[]
for m in cv_models:
    sc=cross_val_score(m,X_cv,y_cv,cv=5)
    means.append(sc.mean()); stds.append(sc.std())
bars=ax.bar(model_names,means,yerr=stds,color=PALETTE[:7],alpha=0.85,
            error_kw=dict(ecolor=TEXT,lw=1.5,capsize=4))
ax.set_ylim(0.6,1.05)
for bar,val in zip(bars,means): ax.text(bar.get_x()+bar.get_width()/2,val+0.012,f'{val:.2f}',ha='center',color=TEXT,fontsize=8)
style_ax(ax,'5-Fold CV Accuracy Comparison','Model','Mean CV Accuracy')

# Validation curve
ax=axes[2]
param_range=np.arange(1,21)
tr_sc,va_sc=validation_curve(DecisionTreeClassifier(random_state=42),X_cv,y_cv,
                              param_name='max_depth',param_range=param_range,cv=5)
ax.plot(param_range,tr_sc.mean(1),color=C1,lw=2,label='Train Score')
ax.plot(param_range,va_sc.mean(1),color=C2,lw=2,label='Val Score')
ax.fill_between(param_range,tr_sc.mean(1)-tr_sc.std(1),tr_sc.mean(1)+tr_sc.std(1),alpha=0.1,color=C1)
ax.fill_between(param_range,va_sc.mean(1)-va_sc.std(1),va_sc.mean(1)+va_sc.std(1),alpha=0.1,color=C2)
style_ax(ax,'Validation Curve (DT max_depth)','Max Depth','Score')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

charts['cross_validation'] = fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════
# 20. HIERARCHICAL CLUSTERING + GAUSSIAN MIXTURE
# ══════════════════════════════════════════════════════════════════════════
from sklearn.mixture import GaussianMixture
X_hc, _ = datasets.make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)

fig, axes = make_fig(1,3,16,5)
# dendrogram
ax=axes[0]
linked=linkage(X_hc[:50],'ward')
ax.set_facecolor(SURFACE)
dendrogram(linked,ax=ax,color_threshold=15,
           link_color_func=lambda k: [C1,C2,C4,C5,C6,C3][k%6],
           above_threshold_color=MUTED)
ax.tick_params(colors=MUTED,labelsize=7)
for sp in ax.spines.values(): sp.set_color(BORDER)
ax.set_title('Hierarchical Clustering – Dendrogram',color=TEXT,fontsize=11,fontweight='bold',pad=8)
ax.set_xlabel('Sample Index',color=MUTED); ax.set_ylabel('Distance',color=MUTED)

# agglomerative
ax=axes[1]
ag=AgglomerativeClustering(n_clusters=4).fit(X_hc)
for c,col in enumerate([C1,C2,C4,C5]):
    m=ag.labels_==c
    ax.scatter(X_hc[m,0],X_hc[m,1],color=col,s=22,alpha=0.8,label=f'Cluster {c}')
style_ax(ax,'Agglomerative Clustering','F1','F2')
ax.legend(fontsize=7,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Gaussian Mixture
ax=axes[2]
gm=GaussianMixture(n_components=4,random_state=42).fit(X_hc)
labels_gm=gm.predict(X_hc)
for c,col in enumerate([C1,C2,C4,C5]):
    m=labels_gm==c
    ax.scatter(X_hc[m,0],X_hc[m,1],color=col,s=22,alpha=0.7,label=f'GMM {c}')
# draw ellipses
from matplotlib.patches import Ellipse
for k in range(4):
    mu=gm.means_[k]; cov=gm.covariances_[k]
    vals,vecs=np.linalg.eigh(cov); order=vals.argsort()[::-1]
    vals,vecs=vals[order],vecs[:,order]
    angle=np.degrees(np.arctan2(*vecs[:,0][::-1]))
    for nsig,alpha in [(1,0.4),(2,0.15)]:
        ell=Ellipse(xy=mu,width=2*nsig*np.sqrt(vals[0]),height=2*nsig*np.sqrt(vals[1]),
                    angle=angle,edgecolor=PALETTE[k],facecolor='none',lw=1.5,alpha=alpha+0.2)
        ax.add_patch(ell)
style_ax(ax,'Gaussian Mixture Model','F1','F2')
ax.legend(fontsize=7,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

charts['clustering_advanced'] = fig_to_b64(fig)

# ── save all charts ───────────────────────────────────────────────────────
with open('/home/user/workspace/ml-visualizer/charts.json','w') as f:
    json.dump(charts,f)
print("DONE – charts:", list(charts.keys()))
