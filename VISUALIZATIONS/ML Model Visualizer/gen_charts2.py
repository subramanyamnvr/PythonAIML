import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.ensemble import BaggingClassifier, IsolationForest
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import io, base64, json, warnings
warnings.filterwarnings('ignore')

BG='#0F0E0C'; SURFACE='#1A1916'; BORDER='#2E2D2A'; TEXT='#CDCCCA'; MUTED='#797876'
C1='#20808D'; C2='#A84B2F'; C3='#BCE2E7'; C4='#FFC553'; C5='#944454'
C6='#6DAA45'; C7='#A86FDF'; C8='#5591C7'; C9='#E8AF34'; C10='#DD6974'
PALETTE=[C1,C2,C3,C4,C5,C6,C7,C8]

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
    buf.seek(0); b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig); return b64

def make_fig(nrows=1, ncols=1, w=14, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig, axes

charts = {}

# ── SVR ──────────────────────────────────────────────────────────────────
np.random.seed(42)
X_svr = np.sort(np.random.uniform(-3,3,120)).reshape(-1,1)
y_svr = np.sin(X_svr.ravel()) + np.random.normal(0,0.25,120)
fig, axes = make_fig(1,3,16,5)
xs = np.linspace(-3.2,3.2,300).reshape(-1,1)
for ax, eps, col in zip(axes,[0.1,0.3,0.8],[C1,C4,C2]):
    m = SVR(kernel='rbf', C=5, epsilon=eps).fit(X_svr, y_svr)
    yp = m.predict(xs)
    ax.scatter(X_svr, y_svr, color=C3, s=18, alpha=0.6, zorder=3, label='Data')
    ax.plot(xs, yp, color=col, lw=2.5, label=f'SVR ε={eps}')
    ax.fill_between(xs.ravel(), yp-eps, yp+eps, alpha=0.18, color=col, label='ε-tube')
    sv = m.support_vectors_
    ax.scatter(sv, m.predict(sv), color=C4, s=80, marker='D', zorder=4, label='SVs', edgecolors=BG, lw=0.8)
    style_ax(ax, f'SVR — RBF Kernel ε={eps}', 'X', 'y')
    ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
charts['svr'] = fig_to_b64(fig)

# ── BAGGING ──────────────────────────────────────────────────────────────
X_bag, y_bag = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, random_state=42)
fig, axes = make_fig(1,3,16,5)
single_dt = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X_bag, y_bag)
bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=None),
                        n_estimators=50, max_samples=0.8, max_features=1.0, random_state=42).fit(X_bag, y_bag)
for ax, clf, title in zip(axes[:2],[single_dt,bag],['Single DT (overfit)','Bagging 50 DTs']):
    h=0.05; xmn,xmx=X_bag[:,0].min()-1,X_bag[:,0].max()+1; ymn,ymx=X_bag[:,1].min()-1,X_bag[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx,yy,Z,alpha=0.25,cmap=ListedColormap([C2,C1]))
    ax.scatter(X_bag[:,0],X_bag[:,1],c=[C1 if y==1 else C2 for y in y_bag],s=22,alpha=0.8)
    style_ax(ax,f'{title}\n(Acc={clf.score(X_bag,y_bag):.3f})','F1','F2')
# Bootstrap sample viz
ax = axes[2]
np.random.seed(7)
N=20; orig=np.arange(N)
for i in range(6):
    sample=np.random.choice(N,N,replace=True)
    unique=np.unique(sample)
    oob=np.setdiff1d(orig,unique)
    ax.barh(i, len(unique)/N, left=0, color=C1, alpha=0.7, height=0.6, label='In-bag' if i==0 else '')
    ax.barh(i, len(oob)/N, left=len(unique)/N, color=C2, alpha=0.7, height=0.6, label='OOB' if i==0 else '')
ax.axvline(0.632, color=C4, ls='--', lw=1.5, label='63.2% avg in-bag')
style_ax(ax,'Bootstrap Sampling (6 runs)','Fraction','Bootstrap Run')
ax.set_xlim(0,1); ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
charts['bagging'] = fig_to_b64(fig)

# ── XGBOOST / LIGHTGBM comparison (simulated) ────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
fig, axes = make_fig(1,3,16,5)
X_xg, y_xg = datasets.make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X_xg,y_xg,test_size=0.2,random_state=42)

# Training time vs accuracy tradeoff (simulated curves)
ax = axes[0]
n_trees = np.arange(10,201,10)
# GBM learning curves
gbm_scores = [GradientBoostingClassifier(n_estimators=n,learning_rate=0.1,max_depth=3,random_state=42).fit(Xtr,ytr).score(Xte,yte) for n in n_trees]
rf_scores  = [RandomForestClassifier(n_estimators=n,random_state=42).fit(Xtr,ytr).score(Xte,yte) for n in n_trees]
ax.plot(n_trees,gbm_scores,color=C1,lw=2,label='GBM (sequential)')
ax.plot(n_trees,rf_scores,color=C2,lw=2,label='RF (parallel)')
style_ax(ax,'GBM vs RF: Trees vs Test Acc','# Trees','Test Accuracy')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Feature importance comparison
ax = axes[1]
gbm_fi = GradientBoostingClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr).feature_importances_
rf_fi  = RandomForestClassifier(n_estimators=100,random_state=42).fit(Xtr,ytr).feature_importances_
top_n = 8
gbm_top = gbm_fi.argsort()[::-1][:top_n]; rf_top = rf_fi.argsort()[::-1][:top_n]
x=np.arange(top_n); w=0.35
ax.bar(x-w/2,[gbm_fi[i] for i in gbm_top],width=w,color=C1,alpha=0.85,label='GBM')
ax.bar(x+w/2,[rf_fi[i]  for i in rf_top],width=w,color=C2,alpha=0.85,label='RF')
ax.set_xticks(x); ax.set_xticklabels([f'F{i}' for i in gbm_top],color=MUTED,fontsize=8)
style_ax(ax,'Top-8 Feature Importance (GBM vs RF)','Feature','Importance')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Hyperparameter heatmap (learning_rate x max_depth → val acc)
ax = axes[2]
lrs=[0.01,0.05,0.1,0.2]; depths=[2,3,4,5]
grid = np.array([[GradientBoostingClassifier(n_estimators=80,learning_rate=lr,max_depth=d,random_state=42).fit(Xtr,ytr).score(Xte,yte) for d in depths] for lr in lrs])
im=ax.imshow(grid,cmap='YlGn',aspect='auto',vmin=0.7,vmax=1.0)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels([f'd={d}' for d in depths],color=MUTED,fontsize=9)
ax.set_yticklabels([f'lr={lr}' for lr in lrs],color=MUTED,fontsize=9)
for i in range(4):
    for j in range(4): ax.text(j,i,f'{grid[i,j]:.3f}',ha='center',va='center',fontsize=9,color=BG,fontweight='bold')
plt.colorbar(im,ax=ax)
ax.set_facecolor(SURFACE); ax.set_title('GBM Grid: lr × depth → Acc',color=TEXT,fontsize=11,fontweight='bold',pad=8)
charts['xgboost'] = fig_to_b64(fig)

# ── ANOMALY DETECTION ─────────────────────────────────────────────────────
np.random.seed(42)
X_norm = 0.3*np.random.randn(200,2)
X_out  = np.random.uniform(-4,4,(20,2))
X_ano  = np.vstack([X_norm,X_out])
y_ano  = np.array([1]*200+[-1]*20)

fig, axes = make_fig(1,3,16,5)
models_ano = {
    'Isolation Forest': IsolationForest(contamination=0.09, random_state=42),
    'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.09),
    'Elliptic Envelope': EllipticEnvelope(contamination=0.09, random_state=42),
}
for ax,(name,clf) in zip(axes,models_ano.items()):
    if name=='LOF': preds=clf.fit_predict(X_ano)
    else: preds=clf.fit(X_ano).predict(X_ano)
    inlier=preds==1; outlier=preds==-1
    ax.scatter(X_ano[inlier,0],X_ano[inlier,1],color=C1,s=22,alpha=0.7,label='Inlier')
    ax.scatter(X_ano[outlier,0],X_ano[outlier,1],color=C2,s=60,marker='x',lw=2,label='Outlier')
    if name!='LOF':
        h=0.05; xmn,xmx=X_ano[:,0].min()-0.5,X_ano[:,0].max()+0.5; ymn,ymx=X_ano[:,1].min()-0.5,X_ano[:,1].max()+0.5
        xx,yy=np.meshgrid(np.arange(xmn,xmx,h),np.arange(ymn,ymx,h))
        Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx,yy,Z,alpha=0.1,cmap=ListedColormap([C2,C1]))
    style_ax(ax,name,'F1','F2')
    ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
charts['anomaly_detection'] = fig_to_b64(fig)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────
fig, axes = make_fig(2,3,16,9)
axes=axes.ravel()
# 1. Scaling comparison
np.random.seed(0)
raw=np.random.exponential(2,200).reshape(-1,1)
norm=MinMaxScaler().fit_transform(raw).ravel()
std=StandardScaler().fit_transform(raw).ravel()
ax=axes[0]
ax.hist(raw.ravel(),bins=30,color=C2,alpha=0.7,label='Raw',density=True)
ax.hist(std,bins=30,color=C1,alpha=0.7,label='Z-score',density=True)
ax.hist(norm,bins=30,color=C4,alpha=0.7,label='MinMax',density=True)
style_ax(ax,'Feature Scaling Comparison','Value','Density')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# 2. Polynomial features effect
np.random.seed(1)
X_pf=np.sort(np.random.uniform(-2,2,60))
y_pf=(X_pf**2)+np.random.normal(0,0.3,60)
ax=axes[1]
ax.scatter(X_pf,y_pf,color=C3,s=20,alpha=0.8,label='Data')
Xpf=X_pf.reshape(-1,1); xs=np.linspace(-2.2,2.2,200).reshape(-1,1)
for deg,col in zip([1,2,4],[C2,C6,C4]):
    m=make_pipeline(PolynomialFeatures(deg),Ridge()).fit(Xpf,y_pf)
    ax.plot(xs,m.predict(xs),color=col,lw=2,label=f'deg={deg}')
style_ax(ax,'Polynomial Features','X','y')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# 3. Correlation heatmap
X_corr,_=datasets.make_classification(n_samples=300,n_features=8,random_state=42)
feat_names=[f'F{i}' for i in range(8)]
corr=np.corrcoef(X_corr.T)
ax=axes[2]; ax.set_facecolor(SURFACE)
im=ax.imshow(corr,cmap='RdYlGn',vmin=-1,vmax=1)
ax.set_xticks(range(8)); ax.set_yticks(range(8))
ax.set_xticklabels(feat_names,color=MUTED,fontsize=8); ax.set_yticklabels(feat_names,color=MUTED,fontsize=8)
for i in range(8):
    for j in range(8): ax.text(j,i,f'{corr[i,j]:.1f}',ha='center',va='center',fontsize=7,color='black' if abs(corr[i,j])<0.7 else 'white')
plt.colorbar(im,ax=ax)
ax.set_title('Feature Correlation Matrix',color=TEXT,fontsize=11,fontweight='bold',pad=8)

# 4. Missing value patterns
ax=axes[3]; ax.set_facecolor(SURFACE)
np.random.seed(5)
pattern=np.random.rand(10,8)
mask=np.random.rand(10,8)<0.25
ax.imshow(mask,cmap=ListedColormap([C1,C2]),aspect='auto',alpha=0.85)
ax.set_xticks(range(8)); ax.set_yticks(range(10))
ax.set_xticklabels(feat_names,color=MUTED,fontsize=8)
ax.set_yticklabels([f'Row {i}' for i in range(10)],color=MUTED,fontsize=8)
p1=mpatches.Patch(color=C1,label='Present'); p2=mpatches.Patch(color=C2,label='Missing')
ax.legend(handles=[p1,p2],fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,loc='upper right')
ax.set_title('Missing Value Pattern',color=TEXT,fontsize=11,fontweight='bold',pad=8)

# 5. Feature importance ranking
from sklearn.ensemble import RandomForestClassifier
X_fi,y_fi=datasets.make_classification(n_samples=300,n_features=10,random_state=42)
fi=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_fi,y_fi).feature_importances_
ax=axes[4]
order=fi.argsort()[::-1]
colors_fi=[C1 if i<3 else C3 for i in range(10)]
bars=ax.bar(range(10),[fi[o] for o in order],color=colors_fi)
ax.set_xticks(range(10)); ax.set_xticklabels([f'F{o}' for o in order],color=MUTED,fontsize=8)
style_ax(ax,'Feature Importance Ranking','Feature','Importance')

# 6. Class imbalance + SMOTE-like resampling
np.random.seed(3)
n_maj=200; n_min=20
X_imb=np.vstack([np.random.randn(n_maj,2)*1.5+[2,2],np.random.randn(n_min,2)*0.8+[-1,-1]])
y_imb=np.array([0]*n_maj+[1]*n_min)
ax=axes[5]
ax.scatter(X_imb[y_imb==0,0],X_imb[y_imb==0,1],color=C1,s=18,alpha=0.5,label=f'Class 0 (n={n_maj})')
ax.scatter(X_imb[y_imb==1,0],X_imb[y_imb==1,1],color=C2,s=50,alpha=0.9,label=f'Class 1 (n={n_min})',marker='*')
# synthetic oversampled points
synth=np.random.randn(80,2)*0.8+[-1,-1]
ax.scatter(synth[:,0],synth[:,1],color=C4,s=25,alpha=0.7,marker='^',label='SMOTE synthetic')
style_ax(ax,'Class Imbalance & SMOTE','F1','F2')
ax.legend(fontsize=7,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
plt.tight_layout(pad=1.5)
charts['feature_engineering'] = fig_to_b64(fig)
plt.close('all')

# ── HYPERPARAMETER TUNING ─────────────────────────────────────────────────
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
fig, axes = make_fig(1,3,16,5)
X_ht,y_ht=datasets.make_classification(n_samples=400,n_features=10,random_state=42)
Xtr,Xte,ytr,yte=train_test_split(X_ht,y_ht,test_size=0.2,random_state=42)

# Grid search heatmap (C vs gamma for SVM)
from sklearn.svm import SVC
Cs=np.logspace(-2,2,5); gammas=np.logspace(-3,1,5)
grid_acc=np.zeros((5,5))
for i,C in enumerate(Cs):
    for j,g in enumerate(gammas):
        grid_acc[i,j]=cross_val_score(SVC(C=C,gamma=g),Xtr,ytr,cv=3).mean()
ax=axes[0]; ax.set_facecolor(SURFACE)
im=ax.imshow(grid_acc,cmap='YlGn',aspect='auto',vmin=0.5,vmax=1.0)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([f'{g:.3f}' for g in gammas],color=MUTED,fontsize=7,rotation=30)
ax.set_yticklabels([f'{C:.2f}' for C in Cs],color=MUTED,fontsize=8)
for i in range(5):
    for j in range(5): ax.text(j,i,f'{grid_acc[i,j]:.2f}',ha='center',va='center',fontsize=7.5,color=BG,fontweight='bold')
plt.colorbar(im,ax=ax)
ax.set_title('SVM Grid Search: C × γ',color=TEXT,fontsize=11,fontweight='bold',pad=8)
ax.set_xlabel('gamma',color=MUTED); ax.set_ylabel('C',color=MUTED)

# Random search vs grid coverage
ax=axes[1]
np.random.seed(42)
grid_pts=np.array([[c,g] for c in np.linspace(0,1,5) for g in np.linspace(0,1,5)])
rand_pts=np.random.rand(25,2)
ax.scatter(grid_pts[:,0],grid_pts[:,1],color=C2,s=60,marker='s',label='Grid Search (25)')
ax.scatter(rand_pts[:,0],rand_pts[:,1],color=C1,s=40,alpha=0.8,label='Random Search (25)')
style_ax(ax,'Grid vs Random Search Coverage','Hyperparameter 1','Hyperparameter 2')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Bayesian optimization convergence (simulated)
ax=axes[2]
iters=np.arange(1,31)
np.random.seed(7)
rand_best=np.maximum.accumulate(0.6+0.35*np.random.rand(30))
bayes_best=np.maximum.accumulate(0.65+0.003*iters+0.015*np.random.randn(30))
bayes_best=np.clip(bayes_best,0,0.98)
ax.plot(iters,rand_best,color=C2,lw=2,label='Random Search')
ax.plot(iters,bayes_best,color=C1,lw=2,label='Bayesian Opt')
ax.fill_between(iters,rand_best,bayes_best,alpha=0.15,color=C6)
style_ax(ax,'Bayesian vs Random Optimization','Iteration','Best Score Found')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
charts['hyperparameter_tuning'] = fig_to_b64(fig)

# ── TIME SERIES ───────────────────────────────────────────────────────────
fig, axes = make_fig(2,3,16,9)
axes=axes.ravel()

# 1. Decomposition
np.random.seed(42)
t=np.arange(200)
trend=0.03*t
seasonal=2*np.sin(2*np.pi*t/25)
noise=np.random.normal(0,0.4,200)
ts=trend+seasonal+noise
ax=axes[0]
ax.plot(t,ts,color=C1,lw=1.5,alpha=0.9,label='Observed')
ax.plot(t,trend,color=C4,lw=2,ls='--',label='Trend')
style_ax(ax,'Time Series with Trend+Seasonality','Time','Value')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

ax=axes[1]
ax.plot(t,seasonal,color=C2,lw=1.5,label='Seasonal')
ax.plot(t,noise,color=MUTED,lw=0.8,alpha=0.6,label='Residual')
style_ax(ax,'Seasonal Component + Residual','Time','Value')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# 2. ACF / PACF (simulated)
ax=axes[2]
from statsmodels.tsa.stattools import acf, pacf
try:
    acf_vals=acf(ts,nlags=30)
    pacf_vals=pacf(ts,nlags=30)
    lags=np.arange(31)
    ax.bar(lags,acf_vals,color=C1,alpha=0.8,width=0.6,label='ACF')
    conf=1.96/np.sqrt(len(ts))
    ax.axhline(conf,color=C4,ls='--',lw=1); ax.axhline(-conf,color=C4,ls='--',lw=1)
except:
    lags=np.arange(31)
    acf_sim=0.85**lags+np.random.normal(0,0.05,31)
    ax.bar(lags,acf_sim,color=C1,alpha=0.8,width=0.6)
style_ax(ax,'Autocorrelation (ACF)','Lag','ACF')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# 3. Moving average smoothing
ax=axes[3]
for w,col in zip([5,15,30],[C2,C4,C6]):
    ma=np.convolve(ts,np.ones(w)/w,mode='valid')
    ax.plot(np.arange(w-1,len(ts)),ma,color=col,lw=2,label=f'MA({w})')
ax.plot(t,ts,color=MUTED,lw=0.8,alpha=0.4,label='Raw')
style_ax(ax,'Moving Average Smoothing','Time','Value')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# 4. Train/Test split for time series
ax=axes[4]
split=160
ax.fill_between(t[:split],ts[:split]-5,ts[:split]+5,alpha=0.1,color=C1)
ax.fill_between(t[split:],ts[split:]-5,ts[split:]+5,alpha=0.1,color=C2)
ax.plot(t[:split],ts[:split],color=C1,lw=1.5,label='Train')
ax.plot(t[split:],ts[split:],color=C2,lw=1.5,label='Test')
ax.axvline(split,color=C4,ls='--',lw=2,label='Train/Test split')
style_ax(ax,'Temporal Train/Test Split','Time','Value')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# 5. Forecast (naive vs exponential smoothing)
ax=axes[5]
test=ts[split:]
naive=np.ones_like(test)*ts[split-1]
alpha=0.3; exp_sm=[ts[split-1]]
for v in ts[split:-1]: exp_sm.append(alpha*v+(1-alpha)*exp_sm[-1])
exp_sm=np.array(exp_sm)
ax.plot(t[split:],test,color=C3,lw=2,label='Actual')
ax.plot(t[split:],naive,color=C2,lw=2,ls='--',label='Naive')
ax.plot(t[split:],exp_sm,color=C1,lw=2,label='Exp. Smoothing α=0.3')
style_ax(ax,'Forecasting Methods','Time','Value')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
plt.tight_layout(pad=1.5)
charts['time_series'] = fig_to_b64(fig)
plt.close('all')

# ── ATTENTION / TRANSFORMER (conceptual) ─────────────────────────────────
fig, axes = make_fig(1,3,16,5)

# Attention weight matrix
ax=axes[0]; ax.set_facecolor(SURFACE)
np.random.seed(42)
seq_len=8
tokens=['The','quick','brown','fox','jumps','over','the','dog']
attn=np.random.dirichlet(np.ones(seq_len)*0.5,size=seq_len)
attn=(attn+attn.T)/2; attn/=attn.sum(axis=1,keepdims=True)
im=ax.imshow(attn,cmap='YlOrRd',aspect='auto',vmin=0,vmax=attn.max())
ax.set_xticks(range(seq_len)); ax.set_yticks(range(seq_len))
ax.set_xticklabels(tokens,color=MUTED,fontsize=7,rotation=30)
ax.set_yticklabels(tokens,color=MUTED,fontsize=7)
plt.colorbar(im,ax=ax)
ax.set_title('Self-Attention Weight Matrix',color=TEXT,fontsize=11,fontweight='bold',pad=8)

# Multi-head attention diagram (conceptual bars)
ax=axes[1]
n_heads=8; head_dims=64
heads=['H1','H2','H3','H4','H5','H6','H7','H8']
head_specializations=['Syntax','Co-ref','Positional','Semantic','Long-range','Local','Negation','Entity']
colors_h=[C1,C2,C4,C5,C6,C7,C8,C3]
vals=np.random.dirichlet(np.ones(8)*2)*100
bars=ax.barh(range(8),vals,color=colors_h,alpha=0.85)
ax.set_yticks(range(8)); ax.set_yticklabels([f'{h}: {s}' for h,s in zip(heads,head_specializations)],color=MUTED,fontsize=8)
style_ax(ax,'Multi-Head Attention (8 heads)','Relative Attention Weight','Head')
ax.set_facecolor(SURFACE)

# Transformer architecture block diagram
ax=axes[2]; ax.set_facecolor(SURFACE); ax.axis('off')
ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.set_title('Transformer Block',color=TEXT,fontsize=11,fontweight='bold',pad=8)
blocks=[
    (1,1,8,1.5,'Input Embedding + Positional Encoding',C3),
    (1,3,8,1.5,'Multi-Head Self-Attention',C1),
    (1,4.8,8,0.6,'Add & Norm (Residual)',MUTED),
    (1,5.6,8,1.5,'Feed-Forward Network (2-layer MLP)',C4),
    (1,7.3,8,0.6,'Add & Norm (Residual)',MUTED),
    (1,8.1,8,1.2,'Output (Softmax / Linear)',C2),
]
for x,y,w,h,label,col in blocks:
    rect=mpatches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',
                                   facecolor=col,edgecolor=BORDER,alpha=0.8,lw=1)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=BG if col!=MUTED else TEXT,fontsize=8,fontweight='bold')
for y1,y2 in zip([2.5,4.5,5.3,7.1,8.0],[3.0,4.8,5.6,7.3,8.1]):
    ax.annotate('',xy=(5,y2),xytext=(5,y1),arrowprops=dict(arrowstyle='->',color=C4,lw=1.5))
charts['attention_transformer'] = fig_to_b64(fig)

# ── UMAP (simulated using t-SNE for digit) ───────────────────────────────
from sklearn.datasets import load_digits
try:
    from umap import UMAP as UMAP_
    has_umap=True
except:
    has_umap=False

dg=load_digits(); Xd=StandardScaler().fit_transform(dg.data[:600]); yd=dg.target[:600]
pca2=PCA(n_components=2).fit_transform(Xd)
tsne2=TSNE(n_components=2,perplexity=30,random_state=42,max_iter=500).fit_transform(Xd)

fig,axes=make_fig(1,3,16,5)
for ax,(emb,title) in zip(axes,[(pca2,'PCA 2D'),(tsne2,'t-SNE (perp=30)'),
                                  (tsne2+np.random.normal(0,0.3,tsne2.shape),'UMAP (approx)')]):
    for c in range(10):
        m=yd==c
        ax.scatter(emb[m,0],emb[m,1],s=14,alpha=0.8,label=str(c))
    style_ax(ax,f'{title} — Digits Dataset','Dim 1','Dim 2')
axes[0].legend(fontsize=6,ncol=2,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT,markerscale=1.5)
charts['dimensionality_methods'] = fig_to_b64(fig)

# ── AUTOENCODER (simulated) ───────────────────────────────────────────────
fig,axes=make_fig(1,3,16,5)

# Architecture diagram
ax=axes[0]; ax.set_facecolor(SURFACE); ax.axis('off')
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_title('Autoencoder Architecture',color=TEXT,fontsize=11,fontweight='bold',pad=8)
layer_sizes=[6,4,2,4,6]
layer_labels=['Input\n(dim=6)','Enc 1\n(dim=4)','Latent\n(dim=2)','Dec 1\n(dim=4)','Output\n(dim=6)']
layer_x=[0.1,0.25,0.5,0.75,0.9]
node_positions=[]
cols_ae=[C3,C1,C4,C2,C3]
for li,(sz,lx,col) in enumerate(zip(layer_sizes,layer_x,cols_ae)):
    ys=np.linspace(0.15,0.85,sz)
    node_positions.append(list(zip([lx]*sz,ys)))
    for y in ys:
        circ=plt.Circle((lx,y),0.032,color=col,zorder=4)
        ax.add_patch(circ)
    ax.text(lx,0.06,layer_labels[li],ha='center',va='center',color=MUTED,fontsize=7.5)
for li in range(len(layer_sizes)-1):
    for (x1,y1) in node_positions[li]:
        for (x2,y2) in node_positions[li+1]:
            ax.plot([x1,x2],[y1,y2],color=BORDER,lw=0.7,zorder=1,alpha=0.6)
ax.annotate('',xy=(0.5,0.94),xytext=(0.1,0.94),arrowprops=dict(arrowstyle='->',color=C1,lw=1.5))
ax.annotate('',xy=(0.9,0.94),xytext=(0.5,0.94),arrowprops=dict(arrowstyle='->',color=C2,lw=1.5))
ax.text(0.3,0.96,'Encoder',ha='center',color=C1,fontsize=8,fontweight='bold')
ax.text(0.7,0.96,'Decoder',ha='center',color=C2,fontsize=8,fontweight='bold')

# Latent space
ax=axes[1]
np.random.seed(42)
for c,col in enumerate([C1,C2,C4,C5]):
    pts=np.random.randn(30,2)*0.5+[np.cos(c*np.pi/2)*2,np.sin(c*np.pi/2)*2]
    ax.scatter(pts[:,0],pts[:,1],color=col,s=30,alpha=0.8,label=f'Class {c}')
style_ax(ax,'Latent Space Representation (2D)','z₁','z₂')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Reconstruction error
ax=axes[2]
np.random.seed(3)
epochs=np.arange(1,51)
train_loss=2.0*np.exp(-0.08*epochs)+0.05+np.random.normal(0,0.02,50)
val_loss=2.1*np.exp(-0.07*epochs)+0.1+np.random.normal(0,0.03,50)
ax.plot(epochs,train_loss,color=C1,lw=2,label='Train Recon Loss')
ax.plot(epochs,val_loss,color=C2,lw=2,ls='--',label='Val Recon Loss')
ax.fill_between(epochs,train_loss,val_loss,alpha=0.1,color=C4)
style_ax(ax,'Autoencoder Reconstruction Loss','Epoch','MSE Loss')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
charts['autoencoder'] = fig_to_b64(fig)

# ── ENSEMBLE COMPARISON (all methods) ────────────────────────────────────
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
X_ens,y_ens=datasets.make_classification(n_samples=600,n_features=10,random_state=42)
Xtr,Xte,ytr,yte=train_test_split(X_ens,y_ens,test_size=0.2,random_state=42)

from sklearn.model_selection import cross_val_score
models_ens={
    'Decision Tree':DecisionTreeClassifier(max_depth=5,random_state=42),
    'Random Forest':__import__('sklearn.ensemble',fromlist=['RandomForestClassifier']).RandomForestClassifier(n_estimators=50,random_state=42),
    'GBM':GradientBoostingClassifier(n_estimators=50,random_state=42),
    'Bagging':BaggingClassifier(n_estimators=50,random_state=42),
    'AdaBoost':__import__('sklearn.ensemble',fromlist=['AdaBoostClassifier']).AdaBoostClassifier(n_estimators=50,random_state=42),
    'Voting (hard)':VotingClassifier(estimators=[('dt',DecisionTreeClassifier(max_depth=5)),
                                                   ('rf',__import__('sklearn.ensemble',fromlist=['RandomForestClassifier']).RandomForestClassifier(n_estimators=30,random_state=42)),
                                                   ('gb',GradientBoostingClassifier(n_estimators=30,random_state=42))],voting='hard'),
}

fig,axes=make_fig(1,2,14,5)
names=list(models_ens.keys()); means=[]; stds=[]
for m in models_ens.values():
    sc=cross_val_score(m,X_ens,y_ens,cv=5)
    means.append(sc.mean()); stds.append(sc.std())

ax=axes[0]
bars=ax.bar(range(len(names)),means,yerr=stds,color=PALETTE[:len(names)],alpha=0.85,
            error_kw=dict(ecolor=TEXT,lw=1.5,capsize=4))
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,color=MUTED,fontsize=8,rotation=15,ha='right')
ax.set_ylim(0.7,1.0)
for bar,val in zip(bars,means): ax.text(bar.get_x()+bar.get_width()/2,val+0.005,f'{val:.3f}',ha='center',color=TEXT,fontsize=7.5)
style_ax(ax,'Ensemble Methods: CV Accuracy','Method','5-Fold CV Accuracy')

# Ensemble diagram
ax=axes[1]; ax.set_facecolor(SURFACE); ax.axis('off')
ax.set_xlim(0,10); ax.set_ylim(0,8); ax.set_title('Stacking Architecture',color=TEXT,fontsize=11,fontweight='bold',pad=8)
base_models=[('DT',C1,1),(('RF',C2,4)),('GBM',C4,7)]
for name,col,x in base_models:
    rect=mpatches.FancyBboxPatch((x,5),1.8,1.5,boxstyle='round,pad=0.1',facecolor=col,edgecolor=BORDER,alpha=0.85,lw=1)
    ax.add_patch(rect)
    ax.text(x+0.9,5.75,name,ha='center',va='center',color=BG,fontsize=10,fontweight='bold')
    ax.annotate('',xy=(5,3.6),xytext=(x+0.9,5),arrowprops=dict(arrowstyle='->',color=BORDER,lw=1.2))
meta=mpatches.FancyBboxPatch((3.5,2.0),3,1.5,boxstyle='round,pad=0.1',facecolor=C5,edgecolor=BORDER,alpha=0.85,lw=1)
ax.add_patch(meta); ax.text(5,2.75,'Meta-Learner\n(Logistic Reg)',ha='center',va='center',color=TEXT,fontsize=9,fontweight='bold')
ax.annotate('',xy=(5,0.8),xytext=(5,2.0),arrowprops=dict(arrowstyle='->',color=C4,lw=2))
out=mpatches.FancyBboxPatch((3.5,0),3,0.8,boxstyle='round,pad=0.1',facecolor=C6,edgecolor=BORDER,alpha=0.85,lw=1)
ax.add_patch(out); ax.text(5,0.4,'Final Prediction',ha='center',va='center',color=BG,fontsize=9,fontweight='bold')
ax.text(5,6.8,'Base Learners',ha='center',color=MUTED,fontsize=9)
charts['ensemble_comparison'] = fig_to_b64(fig)
plt.close('all')

# ── OPTIMIZATION ALGORITHMS ───────────────────────────────────────────────
fig,axes=make_fig(1,3,16,5)

# Loss surface contour
ax=axes[0]
x1=np.linspace(-3,3,200); x2=np.linspace(-3,3,200)
X1,X2=np.meshgrid(x1,x2)
Z=X1**2+3*X2**2+np.sin(3*X1)*np.cos(3*X2)
cs=ax.contourf(X1,X2,Z,levels=20,cmap='RdYlGn_r',alpha=0.7)
ax.set_facecolor(SURFACE)
# SGD path (noisy)
np.random.seed(1)
path_x=[2.5]; path_y=[2.5]
for _ in range(30):
    gx=2*path_x[-1]+3*np.cos(3*path_x[-1])*np.cos(3*path_y[-1])+np.random.randn()*0.3
    gy=6*path_y[-1]-3*np.sin(3*path_x[-1])*np.sin(3*path_y[-1])+np.random.randn()*0.3
    path_x.append(path_x[-1]-0.12*gx); path_y.append(path_y[-1]-0.12*gy)
ax.plot(path_x,path_y,color=C4,lw=2,marker='o',markersize=3,label='SGD')
ax.scatter([0],[0],color=C6,s=150,zorder=5,marker='*',label='Minimum')
style_ax(ax,'Loss Surface & SGD Path','θ₁','θ₂')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Optimizer convergence comparison (simulated)
ax=axes[1]
iters=np.arange(1,101)
np.random.seed(42)
def opt_curve(decay,noise): return 3*np.exp(-decay*iters)+0.05+noise*np.random.randn(100).cumsum()*0.01
sgd=opt_curve(0.02,2); adam=opt_curve(0.06,0.5); rmsprop=opt_curve(0.05,0.8); adagrad=opt_curve(0.04,1)
for curve,col,lbl in zip([sgd,adam,rmsprop,adagrad],[C2,C1,C4,C6],['SGD','Adam','RMSprop','AdaGrad']):
    ax.plot(iters,np.maximum(curve,0.05),color=col,lw=2,label=lbl)
style_ax(ax,'Optimizer Convergence','Iteration','Loss')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)

# Learning rate schedule
ax=axes[2]
epochs=np.arange(1,101)
constant=np.ones(100)*0.1
step_decay=0.1*(0.5**np.floor(epochs/20))
cosine=0.1*0.5*(1+np.cos(np.pi*epochs/100))
warmup_cosine=np.where(epochs<10,0.1*(epochs/10),0.1*0.5*(1+np.cos(np.pi*(epochs-10)/90)))
for sched,col,lbl in zip([constant,step_decay,cosine,warmup_cosine],[MUTED,C2,C1,C4],
                           ['Constant','Step Decay','Cosine','Warmup+Cosine']):
    ax.plot(epochs,sched,color=col,lw=2,label=lbl)
style_ax(ax,'Learning Rate Schedules','Epoch','Learning Rate')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
charts['optimization'] = fig_to_b64(fig)

# ── MODEL INTERPRETABILITY ────────────────────────────────────────────────
from sklearn.inspection import permutation_importance, partial_dependence
fig,axes=make_fig(1,3,16,5)
X_interp,y_interp=datasets.make_regression(n_samples=300,n_features=8,noise=10,random_state=42)
Xtr,Xte,ytr,yte=train_test_split(X_interp,y_interp,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf_interp=RandomForestRegressor(n_estimators=100,random_state=42).fit(Xtr,ytr)

# Permutation importance
ax=axes[0]
pi=permutation_importance(rf_interp,Xte,yte,n_repeats=10,random_state=42)
order=pi.importances_mean.argsort()[::-1]
ax.barh(range(8),[pi.importances_mean[i] for i in order],
        xerr=[pi.importances_std[i] for i in order],
        color=[C1 if pi.importances_mean[i]>0 else C2 for i in order],
        error_kw=dict(ecolor=TEXT,lw=1,capsize=3),alpha=0.85)
ax.set_yticks(range(8)); ax.set_yticklabels([f'F{i}' for i in order],color=MUTED,fontsize=9)
style_ax(ax,'Permutation Importance','Mean Decrease in Score','Feature')

# Partial dependence (top feature)
ax=axes[1]
top_feat=pi.importances_mean.argmax()
pd_result=partial_dependence(rf_interp,Xtr,[top_feat],kind='average')
ax.plot(pd_result['grid_values'][0],pd_result['average'][0],color=C1,lw=2.5)
ax.fill_between(pd_result['grid_values'][0],
                pd_result['average'][0]-10,pd_result['average'][0]+10,alpha=0.15,color=C1)
style_ax(ax,f'Partial Dependence: F{top_feat}',f'Feature F{top_feat}','Partial Dependence')

# SHAP-like waterfall (simulated)
ax=axes[2]; ax.set_facecolor(SURFACE)
feat_names=[f'F{i}' for i in range(8)]
shap_vals=np.array([1.8,-0.9,1.2,-0.4,0.7,0.3,-0.2,0.5])
order2=np.abs(shap_vals).argsort()[::-1]
colors_shap=[C1 if v>0 else C2 for v in shap_vals[order2]]
ax.barh(range(8),[shap_vals[i] for i in order2],color=colors_shap,alpha=0.85)
ax.axvline(0,color=BORDER,lw=1)
ax.set_yticks(range(8)); ax.set_yticklabels([feat_names[i] for i in order2],color=MUTED,fontsize=9)
style_ax(ax,'SHAP Values (single prediction)','SHAP Value','Feature')
charts['interpretability'] = fig_to_b64(fig)
plt.close('all')

# ── PIPELINE & MLOPS ─────────────────────────────────────────────────────
fig,axes=make_fig(1,2,14,5)
ax=axes[0]; ax.set_facecolor(SURFACE); ax.axis('off')
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_title('ML Pipeline Architecture',color=TEXT,fontsize=11,fontweight='bold',pad=8)
pipeline_steps=[
    ('Raw Data',C3,0.5), ('Data Cleaning',C1,2.0), ('Feature Eng.',C4,3.5),
    ('Train/Val/Test\nSplit',C5,5.0), ('Model Training',C2,6.5), ('Evaluation',C6,8.0), ('Deploy',C7,9.3)
]
for name,col,x in pipeline_steps:
    rect=mpatches.FancyBboxPatch((x-0.5,3.5),1.2,2.5,boxstyle='round,pad=0.12',facecolor=col,edgecolor=BORDER,alpha=0.85,lw=1)
    ax.add_patch(rect)
    ax.text(x+0.1,4.75,name,ha='center',va='center',color=BG,fontsize=7.5,fontweight='bold')
for i,((_,_,x1),(_,_,x2)) in enumerate(zip(pipeline_steps[:-1],pipeline_steps[1:])):
    ax.annotate('',xy=(x2-0.5,4.75),xytext=(x1+0.7,4.75),arrowprops=dict(arrowstyle='->',color=C4,lw=1.5))
ax.text(5,1.5,'Feedback Loop: Monitor → Retrain → Deploy',ha='center',color=MUTED,fontsize=9,style='italic')
ax.annotate('',xy=(9.3,3.5),xytext=(9.8,2.0),arrowprops=dict(arrowstyle='->',color=C2,lw=1.2,connectionstyle='arc3,rad=0.3'))
ax.annotate('',xy=(0.5,3.5),xytext=(0.0,2.0),arrowprops=dict(arrowstyle='<-',color=C2,lw=1.2,connectionstyle='arc3,rad=-0.3'))

# MLOps metrics dashboard
ax=axes[1]
np.random.seed(42)
weeks=np.arange(1,17)
train_acc=0.95+np.random.normal(0,0.01,16)
prod_acc=0.91+np.random.normal(0,0.02,16)-0.001*weeks  # slight drift
ax.plot(weeks,train_acc,color=C1,lw=2,marker='o',markersize=4,label='Train Acc')
ax.plot(weeks,prod_acc,color=C2,lw=2,marker='s',markersize=4,label='Prod Acc (drift)')
ax.fill_between(weeks,prod_acc,0.88,alpha=0.12,color=C2)
ax.axhline(0.90,color=C4,ls='--',lw=1.5,label='Retrain threshold')
ax.axvline(12,color=C5,ls=':',lw=2,label='Retrain triggered')
style_ax(ax,'MLOps: Model Performance Monitoring','Week','Accuracy')
ax.legend(fontsize=8,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
ax.set_ylim(0.85,1.0)
charts['pipeline_mlops'] = fig_to_b64(fig)
plt.close('all')

# ── GAUSSIAN PROCESSES ────────────────────────────────────────────────────
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
fig,axes=make_fig(1,3,16,5)
np.random.seed(42)
X_gp=np.array([-2,-1,0,1,2,2.5]).reshape(-1,1)
y_gp=np.sin(X_gp.ravel())+np.random.normal(0,0.1,6)
xs=np.linspace(-3.5,4,300).reshape(-1,1)
for ax,kernel,title in zip(axes,[RBF(length_scale=1.0),Matern(nu=1.5),RBF(length_scale=0.5)+WhiteKernel()],
                            ['GP – RBF Kernel','GP – Matérn Kernel','GP – RBF + Noise']):
    gpr=GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5,random_state=42).fit(X_gp,y_gp)
    mu,sigma=gpr.predict(xs,return_std=True)
    ax.plot(xs,mu,color=C1,lw=2.5,label='Mean')
    ax.fill_between(xs.ravel(),mu-2*sigma,mu+2*sigma,alpha=0.2,color=C1,label='95% CI')
    ax.fill_between(xs.ravel(),mu-sigma,mu+sigma,alpha=0.3,color=C1)
    ax.scatter(X_gp,y_gp,color=C4,s=60,zorder=5,label='Observations',edgecolors=BG,lw=0.8)
    style_ax(ax,title,'X','y')
    ax.legend(fontsize=7,facecolor=SURFACE,edgecolor=BORDER,labelcolor=TEXT)
charts['gaussian_process'] = fig_to_b64(fig)
plt.close('all')

# load existing charts
with open('/home/user/workspace/ml-visualizer/charts.json') as f:
    existing = json.load(f)

existing.update(charts)
with open('/home/user/workspace/ml-visualizer/charts.json','w') as f:
    json.dump(existing, f)
print("New charts:", list(charts.keys()))
print("Total charts:", len(existing))
