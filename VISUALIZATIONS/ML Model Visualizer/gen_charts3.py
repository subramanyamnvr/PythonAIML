"""
gen_charts3.py — Generate charts for 35 new ML/AI/Evaluation concepts
Appends to existing charts.json
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, FancyArrow
import matplotlib.patheffects as pe
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay)
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import io, base64, json, os

BG='#0F0E0C'; SURF='#1A1916'; SURF2='#222120'; BD='#2E2D2A'
TX='#CDCCCA'; TX2='#9A9896'; TX3='#616060'
TEAL='#20808D'; RUST='#A84B2F'; GOLD='#FFC553'; PURPLE='#A86FDF'; GREEN='#6DAA45'
BLUE='#4A90D9'; ORANGE='#E8884A'; PINK='#D9507A'
PALETTE=[TEAL,RUST,GOLD,PURPLE,GREEN,BLUE,ORANGE,PINK]

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':SURF,'axes.edgecolor':BD,
    'axes.labelcolor':TX,'xtick.color':TX2,'ytick.color':TX2,
    'text.color':TX,'grid.color':BD,'grid.linewidth':0.5,
    'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.titlecolor':TX,'axes.labelsize':8.5,
    'legend.facecolor':SURF2,'legend.edgecolor':BD,'legend.fontsize':8,
})

def fig_to_b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=110,bbox_inches='tight',
        facecolor=BG,edgecolor='none'); buf.seek(0)
    b64=base64.b64encode(buf.read()).decode(); plt.close(fig); return b64

charts={}

# ─────────────────────────────────────────────────────────────────────────────
# 1. EVALUATION METRICS (Classification)
# ─────────────────────────────────────────────────────────────────────────────
def chart_eval_classification():
    fig=plt.figure(figsize=(14,10)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Confusion Matrix
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    X,y=make_classification(n_samples=300,n_features=4,random_state=42)
    clf=LogisticRegression(random_state=42).fit(X,y); yp=clf.predict(X)
    cm=confusion_matrix(y,yp)
    im=ax0.imshow(cm,cmap='Blues',aspect='auto')
    ax0.set_xticks([0,1]); ax0.set_yticks([0,1])
    ax0.set_xticklabels(['Pred 0','Pred 1'],fontsize=8)
    ax0.set_yticklabels(['True 0','True 1'],fontsize=8)
    for i in range(2):
        for j in range(2):
            ax0.text(j,i,str(cm[i,j]),ha='center',va='center',
                     fontsize=16,fontweight='bold',color=TX)
    labels=[['TN','FP'],['FN','TP']]
    colors_lbl=[[GREEN,RUST],[RUST,GREEN]]
    for i in range(2):
        for j in range(2):
            ax0.text(j,i+0.35,labels[i][j],ha='center',va='center',
                     fontsize=9,color=colors_lbl[i][j],alpha=0.9)
    ax0.set_title('Confusion Matrix',fontweight='bold',color=TX)

    # ROC Curve
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    yp_prob=clf.predict_proba(X)[:,1]
    fpr,tpr,_=roc_curve(y,yp_prob); roc_auc=auc(fpr,tpr)
    ax1.plot(fpr,tpr,color=TEAL,lw=2,label=f'ROC (AUC={roc_auc:.3f})')
    ax1.plot([0,1],[0,1],'--',color=TX3,lw=1,label='Random')
    ax1.fill_between(fpr,tpr,alpha=0.12,color=TEAL)
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve',fontweight='bold',color=TX)
    ax1.legend(); ax1.grid(True,alpha=0.3)
    ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)

    # Precision-Recall Curve
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    prec,rec,_=precision_recall_curve(y,yp_prob); pr_auc=auc(rec,prec)
    ax2.plot(rec,prec,color=GOLD,lw=2,label=f'PR (AUC={pr_auc:.3f})')
    ax2.fill_between(rec,prec,alpha=0.12,color=GOLD)
    ax2.axhline(y=y.mean(),color=TX3,linestyle='--',lw=1,label=f'Baseline ({y.mean():.2f})')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve',fontweight='bold',color=TX)
    ax2.legend(); ax2.grid(True,alpha=0.3)

    # F1 vs Threshold
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    thresholds=np.linspace(0.01,0.99,100)
    f1_scores=[]; prec_s=[]; rec_s=[]
    for t in thresholds:
        yp_t=(yp_prob>=t).astype(int)
        tp=np.sum((yp_t==1)&(y==1)); fp=np.sum((yp_t==1)&(y==0))
        fn=np.sum((yp_t==0)&(y==1))
        p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9)
        f1_scores.append(2*p*r/(p+r+1e-9))
        prec_s.append(p); rec_s.append(r)
    best_t=thresholds[np.argmax(f1_scores)]
    ax3.plot(thresholds,f1_scores,color=TEAL,lw=2,label='F1')
    ax3.plot(thresholds,prec_s,color=GOLD,lw=1.5,linestyle='--',label='Precision')
    ax3.plot(thresholds,rec_s,color=RUST,lw=1.5,linestyle='--',label='Recall')
    ax3.axvline(x=best_t,color=GREEN,linestyle=':',lw=1.5,label=f'Best t={best_t:.2f}')
    ax3.set_xlabel('Decision Threshold'); ax3.set_ylabel('Score')
    ax3.set_title('Metrics vs Threshold',fontweight='bold',color=TX)
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.3)

    # Metric summary bar
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,balanced_accuracy_score,matthews_corrcoef
    metrics={'Accuracy':accuracy_score(y,yp),'Balanced\nAcc':balanced_accuracy_score(y,yp),
             'Precision':precision_score(y,yp),'Recall':recall_score(y,yp),
             'F1':f1_score(y,yp),'MCC':matthews_corrcoef(y,yp)}
    ax4.barh(list(metrics.keys()),list(metrics.values()),
             color=[TEAL,BLUE,GOLD,RUST,GREEN,PURPLE],alpha=0.85,height=0.6)
    ax4.set_xlim(0,1); ax4.axvline(0.5,color=TX3,linestyle='--',lw=1)
    for i,(k,v) in enumerate(metrics.items()):
        ax4.text(v+0.02,i,f'{v:.3f}',va='center',fontsize=8,color=TX)
    ax4.set_xlabel('Score'); ax4.set_title('Metric Summary',fontweight='bold',color=TX)
    ax4.grid(True,alpha=0.3,axis='x')

    # Calibration Curve
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    from sklearn.calibration import calibration_curve
    frac_pos,mean_pred=calibration_curve(y,yp_prob,n_bins=8)
    ax5.plot(mean_pred,frac_pos,color=TEAL,lw=2,marker='o',markersize=5,label='Calibration')
    ax5.plot([0,1],[0,1],'--',color=TX3,lw=1,label='Perfect calibration')
    ax5.fill_between(mean_pred,frac_pos,mean_pred,alpha=0.1,color=TEAL)
    ax5.set_xlabel('Mean Predicted Probability'); ax5.set_ylabel('Fraction Positives')
    ax5.set_title('Calibration Curve',fontweight='bold',color=TX)
    ax5.legend(); ax5.grid(True,alpha=0.3)
    fig.suptitle('Classification Evaluation Metrics',fontsize=13,fontweight='bold',
                 color=TX,y=1.01)
    return fig_to_b64(fig)

charts['eval_classification']=chart_eval_classification()
print("1. eval_classification done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. REGRESSION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def chart_eval_regression():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)
    np.random.seed(42)
    X,y=make_regression(n_samples=200,n_features=1,noise=20,random_state=42)
    model=LinearRegression().fit(X,y); yp=model.predict(X)
    res=y-yp

    # Actual vs Predicted
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    ax0.scatter(y,yp,alpha=0.6,color=TEAL,s=20,edgecolors='none')
    lim=[min(y.min(),yp.min()),max(y.max(),yp.max())]
    ax0.plot(lim,lim,'--',color=TX3,lw=1.5,label='Perfect')
    ax0.set_xlabel('Actual'); ax0.set_ylabel('Predicted')
    ax0.set_title('Actual vs Predicted',fontweight='bold',color=TX)
    ax0.legend(); ax0.grid(True,alpha=0.3)

    # Residuals vs Fitted
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    ax1.scatter(yp,res,alpha=0.6,color=GOLD,s=20)
    ax1.axhline(0,color=TX3,linestyle='--',lw=1.5)
    ax1.set_xlabel('Fitted Values'); ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted',fontweight='bold',color=TX)
    ax1.grid(True,alpha=0.3)

    # Residual Distribution
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    ax2.hist(res,bins=25,color=PURPLE,alpha=0.7,edgecolor=BD)
    from scipy import stats
    x_range=np.linspace(res.min(),res.max(),100)
    mu,sigma=res.mean(),res.std()
    ax2.plot(x_range,stats.norm.pdf(x_range,mu,sigma)*len(res)*(res.max()-res.min())/25,
             color=GOLD,lw=2,label='Normal fit')
    ax2.set_xlabel('Residual'); ax2.set_ylabel('Count')
    ax2.set_title('Residual Distribution',fontweight='bold',color=TX)
    ax2.legend(); ax2.grid(True,alpha=0.3)

    # QQ Plot
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    from scipy.stats import probplot
    (osm,osr),_=probplot(res,dist='norm')
    ax3.scatter(osm,osr,alpha=0.6,color=RUST,s=15)
    ax3.plot([-3,3],[-3*res.std(),3*res.std()],'--',color=TX3,lw=1.5,label='Normal line')
    ax3.set_xlabel('Theoretical Quantiles'); ax3.set_ylabel('Sample Quantiles')
    ax3.set_title('Q-Q Plot (Normality Check)',fontweight='bold',color=TX)
    ax3.legend(); ax3.grid(True,alpha=0.3)

    # Metric comparison
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
    mse=mean_squared_error(y,yp); rmse=np.sqrt(mse)
    mae=mean_absolute_error(y,yp); r2=r2_score(y,yp)
    mape=mean_absolute_percentage_error(y,yp)
    # Normalized display
    m_names=['RMSE','MAE','MSE\n(÷100)','MAPE\n(%)','R²\n(%)']
    m_vals=[rmse,mae,mse/100,mape*100,r2*100]
    bars=ax4.bar(m_names,m_vals,color=[TEAL,GOLD,RUST,PURPLE,GREEN],alpha=0.85)
    for bar,v in zip(bars,m_vals):
        ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                 f'{v:.1f}',ha='center',fontsize=8,color=TX)
    ax4.set_title('Regression Metrics',fontweight='bold',color=TX)
    ax4.grid(True,alpha=0.3,axis='y')

    # Scale of error: RMSE vs MAE comparison (noise sensitivity)
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    errors=np.linspace(-50,50,200)
    ax5.plot(errors,errors**2/1000,color=TEAL,lw=2,label='MSE (÷1000)')
    ax5.plot(errors,np.abs(errors),color=GOLD,lw=2,label='MAE')
    ax5.plot(errors,np.where(np.abs(errors)<=15,errors**2/(2*15),np.abs(errors)-15/2),
             color=RUST,lw=2,linestyle='--',label='Huber (δ=15)')
    ax5.set_xlabel('Error'); ax5.set_ylabel('Loss')
    ax5.set_title('Loss Function Comparison',fontweight='bold',color=TX)
    ax5.legend(); ax5.grid(True,alpha=0.3)
    ax5.set_xlim(-50,50); ax5.set_ylim(0,55)
    fig.suptitle('Regression Evaluation Metrics',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['eval_regression']=chart_eval_regression()
print("2. eval_regression done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. REINFORCEMENT LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def chart_rl():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Agent-Environment Loop diagram
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.set_xlim(0,10); ax0.set_ylim(0,6)
    ax0.axis('off')
    agent=FancyBboxPatch((0.5,2.5),3,2,boxstyle='round,pad=0.15',
        facecolor=SURF2,edgecolor=TEAL,linewidth=2)
    env=FancyBboxPatch((6.5,2.5),3,2,boxstyle='round,pad=0.15',
        facecolor=SURF2,edgecolor=GOLD,linewidth=2)
    ax0.add_patch(agent); ax0.add_patch(env)
    ax0.text(2,3.5,'Agent',ha='center',va='center',fontsize=13,fontweight='bold',color=TEAL)
    ax0.text(8,3.5,'Environment',ha='center',va='center',fontsize=12,fontweight='bold',color=GOLD)
    ax0.annotate('',xy=(6.5,4.1),xytext=(3.5,4.1),
        arrowprops=dict(arrowstyle='->',color=GREEN,lw=2))
    ax0.text(5,4.35,'Action aₜ',ha='center',fontsize=9,color=GREEN)
    ax0.annotate('',xy=(3.5,3.2),xytext=(6.5,3.2),
        arrowprops=dict(arrowstyle='->',color=RUST,lw=2))
    ax0.text(5,2.9,'State sₜ₊₁',ha='center',fontsize=9,color=RUST)
    ax0.annotate('',xy=(3.5,3.6),xytext=(6.5,3.6),
        arrowprops=dict(arrowstyle='->',color=GOLD,lw=2))
    ax0.text(5,3.75,'Reward rₜ',ha='center',fontsize=9,color=GOLD)
    ax0.set_title('RL Agent-Environment Loop',fontweight='bold',color=TX)

    # Q-learning convergence
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42)
    episodes=np.arange(1,501)
    # Simulated Q-learning reward curve
    rewards=-200+180*(1-np.exp(-episodes/150))+np.random.randn(500)*15
    rewards=np.convolve(rewards,np.ones(20)/20,mode='same')
    ax1.plot(episodes,rewards,color=TEAL,lw=1.5,label='Avg Reward')
    ax1.axhline(-20,color=GOLD,linestyle='--',lw=1,label='Optimal policy')
    ax1.fill_between(episodes,rewards,-200,alpha=0.12,color=TEAL)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward')
    ax1.set_title('Q-Learning Convergence',fontweight='bold',color=TX)
    ax1.legend(); ax1.grid(True,alpha=0.3)

    # Value function heatmap (gridworld)
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    np.random.seed(7)
    V=np.array([[0,  -1, -2, -3],
                [-1, -2, -1, -2],
                [-2, -1,  0, -1],
                [-3, -2, -1,  5]])
    im=ax2.imshow(V,cmap='RdYlGn',aspect='auto',vmin=-3,vmax=5)
    for i in range(4):
        for j in range(4):
            ax2.text(j,i,f'{V[i,j]}',ha='center',va='center',
                     fontsize=12,fontweight='bold',
                     color='black' if -1<V[i,j]<4 else 'white')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title('Value Function V(s) — Gridworld',fontweight='bold',color=TX)
    plt.colorbar(im,ax=ax2)
    ax2.text(3,0,'GOAL\n★',ha='center',va='center',fontsize=9,color='black',fontweight='bold')

    # Exploration vs Exploitation (ε-greedy)
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    eps=np.array([1.0*0.995**i for i in range(500)])
    explore=eps; exploit=1-eps
    ax3.stackplot(episodes,explore,exploit,colors=[RUST,TEAL],alpha=0.7,
                  labels=['Explore (random)','Exploit (greedy)'])
    ax3.set_xlabel('Episode'); ax3.set_ylabel('Proportion')
    ax3.set_title('ε-Greedy: Explore vs Exploit',fontweight='bold',color=TX)
    ax3.legend(loc='center right'); ax3.grid(True,alpha=0.3)

    # Policy comparison
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    algos=['Q-Learning','SARSA','DQN','A3C','PPO','SAC']
    scores=[72,68,88,85,92,94]
    colors=[TEAL,BLUE,GOLD,RUST,GREEN,PURPLE]
    bars=ax4.barh(algos,scores,color=colors,alpha=0.85,height=0.6)
    ax4.set_xlim(0,100)
    for bar,s in zip(bars,scores):
        ax4.text(s+1,bar.get_y()+bar.get_height()/2,
                 f'{s}',va='center',fontsize=9,color=TX)
    ax4.set_xlabel('Performance Score (relative)')
    ax4.set_title('RL Algorithm Comparison',fontweight='bold',color=TX)
    ax4.grid(True,alpha=0.3,axis='x')

    # Bellman equation visualization
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,8)
    ax5.text(5,7.2,'Bellman Optimality Equation',ha='center',fontsize=11,
             fontweight='bold',color=TX)
    formula_box=FancyBboxPatch((0.5,5.2),9,1.5,boxstyle='round,pad=0.2',
        facecolor=SURF2,edgecolor=TEAL,linewidth=1.5)
    ax5.add_patch(formula_box)
    ax5.text(5,5.95,'V*(s) = max_a [ R(s,a) + γ · Σ P(s\'|s,a) · V*(s\') ]',
             ha='center',va='center',fontsize=9.5,color=GOLD,fontfamily='monospace')
    concepts=[('R(s,a)','Immediate reward',RUST),
              ('γ (gamma)','Discount factor (0<γ<1)',GOLD),
              ('P(s\'|s,a)','Transition probability',TEAL),
              ('V*(s\')','Future state value',GREEN),
              ('max_a','Optimal action selection',PURPLE),
              ('TD Error','δ = r+γV(s\')-V(s)',BLUE)]
    for i,(sym,desc,col) in enumerate(concepts):
        row=i//2; col_idx=i%2
        x=1+col_idx*5; y=4.2-row*1.1
        ax5.text(x,y,f'{sym}:',fontsize=8.5,fontweight='bold',color=col)
        ax5.text(x+0.1,y-0.42,desc,fontsize=7.5,color=TX2)
    ax5.set_title('Bellman Equation Components',fontweight='bold',color=TX)
    fig.suptitle('Reinforcement Learning',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['reinforcement_learning']=chart_rl()
print("3. reinforcement_learning done")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CONVOLUTIONAL NEURAL NETWORKS
# ─────────────────────────────────────────────────────────────────────────────
def chart_cnn():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Conv operation visualization
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,6)
    # Input patch
    inp=np.array([[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1]])
    for i in range(5):
        for j in range(5):
            rect=plt.Rectangle((j*0.7,4.5-i*0.7),0.68,0.68,
                facecolor=TEAL if inp[i,j] else SURF2,edgecolor=BD,linewidth=0.5)
            ax0.add_patch(rect)
    # Kernel
    kern=np.array([[1,0,1],[0,1,0],[1,0,1]])
    for i in range(3):
        for j in range(3):
            rect=plt.Rectangle((4.5+j*0.7,3.2-i*0.7),0.68,0.68,
                facecolor=GOLD if kern[i,j] else SURF2,edgecolor=BD,linewidth=0.5,alpha=0.8)
            ax0.add_patch(rect)
    ax0.text(1.7,0.2,'Input (5×5)',ha='center',fontsize=8,color=TX2)
    ax0.text(5.5,0.2,'Kernel (3×3)',ha='center',fontsize=8,color=TX2)
    ax0.annotate('',xy=(9,2.5),xytext=(7.7,2.5),
        arrowprops=dict(arrowstyle='->',color=GREEN,lw=2))
    ax0.text(9.3,2.5,'Feature\nMap',fontsize=7.5,color=GREEN,va='center')
    ax0.set_title('Convolution Operation',fontweight='bold',color=TX)

    # CNN Architecture diagram
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF); ax1.axis('off')
    ax1.set_xlim(0,12); ax1.set_ylim(0,5)
    layers=[('Input\n224×224', 0.5,2.5,0.9,4,BLUE),
            ('Conv+ReLU\n64 filters',2.2,2.5,0.7,3.5,TEAL),
            ('MaxPool\n112×112',3.7,2.5,0.7,2.5,PURPLE),
            ('Conv+ReLU\n128 filters',5.1,2.5,0.6,2.5,TEAL),
            ('MaxPool\n56×56',6.4,2.5,0.5,1.8,PURPLE),
            ('Flatten',7.5,2.5,0.3,1.0,GOLD),
            ('Dense\n+ Softmax',8.5,2.5,0.5,0.6,RUST)]
    for lbl,x,y,w,h,col in layers:
        rect=FancyBboxPatch((x,y-h/2),w,h,boxstyle='round,pad=0.05',
            facecolor=col+'33',edgecolor=col,linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x+w/2,y,lbl,ha='center',va='center',fontsize=6.5,color=TX,fontweight='bold')
    # arrows
    xs=[0.5+0.9,2.2+0.7,3.7+0.7,5.1+0.6,6.4+0.5,7.5+0.3]
    for x in xs:
        ax1.annotate('',xy=(x+0.12,2.5),xytext=(x,2.5),
            arrowprops=dict(arrowstyle='->',color=TX3,lw=1))
    ax1.set_title('CNN Architecture',fontweight='bold',color=TX)

    # Feature map visualization
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    np.random.seed(42)
    # Simulated feature maps
    imgs=[np.random.rand(8,8)*np.outer(np.hanning(8),np.hanning(8)) for _ in range(6)]
    cmaps_=['Blues','Oranges','Greens','Purples','RdYlGn','YlOrRd']
    for i,img in enumerate(imgs):
        inset=ax2.inset_axes([0.33*(i%3)+0.02,0.52-0.5*(i//3)+0.02,0.3,0.44])
        inset.imshow(img,cmap=cmaps_[i],aspect='auto')
        inset.axis('off')
        inset.set_title(f'Filter {i+1}',fontsize=6,pad=1,color=TX)
    ax2.axis('off'); ax2.set_title('Feature Maps',fontweight='bold',color=TX)

    # Pooling operations
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF); ax3.axis('off')
    ax3.set_xlim(0,10); ax3.set_ylim(0,5)
    grid=np.array([[1,3,2,4],[5,6,1,2],[9,8,4,1],[3,7,5,2]])
    for i in range(4):
        for j in range(4):
            col=TEAL if (i<2 and j<2) else (GOLD if (i<2 and j>=2) else (RUST if (i>=2 and j<2) else PURPLE))
            rect=plt.Rectangle((j*0.85+0.2,3.8-i*0.85),0.82,0.82,
                facecolor=col+'44',edgecolor=col,linewidth=1)
            ax3.add_patch(rect)
            ax3.text(j*0.85+0.61,3.8-i*0.85+0.41,str(grid[i,j]),
                     ha='center',va='center',fontsize=10,color=TX)
    # max pool result
    maxp=np.array([[6,4],[9,5]])
    for i in range(2):
        for j in range(2):
            colors_mp=[TEAL,GOLD,RUST,PURPLE]
            rect=plt.Rectangle((6+j*1.2,3.2-i*1.2),1.15,1.15,
                facecolor=colors_mp[i*2+j]+'55',edgecolor=colors_mp[i*2+j],linewidth=1.5)
            ax3.add_patch(rect)
            ax3.text(6+j*1.2+0.575,3.2-i*1.2+0.575,str(maxp[i,j]),
                     ha='center',va='center',fontsize=13,fontweight='bold',color=TX)
    ax3.annotate('',xy=(5.8,2.5),xytext=(3.8,2.5),
        arrowprops=dict(arrowstyle='->',color=GREEN,lw=2))
    ax3.text(4.9,2.8,'MaxPool\n2×2, stride 2',ha='center',fontsize=7.5,color=GREEN)
    ax3.set_title('MaxPooling Operation',fontweight='bold',color=TX)

    # Common architectures timeline
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    archs=[('LeNet','1998'),('AlexNet','2012'),('VGG','2014'),('GoogLeNet','2014'),
           ('ResNet','2015'),('DenseNet','2016'),('EfficientNet','2019'),('ViT','2020')]
    years=[1998,2012,2014,2014,2015,2016,2019,2020]
    top1=[99.2,84.7,92.7,93.3,96.4,97.3,98.2,98.6]
    ax4.scatter(years,top1,color=TEAL,s=80,zorder=5)
    ax4.plot(years,top1,color=TEAL,lw=1,linestyle='--',alpha=0.5)
    for (n,_),x,y in zip(archs,years,top1):
        ax4.annotate(n,(x,y),textcoords='offset points',xytext=(0,8),
                     ha='center',fontsize=7,color=TX2)
    ax4.set_xlabel('Year'); ax4.set_ylabel('Top-5 Accuracy (%)')
    ax4.set_title('CNN Architecture Evolution',fontweight='bold',color=TX)
    ax4.grid(True,alpha=0.3)

    # Receptive field
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    layers_n=[1,2,3,4,5]; rf_sizes=[3,5,7,9,11]
    ax5.bar(layers_n,rf_sizes,color=PURPLE,alpha=0.8)
    ax5.plot(layers_n,rf_sizes,color=GOLD,lw=2,marker='o',markersize=6)
    for x,y in zip(layers_n,rf_sizes):
        ax5.text(x,y+0.1,f'{y}×{y}',ha='center',fontsize=8.5,color=TX)
    ax5.set_xlabel('Conv Layer'); ax5.set_ylabel('Receptive Field Size')
    ax5.set_title('Receptive Field Growth',fontweight='bold',color=TX)
    ax5.grid(True,alpha=0.3,axis='y')
    fig.suptitle('Convolutional Neural Networks (CNN)',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['cnn']=chart_cnn()
print("4. cnn done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. RECURRENT NEURAL NETWORKS / LSTM / GRU
# ─────────────────────────────────────────────────────────────────────────────
def chart_rnn():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Unrolled RNN
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,6)
    for i,t in enumerate(['t-2','t-1','t','t+1']):
        x=1+i*2.2
        box=FancyBboxPatch((x-0.5,2.3),1,1,boxstyle='round,pad=0.1',
            facecolor=TEAL+'44',edgecolor=TEAL,linewidth=1.5)
        ax0.add_patch(box)
        ax0.text(x,2.8,f'h_{t}',ha='center',va='center',fontsize=9,color=TEAL,fontweight='bold')
        ax0.text(x,1.7,f'x_{t}',ha='center',fontsize=8,color=TX2)
        ax0.annotate('',xy=(x,2.3),xytext=(x,1.9),arrowprops=dict(arrowstyle='->',color=TX2,lw=1))
        ax0.text(x,4.05,f'y_{t}',ha='center',fontsize=8,color=GOLD)
        ax0.annotate('',xy=(x,3.95),xytext=(x,3.3),arrowprops=dict(arrowstyle='->',color=GOLD,lw=1))
        if i<3:
            ax0.annotate('',xy=(x+1.7,2.8),xytext=(x+0.5,2.8),
                arrowprops=dict(arrowstyle='->',color=RUST,lw=2))
    ax0.set_title('Unrolled RNN',fontweight='bold',color=TX)

    # LSTM cell
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF); ax1.axis('off')
    ax1.set_xlim(0,10); ax1.set_ylim(0,8)
    gates=[('Forget\nGate σ',1.5,5.5,RUST),('Input\nGate σ',3.5,5.5,TEAL),
           ('Cell\nUpdate tanh',5.5,5.5,GOLD),('Output\nGate σ',7.5,5.5,GREEN)]
    for lbl,x,y,col in gates:
        circ=Circle((x,y),0.7,facecolor=col+'33',edgecolor=col,linewidth=1.5)
        ax1.add_patch(circ)
        ax1.text(x,y,lbl,ha='center',va='center',fontsize=6.5,color=col)
    ax1.annotate('',xy=(9.5,6.5),xytext=(0.3,6.5),
        arrowprops=dict(arrowstyle='->',color=PURPLE,lw=2.5),annotation_clip=False)
    ax1.text(4.8,7.1,'Cell State Cₜ (memory highway)',ha='center',fontsize=8.5,color=PURPLE)
    ax1.annotate('',xy=(9.5,4.5),xytext=(0.3,4.5),
        arrowprops=dict(arrowstyle='->',color=TX2,lw=1.5),annotation_clip=False)
    ax1.text(4.8,4.1,'Hidden State hₜ',ha='center',fontsize=8,color=TX2)
    ax1.set_title('LSTM Cell Gates',fontweight='bold',color=TX)

    # Vanishing gradient
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    steps=np.arange(1,51)
    vanilla_grad=0.9**steps
    lstm_grad=np.ones(50)*0.85+np.random.randn(50)*0.05
    lstm_grad=np.clip(lstm_grad,0.5,1.0)
    ax2.plot(steps,vanilla_grad,color=RUST,lw=2,label='Vanilla RNN (vanishing)')
    ax2.plot(steps,lstm_grad,color=TEAL,lw=2,label='LSTM (stable)')
    ax2.fill_between(steps,vanilla_grad,color=RUST,alpha=0.1)
    ax2.fill_between(steps,lstm_grad,0.5,color=TEAL,alpha=0.1)
    ax2.set_xlabel('Time Steps (backprop)'); ax2.set_ylabel('Gradient Magnitude')
    ax2.set_title('Vanishing Gradient Problem',fontweight='bold',color=TX)
    ax2.legend(); ax2.grid(True,alpha=0.3)

    # Sequence types
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF); ax3.axis('off')
    ax3.set_xlim(0,10); ax3.set_ylim(0,8)
    types=[('One-to-One',1,[4],[4]),
           ('One-to-Many',3,[4],[2,4,6]),
           ('Many-to-One',6,[2,4,6],[4]),
           ('Many-to-Many',8.5,[2,4,6],[2,4,6])]
    for lbl,bx,ins,outs in types:
        ax3.text(bx if lbl!='Many-to-Many' else 8.8,7.5,lbl,ha='center',fontsize=6.5,
                 color=TX2,fontweight='bold')
        for x in ins:
            circ=Circle((bx-1.5+(x-ins[0])*0.7,5.5),0.25,facecolor=RUST+'55',edgecolor=RUST)
            ax3.add_patch(circ)
        for x in outs:
            circ=Circle((bx-1.5+(x-outs[0])*0.7,2.5),0.25,facecolor=TEAL+'55',edgecolor=TEAL)
            ax3.add_patch(circ)
    ax3.set_title('Sequence Architectures',fontweight='bold',color=TX)

    # Loss over training
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    np.random.seed(42); epochs=np.arange(1,101)
    train_loss=2.5*np.exp(-epochs/30)+0.3+np.random.randn(100)*0.05
    val_loss=2.5*np.exp(-epochs/35)+0.5+np.random.randn(100)*0.08
    ax4.plot(epochs,train_loss,color=TEAL,lw=2,label='Train Loss')
    ax4.plot(epochs,val_loss,color=RUST,lw=2,label='Val Loss')
    ax4.fill_between(epochs,val_loss,train_loss,where=val_loss>train_loss,
                     alpha=0.15,color=RUST,label='Generalization Gap')
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('Loss')
    ax4.set_title('LSTM Training Loss',fontweight='bold',color=TX)
    ax4.legend(); ax4.grid(True,alpha=0.3)

    # GRU vs LSTM comparison
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    models=['Vanilla\nRNN','GRU','LSTM','Bi-LSTM','Stacked\nLSTM']
    perf=[68,84,89,92,94]; params=[5,25,35,70,120]
    ax5_twin=ax5.twinx()
    b=ax5.bar(models,perf,color=TEAL,alpha=0.7,label='Accuracy (%)',width=0.4,
              align='center')
    ax5_twin.plot(models,params,color=GOLD,lw=2,marker='D',markersize=6,label='Params (K)')
    ax5.set_ylabel('Accuracy (%)',color=TEAL)
    ax5_twin.set_ylabel('Parameters (K)',color=GOLD)
    ax5.set_title('RNN Variants Comparison',fontweight='bold',color=TX)
    ax5.set_ylim(0,110); ax5_twin.set_ylim(0,150)
    lines1,labels1=ax5.get_legend_handles_labels()
    lines2,labels2=ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1+lines2,labels1+labels2,fontsize=7.5)
    ax5.grid(True,alpha=0.3,axis='y')
    fig.suptitle('RNN / LSTM / GRU — Sequence Models',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['rnn_lstm']=chart_rnn()
print("5. rnn_lstm done")

# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERATIVE ADVERSARIAL NETWORKS (GAN)
# ─────────────────────────────────────────────────────────────────────────────
def chart_gan():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # GAN architecture
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,7)
    # Noise → Generator
    ax0.text(0.5,3.5,'z ~ N(0,I)\nLatent Noise',ha='center',fontsize=8,color=TX2)
    gen=FancyBboxPatch((1.8,2.8),2,1.4,boxstyle='round,pad=0.15',
        facecolor=TEAL+'33',edgecolor=TEAL,linewidth=2)
    ax0.add_patch(gen)
    ax0.text(2.8,3.5,'Generator\nG(z)',ha='center',fontsize=9,fontweight='bold',color=TEAL)
    ax0.annotate('',xy=(1.8,3.5),xytext=(1.2,3.5),arrowprops=dict(arrowstyle='->',color=TX2,lw=1.5))
    # Generator → fake
    ax0.annotate('',xy=(4.4,3.5),xytext=(3.8,3.5),arrowprops=dict(arrowstyle='->',color=TEAL,lw=1.5))
    ax0.text(5,3.5,'Fake\nImage',ha='center',fontsize=7.5,color=TEAL)
    # Real data
    ax0.text(5,5.5,'Real\nImage',ha='center',fontsize=7.5,color=GOLD)
    # → Discriminator
    disc=FancyBboxPatch((6.3,2.5),2.2,3,boxstyle='round,pad=0.15',
        facecolor=RUST+'33',edgecolor=RUST,linewidth=2)
    ax0.add_patch(disc)
    ax0.text(7.4,4,'Discriminator\nD(x)\nReal or Fake?',ha='center',fontsize=8.5,
             fontweight='bold',color=RUST)
    ax0.annotate('',xy=(6.3,3.5),xytext=(5.6,3.5),arrowprops=dict(arrowstyle='->',color=TX2,lw=1.5))
    ax0.annotate('',xy=(6.3,5),xytext=(5.6,5),arrowprops=dict(arrowstyle='->',color=TX2,lw=1.5))
    ax0.text(9.2,4,'Real/Fake',ha='center',fontsize=8,color=TX2)
    ax0.annotate('',xy=(8.5,4),xytext=(8.5,1.5),arrowprops=dict(arrowstyle='->',color=RUST,lw=1.5))
    ax0.text(4.8,1.5,'←Gradient (fool D)',ha='center',fontsize=8,color=RUST)
    ax0.set_title('GAN Architecture',fontweight='bold',color=TX)

    # GAN training loss
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42); steps=np.arange(0,1001)
    g_loss=1.5*np.exp(-steps/400)+0.7+0.3*np.sin(steps/50)+np.random.randn(1001)*0.08
    d_loss=1.5*np.exp(-steps/300)+0.5+0.2*np.sin(steps/50+1)+np.random.randn(1001)*0.06
    ax1.plot(steps,g_loss,color=TEAL,lw=1.5,label='Generator Loss')
    ax1.plot(steps,d_loss,color=RUST,lw=1.5,label='Discriminator Loss')
    ax1.axhline(0.693,color=GOLD,linestyle='--',lw=1,label='Nash Equilibrium (ln2)')
    ax1.set_xlabel('Training Step'); ax1.set_ylabel('Loss')
    ax1.set_title('GAN Training Dynamics',fontweight='bold',color=TX)
    ax1.legend(); ax1.grid(True,alpha=0.3)

    # Mode collapse illustration
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    np.random.seed(42)
    real_data=np.concatenate([np.random.randn(150,2)+[2,2],
                               np.random.randn(150,2)+[-2,-2],
                               np.random.randn(100,2)+[2,-2]])
    good_gen=np.concatenate([np.random.randn(100,2)*0.8+[2.1,2.1],
                              np.random.randn(100,2)*0.8+[-2.1,-1.9],
                              np.random.randn(100,2)*0.8+[2.0,-2.1]])
    collapsed_gen=np.random.randn(300,2)*0.3+[2.1,2.1]
    ax2.scatter(real_data[:,0],real_data[:,1],alpha=0.3,s=8,color=GOLD,label='Real',zorder=2)
    ax2.scatter(good_gen[:,0],good_gen[:,1],alpha=0.5,s=8,color=GREEN,marker='^',
                label='Good GAN',zorder=3)
    ax2.scatter(collapsed_gen[:,0],collapsed_gen[:,1],alpha=0.7,s=8,color=RUST,marker='x',
                label='Mode Collapse',zorder=4)
    ax2.legend(fontsize=7.5); ax2.grid(True,alpha=0.3)
    ax2.set_title('Mode Collapse vs. Good GAN',fontweight='bold',color=TX)

    # GAN variants comparison
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF); ax3.axis('off')
    ax3.set_xlim(0,10); ax3.set_ylim(0,8)
    ax3.text(5,7.5,'GAN Variants',ha='center',fontsize=11,fontweight='bold',color=TX)
    variants=[('DCGAN','Deep Conv layers → stable training, better quality',TEAL),
              ('WGAN','Wasserstein distance → solves mode collapse',BLUE),
              ('cGAN','Conditional on class label → controlled generation',GOLD),
              ('CycleGAN','Unpaired image translation (Horse↔Zebra)',GREEN),
              ('StyleGAN','Style-based generator → high-res face synthesis',PURPLE),
              ('Pix2Pix','Paired image-to-image translation with L1 loss',RUST)]
    for i,(name,desc,col) in enumerate(variants):
        y=6.3-i*1.05
        ax3.text(0.3,y,name,fontsize=9,fontweight='bold',color=col)
        ax3.text(0.3,y-0.45,desc,fontsize=7.5,color=TX2)
    ax3.set_title('GAN Variants',fontweight='bold',color=TX)

    # FID score concept
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    models=['DCGAN','WGAN-GP','BigGAN','StyleGAN2','DALL-E']
    fid=[35,28,18,3.8,np.nan]
    qual=[60,72,82,96,98]
    ax4_b=ax4.bar(models,qual,color=[TEAL,BLUE,GREEN,PURPLE,GOLD],alpha=0.8,width=0.6)
    for bar,f,q in zip(ax4_b,fid,qual):
        ax4.text(bar.get_x()+bar.get_width()/2,q+1,
                 f'FID:{f:.0f}' if not np.isnan(f) else 'FID:-',
                 ha='center',fontsize=7.5,color=TX)
    ax4.set_ylabel('Image Quality Score'); ax4.set_ylim(0,110)
    ax4.set_title('GAN Model Quality (FID ↓ better)',fontweight='bold',color=TX)
    ax4.grid(True,alpha=0.3,axis='y')
    plt.setp(ax4.get_xticklabels(),fontsize=8)

    # Latent space interpolation
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    np.random.seed(42)
    t=np.linspace(0,1,10)
    z_start=np.array([2.0,-1.0]); z_end=np.array([-1.5,2.0])
    z_interp=np.outer(1-t,z_start)+np.outer(t,z_end)
    # Random Gaussian images as placeholder
    circle=plt.Circle((0,0),2.5,color=SURF2,fill=True)
    ax5.add_patch(circle)
    sc=ax5.scatter(z_interp[:,0],z_interp[:,1],c=t,cmap='RdYlGn',
                   s=120,zorder=3,edgecolors=BD,linewidths=0.5)
    ax5.plot(z_interp[:,0],z_interp[:,1],color=TX3,lw=1,linestyle='--',zorder=2)
    plt.colorbar(sc,ax=ax5,label='Interpolation t')
    ax5.text(z_start[0],z_start[1]+0.25,'z_start',fontsize=8,color=RUST,ha='center')
    ax5.text(z_end[0],z_end[1]+0.25,'z_end',fontsize=8,color=GREEN,ha='center')
    ax5.set_xlabel('Latent dim 1'); ax5.set_ylabel('Latent dim 2')
    ax5.set_title('Latent Space Interpolation',fontweight='bold',color=TX)
    ax5.grid(True,alpha=0.3); ax5.set_xlim(-3,3); ax5.set_ylim(-3,3)
    fig.suptitle('Generative Adversarial Networks (GAN)',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['gan']=chart_gan()
print("6. gan done")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VARIATIONAL AUTOENCODERS (VAE)
# ─────────────────────────────────────────────────────────────────────────────
def chart_vae():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # VAE architecture
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,12); ax0.set_ylim(0,6)
    # Encoder
    enc=FancyBboxPatch((0.3,1.5),2.5,3,boxstyle='round,pad=0.1',
        facecolor=TEAL+'33',edgecolor=TEAL,linewidth=2)
    ax0.add_patch(enc)
    ax0.text(1.55,3,'Encoder\nq(z|x)',ha='center',fontsize=10,color=TEAL,fontweight='bold')
    # Latent
    ax0.text(4.5,4.2,'μ (mean)',ha='center',fontsize=8.5,color=GOLD)
    ax0.text(4.5,2.5,'σ (std)',ha='center',fontsize=8.5,color=PURPLE)
    ax0.annotate('',xy=(4,4.2),xytext=(2.8,3.5),arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.5))
    ax0.annotate('',xy=(4,2.5),xytext=(2.8,2.5),arrowprops=dict(arrowstyle='->',color=PURPLE,lw=1.5))
    # Reparameterization
    reparam=FancyBboxPatch((5.2,2.3),2,1.4,boxstyle='round,pad=0.1',
        facecolor=SURF2,edgecolor=RUST,linewidth=1.5)
    ax0.add_patch(reparam)
    ax0.text(6.2,3,'z = μ + σ·ε\nε~N(0,1)',ha='center',fontsize=8,color=RUST)
    ax0.text(6.2,1.7,'Reparameterization\nTrick',ha='center',fontsize=7.5,color=TX2)
    ax0.annotate('',xy=(5.2,3),xytext=(5,3.8),arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.2))
    ax0.annotate('',xy=(5.2,2.8),xytext=(5,2.5),arrowprops=dict(arrowstyle='->',color=PURPLE,lw=1.2))
    # Decoder
    dec=FancyBboxPatch((8.5,1.5),3,3,boxstyle='round,pad=0.1',
        facecolor=GREEN+'33',edgecolor=GREEN,linewidth=2)
    ax0.add_patch(dec)
    ax0.text(10,3,'Decoder\np(x|z)',ha='center',fontsize=10,color=GREEN,fontweight='bold')
    ax0.annotate('',xy=(8.5,3),xytext=(7.2,3),arrowprops=dict(arrowstyle='->',color=RUST,lw=1.5))
    ax0.text(1.55,5.2,'Input x',ha='center',fontsize=8,color=TX2)
    ax0.text(10,5.2,'Reconstruction x̂',ha='center',fontsize=8,color=TX2)
    ax0.set_title('VAE Architecture',fontweight='bold',color=TX)

    # ELBO loss decomposition
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42); epochs=np.arange(1,101)
    recon=80*np.exp(-epochs/25)+15+np.random.randn(100)*2
    kl=20*(1-np.exp(-epochs/40))+np.random.randn(100)*1.5
    elbo=recon+kl
    ax1.plot(epochs,elbo,color=TEAL,lw=2,label='ELBO (total)')
    ax1.plot(epochs,recon,color=RUST,lw=2,linestyle='--',label='Reconstruction loss')
    ax1.plot(epochs,kl,color=GOLD,lw=2,linestyle=':',label='KL Divergence')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('ELBO Loss Decomposition',fontweight='bold',color=TX)
    ax1.legend(); ax1.grid(True,alpha=0.3)

    # Latent space (2D)
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    np.random.seed(42)
    n_classes=5; colors_v=[TEAL,RUST,GOLD,PURPLE,GREEN]
    for i in range(n_classes):
        angle=i*2*np.pi/n_classes
        cx,cy=2*np.cos(angle),2*np.sin(angle)
        pts=np.random.randn(100,2)*0.6+[cx,cy]
        ax2.scatter(pts[:,0],pts[:,1],color=colors_v[i],alpha=0.5,s=15,label=f'Class {i}')
    ax2.set_xlabel('z₁'); ax2.set_ylabel('z₂')
    ax2.set_title('Organized Latent Space',fontweight='bold',color=TX)
    ax2.legend(fontsize=7); ax2.grid(True,alpha=0.3)

    # β-VAE vs standard VAE
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    betas=[0,0.1,0.5,1.0,2.0,5.0,10.0]
    disentangle=[20,35,52,68,82,91,94]
    recon_q=[95,92,87,81,72,58,42]
    ax3.plot(betas,disentangle,color=TEAL,lw=2,marker='o',markersize=6,label='Disentanglement')
    ax3.plot(betas,recon_q,color=RUST,lw=2,marker='s',markersize=6,label='Reconstruction Quality')
    ax3.set_xlabel('β (KL weight)')
    ax3.set_ylabel('Score (%)')
    ax3.set_title('β-VAE: Disentanglement Trade-off',fontweight='bold',color=TX)
    ax3.legend(); ax3.grid(True,alpha=0.3)

    # VAE vs AE vs GAN comparison
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    models=['AE','VAE','β-VAE','GAN','VAE-GAN']
    metrics={'Sample\nQuality':[55,68,62,92,88],
             'Latent\nOrganization':[30,82,95,45,75],
             'Training\nStability':[95,90,88,45,55]}
    x=np.arange(len(models)); w=0.25
    for i,(m,vals) in enumerate(metrics.items()):
        ax4.bar(x+i*w,vals,width=w,label=m,color=PALETTE[i],alpha=0.8)
    ax4.set_xticks(x+w); ax4.set_xticklabels(models,fontsize=8)
    ax4.set_ylabel('Score (0-100)')
    ax4.set_title('Generative Model Comparison',fontweight='bold',color=TX)
    ax4.legend(fontsize=7.5); ax4.grid(True,alpha=0.3,axis='y')

    # KL divergence visualization
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    z=np.linspace(-4,4,200)
    prior=1/np.sqrt(2*np.pi)*np.exp(-0.5*z**2)
    posterior1=1/np.sqrt(2*np.pi*0.5)*np.exp(-0.5*(z-1)**2/0.5)
    posterior2=1/np.sqrt(2*np.pi*0.1)*np.exp(-0.5*(z-2)**2/0.1)
    ax5.plot(z,prior,color=GOLD,lw=2.5,label='Prior N(0,1)')
    ax5.plot(z,posterior1,color=TEAL,lw=2,linestyle='--',label='q(z|x) — good')
    ax5.plot(z,posterior2,color=RUST,lw=2,linestyle=':',label='q(z|x) — high KL')
    ax5.fill_between(z,prior,posterior1,alpha=0.1,color=TEAL,label='KL(q‖p) small')
    ax5.fill_between(z,prior,posterior2,alpha=0.1,color=RUST)
    ax5.set_xlabel('z'); ax5.set_ylabel('Density')
    ax5.set_title('KL Divergence: Prior vs Posterior',fontweight='bold',color=TX)
    ax5.legend(fontsize=7.5); ax5.grid(True,alpha=0.3)
    fig.suptitle('Variational Autoencoder (VAE)',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['vae']=chart_vae()
print("7. vae done")

# ─────────────────────────────────────────────────────────────────────────────
# 8. NATURAL LANGUAGE PROCESSING (NLP)
# ─────────────────────────────────────────────────────────────────────────────
def chart_nlp():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # NLP Pipeline
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,8)
    steps=[('Raw Text','\"The cats are running\"',BLUE),
           ('Tokenize','[The, cats, are, running]',TEAL),
           ('Normalize','[the, cat, are, run]',GREEN),
           ('Embed','[0.2,−0.5,…] vectors',GOLD),
           ('Model','LSTM/Transformer/BERT',PURPLE),
           ('Output','Sentiment/NER/Class',RUST)]
    for i,(step,detail,col) in enumerate(steps):
        y=7-i*1.15
        box=FancyBboxPatch((0.3,y-0.3),9.3,0.6,boxstyle='round,pad=0.08',
            facecolor=col+'22',edgecolor=col,linewidth=1.2)
        ax0.add_patch(box)
        ax0.text(1.5,y,step,fontsize=9,fontweight='bold',color=col,va='center')
        ax0.text(5,y,detail,fontsize=8,color=TX2,va='center')
        if i<5:
            ax0.annotate('',xy=(4.8,y-0.32),xytext=(4.8,y-0.62),
                arrowprops=dict(arrowstyle='->',color=TX3,lw=1))
    ax0.set_title('NLP Pipeline',fontweight='bold',color=TX)

    # Word embeddings t-SNE
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42)
    clusters={'Animals':(['cat','dog','bird','fish','lion'],[-2,0],TEAL),
              'Royalty':(['king','queen','prince','earl','duke'],[2,2],GOLD),
              'Tech':(['AI','GPU','code','data','model'],[2,-2],PURPLE),
              'Food':(['eat','rice','cake','soup','meat'],[-2,-2],RUST)}
    for cat,(words,center,col) in clusters.items():
        pts=np.random.randn(len(words),2)*0.5+center
        ax1.scatter(pts[:,0],pts[:,1],color=col,s=60,alpha=0.8,label=cat)
        for j,w in enumerate(words):
            ax1.annotate(w,(pts[j,0],pts[j,1]),fontsize=7,color=TX2,
                         xytext=(3,3),textcoords='offset points')
    ax1.set_title('Word Embedding Space (t-SNE)',fontweight='bold',color=TX)
    ax1.legend(fontsize=7.5); ax1.grid(True,alpha=0.3)
    ax1.set_xlabel('Dim 1'); ax1.set_ylabel('Dim 2')

    # TF-IDF heatmap
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    np.random.seed(42)
    docs=['Doc 1','Doc 2','Doc 3','Doc 4','Doc 5']
    terms=['machine','learning','neural','data','model','deep','train']
    tfidf=np.random.rand(5,7)*np.random.rand(5,7)
    im=ax2.imshow(tfidf,cmap='YlOrRd',aspect='auto')
    ax2.set_xticks(range(7)); ax2.set_yticks(range(5))
    ax2.set_xticklabels(terms,fontsize=7,rotation=30)
    ax2.set_yticklabels(docs,fontsize=7.5)
    plt.colorbar(im,ax=ax2,shrink=0.8)
    ax2.set_title('TF-IDF Matrix',fontweight='bold',color=TX)

    # NLP tasks performance (GLUE scores)
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    tasks=['Sentiment\n(SST-2)','NER','QA\n(SQuAD)','NLI','Summ.\n(ROUGE)']
    bert=[93,88,88,84,42]; gpt2=[91,80,75,78,38]; roberta=[96,91,91,87,45]
    x=np.arange(len(tasks)); w=0.25
    ax3.bar(x-w,bert,w,label='BERT',color=TEAL,alpha=0.85)
    ax3.bar(x,gpt2,w,label='GPT-2',color=GOLD,alpha=0.85)
    ax3.bar(x+w,roberta,w,label='RoBERTa',color=PURPLE,alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(tasks,fontsize=8)
    ax3.set_ylabel('Score (%)'); ax3.legend(fontsize=8)
    ax3.set_title('NLP Model GLUE Benchmarks',fontweight='bold',color=TX)
    ax3.grid(True,alpha=0.3,axis='y'); ax3.set_ylim(0,105)

    # Attention heatmap
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    np.random.seed(5)
    words=['The','cat','sat','on','the','mat']
    attn=np.random.rand(6,6); attn/=attn.sum(axis=1,keepdims=True)
    im=ax4.imshow(attn,cmap='Blues',aspect='auto')
    ax4.set_xticks(range(6)); ax4.set_yticks(range(6))
    ax4.set_xticklabels(words,fontsize=8,rotation=30)
    ax4.set_yticklabels(words,fontsize=8)
    plt.colorbar(im,ax=ax4,shrink=0.8)
    ax4.set_title('Self-Attention Heatmap',fontweight='bold',color=TX)

    # Language model perplexity
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    models=['n-gram','LSTM','GPT-2\n(sm)','GPT-2\n(lg)','GPT-3','GPT-4']
    ppl=[150,75,35,22,10,5]
    ax5.bar(models,ppl,color=[BLUE,TEAL,GREEN,GOLD,PURPLE,RUST],alpha=0.85)
    for i,(m,p) in enumerate(zip(models,ppl)):
        ax5.text(i,p+1,str(p),ha='center',fontsize=9,color=TX,fontweight='bold')
    ax5.set_ylabel('Perplexity ↓ (lower=better)')
    ax5.set_title('Language Model Perplexity',fontweight='bold',color=TX)
    ax5.grid(True,alpha=0.3,axis='y')
    fig.suptitle('Natural Language Processing (NLP)',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['nlp']=chart_nlp()
print("8. nlp done")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TRANSFER LEARNING & FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────
def chart_transfer():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Transfer learning diagram
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,7)
    # Source domain
    src=FancyBboxPatch((0.3,4),4,2.5,boxstyle='round,pad=0.1',
        facecolor=TEAL+'22',edgecolor=TEAL,linewidth=1.5)
    ax0.add_patch(src)
    ax0.text(2.3,5.25,'Source Domain\nImageNet (1.2M images)',ha='center',fontsize=9,color=TEAL)
    ax0.text(2.3,4.4,'Large dataset → Pre-train model',ha='center',fontsize=7.5,color=TX2)
    # Arrow
    ax0.annotate('',xy=(5.5,5.25),xytext=(4.3,5.25),
        arrowprops=dict(arrowstyle='->,head_width=0.4',color=GOLD,lw=2.5))
    ax0.text(4.9,5.6,'Transfer\nWeights',ha='center',fontsize=7.5,color=GOLD)
    # Target domain
    tgt=FancyBboxPatch((5.5,4),4,2.5,boxstyle='round,pad=0.1',
        facecolor=RUST+'22',edgecolor=RUST,linewidth=1.5)
    ax0.add_patch(tgt)
    ax0.text(7.5,5.25,'Target Domain\nMedical X-rays (500)',ha='center',fontsize=9,color=RUST)
    ax0.text(7.5,4.4,'Small dataset → Fine-tune',ha='center',fontsize=7.5,color=TX2)
    # Layer diagram
    for i,x in enumerate([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]):
        frozen=i<5; col=TX2 if frozen else RUST
        rect=FancyBboxPatch((x-0.3,1.5),0.6,1.8,boxstyle='round,pad=0.05',
            facecolor=col+'33',edgecolor=col,linewidth=1)
        ax0.add_patch(rect)
        ax0.text(x,2.4,'❄' if frozen else '🔥',ha='center',va='center',fontsize=9)
        ax0.text(x,1.2,f'L{i+1}',ha='center',fontsize=7.5,color=col)
    ax0.text(3,0.7,'Frozen layers (feature extraction)',ha='center',fontsize=8,color=TX2)
    ax0.text(7.5,0.7,'Trainable layers',ha='center',fontsize=8,color=RUST)
    ax0.set_title('Transfer Learning Architecture',fontweight='bold',color=TX)

    # Learning curves: from scratch vs transfer
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42); epochs=np.arange(1,51)
    scratch=0.5+0.45*(1-np.exp(-epochs/15))+np.random.randn(50)*0.02
    transfer=0.5+0.45*(1-np.exp(-epochs/5))+np.random.randn(50)*0.02
    scratch=np.clip(scratch,0,0.97); transfer=np.clip(transfer,0,0.98)
    ax1.plot(epochs,scratch,color=RUST,lw=2,label='From scratch')
    ax1.plot(epochs,transfer,color=TEAL,lw=2,label='Transfer learning')
    ax1.axhline(0.95,color=TX3,linestyle='--',lw=1)
    ax1.fill_between(epochs,scratch,transfer,alpha=0.15,color=TEAL,label='Advantage')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Transfer vs From-Scratch Training',fontweight='bold',color=TX)
    ax1.legend(); ax1.grid(True,alpha=0.3)

    # Fine-tuning strategies
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF); ax2.axis('off')
    ax2.set_xlim(0,10); ax2.set_ylim(0,8)
    ax2.text(5,7.5,'Fine-Tuning Strategies',ha='center',fontsize=11,fontweight='bold',color=TX)
    strategies=[('Feature Extraction','Freeze all layers, train only classifier head\nFastest, works with tiny datasets (50-200 samples)',TEAL,7),
                ('Partial Fine-tune','Unfreeze top N layers, small LR for them\nBalance speed vs adaptation',GOLD,5.2),
                ('Full Fine-tune','Train all layers with small LR (1e-5 to 1e-4)\nBest performance, needs 1K+ samples',RUST,3.4),
                ('LoRA / PEFT','Low-rank adapter matrices, frozen backbone\nLLM fine-tuning with <1% of parameters',PURPLE,1.6)]
    for name,desc,col,y in strategies:
        ax2.text(0.3,y,name,fontsize=9.5,fontweight='bold',color=col)
        for j,line in enumerate(desc.split('\n')):
            ax2.text(0.3,y-0.55-j*0.42,line,fontsize=7.5,color=TX2)
    ax2.set_title('Fine-Tuning Strategies',fontweight='bold',color=TX)

    # Dataset size vs accuracy
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    data_sizes=[50,100,500,1000,5000,10000,50000]
    scratch_acc=[45,52,68,75,85,90,95]
    transfer_acc=[75,82,90,93,96,97,98]
    ax3.semilogx(data_sizes,scratch_acc,color=RUST,lw=2,marker='o',markersize=6,label='From scratch')
    ax3.semilogx(data_sizes,transfer_acc,color=TEAL,lw=2,marker='s',markersize=6,label='Transfer learning')
    ax3.fill_between(data_sizes,scratch_acc,transfer_acc,alpha=0.15,color=TEAL)
    ax3.set_xlabel('Training Data Size (log scale)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Data Efficiency of Transfer Learning',fontweight='bold',color=TX)
    ax3.legend(); ax3.grid(True,alpha=0.3)

    # Popular pretrained models
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    models=['ResNet50','VGG16','BERT-base','GPT-2','ViT-B','T5-base']
    params=[25,138,110,117,86,220]; top1=[76,71,None,None,81,None]
    colors_m=[TEAL,BLUE,GOLD,RUST,GREEN,PURPLE]
    bars=ax4.bar(models,params,color=colors_m,alpha=0.85)
    for bar,p in zip(bars,params):
        ax4.text(bar.get_x()+bar.get_width()/2,p+1,f'{p}M',
                 ha='center',fontsize=8,color=TX)
    ax4.set_ylabel('Parameters (Millions)')
    ax4.set_title('Pretrained Model Sizes',fontweight='bold',color=TX)
    ax4.grid(True,alpha=0.3,axis='y')
    plt.setp(ax4.get_xticklabels(),fontsize=8,rotation=30)

    # Domain adaptation
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    np.random.seed(42)
    src_data=np.random.randn(100,2)+[0,0]
    tgt_data=np.random.randn(100,2)+[2,1]
    adapted=np.random.randn(100,2)+[1,0.5]
    ax5.scatter(src_data[:,0],src_data[:,1],color=TEAL,alpha=0.4,s=15,label='Source domain')
    ax5.scatter(tgt_data[:,0],tgt_data[:,1],color=RUST,alpha=0.4,s=15,label='Target domain')
    ax5.scatter(adapted[:,0],adapted[:,1],color=GREEN,alpha=0.6,s=15,marker='^',label='After adaptation')
    ax5.set_xlabel('Feature 1'); ax5.set_ylabel('Feature 2')
    ax5.set_title('Domain Adaptation',fontweight='bold',color=TX)
    ax5.legend(fontsize=7.5); ax5.grid(True,alpha=0.3)
    fig.suptitle('Transfer Learning & Fine-Tuning',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['transfer_learning']=chart_transfer()
print("9. transfer_learning done")

# ─────────────────────────────────────────────────────────────────────────────
# 10. MULTI-ARMED BANDIT & ACTIVE LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def chart_bandit_active():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Bandit distributions
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    np.random.seed(42)
    arms={'Arm A':(3.0,1.0,TEAL),'Arm B':(2.0,1.5,GOLD),'Arm C':(4.5,0.8,RUST),'Arm D':(1.5,2.0,PURPLE)}
    x=np.linspace(-3,10,200)
    from scipy import stats
    for name,(mu,sigma,col) in arms.items():
        ax0.plot(x,stats.norm.pdf(x,mu,sigma),color=col,lw=2,label=f'{name} (μ={mu})')
        ax0.fill_between(x,stats.norm.pdf(x,mu,sigma),alpha=0.1,color=col)
    ax0.axvline(4.5,color=RUST,linestyle='--',lw=1,alpha=0.5)
    ax0.set_xlabel('Reward'); ax0.set_ylabel('Density')
    ax0.set_title('Multi-Armed Bandit: Reward Distributions',fontweight='bold',color=TX)
    ax0.legend(fontsize=7.5); ax0.grid(True,alpha=0.3)

    # UCB vs ε-greedy regret
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42); T=1000; t=np.arange(1,T+1)
    regret_eps=np.cumsum(np.random.exponential(0.5,T))*0.8
    regret_ucb=np.log(t)*3+np.random.randn(T).cumsum()*0.3
    regret_ts=np.log(t)*1.5+np.random.randn(T).cumsum()*0.2
    ax1.plot(t,regret_eps,color=RUST,lw=2,label='ε-greedy (ε=0.1)')
    ax1.plot(t,regret_ucb,color=TEAL,lw=2,label='UCB1')
    ax1.plot(t,regret_ts,color=GOLD,lw=2,label='Thompson Sampling')
    ax1.set_xlabel('Time Step'); ax1.set_ylabel('Cumulative Regret')
    ax1.set_title('Bandit Algorithm Regret',fontweight='bold',color=TX)
    ax1.legend(); ax1.grid(True,alpha=0.3)

    # Active learning pool
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    np.random.seed(42)
    X_pool=np.random.randn(200,2)*2
    labeled_idx=np.random.choice(200,20,replace=False)
    uncertain_idx=np.random.choice(list(set(range(200))-set(labeled_idx)),10,replace=False)
    unlabeled_idx=list(set(range(200))-set(labeled_idx)-set(uncertain_idx))
    ax2.scatter(X_pool[unlabeled_idx,0],X_pool[unlabeled_idx,1],
                color=TX3,alpha=0.4,s=20,label=f'Unlabeled ({len(unlabeled_idx)})')
    ax2.scatter(X_pool[labeled_idx,0],X_pool[labeled_idx,1],
                color=TEAL,s=50,zorder=3,label=f'Labeled ({len(labeled_idx)})')
    ax2.scatter(X_pool[uncertain_idx,0],X_pool[uncertain_idx,1],
                color=RUST,s=100,marker='*',zorder=4,label='Query next (uncertain)')
    ax2.set_xlabel('Feature 1'); ax2.set_ylabel('Feature 2')
    ax2.set_title('Active Learning: Query Strategy',fontweight='bold',color=TX)
    ax2.legend(fontsize=7.5); ax2.grid(True,alpha=0.3)

    # Active learning curves
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    labeled_sizes=np.array([10,20,40,80,160,320,640,1280])
    random_acc=[55,62,70,76,82,87,91,94]
    active_acc=[65,74,82,88,92,95,96,97]
    ax3.plot(labeled_sizes,random_acc,color=RUST,lw=2,marker='o',label='Random sampling')
    ax3.plot(labeled_sizes,active_acc,color=TEAL,lw=2,marker='s',label='Active learning')
    ax3.fill_between(labeled_sizes,random_acc,active_acc,alpha=0.15,color=TEAL)
    ax3.set_xscale('log'); ax3.set_xlabel('# Labeled Examples (log)')
    ax3.set_ylabel('Accuracy (%)'); ax3.legend()
    ax3.set_title('Active vs Random Labeling',fontweight='bold',color=TX)
    ax3.grid(True,alpha=0.3)

    # Query strategies
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF); ax4.axis('off')
    ax4.set_xlim(0,10); ax4.set_ylim(0,8)
    ax4.text(5,7.5,'Active Learning Query Strategies',ha='center',fontsize=10,
             fontweight='bold',color=TX)
    strategies=[('Uncertainty Sampling','Query most uncertain points\n(lowest max class prob)',RUST),
                ('Margin Sampling','Query smallest gap between\ntop-2 class probabilities',GOLD),
                ('Entropy Sampling','Query highest entropy:\nH = -Σ p·log(p)',TEAL),
                ('Core-Set','Greedy k-center: cover\nfeature space uniformly',GREEN),
                ('BALD','Bayesian Active Learning by\nDisagreement (ensemble)',PURPLE)]
    for i,(name,desc,col) in enumerate(strategies):
        y=6.3-i*1.2
        ax4.text(0.3,y,name+':',fontsize=9,fontweight='bold',color=col)
        for j,line in enumerate(desc.split('\n')):
            ax4.text(0.3,y-0.42-j*0.36,line,fontsize=7.5,color=TX2)
    ax4.set_title('Query Strategy Guide',fontweight='bold',color=TX)

    # Thompson sampling visualization
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    np.random.seed(42); rounds=50
    arm_pulls=[0,0,0,0]; arm_rewards=[0,0,0,0]
    arm_true_means=[0.3,0.5,0.7,0.4]
    ts_history=[]
    for _ in range(rounds):
        samples=[np.random.beta(r+1,p-r+1) if p>0 else np.random.beta(1,1)
                 for r,p in zip(arm_rewards,arm_pulls)]
        best=np.argmax(samples)
        reward=np.random.binomial(1,arm_true_means[best])
        arm_pulls[best]+=1; arm_rewards[best]+=reward
        ts_history.append(best)
    from collections import Counter
    pull_counts=Counter(ts_history)
    arms_labels=[f'Arm {i+1}\n(μ={m})' for i,m in enumerate(arm_true_means)]
    colors_ts=[TEAL,GOLD,RUST,PURPLE]
    ax5.bar(arms_labels,[pull_counts.get(i,0) for i in range(4)],
            color=colors_ts,alpha=0.85)
    ax5.set_ylabel('Times Pulled')
    ax5.set_title('Thompson Sampling Pull Distribution',fontweight='bold',color=TX)
    ax5.grid(True,alpha=0.3,axis='y')
    ax5.text(2,25,'Best arm\n(μ=0.7) ✓',ha='center',fontsize=8.5,color=RUST,fontweight='bold')
    fig.suptitle('Multi-Armed Bandit & Active Learning',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['bandit_active']=chart_bandit_active()
print("10. bandit_active done")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SEMI-SUPERVISED & SELF-SUPERVISED LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def chart_semi_self():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Semi-supervised overview
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,7)
    boxes=[('Supervised\nLearning','100% labeled\nHigh cost\nBest performance',TEAL,(0.3,4.5)),
           ('Semi-Supervised\nLearning','5% labeled + 95% unlabeled\nModerate cost\nNear-supervised perf.',GOLD,(3.5,4.5)),
           ('Self-Supervised\nLearning','0% human labels\nContrastive / pretext tasks\nFoundation models',RUST,(6.7,4.5))]
    for name,desc,col,(x,y) in boxes:
        box=FancyBboxPatch((x,y),2.8,2,boxstyle='round,pad=0.12',
            facecolor=col+'22',edgecolor=col,linewidth=1.5)
        ax0.add_patch(box)
        ax0.text(x+1.4,y+1.6,name,ha='center',fontsize=8.5,fontweight='bold',color=col)
        for i,line in enumerate(desc.split('\n')):
            ax0.text(x+1.4,y+1.1-i*0.4,line,ha='center',fontsize=7,color=TX2)
    # Continuum bar
    for i,frac in enumerate([1.0,0.6,0.2]):
        ax0.add_patch(FancyBboxPatch((1.5+i*3,2.7),1.5,0.5,boxstyle='round,pad=0.05',
            facecolor=[TEAL,GOLD,RUST][i]+'33',edgecolor=[TEAL,GOLD,RUST][i]))
        ax0.text(2.25+i*3,2.95,f'Labels: {int(frac*100)}%',ha='center',fontsize=7.5,
                 color=[TEAL,GOLD,RUST][i])
    ax0.text(5,2.3,'← Label Efficiency Continuum →',ha='center',fontsize=8,color=TX2)
    ax0.set_title('Learning Paradigm Continuum',fontweight='bold',color=TX)

    # Label propagation
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42)
    X_ss=np.random.randn(150,2)*1.5
    labels_ssl=np.array([-1]*150)
    # 10 labeled
    labeled_pts=np.array([[2,2],[-2,-2],[2,-2],[-2,2],[0,0],
                          [1.5,1.5],[-1.5,-1.5],[1,-1],[-1,1],[0.5,-0.5]])
    lbl_colors=[TEAL,TEAL,TEAL,TEAL,TEAL,RUST,RUST,RUST,RUST,RUST]
    ax1.scatter(X_ss[:,0],X_ss[:,1],color=TX3,alpha=0.3,s=20,label='Unlabeled')
    for pt,col in zip(labeled_pts,lbl_colors):
        ax1.scatter(*pt,color=col,s=120,zorder=4,edgecolors='white',linewidths=0.8)
    # Pseudo-labels
    pseudo=X_ss[np.random.choice(150,30,replace=False)]
    ax1.scatter(pseudo[:,0],pseudo[:,1],color=GOLD,s=40,marker='^',
                alpha=0.7,label='Pseudo-labeled',zorder=3)
    ax1.set_title('Label Propagation',fontweight='bold',color=TX)
    ax1.legend(fontsize=7.5); ax1.grid(True,alpha=0.3)

    # Contrastive learning (SimCLR)
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF); ax2.axis('off')
    ax2.set_xlim(0,10); ax2.set_ylim(0,7)
    ax2.text(5,6.7,'SimCLR Contrastive Learning',ha='center',fontsize=10,fontweight='bold',color=TX)
    # Original image
    orig=FancyBboxPatch((0.3,4.5),1.5,1.5,boxstyle='round,pad=0.1',
        facecolor=TEAL+'33',edgecolor=TEAL)
    ax2.add_patch(orig); ax2.text(1.05,5.25,'x',ha='center',fontsize=15,color=TEAL,fontweight='bold')
    # Augmentations
    aug1=FancyBboxPatch((0,2),1.5,1.5,boxstyle='round,pad=0.1',facecolor=GOLD+'33',edgecolor=GOLD)
    aug2=FancyBboxPatch((2,2),1.5,1.5,boxstyle='round,pad=0.1',facecolor=GOLD+'33',edgecolor=GOLD)
    ax2.add_patch(aug1); ax2.add_patch(aug2)
    ax2.text(0.75,2.75,'x̃₁\n(crop+flip)',ha='center',fontsize=7.5,color=GOLD)
    ax2.text(2.75,2.75,'x̃₂\n(color+blur)',ha='center',fontsize=7.5,color=GOLD)
    ax2.annotate('',xy=(0.75,2),xytext=(0.9,4.5),arrowprops=dict(arrowstyle='->',color=TX2,lw=1))
    ax2.annotate('',xy=(2.75,2),xytext=(1.2,4.5),arrowprops=dict(arrowstyle='->',color=TX2,lw=1))
    # Encoder
    enc1=FancyBboxPatch((4,3.5),1.5,1,boxstyle='round,pad=0.1',facecolor=PURPLE+'33',edgecolor=PURPLE)
    enc2=FancyBboxPatch((4,1.8),1.5,1,boxstyle='round,pad=0.1',facecolor=PURPLE+'33',edgecolor=PURPLE)
    ax2.add_patch(enc1); ax2.add_patch(enc2)
    ax2.text(4.75,4,'f(·)',ha='center',fontsize=10,color=PURPLE)
    ax2.text(4.75,2.3,'f(·)',ha='center',fontsize=10,color=PURPLE)
    ax2.annotate('',xy=(4,4),xytext=(1.5,2.9),arrowprops=dict(arrowstyle='->',color=TX2,lw=1))
    ax2.annotate('',xy=(4,2.3),xytext=(3.5,2.5),arrowprops=dict(arrowstyle='->',color=TX2,lw=1))
    # Loss
    ax2.text(7.5,3,'NT-Xent Loss:\npull z₁,z₂ together\npush other pairs apart',
             ha='center',fontsize=8,color=RUST)
    ax2.annotate('',xy=(7,4),xytext=(5.5,4),arrowprops=dict(arrowstyle='->',color=RUST,lw=1.5))
    ax2.annotate('',xy=(7,2.3),xytext=(5.5,2.3),arrowprops=dict(arrowstyle='->',color=RUST,lw=1.5))
    ax2.set_title('Contrastive Learning (SimCLR)',fontweight='bold',color=TX)

    # Semi-supervised accuracy vs labeled data
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    pct_labeled=[1,2,5,10,20,50,100]
    supervised=[32,40,55,64,73,85,92]
    semi_sup=[55,65,76,82,87,91,93]
    self_sup=[72,74,78,82,87,91,93]
    ax3.semilogx(pct_labeled,supervised,color=RUST,lw=2,marker='o',label='Supervised only')
    ax3.semilogx(pct_labeled,semi_sup,color=TEAL,lw=2,marker='s',label='Semi-supervised')
    ax3.semilogx(pct_labeled,self_sup,color=GOLD,lw=2,marker='^',label='Self-supervised')
    ax3.fill_between(pct_labeled,supervised,semi_sup,alpha=0.1,color=TEAL)
    ax3.set_xlabel('% Labeled Data'); ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Label Efficiency Comparison',fontweight='bold',color=TX)
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.3)

    # Self-supervised pretext tasks
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF); ax4.axis('off')
    ax4.set_xlim(0,10); ax4.set_ylim(0,8)
    ax4.text(5,7.5,'Self-Supervised Pretext Tasks',ha='center',fontsize=10,fontweight='bold',color=TX)
    pretext=[('Masked Autoencoding','Predict masked patches/tokens (MAE, BERT)',TEAL),
             ('Rotation Prediction','Predict image rotation angle (0°,90°,180°,270°)',GOLD),
             ('Jigsaw Puzzle','Reassemble shuffled image patches → spatial understanding',RUST),
             ('Contrastive (SimCLR)','Match augmented views of same image (NT-Xent loss)',PURPLE),
             ('BYOL','Bootstrap: online network ← target network, no negatives',GREEN),
             ('DINO','Self-distillation with no labels → emergent segmentation',BLUE)]
    for i,(name,desc,col) in enumerate(pretext):
        y=6.3-i*1.05
        ax4.text(0.3,y,name+':',fontsize=8.5,fontweight='bold',color=col)
        ax4.text(0.3,y-0.45,desc,fontsize=7.5,color=TX2)
    ax4.set_title('Pretext Task Guide',fontweight='bold',color=TX)

    # Foundation model scaling
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    models=['BERT\n(110M)','GPT-2\n(1.5B)','GPT-3\n(175B)','PaLM\n(540B)','LLaMA2\n(70B)']
    params_b=[0.11,1.5,175,540,70]; scores=[72,81,92,95,93]
    sc=ax5.scatter(params_b,scores,c=[TEAL,GOLD,RUST,PURPLE,GREEN],
                   s=200,zorder=5,edgecolors='white',linewidths=0.8)
    for m,x,y in zip(models,params_b,scores):
        ax5.annotate(m,(x,y),textcoords='offset points',xytext=(5,5),
                     fontsize=7.5,color=TX2)
    ax5.set_xscale('log'); ax5.set_xlabel('Parameters (Billions, log)')
    ax5.set_ylabel('Benchmark Score (%)')
    ax5.set_title('Scaling Laws: Params vs Performance',fontweight='bold',color=TX)
    ax5.grid(True,alpha=0.3)
    fig.suptitle('Semi-Supervised & Self-Supervised Learning',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['semi_self_supervised']=chart_semi_self()
print("11. semi_self_supervised done")

# ─────────────────────────────────────────────────────────────────────────────
# 12. CLUSTERING EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def chart_cluster_eval():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)
    from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
    from sklearn.cluster import KMeans

    # Elbow method
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    np.random.seed(42)
    X_c,_=make_blobs(n_samples=300,centers=4,cluster_std=0.8,random_state=42)
    ks=range(1,11)
    inertias=[KMeans(n_clusters=k,random_state=42,n_init=5).fit(X_c).inertia_ for k in ks]
    ax0.plot(list(ks),inertias,color=TEAL,lw=2,marker='o',markersize=7)
    ax0.axvline(4,color=RUST,linestyle='--',lw=1.5,label='Elbow (k=4)')
    ax0.fill_between(list(ks),inertias,inertias[-1],alpha=0.1,color=TEAL)
    ax0.set_xlabel('Number of Clusters (k)'); ax0.set_ylabel('Inertia (WCSS)')
    ax0.set_title('Elbow Method',fontweight='bold',color=TX)
    ax0.legend(); ax0.grid(True,alpha=0.3)

    # Silhouette scores
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    sil_scores=[silhouette_score(X_c,KMeans(n_clusters=k,random_state=42,n_init=5).fit_predict(X_c))
                for k in range(2,11)]
    ax1.plot(range(2,11),sil_scores,color=GOLD,lw=2,marker='s',markersize=6)
    ax1.axvline(4,color=RUST,linestyle='--',lw=1.5,label='Best (k=4)')
    ax1.fill_between(range(2,11),sil_scores,0,alpha=0.12,color=GOLD)
    ax1.set_xlabel('Number of Clusters'); ax1.set_ylabel('Silhouette Score (↑ better)')
    ax1.set_title('Silhouette Analysis',fontweight='bold',color=TX)
    ax1.legend(); ax1.grid(True,alpha=0.3)

    # Silhouette plot for k=4
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    from sklearn.metrics import silhouette_samples
    km4=KMeans(n_clusters=4,random_state=42,n_init=5).fit(X_c)
    labels4=km4.labels_; sil_vals=silhouette_samples(X_c,labels4)
    y_lower=0; colors_sil=[TEAL,GOLD,RUST,PURPLE]
    for i in range(4):
        ith=np.sort(sil_vals[labels4==i])
        size=ith.shape[0]
        ax2.fill_betweenx(np.arange(y_lower,y_lower+size),0,ith,
                          facecolor=colors_sil[i],alpha=0.7)
        ax2.text(-0.05,y_lower+size/2,f'{i}',fontsize=8,color=colors_sil[i])
        y_lower+=size+5
    avg=sil_vals.mean()
    ax2.axvline(avg,color=TX3,linestyle='--',lw=1.5,label=f'Avg: {avg:.3f}')
    ax2.set_xlabel('Silhouette Coefficient'); ax2.set_title('Silhouette Plot (k=4)',fontweight='bold',color=TX)
    ax2.legend(); ax2.grid(True,alpha=0.3)

    # DB Index and CH Index
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    ks2=range(2,11)
    db=[davies_bouldin_score(X_c,KMeans(n_clusters=k,random_state=42,n_init=5).fit_predict(X_c)) for k in ks2]
    ch=[calinski_harabasz_score(X_c,KMeans(n_clusters=k,random_state=42,n_init=5).fit_predict(X_c)) for k in ks2]
    ax3_twin=ax3.twinx()
    ax3.plot(list(ks2),db,color=RUST,lw=2,marker='o',markersize=5,label='DB Index (↓)')
    ax3_twin.plot(list(ks2),ch,color=TEAL,lw=2,marker='s',markersize=5,label='CH Index (↑)')
    ax3.axvline(4,color=GOLD,linestyle='--',lw=1)
    ax3.set_xlabel('k'); ax3.set_ylabel('Davies-Bouldin (↓)',color=RUST)
    ax3_twin.set_ylabel('Calinski-Harabász (↑)',color=TEAL)
    ax3.set_title('DB & CH Index',fontweight='bold',color=TX)
    lines1,labs1=ax3.get_legend_handles_labels()
    lines2,labs2=ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1+lines2,labs1+labs2,fontsize=7.5); ax3.grid(True,alpha=0.3)

    # Cluster separation visualized
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    km_final=KMeans(n_clusters=4,random_state=42,n_init=5).fit(X_c)
    scatter=ax4.scatter(X_c[:,0],X_c[:,1],c=km_final.labels_,
                        cmap=plt.cm.Set2,alpha=0.7,s=20,edgecolors='none')
    ax4.scatter(km_final.cluster_centers_[:,0],km_final.cluster_centers_[:,1],
                color='white',s=150,marker='X',zorder=5,label='Centroids')
    ax4.set_xlabel('Feature 1'); ax4.set_ylabel('Feature 2')
    ax4.set_title('Final Clustering (k=4)',fontweight='bold',color=TX)
    ax4.legend(); ax4.grid(True,alpha=0.3)

    # Metric summary table
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,8)
    ax5.text(5,7.5,'Clustering Metrics Reference',ha='center',fontsize=10,fontweight='bold',color=TX)
    metrics_ref=[('Inertia (WCSS)','Sum of squared distances to centroids','↓ better',TEAL),
                 ('Silhouette Score','Cohesion vs separation, range [-1,1]','↑ better',GOLD),
                 ('Davies-Bouldin','Avg ratio of within/between spread','↓ better',RUST),
                 ('Calinski-Harabász','Between-cluster/within-cluster variance ratio','↑ better',PURPLE),
                 ('ARI','Adjusted Rand Index (needs true labels)','↑ better',GREEN),
                 ('NMI','Normalized Mutual Info (needs true labels)','↑ better',BLUE)]
    for i,(name,desc,direction,col) in enumerate(metrics_ref):
        y=6.3-i*1.05
        ax5.text(0.2,y,name,fontsize=8.5,fontweight='bold',color=col)
        ax5.text(0.2,y-0.42,desc,fontsize=7.5,color=TX2)
        ax5.text(9.5,y,direction,fontsize=8,color=col,ha='right',fontweight='bold')
    ax5.set_title('Metric Quick Reference',fontweight='bold',color=TX)
    fig.suptitle('Clustering Evaluation Metrics',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['cluster_evaluation']=chart_cluster_eval()
print("12. cluster_evaluation done")

# ─────────────────────────────────────────────────────────────────────────────
# 13. PROBABILISTIC GRAPHICAL MODELS (Bayesian Networks, HMM)
# ─────────────────────────────────────────────────────────────────────────────
def chart_pgm():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Bayesian network graph
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,7)
    nodes=[('Rain',1.8,5.5,TEAL),('Sprinkler',5,5.5,GOLD),
           ('Wet Grass',3.5,3,PURPLE),('Slippery Road',1.2,3,RUST)]
    for name,x,y,col in nodes:
        circ=Circle((x,y),0.75,facecolor=col+'33',edgecolor=col,linewidth=2)
        ax0.add_patch(circ)
        ax0.text(x,y,name,ha='center',va='center',fontsize=7.5,color=col,fontweight='bold',
                 wrap=True)
    # Edges
    arrows=[((1.8,5.5),(3.5,3),TEAL),((5,5.5),(3.5,3),GOLD),
            ((1.8,5.5),(1.2,3),TEAL),((1.8,5.5),(5,5.5),TX3)]
    for (x1,y1),(x2,y2),col in arrows:
        ax0.annotate('',xy=(x2,y2+0.75),xytext=(x1,y1-0.75),
            arrowprops=dict(arrowstyle='->',color=col,lw=1.5))
    # CPT
    cpt=FancyBboxPatch((5.5,1),4,3.5,boxstyle='round,pad=0.1',
        facecolor=SURF2,edgecolor=BD,linewidth=1)
    ax0.add_patch(cpt)
    ax0.text(7.5,4.3,'P(WG|R,S)',ha='center',fontsize=9,fontweight='bold',color=PURPLE)
    rows=[('R=T,S=T','0.99'),('R=T,S=F','0.90'),('R=F,S=T','0.80'),('R=F,S=F','0.01')]
    for i,(cond,prob) in enumerate(rows):
        ax0.text(6,3.7-i*0.65,cond,fontsize=8,color=TX2)
        ax0.text(9,3.7-i*0.65,prob,fontsize=8,color=GOLD,ha='right')
    ax0.set_title('Bayesian Network: Wet Grass',fontweight='bold',color=TX)

    # HMM diagram
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF); ax1.axis('off')
    ax1.set_xlim(0,10); ax1.set_ylim(0,7)
    states=['S₁\n(Sunny)','S₂\n(Rainy)','S₃\n(Cloudy)']
    obs=['O₁\n(Walk)','O₂\n(Shop)','O₃\n(Clean)']
    state_pos=[(2,5),(5,5),(8,5)]
    obs_pos=[(2,2.5),(5,2.5),(8,2.5)]
    colors_hmm=[TEAL,GOLD,PURPLE]
    for (name,(x,y),col) in zip(states,state_pos,colors_hmm):
        circ=Circle((x,y),0.8,facecolor=col+'33',edgecolor=col,linewidth=2)
        ax1.add_patch(circ)
        ax1.text(x,y,name,ha='center',va='center',fontsize=8,color=col,fontweight='bold')
    for (name,(x,y),col) in zip(obs,obs_pos,colors_hmm):
        rect=FancyBboxPatch((x-0.7,y-0.45),1.4,0.9,boxstyle='round,pad=0.1',
            facecolor=SURF2,edgecolor=TX3,linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x,y,name,ha='center',va='center',fontsize=8,color=TX2)
    # Transition arrows
    for i in range(2):
        ax1.annotate('',xy=(state_pos[i+1][0]-0.8,5),xytext=(state_pos[i][0]+0.8,5),
            arrowprops=dict(arrowstyle='->',color=TX3,lw=1.5))
    # Emission arrows
    for (x,_),col in zip(state_pos,colors_hmm):
        ax1.annotate('',xy=(x,2.95),xytext=(x,4.2),
            arrowprops=dict(arrowstyle='->',color=col,lw=1.2))
    ax1.text(3.4,5.4,'0.7',fontsize=8,color=TX2); ax1.text(6.4,5.4,'0.5',fontsize=8,color=TX2)
    ax1.text(5,6.3,'Transition Probabilities A',ha='center',fontsize=8.5,color=TX,fontweight='bold')
    ax1.text(5,1.5,'Emission Probabilities B',ha='center',fontsize=8.5,color=TX,fontweight='bold')
    ax1.set_title('Hidden Markov Model (HMM)',fontweight='bold',color=TX)

    # Viterbi algorithm illustration
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF); ax2.axis('off')
    ax2.set_xlim(0,10); ax2.set_ylim(0,7)
    ax2.text(5,6.7,'Viterbi Algorithm (Decoding)',ha='center',fontsize=10,fontweight='bold',color=TX)
    time_steps=['t=1','t=2','t=3','t=4']
    states_v=['Sunny','Rainy','Cloudy']
    t_positions=[1.5,4,6.5,9]
    s_positions=[5.2,3.5,1.8]
    for t,tx in zip(time_steps,t_positions):
        ax2.text(tx,0.4,t,ha='center',fontsize=8.5,color=TX2,fontweight='bold')
    np.random.seed(42)
    probs=np.random.rand(4,3); probs/=probs.sum(axis=1,keepdims=True)
    for ti,tx in enumerate(t_positions):
        for si,(sy,col) in enumerate(zip(s_positions,colors_hmm)):
            p=probs[ti,si]
            circ=Circle((tx,sy),0.5*p+0.25,facecolor=col+'44',edgecolor=col,lw=1.5,alpha=0.8)
            ax2.add_patch(circ)
            ax2.text(tx,sy,f'{p:.2f}',ha='center',va='center',fontsize=7.5,color=TX)
    # Best path
    best_path_idx=probs.argmax(axis=1)
    for i in range(3):
        x1,y1=t_positions[i],s_positions[best_path_idx[i]]
        x2,y2=t_positions[i+1],s_positions[best_path_idx[i+1]]
        ax2.plot([x1,x2],[y1,y2],color=GOLD,lw=2.5,linestyle='--',zorder=5)
    ax2.text(5,6.1,'— Best Path (Viterbi)',ha='center',fontsize=8,color=GOLD)
    for s,sy,col in zip(states_v,s_positions,colors_hmm):
        ax2.text(0.3,sy,s,fontsize=8,color=col,va='center',fontweight='bold')
    ax2.set_title('Viterbi Decoding',fontweight='bold',color=TX)

    # Forward-Backward algorithm
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    np.random.seed(42); T=20
    t_arr=np.arange(T)
    forward=np.random.rand(T,2); forward/=forward.sum(axis=1,keepdims=True)
    backward=np.random.rand(T,2); backward/=backward.sum(axis=1,keepdims=True)
    posterior=forward*backward; posterior/=posterior.sum(axis=1,keepdims=True)
    ax3.plot(t_arr,forward[:,0],color=TEAL,lw=1.5,linestyle='--',label='Forward α')
    ax3.plot(t_arr,backward[:,0],color=GOLD,lw=1.5,linestyle=':',label='Backward β')
    ax3.plot(t_arr,posterior[:,0],color=RUST,lw=2.5,label='Posterior γ=αβ')
    ax3.set_xlabel('Time'); ax3.set_ylabel('P(S=Sunny|observations)')
    ax3.set_title('Forward-Backward Algorithm',fontweight='bold',color=TX)
    ax3.legend(); ax3.grid(True,alpha=0.3)

    # Markov chain steady state
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    np.random.seed(42); steps=100
    state=0; history=[0,1,0]  # Sun, Rain, Cloud
    transition=np.array([[0.7,0.2,0.1],[0.3,0.5,0.2],[0.4,0.3,0.3]])
    state_hist=[state]
    for _ in range(steps):
        state=np.random.choice(3,p=transition[state])
        state_hist.append(state)
    from collections import Counter
    counts=Counter(state_hist); total=len(state_hist)
    ax4.bar(['Sunny','Rainy','Cloudy'],
            [counts[0]/total,counts[1]/total,counts[2]/total],
            color=[TEAL,GOLD,PURPLE],alpha=0.85)
    ax4.axhline(0.6,color=TEAL,linestyle='--',lw=1,alpha=0.5)
    ax4.set_ylabel('Steady-State Probability')
    ax4.set_title('Markov Chain Steady State',fontweight='bold',color=TX)
    ax4.grid(True,alpha=0.3,axis='y')

    # PGM types comparison
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,8)
    ax5.text(5,7.5,'PGM Types Comparison',ha='center',fontsize=10,fontweight='bold',color=TX)
    types=[('Bayesian Network','Directed DAG, causal structure\nP(X)=Π P(Xᵢ|parents(Xᵢ))',TEAL),
           ('Markov Random Field','Undirected graph, potential functions\nGood for image segmentation',GOLD),
           ('HMM','Sequence model with latent states\nSpeech, genomics, POS tagging',RUST),
           ('CRF','Discriminative sequence labeling\nNamed entity recognition, OCR',PURPLE),
           ('Factor Graph','Bipartite: variables ↔ factors\nBelief propagation inference',GREEN)]
    for i,(name,desc,col) in enumerate(types):
        y=6.3-i*1.2
        ax5.text(0.3,y,name+':',fontsize=9,fontweight='bold',color=col)
        for j,line in enumerate(desc.split('\n')):
            ax5.text(0.3,y-0.42-j*0.38,line,fontsize=7.5,color=TX2)
    ax5.set_title('PGM Types Guide',fontweight='bold',color=TX)
    fig.suptitle('Probabilistic Graphical Models',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['pgm']=chart_pgm()
print("13. pgm done")

# ─────────────────────────────────────────────────────────────────────────────
# 14. OBJECT DETECTION (Computer Vision)
# ─────────────────────────────────────────────────────────────────────────────
def chart_object_detection():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Detection illustration
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    ax0.set_xlim(0,10); ax0.set_ylim(0,8)
    ax0.set_facecolor('#1a2830')
    # Simulated image background
    ax0.imshow(np.random.randint(30,60,(8,10,3),dtype=np.uint8),
               extent=[0,10,0,8],aspect='auto',alpha=0.4)
    bboxes=[((1,3),(4,7),'Person',0.92,TEAL),
            ((5,4),(8,7),'Car',0.87,GOLD),
            ((2,1),(4,3),'Dog',0.78,RUST)]
    for (x1,y1),(x2,y2),label,conf,col in bboxes:
        rect=plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=col,linewidth=2.5)
        ax0.add_patch(rect)
        ax0.text(x1,y2+0.1,f'{label} {conf:.0%}',fontsize=9,color=col,fontweight='bold',
                 backgroundcolor=SURF+'cc')
    ax0.set_xlim(0,10); ax0.set_ylim(0,8.5)
    ax0.set_title('Object Detection: Bounding Boxes',fontweight='bold',color=TX)
    ax0.axis('off')

    # IoU visualization
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    ax1.set_xlim(0,10); ax1.set_ylim(0,8)
    gt_box=plt.Rectangle((1,2),5,4,fill=False,edgecolor=TEAL,linewidth=2.5,label='Ground Truth')
    pred_box=plt.Rectangle((3,1),5,4.5,fill=False,edgecolor=RUST,linewidth=2.5,label='Prediction')
    # Intersection
    inter=plt.Rectangle((3,2),3,4,color=GOLD,alpha=0.35,label='Intersection')
    ax1.add_patch(gt_box); ax1.add_patch(pred_box); ax1.add_patch(inter)
    ax1.text(4.5,4,'Intersection',ha='center',fontsize=8.5,color=GOLD,fontweight='bold')
    iou=(3*4)/(5*4+5*4.5-3*4)
    ax1.text(5,7.2,f'IoU = |A∩B| / |A∪B| = {iou:.3f}',ha='center',fontsize=9.5,color=TX,fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(True,alpha=0.2)
    ax1.set_title('Intersection over Union (IoU)',fontweight='bold',color=TX)

    # Precision-Recall for detection
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    conf_scores=np.sort(np.random.rand(100))[::-1]
    tp_mask=(conf_scores>0.4).astype(int); tp_mask[:60]=1; tp_mask[60:]=0
    tp_cumsum=np.cumsum(tp_mask)
    fp_cumsum=np.cumsum(1-tp_mask)
    n_pos=60
    prec=tp_cumsum/(tp_cumsum+fp_cumsum+1e-9)
    rec=tp_cumsum/n_pos
    ax2.plot(rec,prec,color=TEAL,lw=2,label='PR Curve')
    ax2.fill_between(rec,prec,alpha=0.15,color=TEAL)
    mAP=np.trapezoid(prec,rec)
    ax2.text(0.5,0.2,f'mAP = {mAP:.3f}',ha='center',fontsize=10,color=GOLD,fontweight='bold')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('mAP (mean Average Precision)',fontweight='bold',color=TX)
    ax2.legend(); ax2.grid(True,alpha=0.3)

    # Algorithm comparison
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    algos=['RCNN','Fast\nRCNN','Faster\nRCNN','SSD','YOLO v3','YOLO v8','DETR']
    fps=[0.05,1.5,17,46,45,80,28]; map50=[66,70,74,74,55,71,63]
    ax3_twin=ax3.twinx()
    ax3.bar(algos,fps,color=TEAL,alpha=0.7,label='FPS ↑',width=0.4,align='edge',bottom=0)
    ax3_twin.plot(algos,map50,color=GOLD,lw=2,marker='D',markersize=7,label='mAP@50 ↑')
    ax3.set_ylabel('FPS (Frames/sec)',color=TEAL)
    ax3_twin.set_ylabel('mAP@50 (%)',color=GOLD)
    ax3.set_title('Detection Models: Speed vs Accuracy',fontweight='bold',color=TX)
    plt.setp(ax3.get_xticklabels(),fontsize=8)
    ax3.set_ylim(0,100); ax3_twin.set_ylim(0,100)
    lines1,labs1=ax3.get_legend_handles_labels()
    lines2,labs2=ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1+lines2,labs1+labs2,fontsize=7.5)

    # Anchor boxes
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    ax4.set_xlim(0,10); ax4.set_ylim(0,10); ax4.axis('off')
    center=(5,5)
    anchors=[(2,2),(4,2),(2,4),(3,4),(4,4),(6,2),(2,6),(5,3)]
    for i,(w,h) in enumerate(anchors):
        rect=plt.Rectangle((center[0]-w/2,center[1]-h/2),w,h,
                            fill=False,edgecolor=PALETTE[i%8],linewidth=1.5,alpha=0.8)
        ax4.add_patch(rect)
    ax4.scatter(*center,color=GOLD,s=100,zorder=5)
    ax4.text(5,9.2,'Anchor Boxes at Single Grid Cell',ha='center',fontsize=9,
             fontweight='bold',color=TX)
    ax4.text(5,0.4,'Multiple scales & aspect ratios',ha='center',fontsize=8,color=TX2)
    ax4.set_title('Anchor Box Strategy',fontweight='bold',color=TX)

    # Non-Maximum Suppression
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,8)
    ax5.text(5,7.5,'Non-Maximum Suppression (NMS)',ha='center',fontsize=10,fontweight='bold',color=TX)
    steps_nms=[('1. Sort boxes','by confidence score (highest first)',TEAL),
               ('2. Select highest-conf box','Add to output list',GREEN),
               ('3. Compute IoU','with remaining candidate boxes',GOLD),
               ('4. Suppress overlapping','Remove boxes with IoU > threshold',RUST),
               ('5. Repeat','Until no candidates remain',PURPLE),
               ('Soft-NMS','Decay scores instead of removing → better recall',BLUE)]
    for i,(step,desc,col) in enumerate(steps_nms):
        y=6.2-i*1.0
        ax5.text(0.3,y,step+':',fontsize=9,fontweight='bold',color=col)
        ax5.text(0.3,y-0.42,desc,fontsize=7.5,color=TX2)
    ax5.set_title('NMS Algorithm Steps',fontweight='bold',color=TX)
    fig.suptitle('Object Detection & Computer Vision',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['object_detection']=chart_object_detection()
print("14. object_detection done")

# ─────────────────────────────────────────────────────────────────────────────
# 15. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def chart_loss_functions():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Classification losses
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    p=np.linspace(0.01,0.99,200)
    bce=-np.log(p); hinge=np.maximum(0,1-p*2+1)
    ax0.plot(p,bce,color=TEAL,lw=2,label='Binary Cross-Entropy')
    ax0.plot(p,np.maximum(0,1-p),color=GOLD,lw=2,label='Hinge Loss')
    ax0.plot(p,-np.log(p)*0.5+np.maximum(0,1-p)*0.5,color=RUST,lw=2,linestyle='--',label='Combined')
    ax0.set_xlabel('Predicted Probability (positive class)')
    ax0.set_ylabel('Loss'); ax0.set_ylim(0,5)
    ax0.set_title('Classification Losses',fontweight='bold',color=TX)
    ax0.legend(); ax0.grid(True,alpha=0.3)

    # Regression losses
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    e=np.linspace(-4,4,200)
    ax1.plot(e,e**2,color=TEAL,lw=2,label='MSE (L2)')
    ax1.plot(e,np.abs(e),color=GOLD,lw=2,label='MAE (L1)')
    delta=1.5
    huber=np.where(np.abs(e)<=delta,0.5*e**2,delta*(np.abs(e)-0.5*delta))
    ax1.plot(e,huber,color=RUST,lw=2,label=f'Huber (δ={delta})')
    ax1.plot(e,np.log(np.cosh(e)),color=PURPLE,lw=2,linestyle='--',label='Log-Cosh')
    ax1.set_xlabel('Error (y - ŷ)'); ax1.set_ylabel('Loss')
    ax1.set_title('Regression Loss Functions',fontweight='bold',color=TX)
    ax1.legend(fontsize=7.5); ax1.grid(True,alpha=0.3); ax1.set_ylim(0,8)

    # Focal loss
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    p2=np.linspace(0.01,0.99,200)
    for gamma,col in [(0,TEAL),(0.5,GOLD),(1,RUST),(2,PURPLE),(5,GREEN)]:
        fl=-(1-p2)**gamma*np.log(p2)
        ax2.plot(p2,fl,color=col,lw=2,label=f'γ={gamma}'+(' (CE)' if gamma==0 else ''))
    ax2.set_xlabel('P(correct class)'); ax2.set_ylabel('Loss')
    ax2.set_title('Focal Loss (class imbalance)',fontweight='bold',color=TX)
    ax2.legend(fontsize=7.5); ax2.grid(True,alpha=0.3); ax2.set_ylim(0,4)

    # KL divergence
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    x=np.linspace(0.01,5,200)
    # P = Gamma(2,1), Q_i = different params
    from scipy.stats import norm
    p_dist=norm.pdf(x,2,0.7)
    ax3.plot(x,p_dist,color=TEAL,lw=2.5,label='P (true)')
    for mu,col,lbl in [(2.5,GOLD,'Q (small KL)'),(4,RUST,'Q (large KL)'),
                       (2,GREEN,'Q ≈ P (KL≈0)')]:
        q=norm.pdf(x,mu,0.7)
        ax3.plot(x,q,color=col,lw=1.5,linestyle='--',label=lbl)
    ax3.set_xlabel('x'); ax3.set_ylabel('Density')
    ax3.set_title('KL Divergence: D_KL(P‖Q)',fontweight='bold',color=TX)
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.3)

    # Contrastive / Triplet
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF); ax4.axis('off')
    ax4.set_xlim(0,10); ax4.set_ylim(0,8)
    ax4.text(5,7.5,'Metric & Contrastive Losses',ha='center',fontsize=10,fontweight='bold',color=TX)
    losses=[('Triplet Loss','L = max(d(a,p) - d(a,n) + margin, 0)\nPull anchor-positive, push anchor-negative',TEAL),
            ('Contrastive','L = (1-y)·d²/2 + y·max(m-d,0)²/2\ny=0 same, y=1 different class',GOLD),
            ('NT-Xent','SimCLR: normalize embeddings, cosine sim,\ntemperature-scaled cross-entropy',RUST),
            ('InfoNCE','Mutual information maximization:\nlog(f(x,c)/Σf(xⱼ,c)) — used in CPC',PURPLE),
            ('ArcFace/CosFace','Angular margin in softmax for face recog.\nAdditive angular margin penalty',GREEN)]
    for i,(name,desc,col) in enumerate(losses):
        y=6.2-i*1.18
        ax4.text(0.3,y,name+':',fontsize=9,fontweight='bold',color=col)
        for j,line in enumerate(desc.split('\n')):
            ax4.text(0.3,y-0.42-j*0.36,line,fontsize=7,color=TX2)
    ax4.set_title('Metric Learning Losses',fontweight='bold',color=TX)

    # Loss landscape
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    np.random.seed(42)
    w1,w2=np.meshgrid(np.linspace(-3,3,60),np.linspace(-3,3,60))
    # Sharp vs flat minima
    sharp=2*(w1**2+w2**2)+0.5*np.sin(5*w1)*np.sin(5*w2)
    flat=0.5*(w1**2+w2**2)+0.05*np.sin(3*w1)*np.sin(3*w2)
    ax5.contourf(w1,w2,sharp,levels=20,cmap='hot',alpha=0.7)
    ax5.contour(w1,w2,sharp,levels=10,colors='white',alpha=0.3,linewidths=0.5)
    ax5.scatter([0],[0],color=TEAL,s=150,marker='*',zorder=5,label='Global min (sharp)')
    ax5.scatter([0.5],[0.3],color=GOLD,s=100,marker='o',zorder=5,label='Local min (flat)')
    ax5.set_xlabel('Weight w₁'); ax5.set_ylabel('Weight w₂')
    ax5.set_title('Loss Landscape: Sharp vs Flat Min',fontweight='bold',color=TX)
    ax5.legend(fontsize=7.5)
    fig.suptitle('Loss Functions & Optimization Objectives',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['loss_functions']=chart_loss_functions()
print("15. loss_functions done")

# ─────────────────────────────────────────────────────────────────────────────
# 16. DATA PREPROCESSING & IMBALANCED DATA
# ─────────────────────────────────────────────────────────────────────────────
def chart_data_preprocessing():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Scaling comparison
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    np.random.seed(42)
    raw=np.random.exponential(5,200)
    std_scaled=(raw-raw.mean())/raw.std()
    minmax=(raw-raw.min())/(raw.max()-raw.min())
    ax0.hist(raw,bins=25,color=RUST,alpha=0.6,label='Raw',density=True)
    ax0.hist(std_scaled,bins=25,color=TEAL,alpha=0.6,label='StandardScaler (z-score)',density=True)
    ax0.hist(minmax,bins=25,color=GOLD,alpha=0.6,label='MinMaxScaler [0,1]',density=True)
    ax0.set_xlabel('Value'); ax0.set_ylabel('Density')
    ax0.set_title('Scaling Methods',fontweight='bold',color=TX)
    ax0.legend(fontsize=7.5); ax0.grid(True,alpha=0.3)

    # Missing value heatmap
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42)
    n_rows,n_cols=10,8
    missing_mask=np.random.rand(n_rows,n_cols)<0.25
    data_vis=np.ones((n_rows,n_cols))
    data_vis[missing_mask]=0
    colors_mv=[[TEAL if v else RUST for v in row] for row in data_vis]
    ax1.imshow(data_vis,cmap=plt.cm.RdYlGn,aspect='auto',vmin=0,vmax=1)
    for i in range(n_rows):
        for j in range(n_cols):
            ax1.text(j,i,'?' if not data_vis[i,j] else '',
                     ha='center',va='center',fontsize=9,color='white')
    ax1.set_xticks(range(n_cols))
    ax1.set_xticklabels([f'F{i+1}' for i in range(n_cols)],fontsize=8)
    ax1.set_yticks(range(n_rows))
    ax1.set_yticklabels([f'R{i+1}' for i in range(n_rows)],fontsize=8)
    ax1.set_title('Missing Value Pattern',fontweight='bold',color=TX)

    # Imbalanced class distribution
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    classes=['Class 0\n(Majority)','Class 1\n(Minority)']
    counts_orig=[950,50]; counts_smote=[950,950]; counts_under=[50,50]
    x=np.arange(2); w=0.25
    ax2.bar(x-w,counts_orig,w,color=RUST,alpha=0.8,label='Original')
    ax2.bar(x,counts_smote,w,color=TEAL,alpha=0.8,label='SMOTE (oversample)')
    ax2.bar(x+w,counts_under,w,color=GOLD,alpha=0.8,label='Undersample')
    ax2.set_xticks(x); ax2.set_xticklabels(classes)
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Class Imbalance Solutions',fontweight='bold',color=TX)
    ax2.legend(fontsize=7.5); ax2.grid(True,alpha=0.3,axis='y')

    # SMOTE visualization
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    np.random.seed(42)
    minority=np.random.randn(20,2)*0.5+[2,2]
    majority=np.random.randn(200,2)*1.5
    # SMOTE: interpolate between minority neighbors
    smote_pts=[]
    for i in range(50):
        idx1,idx2=np.random.choice(20,2,replace=False)
        alpha=np.random.rand()
        smote_pts.append(minority[idx1]*alpha+minority[idx2]*(1-alpha))
    smote_pts=np.array(smote_pts)
    ax3.scatter(majority[:,0],majority[:,1],color=TEAL,alpha=0.3,s=15,label='Majority (N=200)')
    ax3.scatter(minority[:,0],minority[:,1],color=RUST,s=60,label='Minority (N=20)',zorder=4)
    ax3.scatter(smote_pts[:,0],smote_pts[:,1],color=GOLD,s=40,marker='^',
                alpha=0.7,label='SMOTE synthetic',zorder=3)
    ax3.set_xlabel('Feature 1'); ax3.set_ylabel('Feature 2')
    ax3.set_title('SMOTE Oversampling',fontweight='bold',color=TX)
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.3)

    # Outlier detection
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    np.random.seed(42)
    normal=np.random.randn(200,2)
    outliers=np.random.randn(10,2)*3+5
    data_all=np.vstack([normal,outliers])
    from sklearn.ensemble import IsolationForest
    iso=IsolationForest(contamination=0.05,random_state=42).fit(data_all)
    preds=iso.predict(data_all)
    ax4.scatter(data_all[preds==1,0],data_all[preds==1,1],
                color=TEAL,alpha=0.6,s=20,label='Inlier')
    ax4.scatter(data_all[preds==-1,0],data_all[preds==-1,1],
                color=RUST,s=80,marker='X',zorder=4,label='Outlier (Isolation Forest)')
    ax4.set_xlabel('Feature 1'); ax4.set_ylabel('Feature 2')
    ax4.set_title('Outlier Detection',fontweight='bold',color=TX)
    ax4.legend(fontsize=7.5); ax4.grid(True,alpha=0.3)

    # Preprocessing pipeline
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,8)
    ax5.text(5,7.5,'Data Preprocessing Pipeline',ha='center',fontsize=10,fontweight='bold',color=TX)
    steps_pp=[('Data Collection','Load, validate schema, check dtypes',BLUE),
              ('Missing Values','Mean/median/mode impute, KNN impute, or drop',TEAL),
              ('Outlier Handling','IQR clipping, Z-score, Isolation Forest',GOLD),
              ('Encoding','OHE (low card.), TargetEnc (high card.), Ordinal',RUST),
              ('Scaling','StandardScaler (linear), RobustScaler (outliers)',PURPLE),
              ('Class Balancing','SMOTE, class_weight, threshold tuning',GREEN),
              ('Feature Selection','VIF, correlation matrix, RFECV, SHAP',ORANGE)]
    for i,(name,desc,col) in enumerate(steps_pp):
        y=6.5-i*0.9
        ax5.text(0.3,y,f'{i+1}. {name}:',fontsize=8.5,fontweight='bold',color=col)
        ax5.text(0.3,y-0.38,desc,fontsize=7.5,color=TX2)
    ax5.set_title('Preprocessing Checklist',fontweight='bold',color=TX)
    fig.suptitle('Data Preprocessing & Class Imbalance',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['data_preprocessing']=chart_data_preprocessing()
print("16. data_preprocessing done")

# ─────────────────────────────────────────────────────────────────────────────
# 17. STATISTICAL TESTS & HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────────────────────
def chart_statistical_tests():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)
    from scipy import stats

    # Normal distribution + critical region
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF)
    x=np.linspace(-4,4,400)
    y_norm=stats.norm.pdf(x)
    ax0.plot(x,y_norm,color=TEAL,lw=2.5)
    ax0.fill_between(x,y_norm,where=x>1.96,color=RUST,alpha=0.6,label='α/2=0.025 (reject H₀)')
    ax0.fill_between(x,y_norm,where=x<-1.96,color=RUST,alpha=0.6)
    ax0.fill_between(x,y_norm,where=(-1.96<=x)&(x<=1.96),color=TEAL,alpha=0.15,label='Fail to reject H₀')
    ax0.axvline(1.96,color=RUST,linestyle='--',lw=1.5)
    ax0.axvline(-1.96,color=RUST,linestyle='--',lw=1.5)
    ax0.text(2.5,0.1,'α=0.025',color=RUST,fontsize=8,rotation=0)
    ax0.text(-3.5,0.1,'α=0.025',color=RUST,fontsize=8)
    ax0.set_title('Two-Tailed Z-Test (α=0.05)',fontweight='bold',color=TX)
    ax0.legend(fontsize=7.5); ax0.grid(True,alpha=0.3)
    ax0.set_xlabel('Test Statistic (z)'); ax0.set_ylabel('Density')

    # P-value interpretation
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    x=np.linspace(-4,4,400)
    y_norm=stats.norm.pdf(x)
    obs_z=2.4
    ax1.plot(x,y_norm,color=TEAL,lw=2)
    ax1.fill_between(x,y_norm,where=x>obs_z,color=RUST,alpha=0.7,label=f'p-value = {2*stats.norm.sf(obs_z):.4f}')
    ax1.fill_between(x,y_norm,where=x<-obs_z,color=RUST,alpha=0.7)
    ax1.axvline(obs_z,color=GOLD,lw=2,label=f'Observed z={obs_z}')
    ax1.axvline(-obs_z,color=GOLD,lw=2)
    ax1.set_xlabel('Test Statistic'); ax1.set_ylabel('Density')
    ax1.set_title('P-Value Visualization',fontweight='bold',color=TX)
    ax1.legend(fontsize=8); ax1.grid(True,alpha=0.3)

    # Type I and Type II errors
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    x=np.linspace(-2,8,400)
    h0=stats.norm.pdf(x,0,1); h1=stats.norm.pdf(x,3,1)
    threshold=1.65
    ax2.plot(x,h0,color=TEAL,lw=2,label='H₀ distribution')
    ax2.plot(x,h1,color=RUST,lw=2,label='H₁ distribution')
    ax2.fill_between(x,h0,where=x>threshold,color=TEAL,alpha=0.4,label='Type I Error (α, FP)')
    ax2.fill_between(x,h1,where=x<=threshold,color=RUST,alpha=0.4,label='Type II Error (β, FN)')
    ax2.axvline(threshold,color=GOLD,linestyle='--',lw=1.5,label=f'Threshold={threshold}')
    ax2.set_xlabel('Test Statistic'); ax2.set_ylabel('Density')
    ax2.set_title('Type I & Type II Errors',fontweight='bold',color=TX)
    ax2.legend(fontsize=7); ax2.grid(True,alpha=0.3)

    # Statistical power
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    effect_sizes=np.linspace(0,2,100)
    n_values=[10,30,100,500]
    for n,col in zip(n_values,PALETTE):
        from scipy.stats import norm as norm_s
        power=[norm_s.sf(1.645-e*np.sqrt(n))+norm_s.cdf(-1.645-e*np.sqrt(n)) for e in effect_sizes]
        ax3.plot(effect_sizes,power,color=col,lw=2,label=f'n={n}')
    ax3.axhline(0.8,color=TX3,linestyle='--',lw=1,label='Power=0.8 (convention)')
    ax3.set_xlabel("Effect Size (Cohen's d)"); ax3.set_ylabel('Statistical Power (1-β)')
    ax3.set_title('Power Analysis',fontweight='bold',color=TX)
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.3)

    # Test selection guide
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF); ax4.axis('off')
    ax4.set_xlim(0,10); ax4.set_ylim(0,8)
    ax4.text(5,7.5,'Statistical Test Selection Guide',ha='center',fontsize=10,fontweight='bold',color=TX)
    tests=[('t-test (2 groups)','Compare means: Welch (unequal var) or Student',TEAL),
           ('ANOVA','Compare means: 3+ groups; F-statistic',GOLD),
           ('Chi-squared','Categorical independence / goodness-of-fit',RUST),
           ('Mann-Whitney U','Non-parametric 2-group comparison (ranks)',PURPLE),
           ('Kruskal-Wallis','Non-parametric ANOVA equivalent',GREEN),
           ('Wilcoxon Signed','Paired non-parametric t-test alternative',BLUE),
           ('Kolmogorov-Smirnov','Distribution similarity test (2 samples)',ORANGE)]
    for i,(name,desc,col) in enumerate(tests):
        y=6.3-i*0.9
        ax4.text(0.3,y,name+':',fontsize=8.5,fontweight='bold',color=col)
        ax4.text(0.3,y-0.4,desc,fontsize=7.5,color=TX2)
    ax4.set_title('Statistical Tests Reference',fontweight='bold',color=TX)

    # Multiple testing correction
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    n_tests=np.arange(1,51)
    bonferroni_alpha=0.05/n_tests
    fwer_no_correction=1-(1-0.05)**n_tests
    ax5.plot(n_tests,fwer_no_correction,color=RUST,lw=2,label='FWER (no correction)')
    ax5.plot(n_tests,bonferroni_alpha,color=TEAL,lw=2,label='Bonferroni α*')
    ax5.plot(n_tests,np.full_like(n_tests,0.05,dtype=float),color=GOLD,
             linestyle='--',lw=1.5,label='α=0.05 target')
    ax5.fill_between(n_tests,0.05,fwer_no_correction,alpha=0.15,color=RUST,label='Inflation zone')
    ax5.set_xlabel('Number of Tests'); ax5.set_ylabel('Rate')
    ax5.set_title('Multiple Testing Correction',fontweight='bold',color=TX)
    ax5.legend(fontsize=7.5); ax5.grid(True,alpha=0.3)
    fig.suptitle('Statistical Tests & Hypothesis Testing',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['statistical_tests']=chart_statistical_tests()
print("17. statistical_tests done")

# ─────────────────────────────────────────────────────────────────────────────
# 18. GRAPH NEURAL NETWORKS (GNN)
# ─────────────────────────────────────────────────────────────────────────────
def chart_gnn():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Graph structure
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,8)
    nodes={'A':(3,6.5),'B':(1,4),'C':(5,4),'D':(3,1.5),'E':(7,2),'F':(8,5.5)}
    edges=[('A','B'),('A','C'),('B','D'),('C','D'),('C','E'),('E','F'),('C','F')]
    colors_gnn=[TEAL,GOLD,RUST,PURPLE,GREEN,BLUE]
    for (name,(x,y)),col in zip(nodes.items(),colors_gnn):
        circ=Circle((x,y),0.6,facecolor=col+'55',edgecolor=col,linewidth=2)
        ax0.add_patch(circ)
        ax0.text(x,y,name,ha='center',va='center',fontsize=12,fontweight='bold',color=TX)
    for n1,n2 in edges:
        x1,y1=nodes[n1]; x2,y2=nodes[n2]
        ax0.plot([x1,x2],[y1,y2],color=TX3,lw=1.5,zorder=0)
    ax0.set_title('Graph: Nodes + Edges',fontweight='bold',color=TX)

    # Message passing
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF); ax1.axis('off')
    ax1.set_xlim(0,10); ax1.set_ylim(0,7)
    # Show node C aggregating from neighbors A, B, E
    center=(5,3.5)
    neighbors={'A':(2,6),'D':(5,1),'E':(8,2),'F':(8,5.5)}
    circ_c=Circle(center,0.9,facecolor=TEAL+'55',edgecolor=TEAL,linewidth=2.5)
    ax1.add_patch(circ_c)
    ax1.text(*center,'C\n(target)',ha='center',va='center',fontsize=9,color=TEAL,fontweight='bold')
    for i,(name,(x,y)) in enumerate(neighbors.items()):
        col=PALETTE[i+1]
        circ=Circle((x,y),0.6,facecolor=col+'44',edgecolor=col,linewidth=1.5)
        ax1.add_patch(circ)
        ax1.text(x,y,name,ha='center',va='center',fontsize=10,color=col,fontweight='bold')
        ax1.annotate('',xy=(center[0]+(x-center[0])*0.4,center[1]+(y-center[1])*0.4),
            xytext=(x+(center[0]-x)*0.4,y+(center[1]-y)*0.4),
            arrowprops=dict(arrowstyle='->,head_width=0.3',color=col,lw=1.8))
        ax1.text((x+center[0])/2,(y+center[1])/2,f'm_{name}→C',fontsize=7,color=col,ha='center')
    ax1.text(5,6.8,'h_C^(l+1) = σ(W·AGGREGATE({h_N(C)^(l)}))',ha='center',fontsize=8.5,color=GOLD)
    ax1.set_title('Message Passing (GCN)',fontweight='bold',color=TX)

    # GNN variants comparison
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF); ax2.axis('off')
    ax2.set_xlim(0,10); ax2.set_ylim(0,8)
    ax2.text(5,7.5,'GNN Variants',ha='center',fontsize=10,fontweight='bold',color=TX)
    variants_gnn=[('GCN','Spectral: normalized Laplacian conv\nSpectral → spatial equiv.',TEAL),
                  ('GraphSAGE','Sample & aggregate neighborhood\nInductive — works on unseen nodes',GOLD),
                  ('GAT','Attention weights on edges\nα_ij = softmax(a^T[Wh_i‖Wh_j])',RUST),
                  ('GIN','Most expressive (WL isomorphism test)\nh = MLP(h + Σ neighbor_h)',PURPLE),
                  ('MPNN','General message-passing framework\nUnifies GCN, GraphSAGE, GAT',GREEN)]
    for i,(name,desc,col) in enumerate(variants_gnn):
        y=6.3-i*1.2
        ax2.text(0.3,y,name+':',fontsize=9,fontweight='bold',color=col)
        for j,line in enumerate(desc.split('\n')):
            ax2.text(0.3,y-0.42-j*0.38,line,fontsize=7.5,color=TX2)
    ax2.set_title('GNN Architecture Guide',fontweight='bold',color=TX)

    # Node classification performance
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    datasets=['Cora','Citeseer','PubMed','OGB-Arxiv']
    gcn=[81.5,70.3,79.0,71.7]; gat=[83.0,72.5,79.0,73.9]; sage=[81.0,71.0,78.0,71.5]
    x=np.arange(len(datasets)); w=0.25
    ax3.bar(x-w,gcn,w,label='GCN',color=TEAL,alpha=0.85)
    ax3.bar(x,gat,w,label='GAT',color=GOLD,alpha=0.85)
    ax3.bar(x+w,sage,w,label='GraphSAGE',color=RUST,alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(datasets,fontsize=8)
    ax3.set_ylabel('Test Accuracy (%)'); ax3.set_ylim(60,90)
    ax3.set_title('GNN Node Classification',fontweight='bold',color=TX)
    ax3.legend(fontsize=8); ax3.grid(True,alpha=0.3,axis='y')

    # Attention weights visualization
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    np.random.seed(42)
    adj=np.array([[0,1,1,0,0,1],
                  [1,0,1,1,0,0],
                  [1,1,0,1,1,0],
                  [0,1,1,0,1,0],
                  [0,0,1,1,0,1],
                  [1,0,0,0,1,0]],dtype=float)
    attn_weights=adj*np.random.rand(6,6); attn_weights/=attn_weights.sum(axis=1,keepdims=True)+1e-9
    im=ax4.imshow(attn_weights,cmap='YlOrRd',aspect='auto')
    ax4.set_xticks(range(6)); ax4.set_yticks(range(6))
    ax4.set_xticklabels(list(nodes.keys()),fontsize=9)
    ax4.set_yticklabels(list(nodes.keys()),fontsize=9)
    plt.colorbar(im,ax=ax4,shrink=0.8)
    ax4.set_title('GAT Attention Weights',fontweight='bold',color=TX)

    # Applications
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,8)
    ax5.text(5,7.5,'GNN Applications',ha='center',fontsize=10,fontweight='bold',color=TX)
    apps=[('Node Classification','Social network user type prediction',TEAL),
          ('Link Prediction','Recommend friends, knowledge graph completion',GOLD),
          ('Graph Classification','Molecule property prediction (drug discovery)',RUST),
          ('Traffic Prediction','Road network → arrival time (Google Maps)',PURPLE),
          ('Recommendation','User-item graph → collaborative filtering',GREEN),
          ('Protein Structure','AlphaFold-like contact map prediction',BLUE)]
    for i,(task,example,col) in enumerate(apps):
        y=6.3-i*1.05
        ax5.text(0.3,y,task+':',fontsize=9,fontweight='bold',color=col)
        ax5.text(0.3,y-0.42,example,fontsize=7.5,color=TX2)
    ax5.set_title('Applications',fontweight='bold',color=TX)
    fig.suptitle('Graph Neural Networks (GNN)',fontsize=13,fontweight='bold',color=TX,y=1.01)
    return fig_to_b64(fig)

charts['gnn']=chart_gnn()
print("18. gnn done")

# ─────────────────────────────────────────────────────────────────────────────
# 19. MODEL COMPRESSION (Pruning, Quantization, Distillation)
# ─────────────────────────────────────────────────────────────────────────────
def chart_model_compression():
    fig=plt.figure(figsize=(14,9)); fig.patch.set_facecolor(BG)
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)

    # Compression techniques overview
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,8)
    ax0.text(5,7.5,'Model Compression Techniques',ha='center',fontsize=10,fontweight='bold',color=TX)
    techniques=[('Pruning','Remove unimportant weights/neurons\n(magnitude, gradient, structured)',TEAL),
                ('Quantization','Reduce precision: FP32→INT8→INT4→binary\n4x smaller, 2-4x faster inference',GOLD),
                ('Knowledge Distillation','Student learns from teacher softmax\nL = α·CE(y,ŷ_s)+(1-α)·KL(ŷ_t‖ŷ_s)',RUST),
                ('Low-Rank Factorization','Decompose W=UΣVᵀ, keep top-k singular values\nSVD-based weight compression',PURPLE),
                ('Neural Architecture Search','AutoML to find compact, efficient arch\nDifferentiable NAS (DARTS)',GREEN)]
    for i,(name,desc,col) in enumerate(techniques):
        y=6.5-i*1.25
        ax0.text(0.3,y,name+':',fontsize=9,fontweight='bold',color=col)
        for j,line in enumerate(desc.split('\n')):
            ax0.text(0.3,y-0.42-j*0.36,line,fontsize=7.5,color=TX2)
    ax0.set_title('Compression Techniques',fontweight='bold',color=TX)

    # Pruning effect
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    sparsity=[0,10,20,30,40,50,60,70,80,90,95]
    accuracy=[94,94,93.8,93.5,93,92,90.5,88,83,72,58]
    speedup=[1,1.05,1.1,1.18,1.3,1.5,1.8,2.2,3.0,4.5,6.0]
    ax1_twin=ax1.twinx()
    ax1.plot(sparsity,accuracy,color=TEAL,lw=2,marker='o',markersize=5,label='Accuracy (%)')
    ax1_twin.plot(sparsity,speedup,color=GOLD,lw=2,marker='s',markersize=5,label='Speedup (×)')
    ax1.axvline(50,color=RUST,linestyle='--',lw=1.5,label='Sweet spot')
    ax1.set_xlabel('Model Sparsity (%)'); ax1.set_ylabel('Accuracy (%)',color=TEAL)
    ax1_twin.set_ylabel('Inference Speedup (×)',color=GOLD)
    ax1.set_title('Pruning: Accuracy vs Sparsity',fontweight='bold',color=TX)
    lines1,labs1=ax1.get_legend_handles_labels()
    lines2,labs2=ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1+lines2,labs1+labs2,fontsize=7.5); ax1.grid(True,alpha=0.3)

    # Quantization levels
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    bits=['FP32','FP16','BF16','INT8','INT4','Binary']
    size_mb=[100,50,50,25,12.5,3.1]
    top1_drop=[0,0.1,0.2,0.5,1.5,5.0]
    bars=ax2.bar(bits,size_mb,color=[BLUE,TEAL,GREEN,GOLD,RUST,PURPLE],alpha=0.85)
    ax2_twin=ax2.twinx()
    ax2_twin.plot(bits,top1_drop,color=RUST,lw=2,marker='D',markersize=7,label='Accuracy drop (%)')
    for bar,s in zip(bars,size_mb):
        ax2.text(bar.get_x()+bar.get_width()/2,s+0.5,f'{s}MB',ha='center',fontsize=8,color=TX)
    ax2.set_ylabel('Model Size (MB)'); ax2_twin.set_ylabel('Accuracy Drop (%)',color=RUST)
    ax2.set_title('Quantization: Size vs Accuracy',fontweight='bold',color=TX)
    ax2_twin.legend(fontsize=8)

    # Knowledge distillation
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF); ax3.axis('off')
    ax3.set_xlim(0,10); ax3.set_ylim(0,7)
    # Teacher
    teacher=FancyBboxPatch((0.3,4.5),3,2,boxstyle='round,pad=0.1',
        facecolor=RUST+'33',edgecolor=RUST,linewidth=2)
    ax3.add_patch(teacher)
    ax3.text(1.8,5.5,'Teacher\n(large model)',ha='center',fontsize=9,color=RUST,fontweight='bold')
    # Student
    student=FancyBboxPatch((0.3,1.5),2,1.5,boxstyle='round,pad=0.1',
        facecolor=TEAL+'33',edgecolor=TEAL,linewidth=2)
    ax3.add_patch(student)
    ax3.text
    ax3.text(1.3,2.25,'Student\n(small)',ha='center',fontsize=9,color=TEAL,fontweight='bold')
    ax3.annotate('',xy=(1.3,3.1),xytext=(1.8,4.5),arrowprops=dict(arrowstyle='->',color=GOLD,lw=2))
    ax3.text(3.5,3.5,'Soft targets\n(dark knowledge)',fontsize=8.5,color=GOLD,ha='center')
    distill_metrics=[('Teacher Acc','94%',RUST),('Student Acc','91%',TEAL),('Speedup','4x',GREEN),('Size reduction','8x',PURPLE)]
    for j,(lbl,val,c) in enumerate(distill_metrics):
        ax3.text(6.5,5.8-j*1.2,f'{lbl}: {val}',fontsize=9,color=c,fontweight='bold')
    ax3.set_title('Knowledge Distillation',fontweight='bold',color=TX)

    # Pruning types
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    prune_types=['Unstructured\n(weight)','Structured\n(filter)','Magnitude\n(global)','Lottery Ticket\n(subnetwork)']
    sparsity=[0,50,70,80]
    acc_retain=[100,98,95,93]
    bars2=ax4.barh(prune_types,sparsity,color=[BLUE,TEAL,GOLD,RUST],alpha=0.85)
    for bar,a in zip(bars2,acc_retain):
        ax4.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'Acc:{a}%',va='center',fontsize=8,color=TX)
    ax4.set_xlabel('Sparsity %'); ax4.set_xlim(0,100)
    ax4.set_title('Pruning Methods vs Sparsity',fontweight='bold',color=TX); ax4.grid(True,alpha=0.2,axis='x')

    # NAS chart
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    nas_methods=['Manual\nDesign','Random\nSearch','Bayesian\nNAS','Evolutionary\nNAS','DARTS\n(gradient)','One-shot\nNAS']
    search_cost=[0,100,30,50,5,2]
    top1=[76,77,78,79,80.5,81]
    scatter_c=[PURPLE,BLUE,TEAL,GREEN,GOLD,RUST]
    for i,(m,c,t,sc) in enumerate(zip(nas_methods,scatter_c,top1,search_cost)):
        ax5.scatter(sc,t,s=180,color=c,zorder=5,edgecolors='white',linewidth=0.8)
        ax5.annotate(m,(sc,t),textcoords='offset points',xytext=(5,3),fontsize=7,color=c)
    ax5.set_xlabel('Search Cost (GPU days)'); ax5.set_ylabel('Top-1 Accuracy (%)')
    ax5.set_title('Neural Architecture Search',fontweight='bold',color=TX); ax5.grid(True,alpha=0.2)

    plt.suptitle('Model Compression, Pruning & NAS',fontsize=13,fontweight='bold',color=TX,y=1.01)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_causal_inference():
    fig=plt.figure(figsize=(14,8),facecolor=BG)
    gs=fig.add_gridspec(2,3,hspace=0.45,wspace=0.4)

    # DAG
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,7)
    nodes={'X':(2,3.5),'Y':(8,3.5),'Z':(5,6),'W':(5,1)}
    colors_n={'X':TEAL,'Y':RUST,'Z':GOLD,'W':PURPLE}
    for n,(x,y) in nodes.items():
        circ=plt.Circle((x,y),0.7,color=colors_n[n],alpha=0.85,zorder=3)
        ax0.add_patch(circ); ax0.text(x,y,n,ha='center',va='center',fontsize=13,fontweight='bold',color=BG,zorder=4)
    edges=[('Z','X'),('Z','Y'),('X','Y'),('X','W'),('W','Y')]
    for s,e in edges:
        x1,y1=nodes[s]; x2,y2=nodes[e]
        ax0.annotate('',xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle='->',color=TX2,lw=1.8,connectionstyle='arc3,rad=0.05'))
    ax0.text(5,3.9,'Causal Effect\nX→Y',ha='center',fontsize=8,color=GREEN)
    ax0.text(5,5,'Confounder Z',ha='center',fontsize=8,color=GOLD)
    ax0.set_title('Directed Acyclic Graph (DAG)',fontweight='bold',color=TX)

    # ATE estimation
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(0)
    treated=np.random.normal(7,1.5,200); control=np.random.normal(5,1.5,200)
    ax1.hist(control,bins=25,alpha=0.7,color=BLUE,label='Control (no treatment)')
    ax1.hist(treated,bins=25,alpha=0.7,color=RUST,label='Treated')
    ax1.axvline(np.mean(control),color=BLUE,lw=2,ls='--')
    ax1.axvline(np.mean(treated),color=RUST,lw=2,ls='--')
    ax1.annotate('',xy=(np.mean(treated),50),xytext=(np.mean(control),50),
        arrowprops=dict(arrowstyle='<->',color=GOLD,lw=2.5))
    ax1.text(6,53,'ATE ≈ 2.0',fontsize=9,color=GOLD,fontweight='bold',ha='center')
    ax1.set_title('Average Treatment Effect (ATE)',fontweight='bold',color=TX)
    ax1.legend(fontsize=7.5); ax1.set_xlabel('Outcome')

    # RDD
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    np.random.seed(1)
    x_run=np.linspace(-3,3,300)
    y_run=2*x_run+3*(x_run>=0)+np.random.normal(0,0.5,300)
    ax2.scatter(x_run[x_run<0],y_run[x_run<0],alpha=0.5,s=12,color=BLUE,label='Control')
    ax2.scatter(x_run[x_run>=0],y_run[x_run>=0],alpha=0.5,s=12,color=RUST,label='Treated')
    xc=x_run[x_run<0]; xct=x_run[x_run>=0]
    ax2.plot(xc,2*xc,color=BLUE,lw=2.5); ax2.plot(xct,2*xct+3,color=RUST,lw=2.5)
    ax2.axvline(0,color=GOLD,lw=2,ls='--',label='Cutoff')
    ax2.annotate('',xy=(0.1,0.3),xytext=(0.1,-2.7),arrowprops=dict(arrowstyle='<->',color=GREEN,lw=2))
    ax2.text(0.4,-1.2,'LATE\n≈ 3.0',fontsize=8.5,color=GREEN,fontweight='bold')
    ax2.set_title('Regression Discontinuity Design',fontweight='bold',color=TX)
    ax2.legend(fontsize=7.5); ax2.set_xlabel('Running Variable')

    # DiD
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    time=['Pre','Post']
    treated_y=[4,8]; control_y=[3,4]; counterfactual=[4,5]
    ax3.plot(time,treated_y,'o-',color=RUST,lw=2.5,ms=8,label='Treated (actual)')
    ax3.plot(time,control_y,'o-',color=BLUE,lw=2.5,ms=8,label='Control')
    ax3.plot(time,counterfactual,'o--',color=RUST,lw=2,ms=8,alpha=0.5,label='Treated (counterfactual)')
    ax3.fill_between(time,[counterfactual[0],counterfactual[1]],[treated_y[0],treated_y[1]],alpha=0.2,color=GOLD)
    ax3.text(1.05,6.5,'DiD = 3\n(causal effect)',fontsize=8.5,color=GOLD,fontweight='bold')
    ax3.set_title('Difference-in-Differences',fontweight='bold',color=TX); ax3.legend(fontsize=7.5)

    # Propensity score
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    np.random.seed(2)
    ps_c=np.random.beta(2,5,300); ps_t=np.random.beta(3,3,200)
    ax4.hist(ps_c,bins=25,alpha=0.7,color=BLUE,density=True,label='Control')
    ax4.hist(ps_t,bins=25,alpha=0.7,color=RUST,density=True,label='Treated')
    ax4.set_xlabel('Propensity Score P(T=1|X)')
    ax4.set_title('Propensity Score Distribution',fontweight='bold',color=TX)
    ax4.legend(fontsize=8); ax4.axvspan(0.2,0.8,alpha=0.08,color=GREEN,label='Common support')

    # IV
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,7)
    boxes={'Z\n(Instrument)':(1.5,3.5,PURPLE),'X\n(Treatment)':(5,3.5,TEAL),'Y\n(Outcome)':(8.5,3.5,RUST),'U\n(Confounder)':(6.5,5.5,GOLD)}
    for lbl,(x,y,c) in boxes.items():
        rect=FancyBboxPatch((x-0.9,y-0.7),1.8,1.4,boxstyle='round,pad=0.1',facecolor=c+'33',edgecolor=c,lw=2)
        ax5.add_patch(rect); ax5.text(x,y,lbl,ha='center',va='center',fontsize=9,color=c,fontweight='bold')
    ax5.annotate('',xy=(4.1,3.5),xytext=(2.4,3.5),arrowprops=dict(arrowstyle='->',color=TX,lw=2))
    ax5.annotate('',xy=(7.6,3.5),xytext=(5.9,3.5),arrowprops=dict(arrowstyle='->',color=TX,lw=2))
    ax5.annotate('',xy=(5.5,3.5),xytext=(6,4.8),arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.8,linestyle='dashed'))
    ax5.annotate('',xy=(8,3.5),xytext=(7,4.8),arrowprops=dict(arrowstyle='->',color=GOLD,lw=1.8,linestyle='dashed'))
    ax5.text(5,1.5,'IV Conditions:\n1. Relevance: Z→X\n2. Exclusion: Z⊥Y|X\n3. Exogeneity: Z⊥U',
             fontsize=8.5,color=TX2,ha='center',bbox=dict(boxstyle='round',fc=SURF2,ec=BD))
    ax5.set_title('Instrumental Variables (IV)',fontweight='bold',color=TX)

    plt.suptitle('Causal Inference Methods',fontsize=13,fontweight='bold',color=TX,y=1.01)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_multimodal():
    fig=plt.figure(figsize=(14,8),facecolor=BG)
    gs=fig.add_gridspec(2,3,hspace=0.45,wspace=0.4)

    # Modality diagram
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(SURF); ax0.axis('off')
    ax0.set_xlim(0,10); ax0.set_ylim(0,8)
    modalities=[('Image\nEncoder',2,6,TEAL),('Text\nEncoder',2,4,BLUE),('Audio\nEncoder',2,2,PURPLE)]
    for lbl,x,y,c in modalities:
        rect=FancyBboxPatch((x-0.9,y-0.6),1.8,1.2,boxstyle='round,pad=0.1',facecolor=c+'33',edgecolor=c,lw=2)
        ax0.add_patch(rect); ax0.text(x,y,lbl,ha='center',va='center',fontsize=8.5,color=c,fontweight='bold')
        ax0.annotate('',xy=(5,4),xytext=(x+0.9,y),arrowprops=dict(arrowstyle='->',color=c,lw=1.8))
    fusion=FancyBboxPatch((4.1,3.2),1.8,1.6,boxstyle='round,pad=0.1',facecolor=GOLD+'33',edgecolor=GOLD,lw=2)
    ax0.add_patch(fusion); ax0.text(5,4,'Fusion\nLayer',ha='center',va='center',fontsize=9,color=GOLD,fontweight='bold')
    output=FancyBboxPatch((7.1,3.4),1.8,1.2,boxstyle='round,pad=0.1',facecolor=RUST+'33',edgecolor=RUST,lw=2)
    ax0.add_patch(output); ax0.text(8,4,'Output\n(caption/class)',ha='center',va='center',fontsize=8,color=RUST,fontweight='bold')
    ax0.annotate('',xy=(7.1,4),xytext=(5.9,4),arrowprops=dict(arrowstyle='->',color=TX,lw=2))
    ax0.set_title('Multimodal Architecture',fontweight='bold',color=TX)

    # CLIP-style alignment
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    np.random.seed(42)
    n=8
    theta=np.linspace(0,2*np.pi,n,endpoint=False)
    r_img=1+0.1*np.random.randn(n); r_txt=1+0.1*np.random.randn(n)
    x_img=r_img*np.cos(theta); y_img=r_img*np.sin(theta)
    x_txt=r_txt*np.cos(theta+0.05); y_txt=r_txt*np.sin(theta+0.05)
    colors_m=[TEAL,RUST,GOLD,PURPLE,GREEN,BLUE,'#FF8C94','#A8D8EA']
    for i in range(n):
        ax1.scatter(x_img[i],y_img[i],s=120,color=colors_m[i],marker='s',zorder=5,edgecolors='white',lw=0.8)
        ax1.scatter(x_txt[i],y_txt[i],s=120,color=colors_m[i],marker='o',zorder=5,edgecolors='white',lw=0.8)
        ax1.plot([x_img[i],x_txt[i]],[y_img[i],y_txt[i]],color=colors_m[i],alpha=0.5,lw=1.2)
    from matplotlib.lines import Line2D
    legend_elements=[Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',ms=8,label='Image embedding'),
                     Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',ms=8,label='Text embedding')]
    ax1.legend(handles=legend_elements,fontsize=7.5)
    ax1.set_title('CLIP: Contrastive Alignment\n(Image ↔ Text embeddings)',fontweight='bold',color=TX)
    ax1.set_xlabel('Embedding dim 1'); ax1.set_ylabel('Embedding dim 2')
    ax1.grid(True,alpha=0.2)

    # VQA accuracy
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF)
    models_mm=['VQA v1\n(2016)','VilBERT\n(2019)','CLIP\n(2021)','BLIP\n(2022)','GPT-4V\n(2023)','Gemini\n(2024)']
    acc_mm=[63,72,76,82,88,92]
    bars_mm=ax2.bar(models_mm,acc_mm,color=[BLUE,TEAL,GREEN,GOLD,RUST,PURPLE],alpha=0.85)
    for bar,a in zip(bars_mm,acc_mm):
        ax2.text(bar.get_x()+bar.get_width()/2,a+0.3,f'{a}%',ha='center',fontsize=8.5,color=TX,fontweight='bold')
    ax2.set_ylabel('VQA Accuracy (%)'); ax2.set_ylim(55,100)
    ax2.set_title('Multimodal Model Progress (VQA)',fontweight='bold',color=TX)
    ax2.grid(True,alpha=0.2,axis='y')

    # Fusion types
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF); ax3.axis('off')
    ax3.set_xlim(0,10); ax3.set_ylim(0,7)
    fusions=[('Early Fusion\n(concat raw)',2,5.5,TEAL,'Simple, loses modality-specific features'),
             ('Late Fusion\n(combine preds)',2,3.5,GOLD,'Flexible, ignores cross-modal interactions'),
             ('Cross-Attention\n(transformer)',2,1.5,RUST,'Best accuracy, high compute')]
    for name,x,y,c,desc in fusions:
        rect=FancyBboxPatch((0.2,y-0.6),3.5,1.2,boxstyle='round,pad=0.1',facecolor=c+'22',edgecolor=c,lw=1.8)
        ax3.add_patch(rect); ax3.text(x,y,name,ha='center',va='center',fontsize=9,color=c,fontweight='bold')
        ax3.text(5,y,desc,va='center',fontsize=8,color=TX2)
    ax3.set_title('Fusion Strategies',fontweight='bold',color=TX)

    # Image captioning BLEU
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    metrics_cap=['BLEU-1','BLEU-4','METEOR','CIDEr','SPICE']
    old_model=[71,30,25,95,18]; new_model=[80,40,32,135,24]
    x_cap=np.arange(len(metrics_cap)); width=0.35
    ax4.bar(x_cap-width/2,old_model,width,label='BLIP (2022)',color=BLUE,alpha=0.85)
    ax4.bar(x_cap+width/2,new_model,width,label='BLIP-2 (2023)',color=RUST,alpha=0.85)
    ax4.set_xticks(x_cap); ax4.set_xticklabels(metrics_cap,fontsize=9)
    ax4.set_title('Image Captioning Metrics',fontweight='bold',color=TX)
    ax4.legend(fontsize=8); ax4.grid(True,alpha=0.2,axis='y')

    # Modality tokens
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF); ax5.axis('off')
    ax5.set_xlim(0,10); ax5.set_ylim(0,7)
    ax5.text(5,6.3,'LLaVA / GPT-4V Token Flow',ha='center',fontsize=10,color=TX,fontweight='bold')
    token_types=[('[IMG]',1.5,4.5,TEAL),('[IMG]',2.5,4.5,TEAL),('[IMG]',3.5,4.5,TEAL),
                 ('[TXT]',5,4.5,BLUE),('[TXT]',6,4.5,BLUE),('[ANS]',8,4.5,RUST)]
    for tok,x,y,c in token_types:
        rect=FancyBboxPatch((x-0.55,y-0.4),1.1,0.8,boxstyle='round,pad=0.05',facecolor=c+'33',edgecolor=c,lw=1.5)
        ax5.add_patch(rect); ax5.text(x,y,tok,ha='center',va='center',fontsize=7.5,color=c,fontweight='bold')
    ax5.text(2.5,3.5,'Image patches\n(vision encoder)',ha='center',fontsize=8,color=TEAL)
    ax5.text(5.5,3.5,'Question\ntokens',ha='center',fontsize=8,color=BLUE)
    ax5.text(8,3.5,'Generated\nanswer',ha='center',fontsize=8,color=RUST)
    ax5.text(5,1.8,'Projection layer maps visual tokens\ninto LLM token space (MLP/Q-Former)',
             ha='center',fontsize=8.5,color=TX2,bbox=dict(boxstyle='round',fc=SURF2,ec=BD))
    ax5.set_title('Visual Token Integration (LLaVA)',fontweight='bold',color=TX)

    plt.suptitle('Multimodal AI — Vision-Language Models',fontsize=13,fontweight='bold',color=TX,y=1.01)
    plt.tight_layout()
    return fig_to_b64(fig)


def chart_ethics_fairness():
    fig=plt.figure(figsize=(14,8),facecolor=BG)
    gs=fig.add_gridspec(2,3,hspace=0.45,wspace=0.4)

    # Fairness metrics radar
    ax0=fig.add_subplot(gs[0,0],polar=True); ax0.set_facecolor(SURF)
    metrics_fair=['Demographic\nParity','Equal\nOpportunity','Equalized\nOdds','Calibration','Individual\nFairness']
    N=len(metrics_fair)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    model_a=[0.95,0.80,0.75,0.90,0.70]; model_b=[0.70,0.95,0.90,0.85,0.85]
    model_a+=model_a[:1]; model_b+=model_b[:1]
    ax0.plot(angles,model_a,color=TEAL,lw=2,label='Model A'); ax0.fill(angles,model_a,color=TEAL,alpha=0.15)
    ax0.plot(angles,model_b,color=RUST,lw=2,label='Model B'); ax0.fill(angles,model_b,color=RUST,alpha=0.15)
    ax0.set_xticks(angles[:-1]); ax0.set_xticklabels(metrics_fair,fontsize=7.5)
    ax0.set_ylim(0,1); ax0.legend(loc='upper right',bbox_to_anchor=(1.3,1.1),fontsize=8)
    ax0.set_title('Fairness Metric Trade-offs',fontweight='bold',color=TX,pad=15)

    # Disparate impact
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(SURF)
    groups=['Group A\n(majority)','Group B\n(minority)']
    approved=[0.82,0.61]; rejected=[0.18,0.39]
    x1=np.arange(len(groups))
    ax1.bar(x1,approved,label='Approved',color=GREEN,alpha=0.85)
    ax1.bar(x1,rejected,bottom=approved,label='Rejected',color=RUST,alpha=0.85)
    ax1.axhline(0.8,color=GOLD,lw=2,ls='--',label='80% rule threshold')
    ax1.text(0.5,0.65,f'DI = {0.61/0.82:.2f} < 0.8\n→ Disparate Impact!',ha='center',fontsize=9,
             color=RUST,fontweight='bold',bbox=dict(boxstyle='round',fc=SURF2,ec=RUST))
    ax1.set_ylabel('Approval Rate'); ax1.set_xticks(x1); ax1.set_xticklabels(groups)
    ax1.set_title('Disparate Impact Analysis',fontweight='bold',color=TX)
    ax1.legend(fontsize=7.5)

    # Bias sources
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor(SURF); ax2.axis('off')
    ax2.set_xlim(0,10); ax2.set_ylim(0,7)
    bias_sources=[
        ('Historical Bias',1.5,6,'Biased patterns in training data reflecting past discrimination',RUST),
        ('Representation Bias',1.5,4.8,'Underrepresented groups in dataset',GOLD),
        ('Measurement Bias',1.5,3.6,'Proxy variables that encode protected attributes',PURPLE),
        ('Aggregation Bias',1.5,2.4,'One model for all groups ignoring subgroup differences',TEAL),
        ('Deployment Bias',1.5,1.2,'Model used differently than it was trained for',BLUE),
    ]
    for name,x,y,desc,c in bias_sources:
        ax2.text(x,y,f'● {name}',fontsize=9,color=c,fontweight='bold')
        ax2.text(x+0.3,y-0.5,desc,fontsize=7.5,color=TX2)
    ax2.set_title('Sources of AI Bias',fontweight='bold',color=TX)

    # Calibration curve
    ax3=fig.add_subplot(gs[1,0]); ax3.set_facecolor(SURF)
    prob_pred=np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
    perfect=prob_pred
    overconfident=np.array([0.02,0.08,0.15,0.25,0.38,0.52,0.65,0.79,0.90,0.97])
    underconfident=np.array([0.10,0.20,0.30,0.40,0.50,0.58,0.65,0.72,0.78,0.85])
    ax3.plot(prob_pred,perfect,'--',color=TX2,lw=1.5,label='Perfect calibration')
    ax3.plot(prob_pred,overconfident,'o-',color=RUST,lw=2,ms=6,label='Overconfident')
    ax3.plot(prob_pred,underconfident,'s-',color=BLUE,lw=2,ms=6,label='Underconfident')
    ax3.fill_between(prob_pred,perfect,overconfident,alpha=0.12,color=RUST)
    ax3.fill_between(prob_pred,perfect,underconfident,alpha=0.12,color=BLUE)
    ax3.set_xlabel('Mean Predicted Probability'); ax3.set_ylabel('Fraction of Positives')
    ax3.set_title('Calibration Curves (Reliability Diagram)',fontweight='bold',color=TX)
    ax3.legend(fontsize=7.5); ax3.grid(True,alpha=0.2)

    # Debiasing techniques
    ax4=fig.add_subplot(gs[1,1]); ax4.set_facecolor(SURF)
    techniques=['Reweighting\n(pre-proc)','Resampling\n(pre-proc)','Adversarial\nDebiasing','Calibrated\nEOpp (in-proc)','Reject Option\n(post-proc)','Equalized\nOdds (post)']
    bias_red=[40,35,60,55,45,65]; acc_impact=[-2,-3,-5,-4,-1,-6]
    colors_db=[BLUE,TEAL,RUST,GOLD,GREEN,PURPLE]
    scatter_s=[200]*6
    sc=ax4.scatter(acc_impact,bias_red,s=scatter_s,c=colors_db,zorder=5,edgecolors='white',lw=1)
    for i,(t,x,y) in enumerate(zip(techniques,acc_impact,bias_red)):
        ax4.annotate(t,(x,y),textcoords='offset points',xytext=(5,3),fontsize=7,color=colors_db[i])
    ax4.set_xlabel('Accuracy Impact (%)'); ax4.set_ylabel('Bias Reduction (%)')
    ax4.set_title('Debiasing Techniques Trade-offs',fontweight='bold',color=TX); ax4.grid(True,alpha=0.2)
    ax4.invert_xaxis()

    # Privacy techniques
    ax5=fig.add_subplot(gs[1,2]); ax5.set_facecolor(SURF)
    privacy_methods=['Differential\nPrivacy (ε=1)','Federated\nLearning','k-Anonymity\n(k=5)','Homomorphic\nEncryption','SMPC']
    utility=[75,85,80,60,70]; privacy_score=[95,80,75,99,92]
    colors_p=[RUST,TEAL,GOLD,PURPLE,BLUE]
    ax5.scatter(utility,privacy_score,s=200,c=colors_p,zorder=5,edgecolors='white',lw=1)
    for i,(m,u,p) in enumerate(zip(privacy_methods,utility,privacy_score)):
        ax5.annotate(m,(u,p),textcoords='offset points',xytext=(4,3),fontsize=7.5,color=colors_p[i])
    ax5.set_xlabel('Utility (%)'); ax5.set_ylabel('Privacy Score (%)')
    ax5.set_title('Privacy-Preserving ML Trade-offs',fontweight='bold',color=TX)
    ax5.grid(True,alpha=0.2)

    plt.suptitle('AI Ethics, Fairness & Privacy',fontsize=13,fontweight='bold',color=TX,y=1.01)
    plt.tight_layout()
    return fig_to_b64(fig)


# ── Main ──────────────────────────────────────────────────────────────────
import json, os

chart_fns = {
    'eval_classification': chart_eval_classification,
    'eval_regression':     chart_eval_regression,
    'reinforcement_learning': chart_rl,
    'cnn':                 chart_cnn,
    'rnn_lstm':            chart_rnn,
    'gan':                 chart_gan,
    'vae':                 chart_vae,
    'nlp_fundamentals':    chart_nlp,
    'transfer_learning':   chart_transfer,
    'bandits_active':      chart_bandit_active,
    'semi_self_supervised':chart_semi_self,
    'cluster_evaluation':  chart_cluster_eval,
    'pgm':                 chart_pgm,
    'object_detection':    chart_object_detection,
    'loss_functions':      chart_loss_functions,
    'data_preprocessing':  chart_data_preprocessing,
    'statistical_tests':   chart_statistical_tests,
    'gnn':                 chart_gnn,
    'model_compression':   chart_model_compression,
    'causal_inference':    chart_causal_inference,
    'multimodal':          chart_multimodal,
    'ethics_fairness':     chart_ethics_fairness,
}

# Load existing charts
charts_path = '/home/user/workspace/ml-visualizer/charts.json'
with open(charts_path) as f:
    all_charts = json.load(f)

print(f"Existing charts: {len(all_charts)}")
total = len(chart_fns)
for i,(key,fn) in enumerate(chart_fns.items(), 1):
    print(f"[{i:2d}/{total}] Generating {key}...", end=' ', flush=True)
    try:
        all_charts[key] = fn()
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()

with open(charts_path, 'w') as f:
    json.dump(all_charts, f)
print(f"Done. Total charts: {len(all_charts)}")
