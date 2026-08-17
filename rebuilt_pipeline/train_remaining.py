"""Grouped split/model execution for the corrected rebuilt EEG dataset."""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'rebuilt_pipeline/artifacts_v2'; OUT=ROOT/'rebuilt_pipeline/results_v2'; OUT.mkdir(exist_ok=True)
SEED=20260817; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class GRL(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,l): ctx.l=l; return x.view_as(x)
 @staticmethod
 def backward(ctx,g): return -ctx.l*g,None
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.t=nn.Conv2d(1,8,(1,64),padding=(0,31)); self.d=nn.Conv2d(8,16,(6,1),groups=8); self.s=nn.Conv2d(16,16,(1,16),padding=(0,7),groups=16); self.p=nn.Linear(16*510,64); self.dep=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)); self.st=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)); self.dom=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))
 def z(self,x): return torch.relu(self.p(torch.relu(self.s(torch.relu(self.d(torch.relu(self.t(x)))))).flatten(1)))
 def forward(self,x,l=0):
  z=self.z(x); return self.dep(z).squeeze(1),self.st(z).squeeze(1),self.dom(GRL.apply(z,l))
def grouped(meta):
 train=np.zeros(len(meta),bool); test=np.zeros(len(meta),bool)
 for c in meta.corpus.unique():
  ix=np.where(meta.corpus.eq(c))[0]; g=meta.subject_id.iloc[ix].to_numpy(); a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED).split(ix,groups=g)); train[ix[a]]=1;test[ix[b]]=1
 return train,test
def agg_auc(df,key,label,score):
 a=df.assign(_s=score).groupby(key).agg(y=(label,'first'),s=('_s','median'))
 return float(roc_auc_score(a.y,a.s)),a
def boot(a):
 rng=np.random.default_rng(SEED); vals=[]
 for _ in range(1000):
  q=a.iloc[rng.integers(len(a),size=len(a))]
  if q.y.nunique()==2: vals.append(roc_auc_score(q.y,q.s))
 return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def main():
 X=np.load(IN/'X_rebuilt.npy',mmap_mode='r'); m=pd.read_csv(IN/'metadata.csv'); tr,te=grouped(m); np.savez(OUT/'grouped_split_indices.npz',train=tr,test=te)
 assert not set(m.subject_id[tr])&set(m.subject_id[te])
 # architecture shape validation
 n=Net(); dummy=n(torch.zeros(2,1,6,512)); assert all(x.shape[0]==2 for x in dummy)
 # compact, fixed baseline/GRL run; losses are masked according to plan
 def fit(grl):
  model=Net(); opt=torch.optim.Adam(model.parameters(),lr=1e-3); bce=nn.BCEWithLogitsLoss(); ce=nn.CrossEntropyLoss(reduction='none')
  ids=np.where(tr)[0]; loader=DataLoader(TensorDataset(torch.tensor(np.asarray(X[ids])),torch.tensor(m.depression_label.iloc[ids].values),torch.tensor(m.stress_label.iloc[ids].values),torch.tensor(m.domain.iloc[ids].values),torch.tensor(m.anchor_mask.iloc[ids].values)),batch_size=64,shuffle=True)
  hist=[]
  for epoch in range(10):
   losses=[]
   for x,yd,ys,dm,an in loader:
    x=x.unsqueeze(1).float(); pd_,ps,dd=model(x,1.0 if grl else 0.0); loss=torch.tensor(0.)
    q=yd>=0
    if q.any(): loss=loss+bce(pd_[q],yd[q].float())
    q=ys>=0
    if q.any(): loss=loss+bce(ps[q],ys[q].float())
    q=an.bool()
    if grl and q.any():
     counts=torch.bincount(dm[q],minlength=3).float(); w=counts.sum()/(3*counts); loss=loss+(ce(dd[q],dm[q])*w[dm[q]]).mean()
    opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
   hist.append(float(np.mean(losses)))
  return model,hist
 results={'seed':SEED,'device':'cpu','split_subject_overlap':0,'architecture':{'input':[6,512],'latent_dimension':64},'models':{}}
 for name,flag in [('baseline',False),('anchor_grl',True)]:
  model,h=fit(flag); model.eval(); ids=np.where(te)[0]
  with torch.no_grad(): d,s,_=model(torch.tensor(np.asarray(X[ids])).unsqueeze(1).float(),1.0 if flag else 0.0)
  d=d.numpy();s=s.numpy(); sub=m.iloc[ids].copy(); rows={};
  for corpus,key,label,score in [('MODMA','subject_id','depression_label',d),('SAM40','trial_id','stress_label',s),('Cognitive Load','trial_id','stress_label',s)]:
   q=sub.corpus.eq(corpus); a,ag=agg_auc(sub[q],key,label,score[q]); rows[corpus]={'auc':a,'ci95':boot(ag),'n':len(ag)}; ag.assign(corpus=corpus,model=name).to_csv(OUT/f'{name}_{corpus.lower().replace(" ","_")}_predictions.csv',index=False)
  results['models'][name]={'training_loss':h,'grouped':rows}; torch.save(model.state_dict(),OUT/f'{name}.pt')
 (OUT/'results.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
