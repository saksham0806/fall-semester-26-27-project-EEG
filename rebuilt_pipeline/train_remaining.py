"""Execute Phases 5--15 of plan-2.md on artifacts_v2 only (fixed configuration)."""
import json, platform, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'rebuilt_pipeline/artifacts_v2'; OUT=ROOT/'rebuilt_pipeline/results_v2'; OUT.mkdir(exist_ok=True)
SEED,FOLDS,EPOCHS,BATCH_SIZE,LR,LAMBDA=20260817,5,10,64,1e-3,1.0
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed(s=SEED):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
class GRL(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,l):ctx.l=l;return x.view_as(x)
 @staticmethod
 def backward(ctx,g):return -ctx.l*g,None
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.temporal=nn.Conv2d(1,8,(1,64),padding=(0,31));self.depthwise=nn.Conv2d(8,16,(6,1),groups=8);self.sep_dw=nn.Conv2d(16,16,(1,16),padding=(0,7),groups=16);self.sep_pw=nn.Conv2d(16,16,1);self.proj=nn.Linear(16*510,64);self.dep=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1));self.stress=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1));self.domain=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))
 def features(self,x):
  x=torch.relu(self.temporal(x));x=torch.relu(self.depthwise(x));x=torch.relu(self.sep_pw(self.sep_dw(x)));return torch.relu(self.proj(x.flatten(1)))
 def forward(self,x,l=0.):
  z=self.features(x);return self.dep(z).squeeze(1),self.stress(z).squeeze(1),self.domain(GRL.apply(z,l)),z
def loader(X,m,ids,shuffle):
 return DataLoader(TensorDataset(torch.tensor(np.asarray(X[ids]),dtype=torch.float32),torch.tensor(m.depression_label.iloc[ids].to_numpy(),dtype=torch.long),torch.tensor(m.stress_label.iloc[ids].to_numpy(),dtype=torch.long),torch.tensor(m.domain.iloc[ids].to_numpy(),dtype=torch.long),torch.tensor(m.anchor_mask.iloc[ids].to_numpy(),dtype=torch.bool)),batch_size=BATCH_SIZE,shuffle=shuffle)
def folds(m):
 tests=[np.zeros(len(m),bool) for _ in range(FOLDS)]
 for corpus in m.corpus.unique():
  rows=np.flatnonzero(m.corpus.eq(corpus));groups=m.subject_id.iloc[rows].to_numpy()
  for k,(_,q) in enumerate(GroupKFold(FOLDS).split(rows,groups=groups)):tests[k][rows[q]]=1
 checks=[]
 for q in tests:
  assert not(set(m.subject_id[~q])&set(m.subject_id[q]));checks.append({'train_subjects':int(m.loc[~q,'subject_id'].nunique()),'test_subjects':int(m.loc[q,'subject_id'].nunique()),'overlap':0})
 assert np.all(np.sum(tests,axis=0)==1);return tests,checks
def weights(m,ids):
 counts=m.iloc[ids].query('anchor_mask').domain.value_counts().reindex([0,1,2],fill_value=0).to_numpy(float)
 if np.any(counts==0):raise ValueError(f'anchor-GRL requires all training domains: {counts.tolist()}')
 return torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device=DEVICE),counts.tolist()
def fit(X,m,train,val,grl):
 seed();model=Net().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=LR);bce=nn.BCEWithLogitsLoss();ce=nn.CrossEntropyLoss(reduction='none');w,c=weights(m,train) if grl else (None,None);hist=[]
 for epoch in range(EPOCHS):
  model.train();ls=[]
  for x,yd,ys,dm,an in loader(X,m,train,True):
   x,yd,ys,dm,an=(v.to(DEVICE) for v in (x,yd,ys,dm,an));pd_,ps,dd,_=model(x.unsqueeze(1),LAMBDA if grl else 0);loss=torch.zeros((),device=DEVICE);q=yd>=0
   if q.any():loss=loss+bce(pd_[q],yd[q].float())
   q=ys>=0
   if q.any():loss=loss+bce(ps[q],ys[q].float())
   q=an.bool()
   if grl and q.any():loss=loss+LAMBDA*(ce(dd[q],dm[q])*w[dm[q]]).mean()
   opt.zero_grad();loss.backward();opt.step();ls.append(float(loss.detach().cpu()))
  model.eval();vls=[]
  with torch.no_grad():
   for x,yd,ys,dm,an in loader(X,m,val,False):
    x,yd,ys,dm,an=(v.to(DEVICE) for v in (x,yd,ys,dm,an));pd_,ps,dd,_=model(x.unsqueeze(1),LAMBDA if grl else 0);loss=torch.zeros((),device=DEVICE);q=yd>=0
    if q.any():loss+=bce(pd_[q],yd[q].float())
    q=ys>=0
    if q.any():loss+=bce(ps[q],ys[q].float())
    q=an.bool()
    if grl and q.any():loss+=LAMBDA*(ce(dd[q],dm[q])*w[dm[q]]).mean()
    vls.append(float(loss.cpu()))
  hist.append({'epoch':epoch+1,'train_loss':float(np.mean(ls)),'validation_loss':float(np.mean(vls))})
 return model,hist,c
def predict(X,m,ids,model,l=0):
 model.eval();d=[];s=[];z=[]
 with torch.no_grad():
  for x,*_ in loader(X,m,ids,False):
   a,b,_,c=model(x.to(DEVICE).unsqueeze(1),l);d.extend(a.cpu().numpy());s.extend(b.cpu().numpy());z.extend(c.cpu().numpy())
 f=m.iloc[ids].copy();f['dep_logit']=d;f['stress_logit']=s;return f,np.asarray(z)
def aggregate(f,unit,label,score):
 a=f.groupby(unit,sort=False).agg(y=(label,'first'),score=(score,'median')).reset_index()
 if a.y.nunique()!=2:raise ValueError(f'one class after {unit} aggregation')
 return float(roc_auc_score(a.y,a.score)),a
def boot(a):
 rng=np.random.default_rng(SEED);v=[]
 for _ in range(1000):
  q=a.iloc[rng.integers(len(a),size=len(a))]
  if q.y.nunique()==2:v.append(roc_auc_score(q.y,q.score))
 return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]
def grouped(f,name):
 out={}
 for corpus,unit,label,score in [('MODMA','subject_id','depression_label','dep_logit'),('SAM40','trial_id','stress_label','stress_logit'),('Cognitive Load','trial_id','stress_label','stress_logit')]:
  auc,a=aggregate(f[f.corpus.eq(corpus)],unit,label,score);a.assign(corpus=corpus,model=name).to_csv(OUT/f'{name}_{corpus.lower().replace(" ","_")}_predictions.csv',index=False);out[corpus]={'evaluation_unit':unit,'n':len(a),'auc':auc,'ci95':boot(a)}
 return out
def zero_shot(X,m,source,target):
 ids=np.flatnonzero(m.corpus.eq(source));tr,va=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED).split(ids,groups=m.subject_id.iloc[ids]));model,h,_=fit(X,m,ids[tr],ids[va],False);ti=np.flatnonzero(m.corpus.eq(target));f,_=predict(X,m,ti,model);auc,a=aggregate(f,'trial_id','stress_label','stress_logit');tag=f'{source.lower().replace(" ","_")}_to_{target.lower().replace(" ","_")}';a.assign(source=source,target=target,model='baseline').to_csv(OUT/f'zero_shot_{tag}_predictions.csv',index=False);torch.save(model.state_dict(),OUT/f'zero_shot_{tag}.pt');return {'source':source,'target':target,'evaluation_unit':'trial_id','n':len(a),'auc':auc,'ci95':boot(a),'training_loss':h,'target_rows_used_in_training':0}
def main():
 seed();x=torch.tensor([1.],requires_grad=True);GRL.apply(x,1.).sum().backward();assert float(x.grad)==-1.;X=np.load(IN/'X_rebuilt.npy',mmap_mode='r');m=pd.read_csv(IN/'metadata.csv');tests,checks=folds(m);np.savez(OUT/'grouped_split_indices.npz',**{f'test_fold_{i}':q for i,q in enumerate(tests)});n=Net().to(DEVICE);o=n(torch.zeros(2,1,6,512,device=DEVICE));assert o[0].shape==(2,) and o[3].shape==(2,64)
 r={'seed':SEED,'device':str(DEVICE),'python':sys.version,'torch':torch.__version__,'architecture':{'input':[6,512],'temporal':[8,[1,64]],'depthwise':[2,[6,1]],'separable':[16,[1,16]],'latent_dimension':64},'grl_sign_check':'passed (dL/dx=-1)','split_leakage_checks':checks,'models':{},'zero_shot':{}}
 for name,grl in [('baseline',False),('anchor_grl',True)]:
  frames=[];features=[];domains=[];hist=[];counts=[]
  for fold,q in enumerate(tests):
   pool=np.flatnonzero(~q);tr,va=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED+fold).split(pool,groups=m.subject_id.iloc[pool]));train,val=pool[tr],pool[va];assert not(set(m.subject_id.iloc[train])&set(m.subject_id.iloc[val]));model,h,c=fit(X,m,train,val,grl);f,z=predict(X,m,np.flatnonzero(q),model,LAMBDA if grl else 0);torch.save(model.state_dict(),OUT/f'{name}_fold_{fold}.pt');frames.append(f);hist.append(h);counts.append(c);am=f.anchor_mask.to_numpy(bool);features.append(z[am]);domains.extend(f.domain.to_numpy()[am])
  f=pd.concat(frames,ignore_index=True);assert len(f)==len(m);f.to_csv(OUT/f'{name}_segment_predictions.csv',index=False);r['models'][name]={'lambda':LAMBDA if grl else 0,'fold_training':hist,'anchor_domain_counts_training_folds':counts if grl else None,'anchor_domain_silhouette':float(silhouette_score(np.vstack(features),domains)),'grouped':grouped(f,name)}
 r['zero_shot']['SAM40_to_Cognitive_Load']=zero_shot(X,m,'SAM40','Cognitive Load');r['zero_shot']['Cognitive_Load_to_SAM40']=zero_shot(X,m,'Cognitive Load','SAM40');r['zero_shot_grl_status']='NOT RUN: zero-shot forbids target samples; mandated inverse three-domain anchor weights are undefined with one source domain.';(OUT/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
