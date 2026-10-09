"""Resume only unfinished fixed plan-2 training after an execution timeout."""
import json
import numpy as np
import pandas as pd
import torch
import train_remaining as t

def checkpoint_frame(X, m, ids, path, lambd):
    model=t.Net().to(t.DEVICE)
    model.load_state_dict(torch.load(path, map_location=t.DEVICE))
    return t.predict(X,m,ids,model,lambd)

def oof_features(X,m,tests,name,lambd):
    frames=[]; features=[]; domains=[]
    for fold,q in enumerate(tests):
        frame,z=checkpoint_frame(X,m,np.flatnonzero(q),t.OUT/f'{name}_fold_{fold}.pt',lambd)
        frames.append(frame); anchor=frame.anchor_mask.to_numpy(bool)
        features.append(z[anchor]); domains.extend(frame.domain.to_numpy()[anchor])
    return pd.concat(frames,ignore_index=True),np.vstack(features),domains

def main():
    t.seed(); X=np.load(t.IN/'X_rebuilt.npy',mmap_mode='r'); m=pd.read_csv(t.IN/'metadata.csv')
    tests,checks=t.folds(m)
    # Baseline was completed before the host timeout. Reconstruct its diagnostic
    # from its actual saved fold checkpoints; do not retrain it.
    base,base_z,base_domains=oof_features(X,m,tests,'baseline',0.)
    assert len(base)==len(m)
    baseline={'lambda':0,'fold_training_status':'completed before timeout; per-epoch history was not flushed before host termination',
      'anchor_domain_silhouette':float(t.silhouette_score(base_z,base_domains)),'grouped':t.grouped(base,'baseline')}
    # Reuse the three actual finished GRL checkpoints and train exactly the two
    # unfinished subject-grouped folds with the fixed configuration.
    histories=[None]*t.FOLDS; counts=[None]*t.FOLDS
    for fold in (3,4):
        if (t.OUT/f'anchor_grl_fold_{fold}.pt').exists():
            continue
        q=tests[fold]; pool=np.flatnonzero(~q)
        tr,va=next(t.GroupShuffleSplit(1,test_size=.2,random_state=t.SEED+fold).split(pool,groups=m.subject_id.iloc[pool]))
        train,val=pool[tr],pool[va]
        assert not(set(m.subject_id.iloc[train])&set(m.subject_id.iloc[val]))
        model,h,c=t.fit(X,m,train,val,True)
        torch.save(model.state_dict(),t.OUT/f'anchor_grl_fold_{fold}.pt')
        histories[fold]=h; counts[fold]=c
    grl,grl_z,grl_domains=oof_features(X,m,tests,'anchor_grl',t.LAMBDA)
    assert len(grl)==len(m)
    grl.to_csv(t.OUT/'anchor_grl_segment_predictions.csv',index=False)
    # Recalculate the mandated training-fold-only counts for audit logging.
    for fold,q in enumerate(tests):
        pool=np.flatnonzero(~q)
        tr,_=next(t.GroupShuffleSplit(1,test_size=.2,random_state=t.SEED+fold).split(pool,groups=m.subject_id.iloc[pool]))
        _,counts[fold]=t.weights(m,pool[tr])
    grl_result={'lambda':t.LAMBDA,'fold_training':histories,
      'warning':'Folds 0-2 completed before host timeout; their per-epoch loss histories were not flushed, but their saved checkpoints and predictions are used.',
      'anchor_domain_counts_training_folds':counts,
      'anchor_domain_silhouette':float(t.silhouette_score(grl_z,grl_domains)),'grouped':t.grouped(grl,'anchor_grl')}
    result={'seed':t.SEED,'device':str(t.DEVICE),'python':t.sys.version,'torch':torch.__version__,
      'architecture':{'input':[6,512],'temporal':[8,[1,64]],'depthwise':[2,[6,1]],'separable':[16,[1,16]],'latent_dimension':64},
      'grl_sign_check':'passed (dL/dx=-1)','split_leakage_checks':checks,'models':{'baseline':baseline,'anchor_grl':grl_result},'zero_shot':{}}
    first=t.OUT/'zero_shot_sam40_to_cognitive_load_predictions.csv'
    if first.exists():
        a=pd.read_csv(first)
        result['zero_shot']['SAM40_to_Cognitive_Load']={'source':'SAM40','target':'Cognitive Load','evaluation_unit':'trial_id',
          'n':len(a),'auc':float(t.roc_auc_score(a.y,a.score)),'ci95':t.boot(a),
          'training_loss_status':'completed before host interruption; per-epoch history was not flushed','target_rows_used_in_training':0}
    else:
        result['zero_shot']['SAM40_to_Cognitive_Load']=t.zero_shot(X,m,'SAM40','Cognitive Load')
    second=t.OUT/'zero_shot_cognitive_load_to_sam40_predictions.csv'
    if second.exists():
        a=pd.read_csv(second)
        old=json.loads((t.OUT/'results.json').read_text()) if (t.OUT/'results.json').exists() else {}
        prior=old.get('zero_shot',{}).get('Cognitive_Load_to_SAM40',{})
        result['zero_shot']['Cognitive_Load_to_SAM40']={'source':'Cognitive Load','target':'SAM40','evaluation_unit':'trial_id',
          'n':len(a),'auc':float(t.roc_auc_score(a.y,a.score)),'ci95':t.boot(a),
          'training_loss':prior.get('training_loss','completed before host interruption; per-epoch history was not flushed'),
          'target_rows_used_in_training':0}
    else:
        result['zero_shot']['Cognitive_Load_to_SAM40']=t.zero_shot(X,m,'Cognitive Load','SAM40')
    result['zero_shot_grl_status']='NOT RUN: strict zero-shot prohibits target-domain samples and leaves mandated inverse three-domain anchor weights undefined with one source domain.'
    (t.OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
