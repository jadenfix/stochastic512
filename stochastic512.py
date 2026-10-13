"""Seeded stochastic-process simulation and fitted two-regime HMM analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM
from report import write_report


def simulate(rows=512, paths=64, seed=7):
    if not 128<=rows<=10000 or not 2<=paths<=512: raise ValueError('Require128..10000 rows and2..512 paths')
    rng=np.random.default_rng(seed);dt=1/252
    noise=rng.normal(size=(rows-1,paths));gbm=np.empty((rows,paths));ou=np.empty_like(gbm)
    gbm[0]=100;ou[0]=0
    for i in range(1,rows):
        gbm[i]=gbm[i-1]*np.exp((.05-.2**2/2)*dt+.2*np.sqrt(dt)*noise[i-1])
        ou[i]=ou[i-1]+3*(0-ou[i-1])*dt+.3*np.sqrt(dt)*noise[i-1]
    transitions=np.array([[.97,.03],[.06,.94]])
    states=np.zeros(rows,dtype=int);values=np.zeros(rows)
    for i in range(rows):
        if i: states[i]=rng.choice(2,p=transitions[states[i-1]])
        values[i]=rng.normal([-.025,.025][states[i]],[.007,.012][states[i]])
    return gbm,ou,values,states


def filter_hmm(model, observations, prior=None):
    observations=np.asarray(observations,float)
    if observations.ndim!=1 or not np.isfinite(observations).all(): raise ValueError('Need finite scalar observations')
    mu=model.means_.ravel();var=np.asarray(model.covars_).reshape(model.n_components,-1)[:,0]
    if np.any(var<=0):raise ValueError('Invalid fitted variances')
    probability=np.array(model.startprob_ if prior is None else prior,float)
    posterior=[];likelihood=0.0
    for i,value in enumerate(observations):
        if i or prior is not None: probability=probability@model.transmat_
        emission=-.5*(np.log(2*np.pi*var)+(value-mu)**2/var)
        log_weights=np.log(np.maximum(probability,1e-300))+emission
        normalizer=logsumexp(log_weights);probability=np.exp(log_weights-normalizer)
        posterior.append(probability.copy());likelihood+=normalizer
    return np.asarray(posterior),float(likelihood)


def fit(observations,seed=7):
    values=np.asarray(observations,float)
    if values.ndim!=1 or len(values)<128 or not np.isfinite(values).all() or np.std(values)<1e-9:
        raise ValueError('Need128+ finite nonconstant observations')
    cut=int(len(values)*.7)
    model=GaussianHMM(n_components=2,covariance_type='diag',n_iter=80,tol=1e-5,
                      random_state=seed,init_params='stc',params='stmc',min_covar=1e-6,covars_prior=1e-6)
    # Quantile initialization avoids symmetric collapse without using the holdout.
    model.means_=np.quantile(values[:cut],[.2,.8]).reshape(-1,1)
    model.fit(values[:cut,None])
    train,_=filter_hmm(model,values[:cut]);held,score=filter_hmm(model,values[cut:],train[-1])
    if not np.isfinite(held).all() or not np.allclose(held.sum(axis=1),1):raise RuntimeError('Invalid HMM posterior')
    return model,cut,np.vstack([train,held]),score


def run(output='artifacts',rows=512,paths=64,seed=7,csv=None):
    gbm,ou,values,states=simulate(rows,paths,seed)
    provenance={'kind':'seeded_synthetic','seed':seed,'not_market_data':True}
    if csv:
        source=Path(csv);frame=pd.read_csv(source)
        if 'observation' not in frame: raise ValueError('CSV requires observation column')
        values=frame.observation.to_numpy(float)
        provenance={'kind':'user_observations','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    model,cut,probabilities,score=fit(values,seed)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # JSON is explicit data, not an untrusted pickle.
    state={'startprob':model.startprob_.tolist(),'transition':model.transmat_.tolist(),
           'means':model.means_.ravel().tolist(),'variances':np.asarray(model.covars_).reshape(2,-1)[:,0].tolist()}
    (out/'model.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    restored=GaussianHMM(n_components=2,covariance_type='diag')
    restored.n_features=1
    restored.startprob_=np.array(state['startprob']);restored.transmat_=np.array(state['transition'])
    restored.means_=np.array(state['means'])[:,None];restored.covars_=np.array(state['variances'])[:,None]
    roundtrip,_=filter_hmm(restored,values)
    np.testing.assert_allclose(probabilities,roundtrip,atol=1e-12)
    series=pd.DataFrame({'observation':values,'p_state0':probabilities[:,0],'p_state1':probabilities[:,1],
                         'split':np.where(np.arange(len(values))<cut,'train','holdout')})
    if not csv:series['simulated_true_state']=states
    series.to_csv(out/'observations.csv',index=False)
    pd.DataFrame({'step':np.arange(rows),'gbm_q10':np.quantile(gbm,.1,axis=1),'gbm_q50':np.median(gbm,axis=1),
                  'gbm_q90':np.quantile(gbm,.9,axis=1),'ou_q10':np.quantile(ou,.1,axis=1),
                  'ou_q50':np.median(ou,axis=1),'ou_q90':np.quantile(ou,.9,axis=1)}).to_csv(out/'process_quantiles.csv',index=False)
    metrics={'scope':'Stochastic simulation and train-only HMM estimation; no forecast or financial claim',
             'source':provenance,'processes':'GBM exact discrete transition; OU Euler-Maruyama,64 default paths',
             'simulation_rows':rows,'simulation_paths':paths,'hmm_train_rows':cut,'hmm_holdout_rows':len(values)-cut,
             'heldout_filtered_log_likelihood':score,'fitted_means':state['means'],'transition':state['transition'],
             'iterations':model.monitor_.iter,'em_last_log_likelihood_delta':float(model.monitor_.history[-1]-model.monitor_.history[-2]) if len(model.monitor_.history)>1 else None,'posterior_row_sum_max_error':float(np.abs(probabilities.sum(axis=1)-1).max()),
             'model_roundtrip_verified':True,'filtering':'Forward-only filtering; holdout carries final training posterior. No future smoothing.',
             'process_source':'Seeded synthetic ensembles even when user HMM observations supplied'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    write_report(out/'index.html',metrics,gbm,ou,series)
    print(json.dumps(metrics,indent=2));return metrics,series


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='artifacts')
    p.add_argument('--rows',type=int,default=512);p.add_argument('--paths',type=int,default=64)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--csv');a=p.parse_args()
    run(a.output,a.rows,a.paths,a.seed,a.csv)
if __name__=='__main__':main()
