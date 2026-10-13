# stochastic512

A reproducible stochastic-process workbench: seeded GBM and Ornstein–Uhlenbeck ensembles, switching Gaussian observations, and a trained two-state hidden Markov model. Explore the simulated envelopes and forward-filtered regimes in a standalone interactive report.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python stochastic512.py --rows 512 --paths 64 --output artifacts
python -m http.server 8000 --directory artifacts
```

Open http://localhost:8000. All default observations are **seeded synthetic**, not financial history. GBM uses its exact discrete transition; OU uses Euler–Maruyama at dt1/252. The process parameters are fixed educational defaults, not market calibrations. Path quantiles describe the simulated ensemble, not forecast confidence intervals.

The HMM fits only the first70% of observations. Its two Gaussian emissions and transition matrix are actually estimated by EM. The held-out lane uses forward filtering from the last training posterior; it does not use future smoothing or refit on held-out values. Log likelihood measures the fitted observation model, not trading performance. Component IDs are arbitrary and are not investment recommendations.

Supply `--csv observations.csv` with a finite, nonconstant `observation` column and128+ rows for your own ordered scalar sequence. Your file hash is recorded; GBM/OU ensembles remain explicitly synthetic and separate from that input. Outputs include the observation/posterior CSV, process quantile CSV, fitted parameters JSON, metrics and browser report. Reloaded model filtering must reproduce the original result. Tests check prefix causality, holdout isolation, reproducibility and malformed input rejection.
