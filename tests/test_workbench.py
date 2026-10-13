import tempfile
import unittest
import numpy as np
from stochastic512 import simulate,fit,filter_hmm,run

class WorkbenchTests(unittest.TestCase):
    def test_seed_reproduces_processes_and_gbm_positive(self):
        a=simulate(128,8,7);b=simulate(128,8,7)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertTrue((a[0]>0).all())
    def test_holdout_never_changes_fitted_parameters(self):
        values=simulate(192,8,7)[2];first=fit(values,7)
        changed=values.copy();changed[first[1]:]+=10
        second=fit(changed,7)
        np.testing.assert_allclose(first[0].means_,second[0].means_)
        np.testing.assert_allclose(first[0].transmat_,second[0].transmat_)
    def test_filter_is_prefix_causal(self):
        values=simulate(192,8,7)[2];model,_,_,_=fit(values)
        p,_=filter_hmm(model,values)
        values[100:]+=1;q,_=filter_hmm(model,values)
        np.testing.assert_allclose(p[:100],q[:100])
    def test_invalid_constant_or_nan_series_rejected(self):
        for data in [np.ones(128),np.r_[np.zeros(127),np.nan]]:
            with self.assertRaises(ValueError):fit(data)
    def test_actual_fit_reload_artifact_workflow(self):
        with tempfile.TemporaryDirectory() as output:
            metrics,series=run(output,192,8)
            self.assertTrue(metrics['model_roundtrip_verified'])
            self.assertLess(metrics['posterior_row_sum_max_error'],1e-10)
            self.assertEqual(len(series),192)
