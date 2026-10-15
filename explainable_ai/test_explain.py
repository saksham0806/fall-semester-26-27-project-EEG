"""Small numerical checks; no project data or fitted models are loaded."""
import unittest

import numpy as np

from explain import (feature_groups, feature_names, local_effects,
                     permutation_effects, sample_rows, shuffle_group)


class SignalModel:
    def predict_proba(self, X):
        p = 0.1 + 0.8 * X[:, 0]
        return np.column_stack((1 - p, p))


class ExplanationTests(unittest.TestCase):
    def test_permutation_detects_signal_and_ignores_unused_features(self):
        X = np.column_stack((np.tile([0., 1.], 100), np.arange(200)))
        original = X.copy()
        result = permutation_effects(SignalModel(), X, X[:, 0],
                                     np.zeros(200), {"signal": [0], "unused": [1]},
                                     repeats=8, seed=42)
        self.assertGreater(result["groups"]["signal"]["auc_drop_mean"], 0.3)
        self.assertEqual(result["groups"]["unused"]["auc_drop_mean"], 0.)
        np.testing.assert_array_equal(X, original)

    def test_shuffling_keeps_corpus_and_group_relationships(self):
        X = np.column_stack((np.arange(40), np.arange(40) * 2, np.arange(40)))
        corpus = np.repeat([0, 1], 20)
        changed = shuffle_group(X, [0, 1], corpus, np.random.default_rng(4))
        np.testing.assert_array_equal(changed[:, 1], changed[:, 0] * 2)
        np.testing.assert_array_equal(changed[:, 2], X[:, 2])
        for c in [0, 1]:
            np.testing.assert_array_equal(np.sort(changed[corpus == c, 0]), X[corpus == c, 0])

    def test_local_effect_direction_and_nonmutation(self):
        x = np.array([1., 5.])
        p, effects = local_effects(SignalModel(), x, np.array([0., 0.]))
        self.assertAlmostEqual(p, 0.9)
        np.testing.assert_allclose(effects, [0.8, 0.])
        np.testing.assert_array_equal(x, [1., 5.])

    def test_sample_caps_each_subject_and_respects_mask(self):
        subjects = np.repeat(["a", "b", "a", "b"], 10)
        corpus = np.repeat([0, 1], 20)
        mask = np.arange(40) >= 10
        idx = sample_rows(subjects, corpus, mask, per_subject=3, seed=42)
        self.assertTrue(mask[idx].all())
        self.assertEqual(len(idx), 9)
        np.testing.assert_array_equal(idx, sample_rows(subjects, corpus, mask, 3, 42))

    def test_names_and_groups_follow_v1_layout(self):
        names = feature_names()
        groups = feature_groups(names)
        self.assertEqual(len(names), 81)
        self.assertEqual(names[6], "relpsd_theta_Fp1")
        self.assertEqual(names[60], "faa_Fp2_Fp1")
        self.assertEqual(names[69], "std_Fp1")
        self.assertEqual(names[75], "rms_Fp1")
        self.assertEqual(sorted(sum(groups["family"].values(), [])), list(range(81)))
        self.assertEqual(len(groups["band"]["alpha"]), 12)
        self.assertIn(60, groups["channel"]["Fp1"])
        self.assertIn(60, groups["channel"]["Fp2"])


if __name__ == "__main__":
    unittest.main()
