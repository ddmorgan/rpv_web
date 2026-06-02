import importlib.util
import unittest

import pandas as pd

from predictors import MODEL_FILE_ROOT, benchmark_stats, predict


class PredictorTests(unittest.TestCase):
    def test_sample_predictions(self):
        df = pd.read_csv("examples/sample_input.csv")
        result = predict(df.iloc[[0]], ["E900", "EONY"])

        self.assertEqual(result["result_rows"], 2)
        by_model = {row["model"]: row for row in result["results"]}
        self.assertAlmostEqual(by_model["E900"]["predicted_tts_degC"], 85.039, places=3)
        self.assertAlmostEqual(by_model["EONY"]["predicted_tts_degC"], 75.529, places=3)

    def test_benchmark_stats_loaded(self):
        e900 = benchmark_stats("E900")
        eony = benchmark_stats("EONY")
        gbr = benchmark_stats("GBR")
        gkrr = benchmark_stats("GKRR")

        self.assertEqual(e900["n"], 4535)
        self.assertEqual(eony["n"], 4535)
        self.assertEqual(gbr["n"], 4535)
        self.assertEqual(gkrr["n"], 4535)
        self.assertAlmostEqual(e900["residual_std_degC"], 17.451, places=3)
        self.assertAlmostEqual(eony["residual_std_degC"], 22.796, places=3)
        self.assertAlmostEqual(gbr["residual_std_degC"], 7.510, places=3)
        self.assertAlmostEqual(gkrr["residual_std_degC"], 5.167, places=3)

    @unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed locally")
    def test_ml_predictions_run_when_dependencies_are_installed(self):
        df = pd.read_csv("examples/sample_input.csv")
        result = predict(df.iloc[[0]], ["GBR", "GKRR"])

        self.assertEqual(result["result_rows"], 2)
        by_model = {row["model"]: row for row in result["results"]}
        self.assertIsInstance(by_model["GBR"]["predicted_tts_degC"], float)
        self.assertIsInstance(by_model["GKRR"]["predicted_tts_degC"], float)

    def test_ml_artifacts_present(self):
        self.assertTrue((MODEL_FILE_ROOT / "GBR" / "fullfit" / "GradientBoostingRegressor.pkl").exists())
        self.assertTrue((MODEL_FILE_ROOT / "GBR" / "fullfit" / "StandardScaler.pkl").exists())
        self.assertTrue((MODEL_FILE_ROOT / "GKRR" / "fullfit" / "KernelRidge.pkl").exists())
        self.assertTrue((MODEL_FILE_ROOT / "GKRR" / "fullfit" / "StandardScaler.pkl").exists())


if __name__ == "__main__":
    unittest.main()
