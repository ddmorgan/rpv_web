import unittest

import pandas as pd

from predictors import benchmark_stats, predict


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

        self.assertEqual(e900["n"], 4535)
        self.assertEqual(eony["n"], 4535)
        self.assertAlmostEqual(e900["residual_std_degC"], 17.451, places=3)
        self.assertAlmostEqual(eony["residual_std_degC"], 22.796, places=3)


if __name__ == "__main__":
    unittest.main()
