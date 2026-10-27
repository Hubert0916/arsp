"""Focused evaluator tests: weighting, matched evidence, and split isolation."""
import copy
import unittest

from score_proxy import audit, scores

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [.1, .2, .2, .2, .2, .1]


class MetricsTests(unittest.TestCase):
    def test_equal_group_weight_not_equal_row_weight(self):
        rows = [{"source": s, "subject_id": "subject", "item_id": str(i), "y": y,
                 "asrp": [0.0]*6, "empirical_mean": [.5]*6}
                for s, n, y in [("a", 1, 1), ("b", 3, 0)] for i in range(n)]
        result, _ = scores(rows, BUDGETS, WEIGHTS)
        self.assertAlmostEqual(result["asrp_alc"], .5)
        self.assertAlmostEqual(result["empirical_mean_alc"], .25)

    def test_log_budget_trapezoid(self):
        import math
        x = [math.log2(1+n) for n in BUDGETS]
        b = [.25, .3, .22, .20, .18, .17]
        integral = sum((x[i+1]-x[i])*(b[i]+b[i+1])/2 for i in range(5))/(x[-1]-x[0])
        self.assertAlmostEqual(integral, sum(w*p for w, p in zip(WEIGHTS, b)))

    def test_duplicate_item_rejected(self):
        row = {"source": "a", "subject_id": "s", "item_id": "i", "y": 1,
               "asrp": [.5]*6, "empirical_mean": [.5]*6}
        with self.assertRaises(ValueError):
            scores([row, row], BUDGETS, WEIGHTS)

    def test_acquisition_cannot_include_evaluation_item(self):
        group = {"source": "a", "subject_id": "s",
                 "acquisition": [{"item_id": str(i)} for i in range(40)],
                 "evaluation": [{"item_id": "target", "response_id": "response"}]}
        manifest = {"groups": [group], "config": {"budgets": BUDGETS}}
        record = {"source": "a", "subject_id": "s", "item_id": "target", "response_id": "response",
                  "y": 1, "asrp": [.5]*6, "empirical_mean": [.5, 1., 1., 1., 1., 1.]}
        trajectory = {"source": "a", "subject_id": "s", "acquired_item_ids": [str(i) for i in range(31)],
                      "labels": [1]*31,
                      "decisions": [{"position": i, "item_id": str(i), "acquired": True} for i in range(31)]}
        audit([record], [trajectory], manifest)
        bad = copy.deepcopy(trajectory)
        bad["acquired_item_ids"][0] = "target"
        with self.assertRaises(ValueError):
            audit([record], [bad], manifest)
        bad_record = copy.deepcopy(record)
        bad_record["empirical_mean"][1] = 0
        with self.assertRaises(ValueError):
            audit([bad_record], [trajectory], manifest)


if __name__ == "__main__":
    unittest.main()
