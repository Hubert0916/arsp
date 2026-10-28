"""Check official-feedback aggregation, deduplication, and incomplete copies."""
import unittest

from parse_official_results import BUDGETS, parse, summarize


def feedback(groups, missing=None):
    # Constant curves have the same ALC, independent of the budget weights.
    rows = "\n".join(
        f"subject_{i} benchmark_{i} {count} {score:.6f} 0.100000"
        for i, count, score in groups
    )
    sections = ["Formative feedback", "ALC summary", rows]
    for budget in BUDGETS:
        sections.append(f"Label budget: {budget}")
        if budget != missing:
            sections.append(rows)
    return "\n".join(sections)


class OfficialResultsTests(unittest.TestCase):
    def test_equal_group_then_equal_record_weights(self):
        text = feedback([(1, 10, .1), (2, 90, .3)])
        text += "\n" + feedback([(3, 20, .4)])
        runs, issues = parse(text)
        self.assertEqual(issues, [])
        self.assertAlmostEqual(runs[0]['brier_alc'], .2)
        stats = summarize(runs)['complete_six_budget']['brier_alc']
        self.assertAlmostEqual(stats['mean'], .3)
        self.assertAlmostEqual(stats['sample_sd'], .02 ** .5)

    def test_duplicate_and_partial_do_not_change_complete_cohort(self):
        first = feedback([(1, 10, .2)])
        text = first + "\n" + first + "\n" + feedback([(2, 20, .4)], missing=31)
        runs, issues = parse(text)
        self.assertEqual(runs[1]['duplicate_of'], 1)
        self.assertIsNone(runs[2]['brier'][-1])
        summary = summarize(runs)
        self.assertEqual(summary['complete_blocks'], [1])
        self.assertAlmostEqual(summary['all_available_alc_summaries']['mean'], .3)
        self.assertEqual({i['type'] for i in issues}, {'duplicate_feedback', 'incomplete_budget'})

    def test_foreign_rows_are_not_silently_merged(self):
        text = feedback([(1, 10, .2)]) + "\nsubject_9 benchmark_9 10 0.900000 0.100000"
        runs, issues = parse(text)
        self.assertAlmostEqual(runs[0]['brier_alc'], .2)
        self.assertEqual(issues[0]['type'], 'unassigned_rows')

    def test_inconsistent_alc_and_counts_are_rejected(self):
        text = feedback([(1, 10, .2)])
        wrong_alc = text.replace('0.200000', '0.300000', 1)
        with self.assertRaisesRegex(ValueError, 'summary disagrees'):
            parse(wrong_alc)
        wrong_count = text.replace('benchmark_1 10', 'benchmark_1 11', 1)
        with self.assertRaisesRegex(ValueError, 'Changed evaluation count'):
            parse(wrong_count)


if __name__ == '__main__':
    unittest.main()
