import copy
import json
import unittest
from pathlib import Path

from scripts.run_kv_continuation_pilot import summarize


class PilotDecisionTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'configs/kv_continuation_pilot.json').read_text())
        self.config['context_lengths'] = [256]
        self.rows = [{'context_length': 256, 'arms': {
            name: {'kl': [kl]*128, 'nll_delta': [0.01]*128, 'top1_equal': [True]*128}
            for name, kl in [('predicted_all', 0.01), ('zero_all', 1.0), ('original_all', 0.5)]
        }} for _ in range(16)]

    def test_good_candidate_passes(self):
        self.assertEqual(summarize(self.rows, self.config)['256']['decision'], 'GO_TO_GDN_PILOT')

    def test_boundary_failure_not_hidden_by_average(self):
        for row in self.rows:
            row['arms']['predicted_all']['kl'][0] = 1.0
        result = summarize(self.rows, self.config)['256']
        self.assertTrue(result['checks']['mean_kl_acceptable'])
        self.assertFalse(result['checks']['boundary_kl_acceptable'])
        self.assertEqual(result['decision'], 'HOLD_REPAIR_KV')

    def test_late_growth_blocks_go(self):
        for row in self.rows:
            row['arms']['predicted_all']['kl'][96:] = [0.08]*32
        self.assertFalse(summarize(self.rows, self.config)['256']['checks']['no_large_late_growth'])

    def test_baseline_advantage_required(self):
        for row in self.rows:
            row['arms']['original_all'] = copy.deepcopy(row['arms']['predicted_all'])
        self.assertFalse(summarize(self.rows, self.config)['256']['checks']['beats_both_baselines'])


if __name__ == '__main__':
    unittest.main()
