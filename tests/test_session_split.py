import copy
import unittest
from session_split import make_split, validate_grouped_manifest


class SessionSplitTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'classes': ['a', 'b'], 'records': []}
        self.metadata = {}
        for label in ['a', 'b']:
            for sid, v in [('train', 1), ('val', 2), ('test', 3)]:
                for i in range(2):
                    rid = f'{label}-{sid}-{i}'
                    self.manifest['records'].append(dict(record_id=rid, class_name=label, voltage_kv=v))
                    self.metadata[rid] = dict(session_id=sid, voltage_kv=v)
        self.config = dict(holdout_voltages_kv={'a': [3], 'b': [3]}, validation_session_ids=['val'])

    def test_grouped_split_preserves_sources(self):
        original = copy.deepcopy(self.manifest)
        result = make_split(self.manifest, self.metadata, self.config)
        self.assertEqual(validate_grouped_manifest(result), {'train': 4, 'val': 4, 'test': 4})
        self.assertEqual(self.manifest, original)

    def test_missing_session_is_rejected(self):
        self.metadata['a-train-0']['session_id'] = ''
        with self.assertRaises(ValueError):
            make_split(self.manifest, self.metadata, self.config)

    def test_mixed_voltage_session_overlap_is_rejected(self):
        self.metadata['a-train-0']['session_id'] = 'test'
        with self.assertRaises(ValueError):
            make_split(self.manifest, self.metadata, self.config)

    def test_validation_holdout_collision_is_rejected(self):
        self.config['validation_session_ids'] = ['test']
        with self.assertRaises(ValueError):
            make_split(self.manifest, self.metadata, self.config)

    def test_tampered_session_assignment_is_rejected(self):
        result = make_split(self.manifest, self.metadata, self.config)
        result['records'][0]['split'] = 'val'
        with self.assertRaises(ValueError):
            validate_grouped_manifest(result)

    def test_missing_class_coverage_is_rejected(self):
        self.metadata['a-val-0']['session_id'] = 'train'
        self.metadata['a-val-1']['session_id'] = 'train'
        with self.assertRaises(ValueError):
            make_split(self.manifest, self.metadata, self.config)

    def test_voltage_precision_and_metadata_conflicts(self):
        self.config['holdout_voltages_kv']['a'] = ['3.000']
        make_split(self.manifest, self.metadata, self.config)
        self.metadata['a-test-0']['voltage_kv'] = 4
        with self.assertRaises(ValueError):
            make_split(self.manifest, self.metadata, self.config)


if __name__ == '__main__':
    unittest.main()
