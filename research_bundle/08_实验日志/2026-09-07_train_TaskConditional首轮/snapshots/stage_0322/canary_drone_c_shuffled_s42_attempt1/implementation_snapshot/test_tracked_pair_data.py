"""Coordinate/RNG fixtures; the real loader equivalence is checked separately on 94."""
import random
import sys
from pathlib import Path
import unittest
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'legacy_oev1'))
from tracked_pair_data import parameter_matrix, ParameterTap


class TraceTests(unittest.TestCase):
    def test_flip_uses_continuous_boundary(self):
        m = parameter_matrix('RandomFlip', {'flip': True, 'direction': 'horizontal'}, (512, 640))
        np.testing.assert_array_equal(m @ [10, 30, 1], [630, 30, 1])

    def test_composition(self):
        pad = parameter_matrix('LetterBox', {'ratio': (2, 2), 'left': 0, 'top': 64}, (256, 320))
        affine = parameter_matrix('RandomPerspective', {'M': [[.5, 0, 10], [0, .5, 20], [0, 0, 1]]}, (640, 640))
        np.testing.assert_allclose(affine @ pad @ [10, 20, 1], [20, 72, 1])

    def test_tap_preserves_params_and_rng(self):
        def original(labels):
            return {'flip': random.random() < .5, 'direction': 'horizontal'}
        labels = {'img': np.zeros((512, 640, 3)), 'ori_shape': (1024, 1280), 'im_file': 'x'}
        random.seed(42)
        expected = original(labels)
        rng = random.getstate()
        random.seed(42)
        recorder = {}
        actual = ParameterTap(original, recorder, 'RandomFlip')(labels)
        self.assertEqual(actual, expected)
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(recorder[str(Path('x').resolve())]['original_shape'], (1024, 1280))


if __name__ == '__main__':
    unittest.main()
