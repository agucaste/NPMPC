import numpy as np
import pytest

from core.evaluator import Sampler


def test_grid_sampling_order_and_wraparound():
    X0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    s = Sampler(X0=X0)

    # sample sequentially
    assert np.allclose(s.sample(), X0[0])
    assert np.allclose(s.sample(), X0[1])
    assert np.allclose(s.sample(), X0[2])

    # wraps around and warns on full cycle
    with pytest.warns(UserWarning):
        x = s.sample()
    assert np.allclose(x, X0[0])


def test_function_sampler():
    calls = []

    def f():
        calls.append(1)
        return np.array([42.0])

    s = Sampler(f=f)
    assert np.allclose(s.sample(), np.array([42.0]))
    assert len(calls) == 1


def test_invalid_constructor_args():
    with pytest.raises(AssertionError):
        Sampler()  # neither X0 nor f provided

    with pytest.raises(AssertionError):
        Sampler(X0=np.array([[1.0]]), f=lambda: np.array([1.0]))  # both provided
