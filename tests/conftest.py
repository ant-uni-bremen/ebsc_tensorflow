"""Fixture data_batch with the test signals of ebsc_tensorflow.signals, shared by all tests."""

import numpy as np
import pytest

from ebsc_tensorflow import signals, tw_yen_config


@pytest.fixture(scope="session")
def data_batch():
    """One batch of test signals: bandwidth of B(t) 2 Hz, 13 levels in [-4, 4]."""
    t, tB, B, X, LCS, LCS_len = signals.get_data_batch(
        batch_size=tw_yen_config.BATCH_SIZE,
        bandwidth_of_local_bandwidth=2,
        ref_levels=np.linspace(-4, 4, 13),
    )
    return {"t": t, "tB": tB, "B": B, "X": X, "LCS": LCS, "LCS_len": LCS_len}
