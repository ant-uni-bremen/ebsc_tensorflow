"""Tests of NNTW with a small network on the fixture batch."""

import numpy as np
import tensorflow as tf

from ebsc_tensorflow import nntw, tw_yen_config

MODEL_CONF = {"units": 8, "dropout": 0.0, "output_fn": "bounded_output(0.001, 150)"}


def build_model(data_batch):
    """NNTW with 8 units, without dropout, on the grids of the fixture batch."""
    return nntw.NNTW(
        in_shape=(tw_yen_config.NUM_LCS_PADDED, 2),
        out_shape=(tw_yen_config.NUM_BW_SAMPLES,),
        model_conf=MODEL_CONF,
        t=data_batch["t"],
        tB=data_batch["tB"],
        regularization=1e-4,
        temperature=0.5,
    )


def test_nntw_output_shapes(data_batch):
    """Output shapes, and an estimate within the bounds of the output function."""
    model = build_model(data_batch)
    inputs = {"LCS": data_batch["LCS"], "LCS_len": data_batch["LCS_len"]}
    X_hat = model(inputs)
    B_hat = model.predict_B(data_batch["LCS"])
    assert X_hat.shape == (tw_yen_config.BATCH_SIZE, tw_yen_config.NUM_OUTPUT_SAMPLES)
    assert B_hat.shape == (tw_yen_config.BATCH_SIZE, tw_yen_config.NUM_BW_SAMPLES)
    assert np.all(B_hat >= 0.001) and np.all(B_hat <= 150)


def test_nntw_fit_step(data_batch):
    """One training step updates the network and the regularization."""
    model = build_model(data_batch)
    inputs = {"LCS": data_batch["LCS"], "LCS_len": data_batch["LCS_len"]}
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model(inputs)
    weights_before = [w.copy() for w in model.get_weights()]
    regularization_before = float(model.regularization)

    model.fit(inputs, data_batch["X"], batch_size=tw_yen_config.BATCH_SIZE, epochs=1, verbose=0)

    assert float(model.regularization) != regularization_before
    assert any(np.any(a != b) for a, b in zip(weights_before, model.get_weights()))
    assert np.all(np.isfinite(model(inputs)))
