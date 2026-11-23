"""Tests of tw_yen: gamma against ebsc.LocalBandwidth, reconstruction with the true B(t) and gradients."""

import ebsc
import numpy as np
import tensorflow as tf

from ebsc_tensorflow import fast_poly, tw_yen

REGULARIZATION = 1e-3
TEMPERATURE = 0.5
NMSE_THRESHOLD_DB = -20.0  # median of the fixture batch: -22.8 dB


def test_sinc():
    """sinc equals np.sinc."""
    t = np.arange(-5, 5, 0.1)
    np.testing.assert_allclose(tw_yen.sinc(t), np.sinc(t))


def test_calc_gamma(data_batch):
    """The coefficients of calc_gamma give gamma(t) of ebsc.LocalBandwidth."""
    index = 0
    t, tB, B = data_batch["t"], data_batch["tB"], data_batch["B"][index]
    lbw = ebsc.LocalBandwidth(tB, B)
    gamma_coeffs = tw_yen.calc_gamma(B, tB)
    gamma_t = fast_poly.evaluate_piecewise_2nd_order_poly(t, tB, gamma_coeffs)
    np.testing.assert_allclose(gamma_t, lbw.gamma(t), rtol=1e-5)


def test_evaluate_gamma(data_batch):
    """evaluate_gamma at the output times equals ebsc.LocalBandwidth.gamma."""
    index = 1
    t, tB, B = data_batch["t"], data_batch["tB"], data_batch["B"][index]
    lbw = ebsc.LocalBandwidth(tB, B)
    gamma_coeffs = tw_yen.calc_gamma(B, tB)
    gamma_t = tw_yen.evaluate_gamma(t, gamma_coeffs, tB, False)
    np.testing.assert_allclose(gamma_t, lbw.gamma(t), rtol=1e-5)


def test_calc_tau_lcs(data_batch):
    """calc_tau_lcs at the padded crossing times equals ebsc.LocalBandwidth.gamma."""
    index = 2
    tB, B = data_batch["tB"], data_batch["B"][index]
    t_lcs = data_batch["LCS"][index, :, 0]
    lbw = ebsc.LocalBandwidth(tB, B)
    gamma_coeffs = tw_yen.calc_gamma(B, tB)
    tau_lcs = tw_yen.calc_tau_lcs(t_lcs, gamma_coeffs, tB, False)
    np.testing.assert_allclose(tau_lcs, lbw.gamma(t_lcs), rtol=1e-5)


def test_calc_gamma_at_t(data_batch):
    """calc_gamma_at_t equals ebsc.LocalBandwidth.gamma."""
    index = 3
    t, tB, B = data_batch["t"], data_batch["tB"], data_batch["B"][index]
    lbw = ebsc.LocalBandwidth(tB, B)
    gamma_coeffs = tw_yen.calc_gamma(B, tB)
    gamma_t = tw_yen.calc_gamma_at_t(t, gamma_coeffs, tB, False)
    np.testing.assert_allclose(gamma_t, lbw.gamma(t), rtol=1e-5)


def test_batch_approx_tw_yen_true_bandwidth(data_batch):
    """With the true bandwidth, the reconstruction has to be close to the signal."""
    X = data_batch["X"]
    X_hat = tw_yen.batch_approx_tw_yen(
        data_batch["t"],
        data_batch["LCS"],
        data_batch["LCS_len"],
        data_batch["tB"],
        data_batch["B"],
        REGULARIZATION,
        TEMPERATURE,
    ).numpy()
    assert X_hat.shape == X.shape
    nmse_db = 10 * np.log10(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X**2, axis=1))
    assert np.median(nmse_db) < NMSE_THRESHOLD_DB


def test_batch_approx_tw_yen_gradients(data_batch):
    """The reconstruction error has finite, nonzero gradients w.r.t. B and the regularization."""
    B = tf.Variable(data_batch["B"])
    regularization = tf.Variable(REGULARIZATION)
    with tf.GradientTape() as tape:
        X_hat = tw_yen.batch_approx_tw_yen(
            data_batch["t"],
            data_batch["LCS"],
            data_batch["LCS_len"],
            data_batch["tB"],
            B,
            regularization,
            TEMPERATURE,
        )
        loss = tf.reduce_mean((X_hat - data_batch["X"]) ** 2)
    grad_B, grad_regularization = tape.gradient(loss, [B, regularization])
    for grad in (grad_B, grad_regularization):
        assert grad is not None
        assert np.all(np.isfinite(grad))
        assert np.any(grad != 0)
