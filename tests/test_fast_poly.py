"""Tests of fast_poly against scipy.interpolate.PPoly and ebsc.AnalogSignal."""

import ebsc
import numpy as np
from scipy.interpolate import PPoly

from ebsc_tensorflow import fast_poly


def test_quad():
    """quad equals a PPoly with one interval starting at 0, where absolute and local time coincide."""
    a = np.array([[2], [1], [3]])
    t = np.array([0, 5])
    teval = np.arange(0, 5, 0.01)
    poly_res = PPoly(a, t)(teval)

    quad_res = fast_poly.quad(teval, a)
    np.testing.assert_allclose(poly_res, quad_res)


def test_eval_2nd_order_poly():
    """eval_2nd_order_poly equals the antiderivative of a linear PPoly within its interval."""
    coeff = np.array([[2.0], [1.0]])
    t = np.array([0.0, 5.0])
    teval = np.arange(0, 5, 0.01)
    poly = PPoly(coeff, t)
    poly_int = poly.antiderivative()
    poly_int_res = poly_int(teval)
    fast_poly_res, _ = fast_poly.eval_2nd_order_poly(
        teval, t[0], t[1], poly_int.c, False, 0
    )
    np.testing.assert_allclose(poly_int_res, fast_poly_res)


def test_integrate_1st_order_poly():
    """integrate_1st_order_poly gives the antiderivative of the line through (0, 1) and (1, 3)."""
    t = np.array([0.0, 1.0])
    x = np.array([1.0, 3.0])
    coeff = np.array([[2.0], [1.0]])

    teval = np.arange(0, 1, 0.01)
    poly = PPoly(coeff, t)
    poly_int = poly.antiderivative()
    poly_int_res = poly_int(teval)
    fast_poly_int_c, _ = fast_poly.integrate_1st_order_poly(
        t[0], t[1], x[0], x[1], 0
    )
    fast_poly_int_res, _ = fast_poly.eval_2nd_order_poly(
        teval, t[0], t[1], fast_poly_int_c, False, 0
    )
    np.testing.assert_allclose(poly_int_res, fast_poly_int_res)


def test_integrate_1st_order_piecewise_poly():
    """The antiderivative of a piecewise linear function equals that of ebsc.AnalogSignal."""
    t = np.array([0.0, 1.0, 2.0, 4.0])
    x = np.array([2.0, 1.0, 0.0, -1.0])
    teval = np.arange(0, 4, 0.01)
    poly = ebsc.AnalogSignal(t, x)
    poly_int = poly.antiderivative()
    poly_int_res = poly_int(teval)
    fast_poly_c = fast_poly.integrate_1st_order_piecewise_poly(t, x)
    fast_poly_int_res = fast_poly.evaluate_piecewise_2nd_order_poly(
        teval, t, fast_poly_c
    )
    np.testing.assert_allclose(poly_int_res, fast_poly_int_res)


def test_evaluate_piecewise_2nd_order_poly():
    """As test_integrate_1st_order_piecewise_poly, with random values at the breakpoints."""
    t = np.array([0.0, 1.0, 2.0, 4.0])
    x = np.random.default_rng(0).random(4)
    teval = np.arange(0, 4, 0.01)
    poly = ebsc.AnalogSignal(t, x)
    poly_int = poly.antiderivative()
    poly_int_res = poly_int(teval)
    fast_poly_c = fast_poly.integrate_1st_order_piecewise_poly(t, x)
    fast_poly_int_res = fast_poly.evaluate_piecewise_2nd_order_poly(
        teval, t, fast_poly_c
    )
    np.testing.assert_allclose(poly_int_res, fast_poly_int_res)
