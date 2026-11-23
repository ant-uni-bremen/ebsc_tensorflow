"""
Piecewise polynomials in TensorFlow.

Integrates piecewise linear functions and evaluates the resulting piecewise quadratic functions, so that the
time-warping function gamma(t) = 2 * integral of B(t) is differentiable with respect to the samples of B(t).

All coefficients refer to absolute time, a * t**2 + b * t + c, and not to the start of each interval as in
scipy.interpolate.PPoly.
"""

import tensorflow as tf


def quad(t: tf.Tensor, coeff: tf.Tensor) -> tf.Tensor:
    """
    Evaluates the quadratic function coeff[0] * t**2 + coeff[1] * t + coeff[2].

    Args:
        t (tf.Tensor): Evaluation times.
        coeff (tf.Tensor): Coefficients (a, b, c) of shape (3,).

    Returns:
        tf.Tensor: Function values at t.
    """
    return coeff[0] * t**2 + coeff[1] * t + coeff[2]


def eval_2nd_order_poly(
    t: tf.Tensor,
    t0: tf.float32,
    t1: tf.float32,
    coeff: tf.Tensor,
    line_end: tf.bool,
    prev_max: tf.float32,
):
    """
    Evaluates a quadratic function that is valid in the interval [t0, t1).

    Values outside the interval are set to zero before prev_max is added. With line_end, the interval
    includes t1, i.e. [t0, t1].

    Args:
        t (tf.Tensor): Evaluation times.
        t0 (tf.float32): Start of the interval.
        t1 (tf.float32): End of the interval.
        coeff (tf.Tensor): Coefficients (a, b, c) of shape (3,).
        line_end (tf.bool): If true, t1 belongs to the interval.
        prev_max (tf.float32): Offset added to all values.

    Returns:
        tuple: Values at t plus prev_max, and the value at the last evaluation time t[-1] without the offset
        (zero if t[-1] lies outside the interval).
    """
    x = quad(t, coeff)
    x = tf.where(t < t0, 0.0, x)
    x = tf.where(t > t1 if line_end else t >= t1, 0.0, x)
    return x + prev_max, x[-1]


def integrate_1st_order_poly(
    t0: tf.float32,
    t1: tf.float32,
    x0: tf.float32,
    x1: tf.float32,
    start_offset: tf.float32,
):
    """
    Integrates the linear function through (t0, x0) and (t1, x1).

    The antiderivative is a quadratic function in absolute time with the value start_offset at t0.

    Args:
        t0 (tf.float32): Start of the interval.
        t1 (tf.float32): End of the interval.
        x0 (tf.float32): Value at t0.
        x1 (tf.float32): Value at t1.
        start_offset (tf.float32): Value of the antiderivative at t0.

    Returns:
        tuple: Coefficients (a, b, c) of shape (3,), and the value of the antiderivative at t1.
    """
    a = (x1 - x0) / (t1 - t0) / 2
    b = x0 + t0 * (x0 - x1) / (t1 - t0)
    c = start_offset - quad(t0, [a, b, 0])

    cur_max = quad(t1, [a, b, c])
    coeff = tf.stack([a, b, c])
    return coeff, cur_max


def integrate_1st_order_piecewise_poly(t: tf.Tensor, x: tf.Tensor):
    """
    Integrates the piecewise linear function through the points (t[i], x[i]).

    The antiderivative is zero at t[0] and continuous at the breakpoints, so each interval
    [t[i], t[i+1]] continues from the value at the end of the previous one.

    Args:
        t (tf.Tensor): Breakpoints, strictly increasing, shape (N,).
        x (tf.Tensor): Values at the breakpoints, shape (N,).

    Returns:
        tf.Tensor: Coefficients (a, b, c) of the antiderivative in each interval, shape (3, N-1).
    """
    num_points = tf.shape(t)[0]
    num_segments = num_points - 1

    coeffs = tf.zeros((3, num_segments), dtype=x.dtype)
    prev_max = tf.zeros([1], dtype=x.dtype)  # make prev_max a [1]-shaped tensor

    t0 = t[:-1]
    t1 = t[1:]
    x0 = x[:-1]
    x1 = x[1:]

    def cond(i, coeffs, prev_max):
        return i < num_segments

    def body(i, coeffs, prev_max):
        coeff, new_prev_max = integrate_1st_order_poly(
            t0[i], t1[i], x0[i], x1[i], prev_max[0]
        )  # coeff: shape (3,), new_prev_max: scalar

        # Update coeffs
        indices = tf.stack([tf.range(3), tf.fill([3], i)], axis=1)
        coeffs = tf.tensor_scatter_nd_update(coeffs, indices, coeff)

        # Update prev_max at index 0
        prev_max = tf.tensor_scatter_nd_update(prev_max, [[0]], [new_prev_max])

        return i + 1, coeffs, prev_max

    i0 = tf.constant(0)

    _, coeffs, _ = tf.while_loop(
        cond,
        body,
        loop_vars=[i0, coeffs, prev_max],
        shape_invariants=[
            tf.TensorShape([]),  # i
            tf.TensorShape([3, None]),  # coeffs
            tf.TensorShape([1]),  # prev_max tensor shape [1]
        ],
    )

    return coeffs


def evaluate_piecewise_2nd_order_poly(
    t: tf.Tensor, breakpoints: tf.Tensor, coeffs: tf.Tensor
):
    """
    Evaluates a piecewise quadratic function at the times t.

    Interval i is [breakpoints[i], breakpoints[i+1]); the last interval includes its end. The function is
    zero outside [breakpoints[0], breakpoints[-1]].

    The value of each interval at the last evaluation time t[-1] is added to all values of the next interval.
    The result is therefore only correct if t[-1] lies in the last interval, outside the breakpoints, or
    where the function is zero, e.g. at breakpoints[0] for an antiderivative from
    integrate_1st_order_piecewise_poly.

    Args:
        t (tf.Tensor): Evaluation times.
        breakpoints (tf.Tensor): Breakpoints, strictly increasing, shape (N,).
        coeffs (tf.Tensor): Coefficients (a, b, c) of each interval, shape (3, N-1), e.g. from
            integrate_1st_order_piecewise_poly.

    Returns:
        tf.Tensor: Function values at t.
    """
    num_poly = tf.shape(breakpoints)[0]
    num_t_points = tf.shape(t)[0]

    output = tf.zeros((num_poly, num_t_points), dtype=t.dtype)
    line_end = tf.tensor_scatter_nd_update(
        tf.zeros_like(breakpoints, dtype=tf.bool), [[num_poly - 2]], [True]
    )

    t0, t1 = breakpoints[:-1], breakpoints[1:]
    prev_max = 0.0
    for i in tf.range(num_poly - 1):
        x_piece, prev_max = eval_2nd_order_poly(
            t, t0[i], t1[i], coeffs[:, i], line_end[i], prev_max
        )
        indices = tf.stack([tf.fill([num_t_points], i), tf.range(num_t_points)], axis=1)
        output = tf.tensor_scatter_nd_update(output, indices, x_piece)
    return tf.reduce_sum(output, 0)
