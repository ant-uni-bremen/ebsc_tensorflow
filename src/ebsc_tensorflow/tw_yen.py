"""
Approximate time-warped Yen interpolator in TensorFlow.

Reconstructs a signal from its level-crossing samples (t_m, x_m) and a given instantaneous bandwidth B(t):

1. B(t) is the piecewise linear interpolation of the samples b on the grid tB. The time-warping function
   gamma(t) = 2 * integral of B(t) from tB[0] to t is then piecewise quadratic (calc_gamma).
2. The crossing times are warped to tau_m = gamma(t_m) (calc_tau_lcs).
3. The coefficients c_k of the basis functions sinc(tau - k) follow from the Tikhonov-regularized
   least-squares problem (G^T G + regularization * I) c = G^T x with G[m, k] = w_k * sinc(tau_m - k)
   (calc_coefficients).
4. The reconstruction is x_hat(t) = sum_k w_k * c_k * sinc(gamma(t) - k) (calc_gamma_at_t, interpolate_at_t).

The number of basis functions of the time-warped Yen interpolator, floor(gamma(T)) + 1 with T = tB[-1],
changes only in integer steps and provides no gradient with respect to b. Instead, all NUM_OUTPUT_SAMPLES
basis functions are kept and weighted by

    w_k = sigmoid((gamma(T) - k) / temperature),

so the reconstruction is differentiable with respect to b. For temperature -> 0, the weights become one for
k < gamma(T) and zero beyond, which gives the basis functions of the original interpolator.

All functions are compiled for the fixed tensor sizes in tw_yen_config.
"""

import tensorflow as tf
from ebsc_tensorflow import tw_yen_config, fast_poly

@tf.function(
    input_signature=(
        tf.TensorSpec(
            shape=[
                tw_yen_config.NUM_LCS_PADDED,
            ],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=[3, tw_yen_config.NUM_BW_SAMPLES - 1],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=[
                tw_yen_config.NUM_BW_SAMPLES,
            ],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.bool,
        ),
    )
)
def calc_tau_lcs(
    t_lcs: tf.Tensor, c: tf.Tensor, x: tf.Tensor, extrapolate=False
) -> tf.Tensor:
    """
    Warps the level-crossing times, tau_lcs = gamma(t_lcs).

    The values at padded entries are not used later. The last entry should be padding (zero or outside
    [x[0], x[-1]]), see fast_poly.evaluate_piecewise_2nd_order_poly.

    Args:
        t_lcs (tf.Tensor): Level-crossing times, padded, shape (NUM_LCS_PADDED,).
        c (tf.Tensor): Coefficients of gamma from calc_gamma, shape (3, NUM_BW_SAMPLES-1).
        x (tf.Tensor): Breakpoints of gamma (the grid tB), shape (NUM_BW_SAMPLES,).
        extrapolate (bool, optional): Not implemented, ignored. Defaults to False.

    Returns:
        tf.Tensor: Warped level-crossing times, shape (NUM_LCS_PADDED,).
    """
    tau_lcs = evaluate_gamma(t_lcs, c, x, extrapolate=extrapolate)
    return tau_lcs


@tf.function(
    input_signature=(
        tf.TensorSpec(
            shape=[
                tw_yen_config.NUM_OUTPUT_SAMPLES,
            ],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=[3, tw_yen_config.NUM_BW_SAMPLES - 1],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=[
                tw_yen_config.NUM_BW_SAMPLES,
            ],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.bool,
        ),
    )
)
def calc_gamma_at_t(
    t: tf.Tensor, c: tf.Tensor, x: tf.Tensor, extrapolate: bool = False
) -> tf.Tensor:
    """
    Evaluates gamma at the output times, where the reconstruction is computed by interpolate_at_t.

    Args:
        t (tf.Tensor): Output times, shape (NUM_OUTPUT_SAMPLES,).
        c (tf.Tensor): Coefficients of gamma from calc_gamma, shape (3, NUM_BW_SAMPLES-1).
        x (tf.Tensor): Breakpoints of gamma (the grid tB), shape (NUM_BW_SAMPLES,).
        extrapolate (bool, optional): Not implemented, ignored. Defaults to False.

    Returns:
        tf.Tensor: gamma(t), shape (NUM_OUTPUT_SAMPLES,).
    """
    gamma_at_t = evaluate_gamma(t, c, x, extrapolate=extrapolate)
    return gamma_at_t


def evaluate_gamma(
    t: tf.Tensor,
    c: tf.Tensor,
    x: tf.Tensor,
    extrapolate: bool = False,
) -> tf.Tensor:
    """
    Evaluates the piecewise quadratic time-warping function gamma at the times t.

    Args:
        t (tf.Tensor): Evaluation times.
        c (tf.Tensor): Coefficients of gamma from calc_gamma, shape (3, len(x)-1).
        x (tf.Tensor): Breakpoints of gamma (the grid tB).
        extrapolate (bool, optional): Not implemented, ignored. Defaults to False.

    Returns:
        tf.Tensor: gamma(t), zero outside [x[0], x[-1]].
    """
    y = fast_poly.evaluate_piecewise_2nd_order_poly(t, x, c)
    return y


@tf.function(
    input_signature=(
        tf.TensorSpec(
            shape=[
                tw_yen_config.NUM_BW_SAMPLES,
            ],
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=[
                tw_yen_config.NUM_BW_SAMPLES,
            ],
            dtype=tf.float32,
        ),
    )
)
def calc_gamma(inst_bandwidth_samples: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    """
    Calculates the time-warping function gamma(t) = 2 * integral of B(t) from x[0] to t.

    B(t) is the piecewise linear interpolation of inst_bandwidth_samples at x, so gamma is piecewise
    quadratic, zero at x[0], and differentiable with respect to inst_bandwidth_samples.

    Args:
        inst_bandwidth_samples (tf.Tensor): Samples of B(t) in Hz, shape (NUM_BW_SAMPLES,).
        x (tf.Tensor): Sample times of B(t) (the grid tB), strictly increasing, shape (NUM_BW_SAMPLES,).

    Returns:
        tf.Tensor: Coefficients (a, b, c) of gamma in each interval [x[i], x[i+1]] in absolute time,
        shape (3, NUM_BW_SAMPLES-1).
    """
    coeffs = fast_poly.integrate_1st_order_piecewise_poly(x, 2 * inst_bandwidth_samples)
    return coeffs

def tikhonov_solve_tf(G, x, reg_factor):
    """
    Solves the Tikhonov-regularized least-squares problem (G^T G + reg_factor * I) c = G^T x.

    Args:
        G (tf.Tensor): Kernel matrix, shape (M, N).
        x (tf.Tensor): Sample values, shape (M,) or (M, 1).
        reg_factor (tf.Tensor): Regularization, scalar.

    Returns:
        tf.Tensor: Solution c, shape (N,).
    """
    x = tf.reshape(x, (-1, 1))
    A = tf.matmul(G, G, transpose_a=True)
    b = tf.matmul(G, x, transpose_a=True)
    n = tf.shape(A)[0]
    A = A + reg_factor * tf.eye(n, dtype=A.dtype)
    L = tf.linalg.cholesky(A)
    return tf.squeeze(tf.linalg.cholesky_solve(L, b), axis=-1)

def sinc(x: tf.Tensor) -> tf.Tensor:
    """
    Normalized sinc function sin(pi * x) / (pi * x).

    Arguments with |x| < 1e-20 are replaced by 1e-20, which avoids the division by zero at x = 0.

    Args:
        x (tf.Tensor): Arguments.

    Returns:
        tf.Tensor: sinc(x).
    """
    x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
    return tf.sin(x * tf.experimental.numpy.pi) / (x * tf.experimental.numpy.pi)


@tf.function(
    input_signature=(
        tf.TensorSpec([tw_yen_config.NUM_LCS_PADDED], tf.float32),  # x_LCS
        tf.TensorSpec([tw_yen_config.NUM_LCS_PADDED], tf.float32),  # tau_LCS
        tf.TensorSpec((), tf.int32),                          # lcs_len
        tf.TensorSpec((), tf.float32),                        # dof_c = gamma(T)
        tf.TensorSpec((), tf.float32),                        # regularization
        tf.TensorSpec((), tf.float32),                        # temperature
    )
)
def calc_coefficients(
    x_LCS, tau_LCS, lcs_len, dof_c, regularization, temperature
):
    """
    Calculates the coefficients of the basis functions from the level-crossing samples.

    Solves (G^T G + regularization * I) c = G^T x_LCS with G[m, k] = w_k * sinc(tau_LCS[m] - k) for
    k = 0, ..., NUM_OUTPUT_SAMPLES-1 and w_k = sigmoid((dof_c - k) / temperature). Rows of padded samples
    (m >= lcs_len) are zero, so they do not contribute to the solution.

    Args:
        x_LCS (tf.Tensor): Level-crossing amplitudes, padded, shape (NUM_LCS_PADDED,).
        tau_LCS (tf.Tensor): Warped level-crossing times from calc_tau_lcs, shape (NUM_LCS_PADDED,).
        lcs_len (tf.Tensor): Number of level-crossing samples, int32 scalar.
        dof_c (tf.Tensor): gamma(T), the continuous counterpart of the number of basis functions, scalar.
        regularization (tf.Tensor): Tikhonov regularization, scalar.
        temperature (tf.Tensor): Temperature of the basis weights w_k, scalar.

    Returns:
        tf.Tensor: Coefficients c, shape (NUM_OUTPUT_SAMPLES,).
    """
    max_sample_times = tw_yen_config.NUM_OUTPUT_SAMPLES
    max_LCS = tf.shape(x_LCS)[0]
    x_LCS = tf.reshape(x_LCS, (max_LCS, 1))
    n = tf.range(0, max_sample_times, 1, dtype=x_LCS.dtype)
    n_matrix = tf.tile(n, [max_LCS])
    n_matrix = tf.reshape(n_matrix, (max_LCS, max_sample_times))

    tau_LCS_matrix = tf.tile(tf.expand_dims(tau_LCS, 0), [max_sample_times, 1])
    tau_LCS_matrix = tf.transpose(
        tf.reshape(tau_LCS_matrix, (max_sample_times, max_LCS))
    )

    # Rows of padded samples are set to zero. lcs_len does not depend on B, so this
    # hard mask does not affect the gradient.
    row_ids = tf.range(max_LCS)[:, None]
    row_mask = tf.cast(row_ids < lcs_len, tf.float32)  # [rows, 1]

    # Columns are weighted by w_k = sigmoid((gamma(T) - k) / temperature), which is
    # differentiable in B and becomes a hard step for temperature -> 0.
    col_ids = tf.cast(tf.range(max_sample_times)[None, :], tf.float32)  # [1, NUM_OUTPUT_SAMPLES]
    col_mask = tf.sigmoid((dof_c - col_ids) / temperature)             # [1, NUM_OUTPUT_SAMPLES]

    mask = row_mask * col_mask  # [NUM_LCS_PADDED, NUM_OUTPUT_SAMPLES]

    G = sinc(tau_LCS_matrix - n_matrix)
    G_masked = G * mask
    c = tikhonov_solve_tf(G_masked, x_LCS, reg_factor=regularization)
    return tf.squeeze(c)


@tf.function(
    input_signature=(
        tf.TensorSpec([tw_yen_config.NUM_OUTPUT_SAMPLES], tf.float32),  # gamma_at_t
        tf.TensorSpec([tw_yen_config.NUM_OUTPUT_SAMPLES], tf.float32),  # optimal_sample_amplitudes
        tf.TensorSpec((), tf.float32),                         # dof_c = gamma(T)
        tf.TensorSpec((), tf.float32),                         # temperature
    )
)
def interpolate_at_t(
    gamma_at_t, optimal_sample_amplitudes, dof_c, temperature
):
    """
    Evaluates the reconstruction x_hat(t) = sum_k w_k * c_k * sinc(gamma(t) - k) at the output times.

    The weights w_k = sigmoid((dof_c - k) / temperature) are the same as in calc_coefficients.

    Args:
        gamma_at_t (tf.Tensor): gamma at the output times from calc_gamma_at_t, shape (NUM_OUTPUT_SAMPLES,).
        optimal_sample_amplitudes (tf.Tensor): Coefficients c from calc_coefficients,
            shape (NUM_OUTPUT_SAMPLES,).
        dof_c (tf.Tensor): gamma(T), scalar.
        temperature (tf.Tensor): Temperature of the basis weights w_k, scalar.

    Returns:
        tf.Tensor: Reconstructed signal at the output times, shape (NUM_OUTPUT_SAMPLES,).
    """
    n = tf.cast(tf.range(tw_yen_config.NUM_OUTPUT_SAMPLES), tf.float32)
    idx = tf.range(tw_yen_config.NUM_OUTPUT_SAMPLES)
    c = tf.gather(optimal_sample_amplitudes, idx)

    diff = tf.expand_dims(gamma_at_t, 1) - tf.expand_dims(n, 0)
    G = sinc(diff)  # [NUM_OUTPUT_SAMPLES, NUM_OUTPUT_SAMPLES]

    # Same basis weights as in calc_coefficients.
    col_ids = tf.cast(tf.range(tw_yen_config.NUM_OUTPUT_SAMPLES)[None, :], tf.float32)
    col_mask = tf.sigmoid((dof_c - col_ids) / temperature)  # [1, NUM_OUTPUT_SAMPLES]

    G_masked = G * col_mask
    values = tf.linalg.matvec(G_masked, c)
    return values


@tf.function(
    input_signature=(
        tf.TensorSpec([tw_yen_config.NUM_OUTPUT_SAMPLES], tf.float32),      # t
        tf.TensorSpec([tw_yen_config.NUM_LCS_PADDED, 2], tf.float32),    # lcs
        tf.TensorSpec((), tf.int32),                               # lcs_len
        tf.TensorSpec([tw_yen_config.NUM_BW_SAMPLES], tf.float32),         # tB
        tf.TensorSpec([tw_yen_config.NUM_BW_SAMPLES], tf.float32),         # b
        tf.TensorSpec((), tf.float32),                             # regularization
        tf.TensorSpec((), tf.float32),                             # temperature
        tf.TensorSpec((), tf.bool),                                # extrapolate
    )
)
def approx_tw_yen(
    t, lcs, lcs_len, tB, b, regularization, temperature, extrapolate=False
):
    """
    Reconstructs one signal from its level-crossing samples with the approximate time-warped Yen interpolator.

    Args:
        t (tf.Tensor): Output times, shape (NUM_OUTPUT_SAMPLES,).
        lcs (tf.Tensor): Level-crossing samples, columns time and amplitude, padded,
            shape (NUM_LCS_PADDED, 2).
        lcs_len (tf.Tensor): Number of level-crossing samples, int32 scalar.
        tB (tf.Tensor): Sample times of B(t), shape (NUM_BW_SAMPLES,).
        b (tf.Tensor): Samples of B(t) in Hz, positive, shape (NUM_BW_SAMPLES,).
        regularization (tf.Tensor): Tikhonov regularization, scalar.
        temperature (tf.Tensor): Temperature of the basis weights, scalar.
        extrapolate (tf.Tensor, optional): Not implemented, ignored. Defaults to False.

    Returns:
        tf.Tensor: Reconstructed signal at t, shape (NUM_OUTPUT_SAMPLES,).
    """
    c_gamma = calc_gamma(b, tB)

    # gamma(T) at the end of the grid tB, not rounded, so the basis weights are
    # differentiable in b.
    g_tmax = tf.squeeze(
        evaluate_gamma(
            tf.reshape(tB[-1], (1,)),
            c_gamma,
            tB,
            tf.cast(extrapolate, tf.bool),
        )
    )

    tau_lcs = calc_tau_lcs(
        lcs[:, 0],
        c_gamma,
        tB,
        tf.cast(extrapolate, tf.bool),
    )
    optimal_sample_amplitudes = calc_coefficients(
        lcs[:, 1],
        tf.squeeze(tau_lcs),
        tf.cast(lcs_len, dtype=tf.int32),
        g_tmax,
        regularization,
        temperature,
    )
    gamma_at_t = calc_gamma_at_t(t, c_gamma, tB, tf.cast(extrapolate, tf.bool))
    x_recon = interpolate_at_t(
        gamma_at_t, optimal_sample_amplitudes, g_tmax, temperature
    )
    return x_recon


@tf.function(
    input_signature=(
        tf.TensorSpec([tw_yen_config.BATCH_SIZE, tw_yen_config.NUM_OUTPUT_SAMPLES], tf.float32),      # T
        tf.TensorSpec([tw_yen_config.BATCH_SIZE, tw_yen_config.NUM_LCS_PADDED, 2], tf.float32),    # LCS
        tf.TensorSpec([tw_yen_config.BATCH_SIZE], tf.int32),                                 # LCS_len
        tf.TensorSpec([tw_yen_config.BATCH_SIZE, tw_yen_config.NUM_BW_SAMPLES], tf.float32),         # TB
        tf.TensorSpec([tw_yen_config.BATCH_SIZE, tw_yen_config.NUM_BW_SAMPLES], tf.float32),         # B
        tf.TensorSpec([tw_yen_config.BATCH_SIZE], tf.float32),                               # REGULARIZATION
        tf.TensorSpec([tw_yen_config.BATCH_SIZE], tf.float32),                               # TEMPERATURE
        tf.TensorSpec([tw_yen_config.BATCH_SIZE], tf.bool),                                  # EXTRAPOLATE
    ),
)
def mapped_approx_tw_yen(
    T, LCS, LCS_len, TB, B, REGULARIZATION, TEMPERATURE, EXTRAPOLATE
):
    """
    Applies approx_tw_yen to a batch of signals, each with its own parameters (tf.vectorized_map).

    Args:
        T (tf.Tensor): Output times, shape (BATCH_SIZE, NUM_OUTPUT_SAMPLES).
        LCS (tf.Tensor): Level-crossing samples, shape (BATCH_SIZE, NUM_LCS_PADDED, 2).
        LCS_len (tf.Tensor): Numbers of level-crossing samples, int32, shape (BATCH_SIZE,).
        TB (tf.Tensor): Sample times of B(t), shape (BATCH_SIZE, NUM_BW_SAMPLES).
        B (tf.Tensor): Samples of B(t) in Hz, shape (BATCH_SIZE, NUM_BW_SAMPLES).
        REGULARIZATION (tf.Tensor): Tikhonov regularization, shape (BATCH_SIZE,).
        TEMPERATURE (tf.Tensor): Temperature of the basis weights, shape (BATCH_SIZE,).
        EXTRAPOLATE (tf.Tensor): Not implemented, ignored, shape (BATCH_SIZE,).

    Returns:
        tf.Tensor: Reconstructed signals, shape (BATCH_SIZE, NUM_OUTPUT_SAMPLES).
    """
    def f(args):
        t, lcs, lcs_len, tb, b, reg, temp, extr = args
        return approx_tw_yen(
            t, lcs, lcs_len, tb, b, reg, temp, extr
        )

    X_recon = tf.vectorized_map(
        f,
        (T, LCS, LCS_len, TB, B, REGULARIZATION, TEMPERATURE, EXTRAPOLATE),
    )
    return X_recon


def batch_approx_tw_yen(
    t, LCS, LCS_len, tB, B, regularization, temperature=0.5, extrapolate=False
):
    """
    Reconstructs a batch of signals with the approximate time-warped Yen interpolator.

    All signals share the output times, the grid of B(t), the regularization and the temperature. The
    batch has to contain exactly BATCH_SIZE signals.

    Args:
        t (tf.Tensor): Output times, shape (NUM_OUTPUT_SAMPLES,).
        LCS (tf.Tensor): Level-crossing samples, columns time and amplitude, padded,
            shape (BATCH_SIZE, NUM_LCS_PADDED, 2).
        LCS_len (tf.Tensor): Numbers of level-crossing samples, int32, shape (BATCH_SIZE,).
        tB (tf.Tensor): Sample times of B(t), shape (NUM_BW_SAMPLES,).
        B (tf.Tensor): Samples of B(t) in Hz, positive, shape (BATCH_SIZE, NUM_BW_SAMPLES).
        regularization (float or tf.Tensor): Tikhonov regularization, scalar.
        temperature (float or tf.Tensor, optional): Temperature of the basis weights. Defaults to 0.5.
        extrapolate (bool, optional): Not implemented, ignored. Defaults to False.

    Returns:
        tf.Tensor: Reconstructed signals at t, shape (BATCH_SIZE, NUM_OUTPUT_SAMPLES).
    """
    batch = tf.shape(B)[0]
    T = tf.tile(tf.expand_dims(t, 0), (batch, 1))
    TB = tf.tile(tf.expand_dims(tB, 0), (batch, 1))
    REGULARIZATION = tf.tile(tf.expand_dims(tf.cast(regularization, tf.float32), 0), (batch,))
    TEMPERATURE = tf.tile(
        tf.expand_dims(tf.cast(temperature, tf.float32), 0), (batch,)
    )
    EXTRAPOLATE = tf.tile(tf.expand_dims(extrapolate, 0), (batch,))
    X_recon = mapped_approx_tw_yen(
        T, LCS, LCS_len, TB, B, REGULARIZATION, TEMPERATURE, EXTRAPOLATE
    )
    return X_recon
