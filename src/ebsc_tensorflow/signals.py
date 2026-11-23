"""
Variable-bandwidth test signals in [-4, 4] and their level-crossing samples, generated with ebsc.

Used by the tests and the examples. This module needs ebsc (extras test or examples) and is therefore not
imported by ebsc_tensorflow itself; import it with ``from ebsc_tensorflow import signals``.
"""

import ebsc
import numpy as np

from ebsc_tensorflow import tw_yen_config


def get_time_vector(tmin, tmax, fs, mode="x"):
    """
    Uniform time grid with sampling rate fs starting at tmin.

    Args:
        tmin (float): First sample time in s.
        tmax (float): End of the interval in s.
        fs (float): Sampling rate in Hz.
        mode (str, optional): "x" gives floor((tmax - tmin) * fs) samples, excluding tmax; "B" adds one
            sample, so the grid includes tmax. Defaults to "x".

    Returns:
        np.ndarray: Sample times.
    """
    first_sample_time = tmin
    number_samples = np.floor((tmax - tmin) * (fs))
    if mode == "B":
        number_samples = number_samples + 1
    last_sample_time = tmin + (number_samples - 1) / (fs)
    t_s = np.linspace(first_sample_time, last_sample_time, int(number_samples))
    return t_s


def get_feb24_local_bandwidth_generation_time_vector():
    """Grid on which B(t) is generated: 401 samples at 400 Hz in [0, 1] s."""
    return get_time_vector(0, 1, 400, mode="B")


def get_feb24_local_bandwidth_inference_time_vector():
    """Grid tB of the samples of B(t) used by the interpolator: 43 samples at 42 Hz in [0, 1] s."""
    return get_time_vector(0, 1, 42, mode="B")


def get_feb24_signal_time_vector():
    """Output times t: 400 samples at 400 Hz in [0, 1) s."""
    return get_time_vector(0, 1, 400)


def generate_feb24_local_bandwidth(bandwidth, rng):
    """
    Random instantaneous bandwidth B(t) that is itself bandlimited.

    Samples drawn uniformly from [5, 100] Hz at the rate 2 * bandwidth are sinc-interpolated. Draws with
    B(t) <= 0 anywhere on the generation grid are repeated.

    Args:
        bandwidth (float): Bandwidth of B(t) in Hz.
        rng (np.random.Generator): Random number generator.

    Returns:
        ebsc.LocalBandwidth: B(t) on the generation grid.
    """
    t_B_over = get_feb24_local_bandwidth_generation_time_vector()
    fs_B = 2 * bandwidth
    t_B = get_time_vector(0, 1, fs_B, mode="B")
    while True:
        B = rng.uniform(5, 100, len(t_B))
        local_bandwidth = ebsc.BandlimitedFunction(B, fs_B)
        B_over = local_bandwidth(t_B_over)
        if np.min(B_over) > 0:
            break
    local_bandwidth = ebsc.LocalBandwidth(t_B_over, local_bandwidth(t_B_over))
    return local_bandwidth


def generate_feb24_signal(bandwidth, rng):
    """
    Random variable-bandwidth signal with standard normal coefficients.

    Args:
        bandwidth (float): Bandwidth of B(t) in Hz.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple: ebsc.VariableBandwidthFunction of the signal and its ebsc.LocalBandwidth.
    """
    local_bandwidth = generate_feb24_local_bandwidth(bandwidth, rng)
    variable_bandwidth_signal = ebsc.VariableBandwidthFunction(local_bandwidth)
    random_amplitudes = rng.standard_normal(variable_bandwidth_signal.degrees_of_freedom)
    variable_bandwidth_signal.set_sample_amplitudes(random_amplitudes)
    return variable_bandwidth_signal, local_bandwidth


def generate_bounded_feb24_signal(bandwidth, lower_bound=-4, upper_bound=4, rng=None):
    """
    Random variable-bandwidth signal within (lower_bound, upper_bound); other draws are repeated.

    Args:
        bandwidth (float): Bandwidth of B(t) in Hz.
        lower_bound (float, optional): Lower amplitude bound. Defaults to -4.
        upper_bound (float, optional): Upper amplitude bound. Defaults to 4.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple: Signal at the output times t, and its ebsc.LocalBandwidth.
    """
    while True:
        signal, local_bandwidth = generate_feb24_signal(bandwidth, rng=rng)
        analog_signal = signal.to_analog_function()
        if (
            analog_signal.minimum() > lower_bound
            and analog_signal.maximum() < upper_bound
        ):
            break
    t = get_feb24_signal_time_vector()
    return signal(t), local_bandwidth


def generate_LCS_data(amplitudes, ref_levels):
    """
    Level-crossing samples of the piecewise linear interpolation of the signal samples at the output times.

    Args:
        amplitudes (np.ndarray): Signal at the output times t.
        ref_levels (np.ndarray): Reference levels.

    Returns:
        tuple: Crossing times and amplitudes (the crossed levels).
    """
    t = get_feb24_signal_time_vector()
    analog_signal = ebsc.AnalogSignal(t, amplitudes)
    level_crossing_sampler = ebsc.detectors.MultiLevelCrossingDetector(ref_levels, 0)
    events = level_crossing_sampler.get_events_list(analog_signal)
    samples = ebsc.data_types.lc_events_to_samples(events, ref_levels)
    return samples.get_t_as_array(), samples.get_a_as_array()


def get_data_batch(batch_size, bandwidth_of_local_bandwidth, ref_levels):
    """
    Batch of test signals with their samples of B(t) and level-crossing samples.

    Signal i is generated with np.random.default_rng(i), so the batch is reproducible.

    Args:
        batch_size (int): Number of signals.
        bandwidth_of_local_bandwidth (float): Bandwidth of B(t) in Hz.
        ref_levels (np.ndarray): Reference levels.

    Returns:
        tuple: t (NUM_OUTPUT_SAMPLES,), tB (NUM_BW_SAMPLES,), B (batch_size, NUM_BW_SAMPLES),
        X (batch_size, NUM_OUTPUT_SAMPLES), LCS zero padded (batch_size, NUM_LCS_PADDED, 2) and
        LCS_len (batch_size,), all float32 except LCS_len (int32).
    """
    t = get_feb24_signal_time_vector()
    tB = get_feb24_local_bandwidth_inference_time_vector()
    LCS = np.zeros((batch_size, tw_yen_config.NUM_LCS_PADDED, 2), dtype=np.float32)
    LCS_len = np.zeros(batch_size, dtype=np.int32)
    B = np.zeros((batch_size, tB.shape[0]), dtype=np.float32)
    X = np.zeros((batch_size, t.shape[0]), dtype=np.float32)

    for i in range(batch_size):
        rng = np.random.default_rng(i)
        signal_samples, local_bandwidth = generate_bounded_feb24_signal(
            bandwidth_of_local_bandwidth, rng=rng
        )
        t_lcs, x_lcs = generate_LCS_data(signal_samples, ref_levels)
        sequence_len = len(t_lcs)
        B[i, :] = local_bandwidth(tB)
        LCS[i, :sequence_len, 0] = t_lcs
        LCS[i, :sequence_len, 1] = x_lcs
        LCS_len[i] = sequence_len
        X[i, :] = signal_samples
    return (
        t.astype(np.float32),
        tB.astype(np.float32),
        B,
        X,
        LCS,
        LCS_len,
    )
