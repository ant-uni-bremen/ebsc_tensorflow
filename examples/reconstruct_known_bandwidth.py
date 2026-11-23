"""
Reconstruct signals from their level-crossing samples with the approximate time-warped Yen interpolator and the
true bandwidth.

Generates BATCH_SIZE test signals (bandwidth of B(t) 2 Hz, 13 levels in [-4, 4]), reconstructs them with
the samples of the true B(t), prints the median NMSE and plots one signal with its reconstruction, its
level-crossing samples and B(t).
"""

import matplotlib.pyplot as plt
import numpy as np

from ebsc_tensorflow import signals, tw_yen, tw_yen_config

BANDWIDTH_OF_LOCAL_BANDWIDTH = 2  # Hz
REF_LEVELS = np.linspace(-4, 4, 13)
REGULARIZATION = 1e-3
TEMPERATURE = 0.5
PLOT_INDEX = 0

t, tB, B, X, LCS, LCS_len = signals.get_data_batch(
    tw_yen_config.BATCH_SIZE, BANDWIDTH_OF_LOCAL_BANDWIDTH, REF_LEVELS
)
X_hat = tw_yen.batch_approx_tw_yen(
    t, LCS, LCS_len, tB, B, REGULARIZATION, TEMPERATURE
).numpy()

nmse_db = 10 * np.log10(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X**2, axis=1))
print(f"median NMSE {np.median(nmse_db):.1f} dB over {len(nmse_db)} signals")

n = LCS_len[PLOT_INDEX]
fig, (ax_x, ax_b) = plt.subplots(2, 1, sharex=True)
ax_x.plot(t, X[PLOT_INDEX], label="signal")
ax_x.plot(t, X_hat[PLOT_INDEX], "--", label="reconstruction")
ax_x.plot(LCS[PLOT_INDEX, :n, 0], LCS[PLOT_INDEX, :n, 1], "k.", label="level-crossing samples")
ax_x.set_ylabel("amplitude")
ax_x.legend()
ax_b.plot(tB, B[PLOT_INDEX])
ax_b.set_xlabel("time in s")
ax_b.set_ylabel("bandwidth in Hz")
plt.show()
