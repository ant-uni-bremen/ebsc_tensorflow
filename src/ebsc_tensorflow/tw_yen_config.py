"""
Fixed tensor sizes of the approximate time-warped Yen interpolator in tw_yen.

The functions in tw_yen are compiled with these sizes. Inputs of other sizes require changing the values
here before tw_yen is imported.
"""

# Number of output samples, i.e. the length of t. The basis functions are padded to the same number, so it
# also has to be at least floor(gamma(T)) + 1 (at most 301 for B(t) <= 150 Hz and T = 1 s).
NUM_OUTPUT_SAMPLES = 400
# Number of samples of B(t), i.e. the length of tB.
NUM_BW_SAMPLES = 43
# Length to which the level-crossing samples are padded.
NUM_LCS_PADDED = 512
# Number of signals per batch; every batch has to have this size (drop_remainder=True for tf.data).
BATCH_SIZE = 64
