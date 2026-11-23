# ebsc_tensorflow

[![DOI](https://zenodo.org/badge/1462651911.svg)](https://doi.org/10.5281/zenodo.24459230)

TensorFlow counterpart of [ebsc](https://github.com/ant-uni-bremen/ebsc): differentiable time-warping and
time-warped Yen interpolator for event-based sampling.

The package reconstructs a signal from its level-crossing samples with an approximate version of the
time-warped Yen interpolator [1], which combines time-warping [2] with the Yen interpolator [3]. The
interpolator is differentiable with respect to the instantaneous bandwidth B(t), so a neural network that
estimates B(t) can be trained on the reconstruction error. The package contains this network, NN-TW, as
described in the dissertation (see [Citation](#citation)).

## Installation

Python 3.10 to 3.13 (tested with 3.12). ebsc is installed from GitHub, so git has to be available.

Package only:

```bash
pip install .
```

With the dependencies of the tests or the examples (ebsc, pytest, scipy, matplotlib):

```bash
pip install ".[test,examples]"
```

For development:

```bash
pip install --editable ".[test,examples]"
```

## Modules

| Module | Content |
|---|---|
| `tw_yen` | Approximate time-warped Yen interpolator (`approx_tw_yen`, `batch_approx_tw_yen`) and the time-warping function γ(t) (`calc_gamma`, `evaluate_gamma`, `calc_tau_lcs`, `calc_gamma_at_t`) |
| `tw_yen_config` | Fixed tensor sizes of the interpolator |
| `fast_poly` | Integration and evaluation of piecewise polynomials, used for γ(t) |
| `nntw` | `NNB` (encoder-decoder LSTM that estimates B(t) [4]), `NNTWMixin` (adds the interpolator to any network that outputs B(t)) and `NNTW` (both combined) |
| `signals` | Variable-bandwidth test signals and their level-crossing samples, generated with ebsc; used by the tests and the examples and not imported by the package itself |

## Approximate time-warped Yen interpolator

B(t) is given by samples on a time grid and interpolated linearly, so γ(t) = 2∫B(t)dt is piecewise
quadratic and differentiable with respect to the samples of B(t). The interpolator solves the
Tikhonov-regularized least-squares problem of the time-warped Yen interpolator for the coefficients of the
basis functions sinc(γ(t) − k).

The number of basis functions follows from γ(T) and changes only in integer steps. To keep the
reconstruction differentiable, all padded basis functions are kept and weighted by

    w_k = 1 / (1 + exp(-(γ(T) - k) / T_σ)).

For T_σ → 0 the weights select the basis functions of the original interpolator. NN-TW uses T_σ = 0.5
(`temperature`).

```python
import numpy as np
from ebsc_tensorflow import tw_yen

# t: output times (NUM_OUTPUT_SAMPLES,), tB: grid of B(t) (NUM_BW_SAMPLES,)
# LCS: level-crossing samples (BATCH_SIZE, NUM_LCS_PADDED, 2), columns time and amplitude, zero padded
# LCS_len: number of samples per signal (BATCH_SIZE,), B: bandwidth in Hz on tB (BATCH_SIZE, NUM_BW_SAMPLES)
X_hat = tw_yen.batch_approx_tw_yen(t, LCS, LCS_len, tB, B, 1e-3, 0.5)
```

### Fixed tensor sizes

The interpolator is compiled for fixed tensor sizes, set in `tw_yen_config.py`:

| Constant | Value | Meaning |
|---|---|---|
| `NUM_OUTPUT_SAMPLES` | 400 | output samples, also the number of basis functions |
| `NUM_BW_SAMPLES` | 43 | samples of B(t) |
| `NUM_LCS_PADDED` | 512 | padded number of level-crossing samples |
| `BATCH_SIZE` | 64 | signals per batch |

Other sizes require editing `tw_yen_config.py`. Every batch has to contain exactly `BATCH_SIZE` signals, so
use `drop_remainder=True` when batching a `tf.data.Dataset`.

## NN-TW

NN-TW estimates B(t) from the level-crossing samples with the encoder-decoder `NNB` and reconstructs the
signal with the approximate interpolator. It is trained on the reconstruction error, so it needs no samples
of the true B(t). The regularization of the interpolator is a trainable parameter.

The model of the dissertation used:

| Parameter | Value |
|---|---|
| LSTM units | 128 |
| Dropout | 0.2 |
| Output function | `bounded_output(0.001, 150)` (sigmoid scaled to 0.001–150 Hz) |
| Optimizer | Adam, learning rate 3.75e-4, clipnorm 66.9 |
| Batch size | 64 |
| T_σ | 0.5 |
| Initial regularization | 1e-4 |

```python
import tensorflow as tf
from ebsc_tensorflow import nntw, tw_yen_config

model = nntw.NNTW(
    in_shape=(tw_yen_config.NUM_LCS_PADDED, 2),
    out_shape=(tw_yen_config.NUM_BW_SAMPLES,),
    model_conf={"units": 128, "dropout": 0.2, "output_fn": "bounded_output(0.001, 150)"},
    t=t,
    tB=tB,
    regularization=1e-4,
    temperature=0.5,
)
model.compile(optimizer=tf.keras.optimizers.Adam(3.75e-4, clipnorm=66.9), loss="mse")
model.fit(train_dataset, validation_data=val_dataset, epochs=100)

X_hat = model({"LCS": LCS, "LCS_len": LCS_len})  # reconstruction
B_hat = model.predict_B(LCS)                     # estimated bandwidth
```

## Examples

- `reconstruct_known_bandwidth.py`: reconstructs 64 signals with the approximate interpolator and the true
  B(t), prints the median NMSE and plots one signal.

## Tests

```bash
pytest tests
```

The tests compare γ(t) with `ebsc.LocalBandwidth`, check the reconstruction with the true B(t) and the
gradients, and run one training step of NN-TW.

## Citation

If you use this package, please cite the archived release on Zenodo. GitHub shows the citation under
"Cite this repository", generated from `CITATION.cff`.

```bibtex
@software{koenigs_ebsc_tensorflow,
  author  = {Königs, Johannes},
  title   = {ebsc\_tensorflow: A Differentiable Time-Warped Yen Interpolator},
  version = {1.0.0},
  year    = {2026},
  doi     = {10.5281/zenodo.24459230},
  url     = {https://github.com/ant-uni-bremen/ebsc_tensorflow}
}
```

NN-TW and the approximate interpolator are described in detail in the author's dissertation, which is
in preparation.

## References

[1] D. Rzepka, M. Pawlak, D. Kościelnik and M. Miśkowicz, "Reconstruction of varying bandwidth signals
from event-triggered samples," in *Event-Based Control and Signal Processing*, M. Miśkowicz, Ed. Boca Raton,
FL, USA: CRC Press, 2015, ch. 23, pp. 529–545.

[2] J. Clark, M. Palmer and P. Lawrence, "A transformation method for the reconstruction of functions from
nonuniformly spaced samples," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 33, no. 5,
pp. 1151–1165, 1985, doi: [10.1109/TASSP.1985.1164714](https://doi.org/10.1109/TASSP.1985.1164714).

[3] J. L. Yen, "On nonuniform sampling of bandwidth-limited signals," *IRE Trans. Circuit Theory*, vol. 3,
no. 4, pp. 251–257, 1956, doi: [10.1109/TCT.1956.1086325](https://doi.org/10.1109/TCT.1956.1086325).

[4] J. Königs, C. Bockelmann and A. Dekorsy, "Instantaneous bandwidth estimation from level-crossing
samples via LSTM-based encoder-decoder architecture," in *2025 Signal Processing Symposium (SPSympo)*,
Warsaw, Poland, 2025, pp. 78–83, doi:
[10.23919/SPSympo63739.2025.11123995](https://doi.org/10.23919/SPSympo63739.2025.11123995).

## License

GPL-3.0, see `LICENSE`.
