"""
NN-TW: estimation of the instantaneous bandwidth trained through the approximate time-warped Yen interpolator.

NNB estimates the samples of B(t) from the level-crossing samples. NNTWMixin adds the approximate
time-warped Yen interpolator of tw_yen after any network that outputs B(t), so the network is trained on the
reconstruction error and needs no samples of the true B(t). NNTW combines both.
"""

import tensorflow as tf
from functools import partial
from ebsc_tensorflow import tw_yen


def bounded_output(x, lower=0.001, upper=100.0):
    """
    Sigmoid output function scaled to the range (lower, upper).

    Args:
        x (tf.Tensor): Input.
        lower (float, optional): Lower bound. Defaults to 0.001.
        upper (float, optional): Upper bound. Defaults to 100.0.

    Returns:
        tf.Tensor: (upper - lower) * sigmoid(x) + lower.
    """
    scale = upper - lower
    return scale * tf.nn.sigmoid(x) + lower


def get_activation_fn(fn_str):
    """
    Returns the output function described by a string.

    "bounded_output(lower, upper)" gives bounded_output with these bounds; any other string is returned
    unchanged and interpreted by Keras, e.g. "linear".

    Args:
        fn_str (str): Name of the output function, e.g. "bounded_output(0.001, 150)".

    Returns:
        callable or str: Output function.
    """
    name, _, arg_str = fn_str.partition("(")

    if name.strip() != "bounded_output":
        return fn_str

    args = [float(a) for a in arg_str.rstrip(") ").split(",") if a.strip()]
    fn = partial(bounded_output, **dict(zip(("lower", "upper"), args)))
    fn.__name__ = fn_str
    return fn


class NNB(tf.keras.Model):
    """
    Encoder-decoder LSTM that estimates the samples of B(t) from the level-crossing samples.

    The bidirectional LSTM encoder reads the level-crossing samples; padded samples, whose time and
    amplitude both equal mask_val, are skipped. The concatenated final states of both directions initialize
    the decoder LSTM, which runs for out_shape[-1] steps on the repeated encoder output. A dense layer with
    the output function maps each decoder step to one sample of B(t).

    Args:
        in_shape (tuple): Input shape (NUM_LCS_PADDED, 2), stored only.
        out_shape (tuple): Output shape; out_shape[-1] is the number of samples of B(t).
        model_conf (dict): "units" (LSTM units of the encoder; the decoder has twice as many), "dropout"
            and "output_fn" (see get_activation_fn).
        mask_val (float, optional): Padding value of the level-crossing samples. Defaults to -0.0.
        name (str, optional): Model name. Defaults to "NNB".
    """

    def __init__(
        self, in_shape, out_shape, model_conf, mask_val=-0.0, name="NNB"
    ):
        super(NNB, self).__init__(name=name)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.mask_val = mask_val
        self.model_conf = model_conf
        self.encoder_masking_layer = tf.keras.layers.Masking(
            self.mask_val, name="encoder_masking"
        )
        self.enc_bi_lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.model_conf["units"],
                return_state=True,
                dropout=self.model_conf["dropout"],
            ),
            name="enc_bi_lstm",
        )

        self.state_c_concat_layer = tf.keras.layers.Concatenate(name="state_c_concat")
        self.state_h_concat_layer = tf.keras.layers.Concatenate(name="state_h_concat")

        self.repeat_layer = tf.keras.layers.RepeatVector(
            self.out_shape[-1], name="repeat"
        )

        self.dec_lstm_layer = tf.keras.layers.LSTM(
            self.model_conf["units"] * 2,
            name="dec_lstm",
            return_state=True,
            return_sequences=True,
            dropout=self.model_conf["dropout"],
        )

        self.dense_output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                1, activation=get_activation_fn(self.model_conf["output_fn"])
            ),
            name="dense_output",
        )

    def call(self, inputs, training=False):
        """
        Estimates the samples of B(t).

        Args:
            inputs (tf.Tensor): Level-crossing samples, columns time and amplitude, padded with mask_val,
                shape (batch, NUM_LCS_PADDED, 2).
            training (bool, optional): Training mode (dropout). Defaults to False.

        Returns:
            tf.Tensor: Estimated samples of B(t), shape (batch, out_shape[-1], 1).
        """
        x = self.encoder_masking_layer(inputs, training=training)
        x, forward_h, forward_c, backward_h, backward_c = self.enc_bi_lstm_layer(
            x, training=training
        )
        state_h = self.state_h_concat_layer([forward_h, backward_h], training=training)
        state_c = self.state_c_concat_layer([forward_c, backward_c], training=training)
        encoder_states = [state_h, state_c]

        x = self.repeat_layer(x, training=training)

        x, _, _ = self.dec_lstm_layer(
            x, initial_state=encoder_states, training=training
        )
        outputs = self.dense_output_layer(x, training=training)
        return outputs

    def build(self, input_shape):
        super().build(input_shape)


class NNTWMixin:
    """
    Mixin that adds the approximate time-warped Yen interpolator after a Keras model that estimates B(t).

    The model in front of the mixin (e.g. NNB) has to map the level-crossing samples to positive samples
    of B(t) on the grid tB, shape (batch, NUM_BW_SAMPLES, 1). The mixin reconstructs the signal from the
    level-crossing samples with this estimate, so training minimizes the reconstruction error. The
    Tikhonov regularization of the interpolator is trained together with the network weights; it is kept
    positive as softplus of an unconstrained variable.

    Inputs and outputs of call, fit and evaluate:
        inputs: dict with "LCS", the padded level-crossing samples, shape (BATCH_SIZE, NUM_LCS_PADDED, 2),
            and "LCS_len", the numbers of level-crossing samples, shape (BATCH_SIZE,).
        targets / outputs: signal at the output times t, shape (BATCH_SIZE, NUM_OUTPUT_SAMPLES).

    The reconstruction is compiled for batches of exactly BATCH_SIZE signals (tw_yen_config).

    Keyword Args:
        t (array-like): Output times, shape (NUM_OUTPUT_SAMPLES,).
        tB (array-like): Sample times of B(t), shape (NUM_BW_SAMPLES,).
        regularization (float): Initial Tikhonov regularization, positive.
        temperature (float, optional): Temperature of the basis weights. Defaults to 0.5.
        All other arguments are passed to the model.
    """

    def __init__(self, *args, **kwargs):
        self.t = kwargs.pop("t")
        self.tB = kwargs.pop("tB")
        regularization = kwargs.pop("regularization")
        self.temperature = kwargs.pop("temperature", 0.5)
        self.regularization_unconstrained = tf.Variable(
            float(tf.math.log(tf.math.expm1(regularization))),
            trainable=True,
            dtype=tf.float32,
            name="regularization",
        )

        super().__init__(*args, **kwargs, name="NNTW")

    @property
    def regularization(self):
        """tf.Tensor: Current Tikhonov regularization, softplus(regularization_unconstrained)."""
        return tf.nn.softplus(self.regularization_unconstrained)

    def train_step(self, data):
        """
        One training step: estimate B(t), reconstruct, and update the weights and the regularization.

        Args:
            data (tuple): (inputs, targets) or (inputs, targets, sample_weight), see the class docstring.

        Returns:
            dict: Loss (mean over the epoch so far) and compiled metrics.
        """
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        LCS = x["LCS"]
        LCS_len = x["LCS_len"]
        with tf.GradientTape() as tape:
            B_pred = super().call(LCS, training=True)
            B_pred = tf.squeeze(B_pred, axis=-1)
            tf.debugging.assert_all_finite(B_pred, "NaN in B_pred", name="train")

            y_pred = tw_yen.batch_approx_tw_yen(
                self.t,
                LCS,
                tf.cast(LCS_len, dtype=tf.int32),
                self.tB,
                B_pred,
                self.regularization,
                self.temperature,
            )

            loss = self.compute_loss(LCS, y, y_pred, sample_weight=sample_weight)

        # Compute gradients; the regularization is a tf.Variable and not one of the
        # model's trainable variables, so it is added here.
        trainable_vars = self.trainable_variables + [self.regularization_unconstrained]
        grads = tape.gradient(loss, trainable_vars)

        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Metrics; the loss tracker is updated here because compute_loss does not.
        self.compute_metrics(LCS, y, y_pred, sample_weight=sample_weight)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

    def test_step(self, data):
        """
        One evaluation step: estimate B(t), reconstruct, and update loss and metrics.

        Args:
            data (tuple): (inputs, targets) or (inputs, targets, sample_weight), see the class docstring.

        Returns:
            dict: Loss (mean over the evaluation so far) and compiled metrics.
        """
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        LCS = x["LCS"]
        LCS_len = x["LCS_len"]
        B_pred = super().call(LCS, training=False)
        B_pred = tf.squeeze(B_pred, axis=-1)
        y_pred = tw_yen.batch_approx_tw_yen(
            self.t,
            LCS,
            tf.cast(LCS_len, dtype=tf.int32),
            self.tB,
            B_pred,
            self.regularization,
            self.temperature,
        )

        loss = self.compute_loss(LCS, y, y_pred, sample_weight=sample_weight)
        self.compute_metrics(LCS, y, y_pred, sample_weight=sample_weight)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
        metrics = {m.name: m.result() for m in self.metrics}
        return metrics

    def call(self, inputs, training=False):
        """
        Estimates B(t) and reconstructs the signals at the output times t.

        Args:
            inputs (dict): "LCS" and "LCS_len", see the class docstring.
            training (bool, optional): Training mode (dropout). Defaults to False.

        Returns:
            tf.Tensor: Reconstructed signals, shape (BATCH_SIZE, NUM_OUTPUT_SAMPLES).
        """
        LCS = inputs["LCS"]
        LCS_len = inputs["LCS_len"]
        B_pred = super().call(LCS, training=training)
        B_pred = tf.squeeze(B_pred, axis=-1)
        reconstructed_signals = (
            tw_yen.batch_approx_tw_yen(
                self.t,
                LCS,
                tf.cast(LCS_len, dtype=tf.int32),
                self.tB,
                B_pred,
                self.regularization,
                self.temperature,
            )
        )

        return reconstructed_signals

    def predict_B(self, LCS, training=False):
        """
        Estimates the samples of B(t) without reconstruction; works for any batch size.

        Args:
            LCS (tf.Tensor): Level-crossing samples, padded, shape (batch, NUM_LCS_PADDED, 2).
            training (bool, optional): Training mode (dropout). Defaults to False.

        Returns:
            tf.Tensor: Estimated samples of B(t) in Hz on the grid tB, shape (batch, NUM_BW_SAMPLES).
        """
        B_pred = super().call(LCS, training=training)
        B_pred = tf.squeeze(B_pred, axis=-1)
        return B_pred


class NNTW(NNTWMixin, NNB):
    """
    NN-TW: NNB trained through the approximate time-warped Yen interpolator.

    Takes the arguments of NNB (in_shape, out_shape, model_conf, mask_val) and the keyword arguments of
    NNTWMixin (t, tB, regularization, temperature). The name is fixed to "NNTW".
    """

    pass
