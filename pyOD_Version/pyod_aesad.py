"""AE-SAD (AutoEncoder for Semi-supervised Anomaly Detection)
paper:
Reconstruction error-based anomaly detection with few outlying examples.
Fabrizio Angiulli, Fabio Fassetti, Luca Ferragina.
Neurocomputing (2026).
"""
# Author: Luca Ferragina <luca.ferragina@unical.it>
# adaptation to PyOD style

from typing import Callable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from pyod.models.base_dl import BaseDeepLearningDetector
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.utils.torch_utility import LinearBlock


def _get_transform_fn(
    transform: Union[str, Callable[[torch.Tensor], torch.Tensor]]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the transformation function F(x) used by AE-SAD.

    Supported presets (from the paper):
    - 'f0' or 'neg' : F0(x) = 1 - x (default)
    - 'f1'          : indicator-based mapping maximizing |F(x)-x|
    - 'f2'          : piecewise mapping with constant |F(x)-x| = 1/2
    """
    if callable(transform):
        return transform

    name = str(transform).lower()
    if name in ("f0", "neg", "negative", "inverse"):
        return lambda x: 1.0 - x

    if name == "f1":
        # F1(x) = 1_[0,1/2](x)  (paper notation)
        # i.e., 1 if x <= 0.5 else 0
        return lambda x: (x <= 0.5).to(x.dtype)

    if name == "f2":
        # F2(x) = (x+1/2)*1_[0,1/2](x) + (x-1/2)*1_(1/2,1](x)
        return lambda x: torch.where(x <= 0.5, x + 0.5, x - 0.5)

    raise ValueError(
        f"Unknown transform='{transform}'. Use one of: "
        f"['f0'/'neg', 'f1', 'f2'] or pass a callable."
    )


class AESAD(BaseDeepLearningDetector):
    """
    AE–SAD: Semi-supervised Anomaly Detection through Auto-Encoders.

    AE–SAD is a reconstruction error-based detector that leverages a small
    number of labeled anomalies during training by forcing anomalous samples
    to be reconstructed toward a transformation F(x), increasing their
    reconstruction error under the standard MSE score.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, used to define the
        threshold on the decision function.

    preprocessing : bool, optional (default=True)
        If True, apply the preprocessing procedure before training models.

    lr : float, optional (default=1e-3)
        The initial learning rate for the optimizer.

    epoch_num : int, optional (default=10)
        The number of epochs for training.

    batch_size : int, optional (default=32)
        The batch size for training.

    optimizer_name : str, optional (default='adam')
        The name of the optimizer used to train the model.

    device : str, optional (default=None)
        The device to use for the model.

    random_state : int, optional (default=42)
        The random seed for reproducibility.

    use_compile : bool, optional (default=False)
        Whether to compile the model (PyTorch >= 2.0.0, Python < 3.12).

    compile_mode : str, optional (default='default')
        Mode for torch.compile.

    verbose : int, optional (default=1)
        Verbosity mode:
        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

    optimizer_params : dict, optional (default={'weight_decay': 1e-5})
        Additional parameters for the optimizer.

    hidden_neuron_list : list, optional (default=[64, 32])
        Hidden layer sizes, symmetric encoder/decoder.

    hidden_activation_name : str, optional (default='relu')
        The activation function used in hidden layers.

    batch_norm : boolean, optional (default=True)
        Whether to apply Batch Normalization.

    dropout_rate : float in (0., 1), optional (default=0.2)
        Dropout rate across all layers.

    alpha : float, optional (default=1.0)
        Weight of anomalies (paper's alpha). Larger -> anomalies have more
        influence.

    transform : {'f0','neg','f1','f2'} or callable, optional (default='f0')
        Transformation function F(x).

    Notes
    -----
    - Training uses AE–SAD loss:
        (1-y)*MSE(x_hat,x) + y*(alpha/rho)*MSE(x_hat, F(x))
    - Scoring uses standard reconstruction error MSE(x_hat,x).
    """

    def __init__(self,
                 contamination=0.1, preprocessing=True,
                 lr=1e-3, epoch_num=10, batch_size=32,
                 optimizer_name='adam',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 optimizer_params: dict = {'weight_decay': 1e-5},
                 hidden_neuron_list=[64, 32],
                 hidden_activation_name='relu',
                 batch_norm=True, dropout_rate=0.2,
                 alpha: float = 1.0,
                 transform: Union[str, Callable[[torch.Tensor], torch.Tensor]] = 'f0'):
        super(AESAD, self).__init__(contamination=contamination,
                                   preprocessing=preprocessing,
                                   lr=lr, epoch_num=epoch_num,
                                   batch_size=batch_size,
                                   optimizer_name=optimizer_name,
                                   # We'll compute the custom AE-SAD loss manually
                                   # but keep 'mse' available if BaseDeepLearningDetector expects it.
                                   criterion_name='mse',
                                   device=device,
                                   random_state=random_state,
                                   use_compile=use_compile,
                                   compile_mode=compile_mode,
                                   verbose=verbose,
                                   optimizer_params=optimizer_params)

        self.hidden_neuron_list = hidden_neuron_list
        self.hidden_activation_name = hidden_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.alpha = float(alpha)
        self.transform = transform

        # Will be set during fit if y is provided
        self.rho_ = None  # anomaly ratio s/(n-s)
        self._F = _get_transform_fn(transform)

        # We use per-sample MSE (reduction='none') in AE-SAD loss
        self._mse_none = nn.MSELoss(reduction='none')

    def fit(self, X, y=None):
        """Fit detector.

        y is optional but, when provided, it should be 0 for inliers and 1 for
        anomalies (semi-supervised setting).
        """
        if y is not None:
            # Ensure y is 1D and compute rho = s/(n-s)
            y_arr = torch.as_tensor(y).view(-1)
            # We treat any non-zero as anomaly label
            s = int(torch.count_nonzero(y_arr).item())
            n = int(y_arr.numel())
            n_in = n - s

            # If no anomalies are provided, AE-SAD degenerates to standard AE
            if s <= 0 or n_in <= 0:
                # Keep rho_ defined, but training will behave as AE
                self.rho_ = 0.0
            else:
                self.rho_ = float(s) / float(n_in)

        return super(AESAD, self).fit(X, y=y)

    def build_model(self):
        self.model = AutoEncoderModel(
            self.feature_size,
            hidden_neuron_list=self.hidden_neuron_list,
            hidden_activation_name=self.hidden_activation_name,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate)

    def _ae_sad_loss(self, x: torch.Tensor, x_recon: torch.Tensor, y: torch.Tensor):
        """Compute AE-SAD loss for a batch.

        y must be float tensor in {0,1}.
        """
        # Per-sample reconstruction MSE (mean over features)
        # shape: (batch, feature) -> (batch,)
        mse_x = self._mse_none(x_recon, x).mean(dim=1)

        # Transform target for anomalies
        x_t = self._F(x)
        mse_fx = self._mse_none(x_recon, x_t).mean(dim=1)

        # rho may be None (no y passed to fit) or 0 (no anomalies in y)
        rho = self.rho_
        if rho is None or rho <= 0.0:
            # behave like standard AE
            return mse_x.mean()

        # AE-SAD weighting alpha/rho
        w = self.alpha / rho

        # (1-y)*mse(x) + y*w*mse(F(x))
        loss_vec = (1.0 - y) * mse_x + y * (w * mse_fx)
        return loss_vec.mean()

    def training_forward(self, batch_data):
        # Support both unsupervised batches (x) and semi-supervised batches (x,y)
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            x, y = batch_data
        else:
            x = batch_data
            y = None

        x = x.to(self.device)
        self.optimizer.zero_grad()

        x_recon = self.model(x)

        if y is None:
            # No labels available -> standard AE training
            loss = self.criterion(x_recon, x)
        else:
            y = y.to(self.device).view(-1).float()
            loss = self._ae_sad_loss(x, x_recon, y)

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def evaluating_forward(self, batch_data):
        # Score is ALWAYS standard reconstruction error ||x - x_hat||^2
        # (PyOD uses pairwise_distances_no_broadcast here)
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            x, _ = batch_data
        else:
            x = batch_data

        x_gpu = x.to(self.device)
        x_recon = self.model(x_gpu)
        score = pairwise_distances_no_broadcast(x.numpy(),
                                                x_recon.detach().cpu().numpy())
        return score


class AutoEncoderModel(nn.Module):
    def __init__(self,
                 feature_size,
                 hidden_neuron_list=[64, 32],
                 hidden_activation_name='relu',
                 batch_norm=True, dropout_rate=0.2):
        super(AutoEncoderModel, self).__init__()

        self.feature_size = feature_size
        self.hidden_neuron_list = hidden_neuron_list
        self.hidden_activation_name = hidden_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder_layers = []
        last_neuron_size = self.feature_size
        for neuron_size in self.hidden_neuron_list:
            encoder_layers.append(LinearBlock(
                last_neuron_size, neuron_size,
                activation_name=self.hidden_activation_name,
                batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate))
            last_neuron_size = neuron_size
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self):
        decoder_layers = []
        last_neuron_size = self.hidden_neuron_list[-1]
        for neuron_size in reversed(self.hidden_neuron_list[:-1]):
            decoder_layers.append(LinearBlock(
                last_neuron_size, neuron_size,
                activation_name=self.hidden_activation_name,
                batch_norm=self.batch_norm,
                dropout_rate=self.dropout_rate))
            last_neuron_size = neuron_size
        decoder_layers.append(nn.Linear(last_neuron_size, self.feature_size))
        return nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
