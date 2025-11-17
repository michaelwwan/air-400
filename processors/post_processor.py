"""Signal post-processing utilities for respiration estimation outputs."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import scipy.signal as signal
from scipy.sparse import spdiags
from torch import Tensor


class PostProcessor:
    """Apply filtering and spectral analysis to respiration predictions."""

    def __init__(self) -> None:
        """Initialize the logger used for optional diagnostics."""
        self.logger = logging.getLogger(__name__)

    def post_process(
        self,
        sig: Tensor,
        fs: int = 30,
        diff_flag: bool = True,
        infant_flag: bool = False,
        use_bandpass: bool = True,
        eval_method: str = "FFT",
    ) -> Tuple[np.ndarray, float]:
        """Return the filtered signal and respiration rate estimate.

        Args:
            sig: Raw model prediction sequence as a torch tensor.
            fs: Sampling frequency (frames per second).
            diff_flag: Whether predictions/labels represent 1st temporal derivative of signal.
            infant_flag: Whether to use infant-specific frequency bands.
            use_bandpass: Flag controlling the band-pass filter stage.
            eval_method: Either ``"FFT"`` or ``"Peak"`` to select RR method.

        Returns:
            Tuple containing the processed signal (``np.ndarray``) and the
            respiration rate in breaths per minute.

        """
        processed_sig = sig.detach().cpu().numpy()

        if diff_flag:
            processed_sig = self._detrend_signal(np.cumsum(processed_sig), 100)
        else:
            processed_sig = self._detrend_signal(processed_sig, 100)

        if infant_flag:
            low, high = 0.3, 1.0  # infants (18-60 bpm)
        else:
            low, high = 0.08, 0.5  # adults (5-30 bpm)

        if use_bandpass:
            processed_sig = self._bandpass_filter(processed_sig, fs, low, high)

        if eval_method == "FFT":
            sig_rr = self._calculate_fft_rr(processed_sig, fs, low, high)
        elif eval_method == "Peak":
            sig_rr = self._calculate_peak_rr(processed_sig, fs)
        else:
            self.logger.error("Please use FFT or Peak method to calculate RR.")
            raise ValueError("Please use FFT or Peak method to calculate RR.")

        return processed_sig, sig_rr

    @staticmethod
    def _detrend_signal(input_signal: np.ndarray, lambda_value: int = 100) -> np.ndarray:
        """Remove low-frequency trends using a smoothing prior approach."""
        signal_length = input_signal.shape[0]
        # Handle too short input signal
        if signal_length < 3:
            return input_signal
        # observation matrix
        H = np.eye(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        D = spdiags(diags_data, [0, 1, 2], signal_length - 2, signal_length).toarray()
        smoothing = lambda_value**2 * np.dot(D.T, D)
        return np.dot(H - np.linalg.inv(H + smoothing), input_signal)

    @staticmethod
    def _bandpass_filter(
        signal_input: np.ndarray,
        fs: int,
        low: float,
        high: float,
        order: int = 1,
    ) -> np.ndarray:
        """Run a Butterworth band-pass filter over the signal."""
        flattened = np.asarray(signal_input).flatten()
        # Handle too short input signal
        if flattened.ndim == 0 or len(flattened) <= 9:
            return flattened

        [b, a] = signal.butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="bandpass")
        return signal.filtfilt(b, a, np.double(flattened))

    @staticmethod
    def _calculate_fft_rr(
        signal_input: np.ndarray,
        fs: int = 60,
        low_pass: float = 0.75,
        high_pass: float = 2.5,
    ) -> float:
        """Estimate RR using the dominant Fast Fourier Transform (FFT) bin within a passband."""
        ppg_signal = np.expand_dims(signal_input, axis=0)
        n = 1 if ppg_signal.shape[1] == 0 else 2 ** (ppg_signal.shape[1] - 1).bit_length()
        f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=n, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        # Handle empty mask
        if len(mask_pxx) == 0 or len(mask_ppg) == 0:
            return 0.0
        return np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60  # bpm

    @staticmethod
    def _calculate_peak_rr(signal_input: np.ndarray, fs: int) -> float:
        """Estimate RR by counting peaks in the time domain."""
        peaks, _ = signal.find_peaks(signal_input)
        return 60 / (np.mean(np.diff(peaks)) / fs)
