import logging

import numpy as np
import scipy.signal as signal
from scipy.sparse import spdiags


class PostProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def post_process(self, sig, fs=30, diff_flag=True, infant_flag=False, use_bandpass=True, eval_method='FFT'):
        """Calculate respiration rate for PPG signal of each video."""
        # Numpy only works on CPU array, so detach and move from GPU tensors
        sig = sig.detach().cpu().numpy()

        if diff_flag:  # if the predictions/labels are 1st derivative of PPG signal.
            sig = self._detrend_signal(np.cumsum(sig), 100)
        else:
            sig = self._detrend_signal(sig, 100)

        if infant_flag:
            low, high = 0.3, 1.0  # infants (18-60 bpm)
        else:
            low, high = 0.08, 0.5  # adults (5-30 bpm)
        if use_bandpass:
            sig = self._bandpass_filter(sig, fs, low, high)

        if eval_method == "FFT":
            sig_rr = self._calculate_fft_rr(sig, fs, low, high)
        elif eval_method == "Peak":
            sig_rr = self._calculate_peak_rr(sig, fs)
        else:
            self.logger.error("Please use FFT or Peak method to calculate RR.")
            raise ValueError("Please use FFT or Peak method to calculate RR.")

        return sig, sig_rr

    @staticmethod
    def _detrend_signal(input_signal, lambda_value=100):
        signal_length = input_signal.shape[0]
        # Handle too short input signal
        if signal_length < 3:
            return input_signal
        # observation matrix
        H = np.eye(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        D = spdiags(diags_data, [0, 1, 2], signal_length-2, signal_length).toarray()
        return np.dot(H - np.linalg.inv(H + (lambda_value**2) * np.dot(D.T, D)), input_signal)

    @staticmethod
    def _bandpass_filter(signal_input, fs, low, high, order=1):
        signal_input = np.asarray(signal_input).flatten()
        # Handle too short input signal
        if signal_input.ndim == 0 or len(signal_input) <= 9:
            return signal_input
        [b, a] = signal.butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='bandpass')
        return signal.filtfilt(b, a, np.double(signal_input))

    @staticmethod
    def _calculate_fft_rr(signal_input, fs=60, low_pass=0.75, high_pass=2.5):
        """Calculate respiration rate based on PPG using Fast Fourier transform (FFT)."""
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
    def _calculate_peak_rr(signal_input, fs):
        """Calculate respiration rate based on PPG using peak detection."""
        peaks, _ = signal.find_peaks(signal_input)
        return 60 / (np.mean(np.diff(peaks)) / fs)

