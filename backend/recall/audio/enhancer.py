"""Optional DeepFilterNet speech enhancement for utterances sent to the STT provider."""

from collections.abc import Callable

import numpy as np


class DeepFilterNetEnhancer:
    """Lazy-load DeepFilterNet and resample the app's 16 kHz audio to its model rate."""

    def __init__(self):
        try:
            import torch
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise RuntimeError("Install the optional 'denoise' dependencies to use DeepFilterNet") from exc
        self._model, self._state, _ = init_df()
        self._enhance: Callable = enhance
        self._torch = torch
        self._model_rate = int(self._state.sr())

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            raise ValueError("DeepFilterNet adapter currently expects 16 kHz app audio")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        model_audio = _resample(samples, sample_rate, self._model_rate)
        tensor = self._torch.from_numpy(model_audio).unsqueeze(0)
        enhanced = self._enhance(self._model, self._state, tensor)
        if hasattr(enhanced, "detach"):
            enhanced = enhanced.squeeze(0).detach().cpu().numpy()
        return _resample(np.asarray(enhanced, dtype=np.float32).reshape(-1), self._model_rate, sample_rate)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.copy()
    target_size = round(audio.size * target_rate / source_rate)
    old_positions = np.arange(audio.size, dtype=np.float64)
    new_positions = np.arange(target_size, dtype=np.float64) * source_rate / target_rate
    return np.interp(new_positions, old_positions, audio).astype(np.float32)
