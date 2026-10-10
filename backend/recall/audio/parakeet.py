import numpy as np


class ParakeetTranscriber:
    """NVIDIA Parakeet TDT 0.6B via onnx-asr on CPU. The model downloads from Hugging Face on first use.

    `nemo-parakeet-tdt-0.6b-v2` is English-only; `-v3` adds 24 European languages (no Indian ones).
    """

    def __init__(self, model_name: str, quantization: str | None = "int8", model=None):
        if model is None:
            try:
                import onnx_asr
            except ImportError as exc:
                raise RuntimeError("onnx-asr is not installed. Install the backend's [parakeet] extra.") from exc
            model = onnx_asr.load_model(model_name, quantization=quantization, providers=["CPUExecutionProvider"])
        self._model = model

    def transcribe(self, audio: np.ndarray) -> str:
        return self._model.recognize(audio.astype(np.float32, copy=False), sample_rate=16_000).strip()
