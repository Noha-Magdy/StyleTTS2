from typing import Optional, Union
from transformers.utils import TensorType, logging
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
import types

import torch
from transformers import WhisperProcessor, WhisperModel
from transformers.audio_utils import mel_filter_bank

class DifferentiableWhisperFeatureExtractor(WhisperFeatureExtractor):
  def __init__(self, wfe:WhisperFeatureExtractor):
    self.hop_length=wfe.hop_length
    self.n_fft=wfe.n_fft
    self.dither=wfe.dither
    self.sampling_rate=wfe.sampling_rate
    self.feature_size=wfe.feature_size
    self.mel_filters = wfe.mel_filters

  def _hf_differentiable_extract_fbank_features(self, waveform, device=None):
    hop_length=self.hop_length
    n_fft=self.n_fft
    dither=self.dither
    sampling_rate=self.sampling_rate
    feature_size=self.feature_size
    device = waveform.device

    mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)



    window = torch.hann_window(n_fft, device=device)


    if dither != 0.0:
        waveform += dither * torch.randn(waveform.shape, dtype=waveform.dtype, device=waveform.device)

    print("stft.hop_length", hop_length)
    print("stft.n_fft", n_fft)

    # print("stft.window", window)
    print("stft.waveform", waveform.shape)
    stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2 # WTF???
    # mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
    mel_filters = mel_filters.to(device, torch.float32)
    # fix to Batch (maybe no need to edit...but double check, anyway)
    mel_spec = mel_filters.T @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()

    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec




  def __call__(
      self,
      raw_speech: torch.Tensor,
      truncation: bool = True,
      pad_to_multiple_of: Optional[int] = None,
      return_tensors: Optional[Union[str, TensorType]] = None,
      return_attention_mask: Optional[bool] = None,
      padding: Optional[str] = "max_length",
      max_length: Optional[int] = None,
      sampling_rate: Optional[int] = None,
      do_normalize: Optional[bool] = None,
      device: Optional[str] = "cpu",
      return_token_timestamps: Optional[bool] = None,
      **kwargs,
  ) -> BatchFeature:
      """
      New diff
      Main method to featurize and prepare for the model one or several sequence(s). Implementation uses PyTorch for
      the STFT computation if available, otherwise a slower NumPy based one.

      Args:
          raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
              The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
              values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
              stereo, i.e. single float per timestep.
          truncation (`bool`, *optional*, default to `True`):
              Activates truncation to cut input sequences longer than *max_length* to *max_length*.
          pad_to_multiple_of (`int`, *optional*, defaults to None):
              If set will pad the sequence to a multiple of the provided value.

              This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
              `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
          return_attention_mask (`bool`, *optional*):
              Whether to return the attention mask. If left to the default, will return the attention mask according
              to the specific feature_extractor's default.

              [What are attention masks?](../glossary#attention-mask)

              <Tip>

              For Whisper models, `attention_mask` should always be passed for batched inference, to avoid subtle
              bugs.

              </Tip>

          return_tensors (`str` or [`~utils.TensorType`], *optional*):
              If set, will return tensors instead of list of python integers. Acceptable values are:

              - `'tf'`: Return TensorFlow `tf.constant` objects.
              - `'pt'`: Return PyTorch `torch.Tensor` objects.
              - `'np'`: Return Numpy `np.ndarray` objects.
          sampling_rate (`int`, *optional*):
              The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
              `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
              pipeline.
          padding_value (`float`, *optional*, defaults to 0.0):
              The value that is used to fill the padding values / vectors.
          do_normalize (`bool`, *optional*, defaults to `False`):
              Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
              improve the performance of the model.
          device (`str`, *optional*, defaults to `'cpu'`):
              Specifies the device for computation of the log-mel spectrogram of audio signals in the
              `_torch_extract_fbank_features` method. (e.g., "cpu", "cuda")
          return_token_timestamps (`bool`, *optional*, defaults to `None`):
              Whether or not to return the number of frames of the input raw_speech.
              These num_frames can be used by the model to compute word level timestamps.
      """
      # if sampling_rate is not None:
      #     if sampling_rate != self.sampling_rate:
      #         raise ValueError(
      #             f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
      #             f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
      #             f" was sampled with {self.sampling_rate} and not {sampling_rate}."
      #         )
      # else:
      #     logger.warning(
      #         f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
      #         "Failing to do so can result in silent errors that might be hard to debug."
      #     )

      print("self", type(self))

      is_batched = True # Only batched data is supported. Batch is index 0. It is assumed that they are padded as well





      # zero-mean and unit-variance normalization. As this is the output of the gen model it is not needed
      # if do_normalize:
      #     padded_inputs["input_features"] = self.zero_mean_unit_var_norm(
      #         padded_inputs["input_features"],
      #         attention_mask=padded_inputs["attention_mask"],
      #         padding_value=self.padding_value,
      #     )
      #     padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], axis=0)

      # # make sure list is in array format
      # input_features = padded_inputs.get("input_features").transpose(2, 0, 1) # WHY? IDK

      input_features = self._hf_differentiable_extract_fbank_features(raw_speech, device)

      # if isinstance(input_features[0], List):
      #     padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]

      # else:
      #     padded_inputs["input_features"] = input_features

      # if return_attention_mask:
      #     # rescale from sample (48000) to feature (3000)
      #     padded_inputs["attention_mask"] = padded_inputs["attention_mask"][:, :: self.hop_length]

      # if return_token_timestamps is not None:
      #     padded_inputs["num_frames"] = [len(raw_speech_i) // self.hop_length for raw_speech_i in raw_speech]

      res = BatchFeature({
          "input_features": input_features
      })

      return res


