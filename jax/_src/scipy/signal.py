# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import scipy.signal as osp_signal
import operator
import warnings

import numpy as np

import jax
import jax.numpy.fft
from jax import lax
from jax._src.numpy.lax_numpy import _check_arraylike
from jax._src.numpy import lax_numpy as jnp
from jax._src.numpy import linalg
from jax._src.numpy.lax_numpy import _promote_dtypes_inexact
from jax._src.numpy.util import _wraps
from jax._src.third_party.scipy import signal_helper
from jax._src.util import canonicalize_axis, tuple_delete, tuple_insert


# Note: we do not re-use the code from jax.numpy.convolve here, because the handling
# of padding differs slightly between the two implementations (particularly for
# mode='same').
def _convolve_nd(in1, in2, mode, *, precision):
  if mode not in ["full", "same", "valid"]:
    raise ValueError("mode must be one of ['full', 'same', 'valid']")
  if in1.ndim != in2.ndim:
    raise ValueError("in1 and in2 must have the same number of dimensions")
  if in1.size == 0 or in2.size == 0:
    raise ValueError(f"zero-size arrays not supported in convolutions, got shapes {in1.shape} and {in2.shape}.")
  in1, in2 = _promote_dtypes_inexact(in1, in2)

  no_swap = all(s1 >= s2 for s1, s2 in zip(in1.shape, in2.shape))
  swap = all(s1 <= s2 for s1, s2 in zip(in1.shape, in2.shape))
  if not (no_swap or swap):
    raise ValueError("One input must be smaller than the other in every dimension.")

  shape_o = in2.shape
  if swap:
    in1, in2 = in2, in1
  shape = in2.shape
  in2 = jnp.flip(in2)

  if mode == 'valid':
    padding = [(0, 0) for s in shape]
  elif mode == 'same':
    padding = [(s - 1 - (s_o - 1) // 2, s - s_o + (s_o - 1) // 2)
               for (s, s_o) in zip(shape, shape_o)]
  elif mode == 'full':
    padding = [(s - 1, s - 1) for s in shape]

  strides = tuple(1 for s in shape)
  result = lax.conv_general_dilated(in1[None, None], in2[None, None], strides,
                                    padding, precision=precision)
  return result[0, 0]


@_wraps(osp_signal.convolve)
def convolve(in1, in2, mode='full', method='auto',
             precision=None):
  if method != 'auto':
    warnings.warn("convolve() ignores method argument")
  return _convolve_nd(in1, in2, mode, precision=precision)


@_wraps(osp_signal.convolve2d)
def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0,
               precision=None):
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("convolve2d() only supports boundary='fill', fillvalue=0")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError("convolve2d() only supports 2-dimensional inputs.")
  return _convolve_nd(in1, in2, mode, precision=precision)


@_wraps(osp_signal.correlate)
def correlate(in1, in2, mode='full', method='auto',
              precision=None):
  if method != 'auto':
    warnings.warn("correlate() ignores method argument")
  return _convolve_nd(in1, jnp.flip(in2.conj()), mode, precision=precision)


@_wraps(osp_signal.correlate2d)
def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0,
                precision=None):
  if boundary != 'fill' or fillvalue != 0:
    raise NotImplementedError("correlate2d() only supports boundary='fill', fillvalue=0")
  if jnp.ndim(in1) != 2 or jnp.ndim(in2) != 2:
    raise ValueError("correlate2d() only supports 2-dimensional inputs.")

  swap = all(s1 <= s2 for s1, s2 in zip(in1.shape, in2.shape))
  same_shape =  all(s1 == s2 for s1, s2 in zip(in1.shape, in2.shape))

  if mode == "same":
    in1, in2 = jnp.flip(in1), in2.conj()
    result = jnp.flip(_convolve_nd(in1, in2, mode, precision=precision))
  elif mode == "valid":
    if swap and not same_shape:
      in1, in2 = jnp.flip(in2), in1.conj()
      result = _convolve_nd(in1, in2, mode, precision=precision)
    else:
      in1, in2 = jnp.flip(in1), in2.conj()
      result = jnp.flip(_convolve_nd(in1, in2, mode, precision=precision))
  else:
    if swap:
      in1, in2 = jnp.flip(in2), in1.conj()
      result = _convolve_nd(in1, in2, mode, precision=precision).conj()
    else:
      in1, in2 = jnp.flip(in1), in2.conj()
      result = jnp.flip(_convolve_nd(in1, in2, mode, precision=precision))
  return result


@_wraps(osp_signal.detrend)
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=None):
  if overwrite_data is not None:
    raise NotImplementedError("overwrite_data argument not implemented.")
  if type not in ['constant', 'linear']:
    raise ValueError("Trend type must be 'linear' or 'constant'.")
  data, = _promote_dtypes_inexact(jnp.asarray(data))
  if type == 'constant':
    return data - data.mean(axis, keepdims=True)
  else:
    N = data.shape[axis]
    # bp is static, so we use np operations to avoid pushing to device.
    bp = np.sort(np.unique(np.r_[0, bp, N]))
    if bp[0] < 0 or bp[-1] > N:
      raise ValueError("Breakpoints must be non-negative and less than length of data along given axis.")
    data = jnp.moveaxis(data, axis, 0)
    shape = data.shape
    data = data.reshape(N, -1)
    for m in range(len(bp) - 1):
      Npts = bp[m + 1] - bp[m]
      A = jnp.vstack([
        jnp.ones(Npts, dtype=data.dtype),
        jnp.arange(1, Npts + 1, dtype=data.dtype) / Npts
      ]).T
      sl = slice(bp[m], bp[m + 1])
      coef, *_ = linalg.lstsq(A, data[sl])
      data = data.at[sl].add(-jnp.matmul(A, coef, precision=lax.Precision.HIGHEST))
    return jnp.moveaxis(data.reshape(shape), 0, axis)


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
  """Calculate windowed FFT in the same way the original SciPy does.
  """
  if x.dtype.kind == 'i':
    x = x.astype(win.dtype)

  *batch_shape, signal_length = x.shape
  # Created strided array of data segments
  if nperseg == 1 and noverlap == 0:
    result = x[..., np.newaxis]
  else:
    step = nperseg - noverlap
    batch_shape = tuple(batch_shape)
    x = x.reshape((int(np.prod(batch_shape)), signal_length))[..., np.newaxis]
    result = jax.lax.conv_general_dilated_patches(
        x, (nperseg,), (step,),
        'VALID',
        dimension_numbers=('NTC', 'OIT', 'NTC'))
    result = result.reshape(batch_shape + result.shape[-2:])

  # Detrend each data segment individually
  result = detrend_func(result)

  # Apply window by multiplication
  result = win.reshape((1,) * len(batch_shape) + (1, nperseg)) * result

  # Perform the fft on last axis. Zero-pads automatically
  if sides == 'twosided':
    return jax.numpy.fft.fft(result, n=nfft)
  else:
    return jax.numpy.fft.rfft(result.real, n=nfft)


def odd_ext(x, n, axis=-1):
  """Extends `x` along with `axis` by odd-extension.

  This function was previously a part of "scipy.signal.signaltools" but is no
  longer exposed.

  Args:
    x : input array
    n : the number of points to be added to the both end
    axis: the axis to be extended
  """
  if n < 1:
    return x
  if n > x.shape[axis] - 1:
    raise ValueError(
        f"The extension length n ({n}) is too big. "
        f"It must not exceed x.shape[axis]-1, which is {x.shape[axis] - 1}.")
  left_end = lax.slice_in_dim(x, 0, 1, axis=axis)
  left_ext = jnp.flip(lax.slice_in_dim(x, 1, n + 1, axis=axis), axis=axis)
  right_end = lax.slice_in_dim(x, -1, None, axis=axis)
  right_ext = jnp.flip(lax.slice_in_dim(x, -(n + 1), -1, axis=axis), axis=axis)
  ext = jnp.concatenate((2 * left_end - left_ext,
                         x,
                         2 * right_end - right_ext),
                         axis=axis)
  return ext


def _spectral_helper(x, y,
                     fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend_type='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False):
  """LAX-backend implementation of `scipy.signal._spectral_helper`.

  Unlike the original helper function, `y` can be None for explicitly
  indicating auto-spectral (non cross-spectral) computation.  In addition to
  this, `detrend` argument is renamed to `detrend_type` for avoiding internal
  name overlap.
  """
  if mode not in ('psd', 'stft'):
    raise ValueError(f"Unknown value for mode {mode}, "
                     "must be one of: ('psd', 'stft')")

  def make_pad(mode, **kwargs):
    def pad(x, n, axis=-1):
      pad_width = [(0, 0) for unused_n in range(x.ndim)]
      pad_width[axis] = (n, n)
      return jnp.pad(x, pad_width, mode, **kwargs)
    return pad

  boundary_funcs = {
      'even': make_pad('reflect'),
      'odd': odd_ext,
      'constant': make_pad('edge'),
      'zeros': make_pad('constant', constant_values=0.0),
      None: lambda x, *args, **kwargs: x
  }

  # Check/ normalize inputs
  if boundary not in boundary_funcs:
    raise ValueError(
        f"Unknown boundary option '{boundary}', "
        f"must be one of: {list(boundary_funcs.keys())}")

  axis = jax.core.concrete_or_error(operator.index, axis,
                                    "axis of windowed-FFT")
  axis = canonicalize_axis(axis, x.ndim)

  if nperseg is not None:  # if specified by user
    nperseg = jax.core.concrete_or_error(int, nperseg,
                                         "nperseg of windowed-FFT")
    if nperseg < 1:
      raise ValueError('nperseg must be a positive integer')
  # parse window; if array like, then set nperseg = win.shape
  win, nperseg = signal_helper._triage_segments(
      window, nperseg, input_length=x.shape[axis])

  if noverlap is None:
    noverlap = nperseg // 2
  else:
    noverlap = jax.core.concrete_or_error(int, noverlap,
                                          "noverlap of windowed-FFT")
  if nfft is None:
    nfft = nperseg
  else:
    nfft = jax.core.concrete_or_error(int, nfft,
                                      "nfft of windowed-FFT")

  _check_arraylike("_spectral_helper", x)
  x = jnp.asarray(x)

  if y is None:
    outdtype = jax.dtypes.canonicalize_dtype(np.result_type(x, np.complex64))
  else:
    _check_arraylike("_spectral_helper", y)
    y = jnp.asarray(y)
    outdtype = jax.dtypes.canonicalize_dtype(
        np.result_type(x, y, np.complex64))
    if mode != 'psd':
      raise ValueError("two-argument mode is available only when mode=='psd'")
    if x.ndim != y.ndim:
      raise ValueError(
          "two-arguments must have the same rank ({x.ndim} vs {y.ndim}).")

    # Check if we can broadcast the outer axes together
    try:
      outershape = jnp.broadcast_shapes(tuple_delete(x.shape, axis),
                                        tuple_delete(y.shape, axis))
    except ValueError as e:
      raise ValueError('x and y cannot be broadcast together.') from e

  # Special cases for size == 0
  if y is None:
    if x.size == 0:
      return jnp.zeros(x.shape), jnp.zeros(x.shape), jnp.zeros(x.shape)
  else:
    if x.size == 0 or y.size == 0:
      outshape = tuple_insert(
          outershape, min([x.shape[axis], y.shape[axis]]), axis)
      emptyout = jnp.zeros(outshape)
      return emptyout, emptyout, emptyout

  # Move time-axis to the end
  if x.ndim > 1:
    if axis != -1:
      x = jnp.moveaxis(x, axis, -1)
      if y is not None and y.ndim > 1:
        y = jnp.moveaxis(y, axis, -1)

  # Check if x and y are the same length, zero-pad if necessary
  if y is not None:
    if x.shape[-1] != y.shape[-1]:
      if x.shape[-1] < y.shape[-1]:
        pad_shape = list(x.shape)
        pad_shape[-1] = y.shape[-1] - x.shape[-1]
        x = jnp.concatenate((x, jnp.zeros(pad_shape)), -1)
      else:
        pad_shape = list(y.shape)
        pad_shape[-1] = x.shape[-1] - y.shape[-1]
        y = jnp.concatenate((y, jnp.zeros(pad_shape)), -1)

  if nfft < nperseg:
    raise ValueError('nfft must be greater than or equal to nperseg.')
  if noverlap >= nperseg:
    raise ValueError('noverlap must be less than nperseg.')
  nstep = nperseg - noverlap

  # Apply paddings
  if boundary is not None:
    ext_func = boundary_funcs[boundary]
    x = ext_func(x, nperseg // 2, axis=-1)
    if y is not None:
      y = ext_func(y, nperseg // 2, axis=-1)

  if padded:
    # Pad to integer number of windowed segments
    # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
    nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
    zeros_shape = list(x.shape[:-1]) + [nadd]
    x = jnp.concatenate((x, jnp.zeros(zeros_shape)), axis=-1)
    if y is not None:
      zeros_shape = list(y.shape[:-1]) + [nadd]
      y = jnp.concatenate((y, jnp.zeros(zeros_shape)), axis=-1)

  # Handle detrending and window functions
  if not detrend_type:
    def detrend_func(d):
      return d
  elif not hasattr(detrend_type, '__call__'):
    def detrend_func(d):
      return detrend(d, type=detrend_type, axis=-1)
  elif axis != -1:
    # Wrap this function so that it receives a shape that it could
    # reasonably expect to receive.
    def detrend_func(d):
      d = jnp.moveaxis(d, axis, -1)
      d = detrend_type(d)
      return jnp.moveaxis(d, -1, axis)
  else:
    detrend_func = detrend_type

  if np.result_type(win, np.complex64) != outdtype:
    win = win.astype(outdtype)

  # Determine scale
  if scaling == 'density':
    scale = 1.0 / (fs * (win * win).sum())
  elif scaling == 'spectrum':
    scale = 1.0 / win.sum()**2
  else:
    raise ValueError(f'Unknown scaling: {scaling}')
  if mode == 'stft':
    scale = jnp.sqrt(scale)

  # Determine onesided/ two-sided
  if return_onesided:
    sides = 'onesided'
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
      sides = 'twosided'
      warnings.warn('Input data is complex, switching to '
                    'return_onesided=False')
  else:
    sides = 'twosided'

  if sides == 'twosided':
    freqs = jax.numpy.fft.fftfreq(nfft, 1/fs)
  elif sides == 'onesided':
    freqs = jax.numpy.fft.rfftfreq(nfft, 1/fs)

  # Perform the windowed FFTs
  result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

  if y is not None:
    # All the same operations on the y data
    result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
                           sides)
    result = jnp.conjugate(result) * result_y
  elif mode == 'psd':
    result = jnp.conjugate(result) * result

  result *= scale

  if sides == 'onesided' and mode == 'psd':
    end = None if nfft % 2 else -1
    result = result.at[..., 1:end].mul(2)

  time = jnp.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1,
                    nperseg - noverlap) / fs
  if boundary is not None:
    time -= (nperseg / 2) / fs

  result = result.astype(outdtype)

  # All imaginary parts are zero anyways
  if y is None and mode != 'stft':
    result = result.real

  # Move frequency axis back to axis where the data came from
  result = jnp.moveaxis(result, -1, axis)

  return freqs, time, result


@_wraps(osp_signal.stft)
def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
         detrend=False, return_onesided=True, boundary='zeros', padded=True,
         axis=-1):
  freqs, time, Zxx = _spectral_helper(x, None, fs, window, nperseg, noverlap,
                                      nfft, detrend, return_onesided,
                                      scaling='spectrum', axis=axis,
                                      mode='stft', boundary=boundary,
                                      padded=padded)

  return freqs, time, Zxx


_csd_description = """
The original SciPy function exhibits slightly different behavior between
``csd(x, x)``` and ```csd(x, x.copy())```.  The LAX-backend version is designed
to follow the latter behavior.  For using the former behavior, call this
function as `csd(x, None)`."""


@_wraps(osp_signal.csd, lax_description=_csd_description)
def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density',
        axis=-1, average='mean'):
  freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,
                                  detrend, return_onesided, scaling, axis,
                                  mode='psd')
  if y is not None:
    Pxy = Pxy + 0j  # Ensure complex output when x is not y

  # Average over windows.
  if Pxy.ndim >= 2 and Pxy.size > 0:
    if Pxy.shape[-1] > 1:
      if average == 'median':
        bias = signal_helper._median_bias(Pxy.shape[-1]).astype(Pxy.dtype)
        if jnp.iscomplexobj(Pxy):
          Pxy = (jnp.median(jnp.real(Pxy), axis=-1)
                  + 1j * jnp.median(jnp.imag(Pxy), axis=-1))
        else:
          Pxy = jnp.median(Pxy, axis=-1)
        Pxy /= bias
      elif average == 'mean':
        Pxy = Pxy.mean(axis=-1)
      else:
        raise ValueError(f'average must be "median" or "mean", got {average}')
    else:
      Pxy = jnp.reshape(Pxy, Pxy.shape[:-1])

  return freqs, Pxy


@_wraps(osp_signal.welch)
def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          detrend='constant', return_onesided=True, scaling='density',
          axis=-1, average='mean'):
  freqs, Pxx = csd(x, None, fs=fs, window=window, nperseg=nperseg,
                   noverlap=noverlap, nfft=nfft, detrend=detrend,
                   return_onesided=return_onesided, scaling=scaling,
                   axis=axis, average=average)

  return freqs, Pxx.real
