# Copyright 2019 Google LLC
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
"""
cusparse wrappers for performing sparse matrix computations in JAX
"""

import numpy as np

from jax.lib import xla_client

try:
  from . import cusparse_kernels
except ImportError:
  cusparse_kernels = None
else:
  for _name, _value in cusparse_kernels.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")


kernels_available = cusparse_kernels and cusparse_kernels.kernels_available()


_ops = xla_client.ops
_Shape = xla_client.Shape


# TODO(phawkins): remove after we no longer need to support old jax releases.
def _unpack_builder(c):
  # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
  return getattr(c, "_builder", c)


def csr_todense(c, data, indices, indptr, *, shape):
  """CSR to dense matrix."""
  c = _unpack_builder(c)
  data_dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(indices).element_type())
  rows, cols = shape
  nnz = c.get_shape(data).dimensions()[0]

  buffer_size, opaque = cusparse_kernels.build_csr_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_todense",
      operands=(data, indices, indptr),
      operand_shapes_with_layout=(
          c.get_shape(data),
          c.get_shape(indices),
          c.get_shape(indptr),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, shape, (1, 0)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
  )
  return _ops.GetTupleElement(out, 0)


def csr_fromdense(c, mat, *, nnz, index_dtype):
  """CSR from dense matrix."""
  c = _unpack_builder(c)
  data_dtype = np.dtype(c.get_shape(mat).element_type())
  shape = c.get_shape(mat).dimensions()
  rows, cols = shape

  buffer_size, opaque = cusparse_kernels.build_csr_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_fromdense",
      operands=(mat,),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, shape, (1, 0)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (shape[0] + 1,), (0,)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
  )

  return tuple(_ops.GetTupleElement(out, i) for i in range(3))


def csr_matvec(c, data, indices, indptr, x, *, shape, transpose=False, compute_dtype=None):
  """CSR matrix/vector multiply."""
  c = _unpack_builder(c)
  dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(indices).element_type())
  x_dtype = np.dtype(c.get_shape(x).element_type())
  rows, cols = shape
  nnz, = c.get_shape(data).dimensions()

  if compute_dtype is None:
    compute_dtype = dtype

  buffer_size, opaque = cusparse_kernels.build_csr_matvec_descriptor(
      dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_matvec",
      operands=(data, indices, indptr, x),
      operand_shapes_with_layout=(
          c.get_shape(data),
          c.get_shape(indices),
          c.get_shape(indptr),
          c.get_shape(x),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size,), (0,)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
  )
  return _ops.GetTupleElement(out, 0)


def csr_matmat(c, data, indices, indptr, B, *, shape, transpose=False, compute_dtype=None):
  """CSR from dense matrix."""
  c = _unpack_builder(c)
  dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(indices).element_type())
  B_dtype = np.dtype(c.get_shape(B).element_type())
  rows, cols = shape
  _, Ccols = c.get_shape(B).dimensions()
  nnz, = c.get_shape(data).dimensions()

  if compute_dtype is None:
    compute_dtype = dtype

  buffer_size, opaque = cusparse_kernels.build_csr_matmat_descriptor(
      dtype, B_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_csr_matmat",
      operands=(data, indices, indptr, B),
      operand_shapes_with_layout=(
          c.get_shape(data),
          c.get_shape(indices),
          c.get_shape(indptr),
          c.get_shape(B),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size, Ccols), (1, 0)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
  )
  return _ops.GetTupleElement(out, 0)


def coo_todense(c, data, row, col, *, shape):
  """COO to dense matrix."""
  c = _unpack_builder(c)
  data_dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(row).element_type())
  rows, cols = shape
  nnz = c.get_shape(data).dimensions()[0]

  buffer_size, opaque = cusparse_kernels.build_coo_todense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_todense",
      operands=(data, row, col),
      operand_shapes_with_layout=(
          c.get_shape(data),
          c.get_shape(row),
          c.get_shape(col),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, shape, (1, 0)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
  )
  return _ops.GetTupleElement(out, 0)


def coo_fromdense(c, mat, *, nnz, index_dtype):
  """COO from dense matrix."""
  c = _unpack_builder(c)
  data_dtype = np.dtype(c.get_shape(mat).element_type())
  shape = c.get_shape(mat).dimensions()
  rows, cols = shape

  buffer_size, opaque = cusparse_kernels.build_coo_fromdense_descriptor(
      data_dtype, index_dtype, rows, cols, nnz)

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_fromdense",
      operands=(mat,),
      operand_shapes_with_layout=(
          _Shape.array_shape(data_dtype, shape, (1, 0)),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(data_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(index_dtype, (nnz,), (0,)),
          _Shape.array_shape(np.dtype(np.int8), (buffer_size,), (0,)),
      )),
      opaque=opaque,
  )

  return tuple(_ops.GetTupleElement(out, i) for i in range(3))

def coo_matvec(c, data, row, col, x, *, shape, transpose=False, compute_dtype=None):
  """CSR matrix/vector multiply."""
  c = _unpack_builder(c)
  dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(row).element_type())
  x_dtype = np.dtype(c.get_shape(x).element_type())
  rows, cols = shape
  nnz, = c.get_shape(data).dimensions()

  if compute_dtype is None:
    compute_dtype = dtype

  buffer_size, opaque = cusparse_kernels.build_coo_matvec_descriptor(
      dtype, x_dtype, compute_dtype, index_dtype,
      rows, cols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_matvec",
      operands=(data, row, col, x),
      operand_shapes_with_layout=(
          c.get_shape(data),
          c.get_shape(row),
          c.get_shape(col),
          c.get_shape(x),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size,), (0,)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
  )
  return _ops.GetTupleElement(out, 0)


def coo_matmat(c, data, row, col, B, *, shape, transpose=False, compute_dtype=None):
  """CSR from dense matrix."""
  c = _unpack_builder(c)
  dtype = np.dtype(c.get_shape(data).element_type())
  index_dtype = np.dtype(c.get_shape(row).element_type())
  B_dtype = np.dtype(c.get_shape(B).element_type())
  rows, cols = shape
  _, Ccols = c.get_shape(B).dimensions()
  nnz, = c.get_shape(data).dimensions()

  if compute_dtype is None:
    compute_dtype = dtype

  buffer_size, opaque = cusparse_kernels.build_coo_matmat_descriptor(
      dtype, B_dtype, compute_dtype, index_dtype,
      rows, cols, Ccols, nnz, transpose)
  out_size = cols if transpose else rows

  out = xla_client.ops.CustomCallWithLayout(
      c,
      b"cusparse_coo_matmat",
      operands=(data, row, col, B),
      operand_shapes_with_layout=(
          c.get_shape(data),
          c.get_shape(row),
          c.get_shape(col),
          c.get_shape(B),
      ),
      shape_with_layout=_Shape.tuple_shape((
          _Shape.array_shape(compute_dtype, (out_size, Ccols), (1, 0)),
          _Shape.array_shape(np.dtype(np.uint8), (buffer_size,), (0,)))),
      opaque=opaque,
  )
  return _ops.GetTupleElement(out, 0)
