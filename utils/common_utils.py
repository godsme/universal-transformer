from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

#####################################################################################
def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    
    Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.
    
    Raises:
    ValueError: If the expected shape doesn't match the actual shape.
    """
    
    if name is None:
        name = tensor.name
        
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
            
    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

#####################################################################################
def shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
      
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    with tf.name_scope("get_shape"):
      # static dimensions
      shape = tensor.shape.as_list()

      # collect non-static dimensions
      non_static_indexes = []
      for (index, dim) in enumerate(shape):
        if dim is None:
          non_static_indexes.append(index)

      # get missing dimensions in dynamic way 
      if non_static_indexes:
        dyn_shape = tf.shape(tensor)
        for index in non_static_indexes:
            shape[index] = dyn_shape[index]
            
      return shape

#####################################################################################
def numpy_ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    # Needed info is constant, so we construct in numpy
    num_lower = rows - 1 if num_lower < 0 else num_lower
    num_upper = cols - 1 if num_upper < 0 else num_upper

    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)

    band = np.ones((rows, cols)) * lower_mask * upper_mask

    if out_shape:
      band = band.reshape(out_shape)
    
    return tf.constant(band, tf.float32)

#####################################################################################
def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.

  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.

  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    return numpy_ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape)
  else:
    band = tf.matrix_band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band