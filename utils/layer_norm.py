from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import common_utils

#####################################################################################
def _layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.get_variable(
      "layer_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias

#####################################################################################
def _layer_norm_compute(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias

#####################################################################################
def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = common_utils.shape_list(x)[-1]

  with tf.variable_scope(name, default_name="layer_norm", values=[x], reuse=reuse):
    scale, bias = _layer_norm_vars(filters)
    return _layer_norm_compute(x, epsilon, scale, bias)