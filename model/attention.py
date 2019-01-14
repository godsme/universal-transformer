
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import common_utils

###############################################################
def _split_heads(x, hparams):
  """Split x into different heads, and transpose the resulting value.
  The tensor is transposed to insure the inner dimensions hold the correct
  values during the matrix multiplication.
  Args:
    x: A tensor with shape [batch_size, length, hidden_size]
  Returns:
    A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
  """
  with tf.name_scope("split_heads"):
    shape = common_utils.shape_list(x, expected_rank=[3])

    # Split the last dimension
    x = tf.reshape(x, [shape[0], shape[1], hparams.num_heads, hparams.head_size])

    # Transpose the result
    # [batch_size, num_heads, seq_len, head_size]
    return tf.transpose(x, [0, 2, 1, 3])

###############################################################
def _combine_heads(x, hparams):
  """Combine tensor that has been split.
  Args:
    x: A tensor [batch_size, num_heads, length, head_size]

  Returns:
    A tensor with shape [batch_size, length, hidden_size]
  """
  with tf.name_scope("combine_heads"):
    shape = common_utils.shape_list(x)
      
    x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
    return tf.reshape(x, [shape[0], shape[2], hparams.hidden_size])

###############################################################
def multihead_attention(kv, q, bias, hparams):
    q_o = dense(q,  hparams.hidden_size, use_bias=False, name="q")
    k_o = dense(kv, hparams.hidden_size, use_bias=False, name="k")
    v_o = dense(kv, hparams.hidden_size, use_bias=False, name="v")

    # Split q, k, v into heads.
    # [B, N, Q, S]
    q_o = _split_heads(q_o, hparams)
    # [B, N, K, S]
    k_o = _split_heads(k_o, hparams)
    # [B, N, K, S]
    v_o = _split_heads(v_o, hparams)

    q_o *= float(hparams.hidden_size) ** -0.5

    # Calculate dot product attention
    # [B, N, Q, K]
    logits = tf.matmul(q_o, k_o, transpose_b=True)
    logits += bias

    weights = tf.nn.softmax(logits, name="attention_weights")

    weights = dropout(weights, hparams.attention_dropout)

    # [B, N, Q, S]
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    # [B, Q, H]
    return _combine_heads(attention_output, hparams)

