

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import common_utils

def post_layers(x, y, hparams):
    layer_norm(x, filters=None, epsilon=hparams.norm_epsilon, name=None, reuse=None)
    
#####################################################################################
def dense_relu_dense(inputs,
                     hparams,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  layer_name = "%s_{}" % name if name else "{}"
  h = common_utils.dense(
      inputs,
      hparams.filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      name=layer_name.format("conv1"))

  if dropout_rate != 0.0:
    h = common_utils.dropout(h, hparams.relu_dropout)

  o = common_utils.dense(
      h,
      hparams.hidden_size,
      use_bias=True,
      name=layer_name.format("conv2"))

  return o

#####################################################################################
def transformer_ffn_unit(x, hparams):
  """Applies a feed-forward function which is parametrised for encoding.

  Args:
    x: input
    hparams: model hyper-parameters
    nonpadding_mask: optional Tensor with shape [batch_size, encoder_length]
    indicating what positions are not padding.  This is used
    to mask out padding in convoltutional layers.  We generally only
    need this mask for "packed" datasets, because for ordinary datasets,
    no padding is ever followed by nonpadding.
    pad_remover: to mask out padding in convolutional layers (efficiency).

  Returns:
    the output tensor
  """

  with tf.variable_scope("ffn"):
    y = dense_relu_dense(x, hparams)
    x = layer_postprocess(x, y, hparams)

  return x

#####################################################################################
def transformer_encoder_attention_unit(x,
                                       encoder_self_attention_bias,
                                       hparams):
  """Applies multihead attention function which is parametrised for encoding.

  Args:
    x: input
    hparams: model hyper-parameters
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    attention_dropout_broadcast_dims: Fpr noise broadcasting in the dropout
      layers to save memory during training
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    the output tensor

  """

  with tf.variable_scope("self_attention"):
    y = multihead_attention(
        x,
        x,
        encoder_self_attention_bias,
        hparams)

    x = layer_postprocess(x, y, hparams)

  return x

#####################################################################################
def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      tf.maximum(tf.to_float(num_timescales) - 1, 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal

#####################################################################################
def add_position_timing_signal(x, step, hparams):
  """Add n-dimensional embedding as the position (horizontal) timing signal.

  Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    hparams: model hyper parameters

  Returns:
    a Tensor with the same shape as x.

  """
  shape = common_utils.shape_list(x)
  length = shape[1]
  channels = shape[2]

  signal = get_timing_signal_1d(length, channels)

  x_with_timing = x + signal

  return x_with_timing

#####################################################################################
def add_step_timing_signal(x, step, hparams):
  """Add n-dimensional embedding as the step (vertical) timing signal.

  Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    hparams: model hyper parameters

  Returns:
    a Tensor with the same shape as x.

  """
  num_steps = hparams.act_max_steps
  channels = common_utils.shape_list(x)[-1]
  
  timing_signal = get_timing_signal_1d(num_steps, channels)
  signal = tf.expand_dims(timing_signal[:, step, :], axis=1)

  x_with_timing = x + signal

  return x_with_timing

#####################################################################################
def step_preprocess(x, step, hparams):
  """Preprocess the input at the beginning of each step.

  Args:
    x: input tensor
    step: step
    hparams: model hyper-parameters

  Returns:
    preprocessed input.

  """
  x = add_position_timing_signal(x, step, hparams)
  x = add_step_timing_signal(x, step, hparams)

  return x

#####################################################################################
def universal_transformer_act_basic(x, hparams, ffn_unit, attention_unit):
  """Basic universal_transformer with ACT based on remainder-distribution ACT.

  Args:
    x: input
    hparams: model hyper-parameters
    ffn_unit: feed-forward unit
    attention_unit: multi-head attention unit

  Returns:
    the output tensor,  (ponder_times, remainders)

  """

  state = x
  act_max_steps = hparams.act_max_steps
  threshold = 1.0 - hparams.act_epsilon

  shape = tf.shape(state)
  batch_size = shape[0]
  length     = shape[1]

  # Halting probabilities (p_t^n in the paper)
  halting_probability = tf.zeros((batch_size, length), name="halting_probability")
  
  # Remainders (R(t) in the paper)
  remainders = tf.zeros((batch_size, length), name="remainder")

  # Number of updates performed (N(t) in the paper)
  n_updates = tf.zeros((batch_size, length), name="n_updates")

  # Previous cell states (s_t in the paper)
  previous_state = tf.zeros_like(state, name="previous_state")
  step = tf.constant(0, dtype=tf.int32)

  def ut_function(state, 
                  step, 
                  halting_probability, 
                  remainders, 
                  n_updates,
                  previous_state):
    """implements act (position-wise halting).

    Args:
      state: 3-D Tensor: [batch_size, length, channel]
      step: indicates number of steps taken so far
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      previous_state: previous state

    Returns:
      transformed_state: transformed state
      step: step+1
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      new_state: new state
    """
    state_shape = state.get_shape()
    state = step_preprocess(state, step, hparams)

    with tf.variable_scope("sigmoid_activation_for_pondering"):
      p = common_layers.dense(
          state,
          1,
          activation=tf.nn.sigmoid,
          use_bias=True,
          bias_initializer=tf.constant_initializer(hparams.act_halting_bias_init))
      p = tf.squeeze(p, axis=-1)

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    new_halting_prob = halting_probability + p * still_running
    
    # Mask of inputs which halted at this step
    new_halted = tf.cast(
        tf.greater(new_halting_prob, threshold),
        tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
        tf.less_equal(new_halting_prob, threshold),
        tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = tf.expand_dims(p * still_running + new_halted * remainders,
                                    -1)

    # apply transformation on the state
    transformed_state = ffn_unit(attention_unit(state))

    # update running part in the weighted state and keep the rest
    new_state = ((transformed_state * update_weights) +
                 (previous_state * (1 - update_weights)))

    # remind TensorFlow of everything's shape
    transformed_state.set_shape(state_shape)
    for x in [halting_probability, remainders, n_updates]:
      x.set_shape([
          state_shape[0],
          state_shape[1],
      ])
    new_state.set_shape(state_shape)
    step += 1
    return (transformed_state, step, halting_probability, remainders, n_updates,
            new_state)

  # While loop stops when this predicate is FALSE.
  # Ie all (probability < 1-eps AND counter < N) are false.
  def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
    del u0, u1, u2, u3
    return tf.reduce_any(
        tf.logical_and(
            tf.less(halting_probability, threshold),
            tf.less(n_updates, act_max_steps)))

  # Do while loop iterations until predicate above is false.
  (_, _, _, remainder, n_updates, new_state) = tf.while_loop(
      should_continue, ut_function,
      (state, step, halting_probability, remainders, n_updates, previous_state),
      maximum_iterations=act_max_steps + 1)

  ponder_times = n_updates
  remainders = remainder

  tf.contrib.summary.scalar("ponder_times", tf.reduce_mean(ponder_times))

  return new_state, (ponder_times, remainders)

#####################################################################################
def universal_transformer_layer(x,
                                hparams,
                                ffn_unit,
                                attention_unit):
  with tf.variable_scope("universal_transformer_act"):
    return universal_transformer_act_basic(x, hparams, ffn_unit, attention_unit)




#####################################################################################
def _encoder(x, hparams):
    pass

#####################################################################################
def encoder(x, hparams):
  '''
  Args:
    x: input: [batch_size, length]
    hparams: model hyper-parameters
  '''
  lookup_table = embedding_layer(hparams)
  embeddings = tf.nn.embedding_lookup(lookup_table, x)
    
  with tf.variable_scope("encoder"):
    memory = _encoder(embeddings, hparams)

  return memory

#####################################################################################
def universal_transformer(x, y, hparams):
    pass