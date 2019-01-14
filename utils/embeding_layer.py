from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#####################################################################################
def embedding_layer(hparams):
  with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
    # Create and initialize weights. The random normal initializer was chosen
    # randomly, and works well.
      lookup_table = tf.get_variable(
          "weights",
          dtype = tf.float32, 
          shape = [hparams.vocab_size, hparams.hidden_size],
          initializer=tf.random_normal_initializer(0., hparams.hidden_size ** -0.5))

  return lookup_table