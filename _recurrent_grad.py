import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
recurrent_grad_module = tf.load_op_library('build/librecurrent.so')

@ops.RegisterGradient("Recurrent")
# def _recurrent_grad_cc(op, grad):
#     return recurrent_grad_module.recurrent_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])