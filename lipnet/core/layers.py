from tensorflow.keras.layers import Lambda  # Updated import
from lipnet.core.loss import ctc_lambda_func

# CTC Layer implementation using Lambda layer
# (because Keras doesn't support extra params on loss function)
def CTC(name, args):
    return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)
