import tensorflow.keras.utils as utils

orig = 7
NUM_DIGITS = 20
print("before conversion", orig)
converted = utils.to_categorical(orig, NUM_DIGITS)
print("after conversion:", converted)
