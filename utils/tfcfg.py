import tensorflow as tf
from keras import backend as K
#import keras.backend.tensorflow_backend as KTF

def SET_GPU_MEM(value=0.5):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = value
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.tensorflow_backend.set_session(session)

def CLEAR_SESSION():
	tf.reset_default_graph()
	K.clear_session()
