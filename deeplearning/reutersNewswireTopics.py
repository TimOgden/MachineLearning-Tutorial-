import tensorflow as tf
from tensorflow import keras
from keras.datasets import reuters
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
(x_train, y_train), (x_test, y_test) = reuters.load_data()

# Word index example:
# {'cat': 1, 'dog': 2, 'the': 3}
word_index = reuters.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(key,value) for (value, key) in word_index.items()])

def decode_string(array):
	return ' '.join([reverse_word_index[i] for i in array])



def create_word_frequency(array):
	freq = {}
	for word in array:
		if word in freq:
			freq[word] += 1
		else:
			freq[word] = 1
	temp = []
	for value, key in freq.items():
		temp.append(value)
	return np.array(temp)

# Converting semantics into word frequency
#x_train = [create_word_frequency(x_train[i]) for i in range(len(x_train))]
#print(len(x_train))
#x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=word_index["<PAD>"], padding='post', maxlen=256)
#x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=word_index["<PAD>"], padding='post', maxlen=256)
def define_model():
	model = keras.Sequential([
		
		layers.Dense(8982, activation=tf.nn.relu, input_shape=[8982,]),
		layers.Dense(87, activation=tf.nn.relu),
		layers.Flatten(),
		layers.Dense(46, activation=tf.nn.softmax)
		])
	optimizer = tf.train.AdamOptimizer(.001)
	model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
	return model

model = define_model()

history = model.fit(x_train, y_train, epochs=10, verbose=2)

# Testing model
example_batch = y_train[:50]
#example_result = model.predict(example_batch)
#print(model.predict(example_batch))
'''
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())

#Let's make a function to graph the trends of the error
def plot_history():
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [MPG]')
	plt.plot(hist['epoch'], hist['mean_absolute_error'], label=['Train Error'])
	plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label=['Val Error'])
	plt.legend()
	plt.ylim([0,5])
	plt.show()
	
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Square Error [$MPG^2$]')
	plt.plot(hist['epoch'], hist['mean_squared_error'], label=['Train Error'])
	plt.plot(hist['epoch'], hist['val_mean_squared_error'], label=['Val Error'])
	plt.ylim(0,20)
	plt.legend()
	plt.show()

plot_history()

loss, mae, mse = model.evaluate(x_test, y_test, verbose=0)

#print("Testing set Mean Abs Error: {:5.2f}".format(mae))
'''