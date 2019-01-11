import gym
import random
import numpy as np
import keras
from keras.layers import Dense, Dropout, Input
from statistics import mean, median
from collections import Counter

lr = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				break
#some_random_games_first()
#print('done')

def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])
			prev_observation = observation
			score += reward
			if done:
				break
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]
				training_data.append([data[0], output])
		env.reset()
		scores.append(score)
	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)
	print('Average accepted score:', mean(accepted_scores))
	print('Median accepted score:', median(accepted_scores))
	print(Counter(accepted_scores))
	return training_data

#initial_population()

def neural_network_model(input_size):
	model = keras.Sequential()
	#model.add(Input(shape=(None,input_size,1)))
	model.add(Dense(128, activation='relu', input_shape=(None,input_size,1)))
	model.add(Dropout(.2))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='Adam', learning_rate=lr,
				loss='categorical_crossentropy')
	return model

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y = [i[1] for i in training_data]
	if not model:
		model = neural_network_model(input_size = len(X[0]))
	model.fit(X, y, epochs=5, verbose=2)

training_data = initial_population()
model = train_model(training_data)