import gym
import random
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
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
				#if data[1] == 1:
				#	output = [0,1]
				#elif data[1] == 0:
				#	output = [1,0]
				#training_data.append([data[0], output])
				training_data.append([data[0], data[1]])
		env.reset()
		scores.append(score)
	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)
	print('Average accepted score:', mean(accepted_scores))
	print('Median accepted score:', median(accepted_scores))
	#print(Counter(accepted_scores))
	return training_data

#initial_population()

def neural_network_model(input_size):
	model = keras.Sequential()
	#model.add(Input(shape=(None,input_size,1)))
	model.add(Dense(128, activation='relu', input_shape=[input_size,1]))
	model.add(Dropout(.2))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(.2))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer=keras.optimizers.Adam(lr=lr),
				loss='binary_crossentropy', metrics=['accuracy'])
	print(type(model))
	return model

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y = [i[1] for i in training_data]
	if not model:
		model = neural_network_model(input_size = len(X[0]))
	model.fit(X, y, epochs=5, verbose=2)
	return model

training_data = initial_population()
model = train_model(training_data)
n_games = 10000
for _ in range(2):
	scores = []
	choices = []
	training_data = []
	accepted_scores = []
	for each_game in range(n_games):
		score = 0
		game_memory = []
		prev_obs = []
		env.reset()
		for _ in range(goal_steps):
			#env.render()
			if len(prev_obs) == 0:
				action = random.randrange(0,2)
			else:
				action = int(np.round(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])[0])
			choices.append(action)
			new_observation, reward, done, info = env.step(action)
			prev_obs = new_observation
			game_memory.append([new_observation, action])
			score += reward
			if done:
				break
		if score >= 150:
			#print('test')
			accepted_scores.append(score)
			for data in game_memory:
				training_data.append([data[0], data[1]])
		scores.append(score)
	print('Almost done')
	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)
	if len(accepted_scores)>0:
		print('Average accepted score:', mean(accepted_scores))
		print('Median accepted score:', median(accepted_scores))
	else:
		print('0 accepted scores????')
	print('Average score:', sum(scores)/len(scores))
	#print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
	model = train_model(training_data)
