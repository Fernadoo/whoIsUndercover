import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gensim.downloader as api
from random import randrange
import os
from functools import reduce

class agent:

	def __init__(self,init_word, num_sim, num_player):
		self.out = False
		self.input_list = [] # other players' words
		self.belief = {i: 1.0/num_player for i in range(1, num_player+1)} # the probabilities of other agents being undercover
		self.my_word = init_word # the word hold by this agent
		self.num_sim = num_sim # number of similiar word to get in each round
		self.num_player = num_player # number of players
		self.num_playing = num_player # number of players still playing
		self.word_vectors = api.load("glove-wiki-gigaword-100")
		self.words = self.similarity([self.my_word], 100)
		self.p1 = 1.0/num_player
		self.p2_list = {i:1-2.0/num_player for i in range(2, num_player+1)}
		self.sweet_interval = (0.3, 0.8)
		self.used_words = []

	# input: [cos1, cos2, ...]
	# output: np.random.normal()
	def guass_init(self, input_cos):
		old_len = len(input_cos)
		for i in range(old_len):
			input_cos.append(2 - input_cos[i])
		mean = np.mean(input_cos)
		std = np.std(input_cos)
		output = np.random.normal(mean, std, int(1e5))
		sns.distplot(output, hist=False ,kde=True)
		plt.savefig("fig_gauss_init.png")
		# print(output)
		return output

	# input: the random samples given by the prob density function, interval
	# output: the sampled point
	def sample(self, input_samples, interval=[0,2]):
		lower = interval[0]
		upper = interval[1]
		samples = []
		cnt = 0
		for i in range(len(input_samples)):
			if lower <= input_samples[i] and input_samples[i] <= upper:
				samples.append(input_samples[i])
				cnt += 1
		rand_sample = np.random.choice(samples)
		return rand_sample

	# input: word, number of similar words to be output
	# output: wordvec, cosines of similar words
	def similarity(self, words, n):
		result = self.word_vectors.most_similar(positive=words, topn=n)
		return result

	def gauss_merge(self, w1, gauss1, w2, gauss2):
		mean1 = np.mean(gauss1)
		std1 = np.std(gauss1)
		mean2 = np.mean(gauss2)
		std2 = np.std(gauss2)
		mean_mix = w1*mean1 + w2*mean2
		std_mix = pow((pow(w1*std1,2) + pow(w1*std2,2)),0.5)
		output = np.random.normal(mean_mix, std_mix, int(1e5))
		sns.distplot(output, hist=False ,kde=True)
		plt.savefig("fig_gauss_mix.png")
		# print(output)
		return output

	def observe(self, inputpt_word_list, alpha): # dictionary
		pos_rel = []
		neg_rel = [[] for _ in range(alpha)]
		for i in range(2, self.num_playing+1):
			rel = self.word_vectors.similarity(inputpt_word_list[i-2], self.my_word)
			if rel < self.sweet_interval[0]:
				self.p2_list[i] *= (1 - self.sweet_interval[0] + rel)
			elif rel > self.sweet_interval[1]:
				self.p2_list[i] *= (1 - rel + self.sweet_interval[1])
			pos_rel.append(rel)
			for j in range(alpha):
				neg_rel[j].append(self.word_vectors.similarity(inputpt_word_list[i-2], self.words[j][0])) # relevant words to the my_word
		pos_rel_val = reduce(lambda x, y: x*y, pos_rel)
		neg_rel_val = max([reduce(lambda x, y: x*y, neg_rel[i]) for i in range(alpha)])
		self.p1 *= (float(neg_rel_val) / pos_rel_val)
		self.belief[1] = self.p1
		for i in range(2, self.num_playing+1):
			if i in self.belief.keys():
				self.belief[i] = 1 - self.p1 - self.p2_list[i]

	def vote(self): # return the potential undercover
		if self.num_playing > 2 and self.out == False:
			'''do the voting algorithm here
			to get a vote from the AI'''
			self.observe(self.input_list, 4) # parameters?
			s = sum(self.belief.values())
			self.belief = {i: float(self.belief[i])/s for i in self.belief.keys()} # normalize
			tmp = {i: self.belief[i] for i in list(self.belief.keys())[1:]}
			player = max(tmp, key=lambda x:tmp[x])
			# print(tmp)
			print("Mr.D votes player {}!".format(player))
			return player

	def gauss_mix(self):
		mylist = self.input_list
		mylist.append(self.my_word)
		v = self.word_vectors

		# handle badword
		badword = self.word_vectors.doesnt_match(mylist) # the remotest one
		new_word_list = self.similarity([badword], 10) # parameters?
		gauss2 = [v.similarity(self.my_word, x) for x,_ in new_word_list] + [v.similarity(self.my_word, badword)]
		
		# handle good words
		mylist.remove(badword)
		new_word_list = self.similarity(mylist, 20) # [(word, cos), ...] parameters?
		if badword != self.my_word:
			mylist.remove(self.my_word)
		gauss1 = [v.similarity(self.my_word, x) for x,_ in new_word_list] + [v.similarity(self.my_word, x) for x in mylist]

		# mix!!
		p = self.belief[1]
		return self.sample(self.gauss_merge(1-p, gauss1, p, gauss2),[0.6,1])

	def speak(self):
		if len(self.input_list) == 0:
			cosin = self.sample(self.guass_init([x[1] for x in self.similarity([self.my_word], self.num_sim)]), [0,0.5]) # first loop, parameter??
		else:
			cosin = self.gauss_mix()

		# get the word in the word base
		words = sorted(self.words, key=lambda x:abs(cosin-x[1]))
		fpick = ''
		for word,cosin in words:
			if word in self.used_words:
				continue
			self.used_words.append(word)
			fpick = word
			break
		print("Mr.D says {}".format(fpick))



if __name__ == '__main__':

	'''game design basic version'''
	num_of_players = int(input("The number of players you want: "))
	num_of_playing = num_of_players

	# two similar words for the game, one for undercover, one for civilian
	normal_word, undercover_word = input("Give two words to play with: ").split()
	os.system('clear')

	# word distribution
	players_card = {i: normal_word for i in range(1, num_of_players+1)} # eg. {1:cat, 2:cat, 3:cat}
	# r = randrange(1, num_of_players+1) # eg. r = 3
	r = 1
	players_card[r] = undercover_word # eg. {1:cat, 2:cat, 3:dog}
	for i in range(2, num_of_players+1):
		print("Player {}, the word for you is: {}".format(i, players_card[i]))
		loops = True
		while(loops):
			t = input("Player {}, now please close your eyes(Y/N): ".format(i))
			if t == 'Y' or t == 'y':
				loops = False
		os.system('clear')

	os.system('clear')
	print('*******Game Started*******')
	print('Mr.D is thinking...')
	mr_D = agent(players_card[1], 10, num_of_players)
	# print("Mr.D's belief is {}".format(mr_D.belief))
	# each round, one player out
	R = 1
	while num_of_playing > 2 and r in players_card.keys():
		if mr_D.out == True:
			print("Oh no! AI loses.")
			break
		print('Round {}'.format(R))
		print('----------------------------------------')
		mr_D.num_playing = num_of_playing # current number of players
		mr_D.speak()
		mr_D.input_list = [input("Player {}\'s turn to give a word: ".format(i)) for i in list(players_card.keys())[1:]]
		mr_D.used_words += mr_D.input_list

		# initialize the votes
		votes = {i: 0 for i in range(1, num_of_players+1)}
		
		# player still alive votes [1, 2, 1]
		votes[mr_D.vote()] += 1
		unlucky_list = [int(input("Player {}, vote for the Undercover(give a number between 1~{}): ".format(i, num_of_players))) for i in list(players_card.keys())[1:]]

		# update the votes of humans
		for unlucky in unlucky_list:
			votes[unlucky] += 1
		
		p = max(votes, key=lambda x:votes[x]) # player with the most votes
		print("Player {} out!".format(p))
		players_card.pop(p) # delete him
		mr_D.belief.pop(p)
		num_of_playing -= 1
		R += 1
	if r in players_card.keys():
		print("Undercover wins!")
		if r == 1:
			print("Mr.D wins! He is undercover, his word is {}!".format(mr_D.my_word))
		else:
			print("Mr.D loses! He is civilian, his word is {}!".format(mr_D.my_word))
	else:
		print("Undercover loses!")
		if r == 1:
			print("Mr.D loses! He is undercover, his word is {}!".format(mr_D.my_word))
		else:
			print("Mr.D wins! He is civilian, his word is {}!".format(mr_D.my_word))