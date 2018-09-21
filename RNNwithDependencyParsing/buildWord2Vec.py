import json
import gensim
from collections import OrderedDict, defaultdict

import numpy as np

from adagrad import Adagrad

data = open('hist_split.json').read()

parsedData = json.loads(data)

training_data = parsedData["train"]
sentences = []

for wordlist in training_data:
	indexed = wordlist[0]
	words = []

	for sublist in indexed:
		if sublist[0] == None:
			words.append("*")
		else:
			words.append(sublist[0])
	sentences.append(words)

model = gensim.models.Word2Vec(size=100, window=5, min_count=1)
model.build_vocab(sentences) 
alpha, min_alpha, passes = (0.025, 0.001, 20) 
alpha_delta = (alpha - min_alpha) / passes

for epoch in range(passes):
	model.alpha, model.min_alpha = alpha, alpha 
	model.train(sentences)
	print('completed pass %i at alpha %f' % (epoch + 1, alpha)) 
	alpha -= alpha_delta

	np.random.shuffle(sentences)

model.save('gensimModel')