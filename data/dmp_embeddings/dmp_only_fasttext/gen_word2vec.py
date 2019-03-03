from gensim.models import Word2Vec
import codecs

import gzip
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = []


with codecs.open("dmp_data.txt", "r", "utf-8") as f:
	for i, line in enumerate(f):
		line = line.strip().split()
		
		for j, word in enumerate(line):
			if word == "'s" or word == "n't":
				print line
				line[j-1 : j + 1] = [''.join(line[j - 1 : j + 1])]
				print line


		documents.append(line)
		print "\r%d" % i,

#documents = [["hello"], ["there"]]
#model = Word2Vec(documents, size=300, window=5, min_count=1, workers=10)

model = Word2Vec(documents, size=512, window=7, min_count=1, workers=10)
model.train(documents, total_examples=len(documents), epochs=50)

model.wv.save_word2vec_format("word2vec/model.vec", binary=False)



