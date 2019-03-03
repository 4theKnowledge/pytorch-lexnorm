from gensim.models import KeyedVectors

m = KeyedVectors.load_word2vec_format('model.vec')


while True:
	word = raw_input(">")
	try:
		print m.most_similar(word, topn=20)
	except:
		print " %s is not in vocabulary." % word
