from gensim.models import KeyedVectors

#mw2v = KeyedVectors.load_word2vec_format('50_epochs_ft/model.vec')
#mft = KeyedVectors.load_word2vec_format('word2vec/model.vec')

m = KeyedVectors.load_word2vec_format('word2vec/model.vec', binary=False)

#m = KeyedVectors.load_word2vec_format('model.vec', binary=False)
print "Loaded model."

while True:
	word = raw_input(">")
	try:
		#print "Word2Vec:"
		#print mw2v.most_similar(word, topn=10)
		
		#print "-" * 50
		#print "FastText:"		
		#print mft.most_similar(word, topn=10)

		print m.most_similar(word, topn=10)
	except Exception as e:
		print " %s is not in vocabulary." % word
		print e
