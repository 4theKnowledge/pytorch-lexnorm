from gensim.models import KeyedVectors
import time
m = KeyedVectors.load_word2vec_format('model.vec')


print m.distances("person", ["bogger", "walking", "operator", "boilermaker"])

a = "person"
b = ["bogger", "walking", "operator", "boilermaker", "personel"] * 100000

t1 = time.time()

#for bb in b:
#	s = m.similarity(a, bb)
print "Done"
t2 = time.time()
print t2-t1


while True:
	word = raw_input("> ")
	try:
		
		print m.most_similar(word, topn=100)
	except:
		print " %s is not in vocabulary." % word
