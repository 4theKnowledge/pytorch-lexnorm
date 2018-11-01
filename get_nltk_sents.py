from config import *
cf = Config()
from nltk.corpus import gutenberg, brown, reuters

def get_nltk_sents():
	ids = gutenberg.fileids()
	sents = []
	for i in ids:
		logger.debug(i)
		for sent in gutenberg.sents(i):
			sents.append(sent)

	# Load brown and reuters as well
	logger.debug("Brown corpus")
	for s in brown.sents():
		sents.append(s)
	#logger.debug("Reuters corpus")
	#for s in reuters.sents():
	#	sents.append(s)

	logger.info("Found %d sentences total" % len(sents))
	return sents

