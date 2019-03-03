from config import * 


import pickle as pkl
import numpy as np
#from get_nltk_sents import get_nltk_sents
from nltk.corpus.reader import ConllCorpusReader
from subprocess import call
import os
import codecs
from data_utils import TabbedCorpusReader


# Get all unique word tags and char tags in the test set.
# This prevents them from being added to the list of possible prediction tags later on.
# Theoretically the model would never predict them anyway, but it's probably best to remove them just to make sure.
def get_unique_test_tag_set():

	logger.info("Building set of testset-unique tags...")

	corpusReaderTrain = ConllCorpusReader(cf.DATA_FOLDER, [cf.TRAIN_FILENAME], ['words', 'pos'])
	corpusReaderTest  = ConllCorpusReader(cf.DATA_FOLDER, [cf.TEST_FILENAME], ['words', 'pos'])

	tagged_sents_train = corpusReaderTrain.tagged_sents()
	tagged_sents_test  = corpusReaderTest.tagged_sents()
	
	train_wordtags = set()
	train_chartags = set()
	for sent in tagged_sents_train:
		for word, tag in sent:
			if tag != "<PAD>" and tag != "<SELF>":
				train_wordtags.add(tag)
				for char in tag:
					train_chartags.add(char)

	test_unique_wordtags = set()
	test_unique_chartags = set()
	for sent in tagged_sents_test:
		for word, tag in sent:
			if tag != "<PAD>" and tag != "<SELF>":
				if tag not in train_wordtags:
					test_unique_wordtags.add(tag)				
					for char in tag:
						if char not in train_chartags:
							test_unique_chartags.add(char)
	
	logger.info("%d unique word tags and %d unique char tags found in the test dataset." % (len(test_unique_wordtags), len(test_unique_chartags)))
	return test_unique_wordtags, test_unique_chartags

def get_char_and_chartag_ids(tagged_sents):
	char_to_ix = { "<PAD>": 0 }
	ix_to_char = [ "<PAD>" ]

	if cf.MODEL_TYPE == S2S:
		ctag_to_ix = { "<PAD>" : 0}
		ix_to_ctag = [ "<PAD>" ]
	else:
		ctag_to_ix = {}
		ix_to_ctag = []

	for sent in tagged_sents:		
		for word, tag in sent:
			for c in word:
				if c not in char_to_ix:
					char_to_ix[c] = len(char_to_ix)
					ix_to_char.append(c) 
			for c in tag:
				if c not in ctag_to_ix: # and c not in test_unique_chartags:
					ctag_to_ix[c] = len(ctag_to_ix)
					ix_to_ctag.append(c)
	return char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag


# Replace any words that appear in the data with the corresponding replacement term,
# such as replacing all hashtags with "<HASHTAG>".
def auto_replace_word(w):
	for k, v in cf.ignored_words_replacement_map.items():		
		if w.startswith(k):
			w = v
	return w

# Generate word_to_ix and wtag_to_ix for tagged_sents (which are in Conll2000 format).
# Also generate the inverse ix_to_word and ix_to_wtag.
def get_word_and_wordtag_ids(tagged_sents):
	if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
		word_to_ix = { "<PAD>": 0 }
		ix_to_word = [ "<PAD>" ]
	elif cf.GRANULARITY == WORD_LEVEL:
		word_to_ix = { "<PAD>": 0}
		ix_to_word = [ "<PAD>" ]		
	unk_terms = 0 # Terms that were replaced with UNK due to being undesirable (hashtags, etc)

	if cf.MODEL_TYPE == S2S:
		wtag_to_ix = { "<PAD>" : 0}
		ix_to_wtag = [ "<PAD>" ]
	else:
		wtag_to_ix = {}
		ix_to_wtag = []
	for sent in tagged_sents:
		#if cf.MODEL_TYPE == S2S:
		for word, tag in sent:
			word = auto_replace_word(word)
			#if any(word.startswith(s) for s in cf.ignore_words_starting_with):
			#	unk_terms += 1
			#	word = "<UNK>"
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
				ix_to_word.append(word)
			if tag not in wtag_to_ix: # and tag not in test_unique_wordtags:
				wtag_to_ix[tag] = len(wtag_to_ix)
				ix_to_wtag.append(tag)

		#elif cf.MODEL_TYPE == S21:			
		#	for word in sent[0]:
		#		if word not in word_to_ix:
		#			word_to_ix[word] = len(word_to_ix)
		#			ix_to_word.append(word)
		#	tag = sent[1]
		#	if tag not in wtag_to_ix:
		#		wtag_to_ix[tag] = len(wtag_to_ix)
		#		ix_to_wtag.append(tag)
	#word_to_ix[SOS_TOKEN] = len(word_to_ix)
	#word_to_ix[EOS_TOKEN] = len(word_to_ix)
	#ix_to_word = [k for k, v in word_to_ix.iteritems()]
	#ix_to_wtag = [k for k, v in wtag_to_ix.iteritems()]
	if unk_terms > 0:
		logger.debug("%d undesirable terms were replaced with <UNK>." % unk_terms)
	return word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag

# Remove any sents that are too short or long.
def clean_sentences(sentences):
	if cf.MODEL_TYPE == S2S:
		return [sent for sent in sentences if len(sent) >= cf.MIN_SENT_LENGTH and len(sent) <= cf.MAX_SENT_LENGTH]
	elif cf.MODEL_TYPE == S21:
		return [sent for sent in sentences if len(sent[0]) >= cf.MIN_SENT_LENGTH and len(sent[0]) <= cf.MAX_SENT_LENGTH]

# Convert a tokenized sentence into a tensor of word indexes.
# def prepare_sequence(seq, to_ix):
# 	idxs = [to_ix[w] for w in seq]
# 	return torch.tensor(idxs, dtype=torch.long, device=device)


# Found at https://github.com/guillaumegenthial/sequence_wtagging
def get_emb_vocab(filename):
	logger.info("Loading embedding vocab...")
	vocab = set()
	#if cf.GRANULARITY in [CHAR_AND_WORD_LEVEL, WORD_LEVEL]:
	#	vocab.add("<UNK>")
	#if cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
	#	vocab.add("<SOS>")
	#	vocab.add("<EOS>")	
	with open(filename) as f:
		for line in f:
			word = line.strip().split(' ')[0]
			vocab.add(word)
	logger.info("Done. Embedding vocab contains {} tokens".format(len(vocab)))
	return vocab


# Found at https://github.com/guillaumegenthial/sequence_wtagging
def export_trimmed_embedding_vectors(vocab, embedding_dim, oov_embeddings_filename, emb_filename, trimmed_filename, dim):
	"""
	Saves glove vectors in numpy array

	Args:
		vocab: dictionary vocab[word] = index
		emb_filename: a path to an embeddings file
		trimmed_filename: a path where to store a matrix in npy
		dim: (int) dimension of embeddings
	"""
	logger.info("Generating trimmed embedding vectors...")
	embeddings = np.zeros([len(vocab), dim])

	embeddings[0] = np.zeros(embedding_dim) # add zero embeddings for padding
	#if cf.GRANULARITY in [WORD_LEVEL, CHAR_AND_WORD_LEVEL]:
	#	embeddings[1] = np.random.uniform(size=embedding_dim) # add random embeddings for the UNK tag
	#if cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
	#	embeddings[2] = np.random.uniform(size=embedding_dim) # add random embeddings for the SOS tag
	#	embeddings[3] = np.random.uniform(size=embedding_dim) # add random embeddings for the EOS tag

	for filename in [emb_filename, oov_embeddings_filename]:
		if not os.path.isfile(filename):
			logger.warn("File %s does not exist." % filename)
			continue
		logger.info("Loading embeddings from %s" % filename)
		with codecs.open(filename, 'r', 'utf-8') as f:
			for line in f:
				line = line.strip().split(' ')
				word = line[0]
				embedding = [float(x) for x in line[1:]]
				if word in vocab:
					word_idx = vocab[word]
					embeddings[word_idx] = np.asarray(embedding)
			# Check how many embeddings found so far
			found_embs = [np.any(a) for a in embeddings].count(True)
			logger.debug("Loading %d embeddings so far" % found_embs)

	np.savez_compressed(trimmed_filename, embeddings=embeddings)
	logger.info("Saved %d embedding vectors to %s." % (len(embeddings), trimmed_filename))


# Use FastText to generate embeddings for any words that are OOV.
# TODO: Allow other emb models
def generate_oov_embeddings(ix_to_word, emb_vocab, emb_bin_filename, oov_tokens_filename, emb_oov_filename, emb_model="fasttext"):
	oov_tokens = [word for word in ix_to_word[1:] if word not in emb_vocab] # Ignore the first token (the pad token).

	print(oov_tokens)
	logger.debug("Vocab size: %d\n      Emb vocab size: %d\n      OOV Tokens: %d" % (len(ix_to_word), len(emb_vocab), len(oov_tokens)))

	if len(oov_tokens) == 0:
		logger.info("No OOV tokens were found.")
		if(os.path.exists(emb_oov_filename)):
			os.remove(emb_oov_filename)
		if(os.path.exists(oov_tokens_filename)):
			os.remove(oov_tokens_filename)
		logger.info("The OOV embedding files (%s and %s) were removed." % (emb_oov_filename, oov_tokens_filename))
		return

	logger.info("Generating embeddings for tokens not found in the pretrained embeddings...")
	with codecs.open(oov_tokens_filename, 'w', 'utf-8') as f:
		f.write("\n".join(oov_tokens))
	os.system("fasttext print-word-vectors \"%s\" < \"%s\" > \"%s\"" % (emb_bin_filename, oov_tokens_filename, emb_oov_filename))


# Save all data to the relevant files.
def save_data_to_files(tagged_sents, word_to_ix, wtag_to_ix, ix_to_word, ix_to_wtag, char_to_ix, ctag_to_ix, ix_to_char, ix_to_ctag):
	with open("%s/tagged_sents_all.pkl" % cf.ASSET_FOLDER, 'wb') as f:
	 	pkl.dump(list(tagged_sents), f, protocol=2)
	with open("%s/word_to_ix.pkl" % cf.ASSET_FOLDER, 'wb') as f:
		pkl.dump(word_to_ix, f, protocol=2)
	with open("%s/wtag_to_ix.pkl" % cf.ASSET_FOLDER, 'wb') as f:
		pkl.dump(wtag_to_ix, f, protocol=2)
	with codecs.open("%s/ix_to_word.txt" % cf.ASSET_FOLDER, 'w', 'utf-8') as f:
		f.write("\n".join(ix_to_word))	
	with codecs.open("%s/ix_to_wtag.txt" % cf.ASSET_FOLDER, 'w', 'utf-8') as f:
		f.write("\n".join(ix_to_wtag))
	with open("%s/char_to_ix.pkl" % cf.ASSET_FOLDER, 'wb') as f:
		pkl.dump(char_to_ix, f, protocol=2)
	with open("%s/ctag_to_ix.pkl" % cf.ASSET_FOLDER, 'wb') as f:
		pkl.dump(ctag_to_ix, f, protocol=2)
	with codecs.open("%s/ix_to_char.txt" % cf.ASSET_FOLDER, 'w', 'utf-8') as f:
		f.write("\n".join(ix_to_char))	
	with codecs.open("%s/ix_to_ctag.txt" % cf.ASSET_FOLDER, 'w', 'utf-8') as f:
		f.write("\n".join(ix_to_ctag))


def main():

	if cf.EMBEDDING_MODEL == "Elmo":
		raise Exception("Please use build_data_elmo instead.")
	#if cf.MODEL_TYPE == S2S:
	corpusReader = ConllCorpusReader(cf.DATA_FOLDER, [cf.TRAIN_FILENAME, cf.TEST_FILENAME], ['words', 'pos'])
	#elif cf.MODEL_TYPE == S21:
	#	corpusReader = TabbedCorpusReader(cf.DATA_FOLDER, [cf.TRAIN_FILENAME, cf.TEST_FILENAME])

	tagged_sents = corpusReader.tagged_sents()

	test_unique_wordtags, test_unique_chartags = get_unique_test_tag_set()

	logger.info("%d sentences loaded." % len(tagged_sents))
	#tagged_sents = clean_sentences(tagged_sents)
	#logger.info("%d sentences after cleaning (removing short/long sentences)." % len(tagged_sents))

	word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag = get_word_and_wordtag_ids(tagged_sents) #, test_unique_wordtags)
	char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag = get_char_and_chartag_ids(tagged_sents) #, test_unique_chartags)

	save_data_to_files(tagged_sents, word_to_ix, wtag_to_ix, ix_to_word, ix_to_wtag, char_to_ix, ctag_to_ix, ix_to_char, ix_to_ctag)

	if cf.USE_PRETRAINED_WORD_EMBEDDINGS:
		# Get all words in the embedding vocab
		emb_vocab = get_emb_vocab(cf.EMB_VEC_FILENAME)

		# Generate OOV embeddings for any words in ix_to_word that aren't in emb_vocab
		#generate_oov_embeddings(ix_to_word, emb_vocab, cf.EMB_BIN_FILENAME, cf.OOV_TOKENS_FILENAME, cf.EMB_OOV_FILENAME)

		# Combine OOV embeddings with IV embeddings and export them to a file
		export_trimmed_embedding_vectors(word_to_ix, cf.WORD_EMBEDDING_DIM, cf.EMB_OOV_FILENAME, cf.EMB_VEC_FILENAME, cf.EMB_TRIMMED_FILENAME, cf.WORD_EMBEDDING_DIM)

	if cf.USE_PRETRAINED_CHAR_EMBEDDINGS:
		char_emb_vocab = get_emb_vocab(cf.CHAR_EMB_VEC_FILENAME)
		generate_oov_embeddings(ix_to_char, char_emb_vocab, cf.CHAR_EMB_BIN_FILENAME, cf.CHAR_OOV_TOKENS_FILENAME, cf.CHAR_EMB_OOV_FILENAME)
		export_trimmed_embedding_vectors(char_to_ix, cf.CHAR_EMBEDDING_DIM, cf.CHAR_EMB_OOV_FILENAME, cf.CHAR_EMB_VEC_FILENAME, cf.CHAR_EMB_TRIMMED_FILENAME, cf.CHAR_EMBEDDING_DIM)

	logger.info("Data building complete.")

if __name__ == "__main__":
	main()
