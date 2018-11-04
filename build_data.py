from config import * 
cf = Config()

import pickle as pkl
import numpy as np
from get_nltk_sents import get_nltk_sents
from nltk.corpus.reader import ConllCorpusReader
from subprocess import call
import os
import codecs
from data_utils import TabbedCorpusReader

# Generate word_to_ix and tag_to_ix for tagged_sents (which are in Conll2000 format).
# Also generate the inverse ix_to_word and ix_to_tag.
def get_word_and_tag_ids(tagged_sents):
	word_to_ix = { "<PAD>": 0 }
	ix_to_word = [ "<PAD>" ]

	if cf.MODEL_TYPE == "S2S":
		tag_to_ix = { "<PAD>" : 0}
		ix_to_tag = [ "<PAD>" ]
	else:
		tag_to_ix = {}
		ix_to_tag = []
	for sent in tagged_sents:
		if cf.MODEL_TYPE == S2S:
			for word, tag in sent:
				if word not in word_to_ix:
					word_to_ix[word] = len(word_to_ix)
					ix_to_word.append(word)
				if tag not in tag_to_ix:
					tag_to_ix[tag] = len(tag_to_ix)
					ix_to_tag.append(tag)
		elif cf.MODEL_TYPE == S21:			
			for word in sent[0]:
				if word not in word_to_ix:
					word_to_ix[word] = len(word_to_ix)
					ix_to_word.append(word)
			tag = sent[1]
			if tag not in tag_to_ix:
				tag_to_ix[tag] = len(tag_to_ix)
				ix_to_tag.append(tag)
	#word_to_ix[SOS_TOKEN] = len(word_to_ix)
	#word_to_ix[EOS_TOKEN] = len(word_to_ix)
	#ix_to_word = [k for k, v in word_to_ix.iteritems()]
	#ix_to_tag = [k for k, v in tag_to_ix.iteritems()]
	return word_to_ix, ix_to_word, tag_to_ix, ix_to_tag

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


# Found at https://github.com/guillaumegenthial/sequence_tagging
def get_emb_vocab(filename):
	logger.info("Loading embedding vocab...")
	vocab = set()
	with open(filename) as f:
		for line in f:
			word = line.strip().split(' ')[0]
			vocab.add(word)
	logger.info("Done. Embedding vocab contains {} tokens".format(len(vocab)))
	return vocab


# Found at https://github.com/guillaumegenthial/sequence_tagging
def export_trimmed_embedding_vectors(vocab, oov_embeddings_filename, emb_filename, trimmed_filename, dim):
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

	embeddings[0] = np.zeros(cf.EMBEDDING_DIM) # add zero embeddings for padding

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
def generate_oov_embeddings(ix_to_word, emb_vocab, emb_model="fasttext"):
	oov_tokens = [word for word in ix_to_word[1:] if word not in emb_vocab] # Ignore the first token (the pad token)

	logger.debug("Vocab size: %d\n      Emb vocab size: %d\n      OOV Tokens: %d" % (len(ix_to_word), len(emb_vocab), len(oov_tokens)))

	if len(oov_tokens) == 0:
		logger.info("No OOV tokens were found.")
		if(os.path.exists(cf.EMB_OOV_FILENAME)):
			os.remove(cf.EMB_OOV_FILENAME)
		if(os.path.exists(cf.OOV_TOKENS_FILENAME)):
			os.remove(cf.OOV_TOKENS_FILENAME)
		logger.info("The OOV embedding files (%s and %s) were removed." % (cf.EMB_OOV_FILENAME, cf.OOV_TOKENS_FILENAME))
		return

	logger.info("Generating embeddings for tokens not found in the pretrained embeddings...")
	with codecs.open(cf.OOV_TOKENS_FILENAME, 'w', 'utf-8') as f:
		f.write("\n".join(oov_tokens))
	os.system("fasttext print-word-vectors %s < %s > %s" % (cf.EMB_BIN_FILENAME, cf.OOV_TOKENS_FILENAME, cf.EMB_OOV_FILENAME))


# Save all data to the relevant files.
def save_data_to_files(tagged_sents, word_to_ix, tag_to_ix, ix_to_word, ix_to_tag):
	with open("asset/tagged_sents_all.pkl", 'w') as f:
	 	pkl.dump(list(tagged_sents), f)
	with open("asset/word_to_ix.pkl", 'w') as f:
		pkl.dump(word_to_ix, f)
	with open("asset/tag_to_ix.pkl", 'w') as f:
		pkl.dump(tag_to_ix, f)
	with codecs.open("asset/ix_to_word.txt", 'w', 'utf-8') as f:
		f.write("\n".join(ix_to_word))	
	with codecs.open("asset/ix_to_tag.txt", 'w', 'utf-8') as f:
		f.write("\n".join(ix_to_tag))




def main():
	if cf.MODEL_TYPE == S2S:
		corpusReader = ConllCorpusReader(cf.DATA_FOLDER, [cf.TRAIN_FILENAME, cf.DEV_FILENAME, cf.TEST_FILENAME], ['words', 'pos'])
	elif cf.MODEL_TYPE == S21:
		corpusReader = TabbedCorpusReader(cf.DATA_FOLDER, [cf.TRAIN_FILENAME, cf.DEV_FILENAME, cf.TEST_FILENAME])

	tagged_sents = corpusReader.tagged_sents()

	logger.info("%d sentences loaded." % len(tagged_sents))
	tagged_sents = clean_sentences(tagged_sents)
	logger.info("%d sentences after cleaning." % len(tagged_sents))

	word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = get_word_and_tag_ids(tagged_sents)

	save_data_to_files(tagged_sents, word_to_ix, tag_to_ix, ix_to_word, ix_to_tag)

	# Get all words in the embedding vocab
	emb_vocab = get_emb_vocab(cf.EMB_VEC_FILENAME)

	# Generate OOV embeddings for any words in ix_to_word that aren't in emb_vocab
	generate_oov_embeddings(ix_to_word, emb_vocab)

	# Combine OOV embeddings with IV embeddings and export them to a file
	export_trimmed_embedding_vectors(word_to_ix, cf.EMB_OOV_FILENAME, cf.EMB_VEC_FILENAME, cf.EMB_TRIMMED_FILENAME, cf.EMBEDDING_DIM)

	logger.info("Data building complete.")

if __name__ == "__main__":
	main()