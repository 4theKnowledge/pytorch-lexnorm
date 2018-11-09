import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader

import codecs
from config import *
cf = Config()

from data_utils import TabbedCorpusReader

from nltk.corpus.reader import ConllCorpusReader

import torch




# Converts a dataset to a numpy format so it can be loaded into the DataLoader.
# if test_set is True, ignore min/max lengths
def tagged_sents_to_numpy(tagged_sents, word_to_ix, wtag_to_ix, char_to_ix, ctag_to_ix, ix_to_char, ix_to_word):

	data_w = []
	data_x = []
	data_y = []
	rejected_sents = [] # A list to store 'rejected' sents, i.e. ones that were too short/long.
	rejected_words = []
	for sent in tagged_sents:

		if cf.MODEL_TYPE == S2S:
			words, tags = zip(*sent)
			if len(words) != len(tags):
				raise Exception("Words and tags are not the same length.")
		elif cf.MODEL_TYPE == S21:
			words, tag = sent[0], sent[1]

		if cf.GRANULARITY == WORD_LEVEL and (len(words) > cf.MAX_SENT_LENGTH or len(words) < cf.MIN_SENT_LENGTH):
			rejected_sents.append(sent)
			#continue

		wordz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)
		wordz[:len(words)] = np.asarray([word_to_ix[w] for w in words[:cf.MAX_SENT_LENGTH]], )

		if cf.GRANULARITY == WORD_LEVEL:			
			if cf.MODEL_TYPE == S2S:
				data_w.append([0])
				data_x.append(wordz)

				tagz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)	
				tagz[:len(tags)] = np.asarray([wtag_to_ix[w] for w in tags[:cf.MAX_SENT_LENGTH]], )

				data_y.append(tagz)	
			elif cf.MODEL_TYPE == S21:
				tag   = sent[1]
				data_y.append(wtag_to_ix[tag])

		elif cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
			for i, (word, tag) in enumerate(sent):
				if len(word) > cf.MAX_WORD_LENGTH:
					rejected_words.append(sent)
					continue
				charz = np.zeros(cf.MAX_WORD_LENGTH, dtype=int)
				charz[:min(len(word), cf.MAX_WORD_LENGTH)] = np.asarray([char_to_ix[c] for c in word[:cf.MAX_WORD_LENGTH]], ) # Trim words that are too long
				ctagz = np.zeros(cf.MAX_WORD_LENGTH, dtype=int)	
				ctagz[:min(len(tag), cf.MAX_WORD_LENGTH)] = np.asarray([ctag_to_ix[c] for c in tag[:cf.MAX_WORD_LENGTH]], )	# Trim tags that are too long

				

				# print i
				# print max(0, i) , min(len(words), i+5)
				# print word
				# print [ix_to_word[w] for w in context_wordz]
				# print ""
				if cf.GRANULARITY == CHAR_LEVEL:
					data_w.append([0])
				elif cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
					words = [0, 0] + [word_to_ix[w[0]] for w in sent] + [0, 0]
					context_wordz = words[max(0, i) : min(len(words), i+5)]	 # hardcoded to 5 for now
					data_w.append(context_wordz)
				data_x.append(charz)
				data_y.append(ctagz)
	
	# If there are not enough sentences to create a set of perfectly-sized batches, append some batches that are all 0s.
	# These will be ignored by the model.
	if cf.MODEL_TYPE == S2S:
		if cf.GRANULARITY == WORD_LEVEL:
			to_pad = cf.BATCH_SIZE - (len(data_x) % cf.BATCH_SIZE)
			for i in range(to_pad):
				data_w.append( np.asarray([0] ))
				data_x.append( np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
				data_y.append( np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
		elif cf.GRANULARITY == CHAR_LEVEL:
			to_pad = cf.BATCH_SIZE - (len(data_x) % cf.BATCH_SIZE)
			for i in range(to_pad):
				data_w.append( np.asarray([0] ))
				data_x.append( np.zeros(cf.MAX_WORD_LENGTH, dtype = int))
				data_y.append( np.zeros(cf.MAX_WORD_LENGTH, dtype = int))
		elif cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
			logger.warn("Batch padding not yet implemented for CHAR_AND_WORD_LEVEL model.")

	# 	print len(data_x)
	# 	print len(data_w)

	return np.asarray(data_w), np.asarray(data_x), np.asarray(data_y), rejected_sents, rejected_words

# Found at https://github.com/guillaumegenthial/sequence_tagging
def get_trimmed_emb_vectors(filename):
	"""
	Args:
		filename: path to the npz file
	Returns:
		matrix of embeddings (np array)
	"""
	with np.load(filename) as data:
		return data["embeddings"]


# Build two separate data iterators: one for the training data, another for dev (validation).
# Also retrieve a test dataset.
def load_datasets(word_to_ix, wtag_to_ix, char_to_ix, ctag_to_ix, ix_to_char, ix_to_word):
	data_iterators = { "train": None, "dev": None }
	test_dataset = []
	for i, dataset in enumerate(["train", "dev", "test"]):

		if cf.MODEL_TYPE == S2S:
			corpusReader = ConllCorpusReader(cf.DATA_FOLDER, [[cf.TRAIN_FILENAME, cf.DEV_FILENAME, cf.TEST_FILENAME][i]], ['words', 'pos'])
		elif cf.MODEL_TYPE == S21:
			corpusReader = TabbedCorpusReader(cf.DATA_FOLDER, [[cf.TRAIN_FILENAME, cf.DEV_FILENAME, cf.TEST_FILENAME][i]])

		tagged_sents = corpusReader.tagged_sents()
		data_w, data_x, data_y, rejected_sents, rejected_words = tagged_sents_to_numpy(tagged_sents, word_to_ix, wtag_to_ix, char_to_ix, ctag_to_ix, ix_to_char, ix_to_word)
		myDataset = MyDataset(data_w, data_x, data_y)


		if dataset == "test":
			test_dataset = myDataset
		else:			
			data_iterator = DataLoader(myDataset, batch_size=cf.BATCH_SIZE, pin_memory=True)
			data_iterators[dataset] = data_iterator
			#for d in data_iterator:
		#		torch.set_printoptions(threshold = 5000000)
	#			print d 
	#			exit()
			logger.info("Loaded %d %s batches.\n" % (len(data_iterator), dataset) +
				"      (%d x %d = ~%d sentences total)" % (len(data_iterator), cf.BATCH_SIZE, len(data_iterator) * cf.BATCH_SIZE))
		if len(rejected_sents) > 0:
			logger.warning("%d of %d sentences from the %s set were trimmed due to being too long or short." % (len(rejected_sents), len(tagged_sents) + len(rejected_sents), dataset))
		if len(rejected_words) > 0:
			logger.warning("%d words from the %s set were trimmed due to being too long." % (len(rejected_words), dataset))

	return data_iterators, test_dataset

class MyDataset(Dataset):
	def __init__(self, w, x, y):
		super(MyDataset, self).__init__()
		self.w = w
		self.x = x
		self.y = y

	def __getitem__(self, ids):
		return self.w[ids], self.x[ids], self.y[ids]

	def __len__(self):
		return self.x.shape[0]

def load_data():
	with open("%s/word_to_ix.pkl" % cf.ASSET_FOLDER, 'r') as f:
		word_to_ix = pkl.load(f)
	with codecs.open("%s/ix_to_word.txt" % cf.ASSET_FOLDER, 'r', 'utf-8') as f:
		ix_to_word = [line.strip() for line in f]
	with open("%s/wtag_to_ix.pkl" % cf.ASSET_FOLDER, 'r') as f:
		wtag_to_ix = pkl.load(f)
	with codecs.open("%s/ix_to_wtag.txt" % cf.ASSET_FOLDER, 'r', 'utf-8') as f:
		ix_to_wtag = [line.strip() for line in f]
	with open("%s/char_to_ix.pkl" % cf.ASSET_FOLDER, 'r') as f:
		char_to_ix = pkl.load(f)
	with codecs.open("%s/ix_to_char.txt" % cf.ASSET_FOLDER, 'r', 'utf-8') as f:
		ix_to_char = [line.strip() for line in f]
	with open("%s/ctag_to_ix.pkl" % cf.ASSET_FOLDER, 'r') as f:
		ctag_to_ix = pkl.load(f)
	with codecs.open("%s/ix_to_ctag.txt" % cf.ASSET_FOLDER, 'r', 'utf-8') as f:
		ix_to_ctag = [line.strip() for line in f]


	data_iterators, test_dataset = load_datasets(word_to_ix, wtag_to_ix, char_to_ix, ctag_to_ix, ix_to_char, ix_to_word)

	if cf.USE_PRETRAINED_WORD_EMBEDDINGS:
		pretrained_embeddings = get_trimmed_emb_vectors(cf.EMB_TRIMMED_FILENAME)
		logger.info("Loaded %d pretrained embeddings." % len(pretrained_embeddings)) 
	else:
		pretrained_embeddings = None

	return data_iterators, test_dataset, pretrained_embeddings, word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag
