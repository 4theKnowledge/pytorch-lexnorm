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
def tagged_sents_to_numpy(tagged_sents, word_to_ix, tag_to_ix):
	data_x = []
	data_y = []
	rejected_sents = [] # A list to store 'rejected' sents, i.e. ones that were too short/long.
	for sent in tagged_sents:

		if cf.MODEL_TYPE == S2S:
			words, tags = zip(*sent)
			if len(words) != len(tags):
				raise Exception("Words and tags are not the same length.")
		elif cf.MODEL_TYPE == S21:
			words, tag = sent[0], sent[1]

		if len(words) > cf.MAX_SENT_LENGTH or len(words) < cf.MIN_SENT_LENGTH:
			rejected_sents.append(sent)
			continue

		wordz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)			
		wordz[:len(words)] = np.asarray([word_to_ix[w] for w in words], )
		data_x.append(wordz)

		if cf.MODEL_TYPE == S2S:
			tagz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)	
			tagz[:len(tags)] = np.asarray([tag_to_ix[w] for w in tags], )
			data_y.append(tagz)	
		elif cf.MODEL_TYPE == S21:
			tag   = sent[1]
			data_y.append(tag_to_ix[tag])

	
	# If there are not enough sentences to create a set of perfectly-sized batches, append some batches that are all 0s.
	# These will be ignored by the model.
	if cf.MODEL_TYPE == S2S:
		to_pad = cf.BATCH_SIZE - (len(data_x) % cf.BATCH_SIZE)
		for i in range(to_pad):
			data_x.append( np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
			data_y.append( np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
	return np.asarray(data_x), np.asarray(data_y), rejected_sents

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
def load_datasets(word_to_ix, tag_to_ix):
	data_iterators = { "train": None, "dev": None }
	test_dataset = []
	for i, dataset in enumerate(["train", "dev", "test"]):

		if cf.MODEL_TYPE == S2S:
			corpusReader = ConllCorpusReader(cf.DATA_FOLDER, [[cf.TRAIN_FILENAME, cf.DEV_FILENAME, cf.TEST_FILENAME][i]], ['words', 'pos'])
		elif cf.MODEL_TYPE == S21:
			corpusReader = TabbedCorpusReader(cf.DATA_FOLDER, [[cf.TRAIN_FILENAME, cf.DEV_FILENAME, cf.TEST_FILENAME][i]])

		tagged_sents = corpusReader.tagged_sents()
		data_x, data_y, rejected_sents = tagged_sents_to_numpy(tagged_sents, word_to_ix, tag_to_ix)
		myDataset = MyDataset(data_x, data_y)

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
			logger.warning("%d of %d sentences were discluded from the %s set due to being too long or short." % (len(rejected_sents), len(tagged_sents) + len(rejected_sents), dataset))
	return data_iterators, test_dataset

class MyDataset(Dataset):
	def __init__(self, x, y):
		super(MyDataset, self).__init__()
		self.x = x
		self.y = y

	def __getitem__(self, ids):
		return self.x[ids], self.y[ids]

	def __len__(self):
		return self.x.shape[0]

def load_data():
	with open("asset/word_to_ix.pkl", 'r') as f:
		word_to_ix = pkl.load(f)
	with codecs.open("asset/ix_to_word.txt", 'r', 'utf-8') as f:
		ix_to_word = [line.strip() for line in f]
	with open("asset/tag_to_ix.pkl", 'r') as f:
		tag_to_ix = pkl.load(f)
	with codecs.open("asset/ix_to_tag.txt", 'r', 'utf-8') as f:
		ix_to_tag = [line.strip() for line in f]

	data_iterators, test_dataset = load_datasets(word_to_ix, tag_to_ix)

	if cf.USE_PRETRAINED_EMBEDDINGS:
		pretrained_embeddings = get_trimmed_emb_vectors(cf.EMB_TRIMMED_FILENAME)
		logger.info("Loaded %d pretrained embeddings." % len(pretrained_embeddings)) 
	else:
		pretrained_embeddings = None

	return data_iterators, test_dataset, pretrained_embeddings, word_to_ix, ix_to_word, tag_to_ix, ix_to_tag
