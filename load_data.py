import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader

import codecs
from config import *

from data_utils import TabbedCorpusReader

from nltk.corpus.reader import ConllCorpusReader

import torch


# Converts a dataset to a numpy format so it can be loaded into the DataLoader.
# if test_set is True, ignore min/max lengths
def tagged_sents_to_numpy(tagged_sents, word_to_ix, wtag_to_ix, char_to_ix, ctag_to_ix, ix_to_char, ix_to_word, dataset, word_index):

	# Replace any words that appear in the data with the corresponding replacement term,
	# such as replacing all hashtags with "<HASHTAG>".
	def auto_replace_word(w):
		#if any(w.startswith(s) for s in cf.ignore_words_starting_with):
		#	return "<UNK>"
		for k, v in cf.ignored_words_replacement_map.items():
			if w.startswith(k):
				w = v
		return w

	data_f = [] # Flagger data
	data_w = []
	data_x = []
	data_y = []
	rejected_sents = [] # A list to store 'rejected' sents, i.e. ones that were too short/long.
	rejected_words = []
	filtered_words = []
	rejected_tags  = []
	non_alphabetical_words = []

	if cf.USE_PRETRAINED_WORD_EMBEDDINGS:
		pretrained_embeddings = get_trimmed_emb_vectors(cf.EMB_TRIMMED_FILENAME)

	debugf = codecs.open("debug_bert3.txt", "w", "utf-8")

	for sent in tagged_sents:

		words, tags = zip(*sent)
		if len(words) != len(tags):
			raise Exception("Words and tags are not the same length.")
		

		if cf.GRANULARITY == WORD_LEVEL and (len(words) > cf.MAX_SENT_LENGTH or len(words) < cf.MIN_SENT_LENGTH):
			rejected_sents.append(sent)
		
		#if len(cf.ignored_words_replacement_map) > 0:
		words = [auto_replace_word(w) for w in words]
		
		

		if cf.GRANULARITY == WORD_LEVEL:			
			if cf.MODEL_TYPE == S2S:

				wordz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)

				if cf.EMBEDDING_MODEL in ["Elmo", "Bert"]:
					for jj, (word, tag) in enumerate(sent):


						debugf.write("%s %s %s\n" % (word, tag, str(pretrained_embeddings[word_index][:5])))

						#if word == "the":
						#	print (word, tag, pretrained_embeddings[word_index][:5])

						#print ix_to_word[word_index], word
						if jj >= cf.MAX_SENT_LENGTH:							
							
							#print "(SKIP)", ix_to_word[word_index], word
							word_index += 1
							continue

						#print ix_to_word[word_index], word

						wordz[jj] = word_index
						#print ix_to_word[word_index], word, word_index
						if ix_to_word[word_index] != word:
							print(sent)
							raise Exception("Word and index mismatch at word index %d (%s / %s)" % (word_index, ix_to_word[word_index], word))
						word_index += 1
				else:
					wordz[:len(words)] = np.asarray([word_to_ix[w] for w in words[:cf.MAX_SENT_LENGTH]], )

				data_w.append([0])	# data_w is a placeholder, no need to use it for word-level model.
				data_x.append(wordz)

				tagz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)	
				
				if cf.FLAGGER_MODE:
					# TODO: Am not sure if this part works yet
					tagz[:len(tags)] = np.asarray([(0 if tags[i] == words[i] else 1) for i, t in enumerate(tags[:cf.MAX_SENT_LENGTH])], )
				else:
					tagz[:len(tags)] = np.asarray([wtag_to_ix[w] for w in tags[:cf.MAX_SENT_LENGTH]], )

				if cf.WORD_LEVEL_WITH_FLAGGER:
					flagz = np.zeros(cf.MAX_SENT_LENGTH, dtype=int)	
					flagz[:len(tags)] = np.asarray([(0 if tags[i] == "<SELF>" else 1) for i, t in enumerate(tags[:cf.MAX_SENT_LENGTH])], )

					data_f.append(flagz)
				data_y.append(tagz)	
			elif cf.MODEL_TYPE == S21:
				raise Exception("Word level S21 not supported")
				#tag   = sent[1]
				#data_y.append(wtag_to_ix[tag])

		elif cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
			word_start_index = word_index
			for i, (word, tag) in enumerate(sent):

				


				if len(word) > cf.MAX_WORD_LENGTH:
					rejected_words.append(word)
				if len(tag) > cf.MAX_WORD_LENGTH:
					rejected_tags.append(word)

				# Any words starting with #, @ etc are not included in the training set.
				if any(word.startswith(s) for s in cf.ignore_words_starting_with) and dataset == "train":
					filtered_words.append(word)
					word_index += 1
					continue
				if cf.ignore_non_alphabetical and not any(c.isalpha() for c in word) and dataset == "train":
					non_alphabetical_words.append(word)
					word_index += 1					
					continue

						
			
				charz = np.zeros(cf.MAX_WORD_LENGTH, dtype=int)
				charz[:min(len(word), cf.MAX_WORD_LENGTH)] = np.asarray([char_to_ix[c] for c in word[:cf.MAX_WORD_LENGTH]], )
				
				if cf.MODEL_TYPE == S2S:
					ctagz = np.zeros(cf.MAX_WORD_LENGTH, dtype=int)	
					ctagz[:min(len(tag), cf.MAX_WORD_LENGTH)] = np.asarray([ctag_to_ix[c] for c in tag[:cf.MAX_WORD_LENGTH]], )
					data_y.append(ctagz)
				elif cf.MODEL_TYPE == S21:
					if cf.FLAGGER_MODE:
						output_tag = 0 if word == tag else 1
					else:
						output_tag = wtag_to_ix[tag]
					data_y.append(output_tag)
				
					
				if cf.GRANULARITY == CHAR_LEVEL:
					data_w.append([0])
				elif cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
					word = auto_replace_word(word)
				
					
					if cf.EMBEDDING_MODEL in ["Elmo", "Bert"]:
						word_ids = [0, 0, 0] + [(word_start_index + iii) for iii, ww in enumerate(words)] + [0, 0, 0]
					else:
						word_ids = [0, 0, 0] + [word_to_ix[ww] for ww in words] + [0, 0, 0]

					context_left = []
					context_right = []

					#print word_ids 

					context_wordz = word_ids[i-3+3: i+4+3]

					#for xi in range(i+1, -1, -1):	# Left context, ignoring ignored terms such as hashtags/at mentions
					#	if len(context_left) == 3: continue
					#	#if not ix_to_word[word_ids[xi]] in cf.ignored_words_replacement_map.values():
					#	context_left.append(word_ids[xi])
					#for xi in range(i+3, len(word_ids)):	# Right context
					#	if len(context_right) == 3: continue
					#	#if not ix_to_word[word_ids[xi]] in cf.ignored_words_replacement_map.values():
					#	context_right.append(word_ids[xi])
					#context_wordz = context_left[::-1] + [word_to_ix[word]] + context_right


					#context_wordz = words[max(0, i) : min(len(words), i+5)]	 # hardcoded to 5 for now

					#if len(context_wordz) != 5:
					#print ""
					#print "SENT:", sent 
					#print "WORD:", word
					#print "CTX: ", context_wordz
					#print "CTX: ", [ix_to_word[ix] for ix in context_wordz]
					data_w.append(context_wordz)

					#if cf.EMBEDDING_MODEL == "Elmo":
					word_index += 1
					#if word_index > 100:
					#	exit()
				data_x.append(charz)
				
	
	# If there are not enough sentences to create a set of perfectly-sized batches, append some batches that are all 0s.
	# These will be ignored by the model.
	if cf.MODEL_TYPE == S2S:
		if cf.GRANULARITY == WORD_LEVEL:
			to_pad = cf.BATCH_SIZE - (len(data_x) % cf.BATCH_SIZE)
			for i in range(to_pad):
				data_w.append( np.asarray([0] ))				
				data_x.append( np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
				data_y.append( np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
				if cf.WORD_LEVEL_WITH_FLAGGER:
					data_f.append(np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
				
		elif cf.GRANULARITY == CHAR_LEVEL:
			to_pad = cf.BATCH_SIZE - (len(data_x) % cf.BATCH_SIZE)
			for i in range(to_pad):
				data_w.append( np.asarray([0] ))
				data_x.append( np.zeros(cf.MAX_WORD_LENGTH, dtype = int))
				data_y.append( np.zeros(cf.MAX_WORD_LENGTH, dtype = int))
		elif cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
			to_pad = cf.BATCH_SIZE - (len(data_x) % cf.BATCH_SIZE)
			for i in range(to_pad):
				data_w.append( np.zeros(cf.MAX_SENT_LENGTH, dtype = int))
				data_x.append( np.zeros(cf.MAX_WORD_LENGTH, dtype = int))
				data_y.append( np.zeros(cf.MAX_WORD_LENGTH, dtype = int))
	elif cf.MODEL_TYPE == S21:
		if cf.GRANULARITY == CHAR_LEVEL:
			to_pad = cf.BATCH_SIZE - (len(data_x) % cf.BATCH_SIZE)
			for i in range(to_pad):
				data_w.append( np.asarray([0] ))
				data_x.append( np.zeros(cf.MAX_WORD_LENGTH, dtype = int))
				data_y.append( 0 )			

	# 	print len(data_x)
	# 	print len(data_w)

	return np.asarray(data_w), np.asarray(data_x), np.asarray(data_y), np.asarray(data_f), rejected_sents, rejected_words, filtered_words, non_alphabetical_words, rejected_tags, word_index

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
	word_index = 1 # Used for elmo models only
	for i, dataset in enumerate(["train", "test"]):

		#if cf.MODEL_TYPE == S2S:
		corpusReader = ConllCorpusReader(cf.DATA_FOLDER, [[cf.TRAIN_FILENAME, cf.TEST_FILENAME][i]], ['words', 'pos'])
		#elif cf.MODEL_TYPE == S21:
		#	corpusReader = TabbedCorpusReader(cf.DATA_FOLDER, [[cf.TRAIN_FILENAME, cf.TEST_FILENAME][i]])

		tagged_sents = corpusReader.tagged_sents()
		data_w, data_x, data_y, data_f, rejected_sents, rejected_words, filtered_words, non_alphabetical_words, rejected_tags, word_index = tagged_sents_to_numpy(tagged_sents, word_to_ix, wtag_to_ix, char_to_ix, ctag_to_ix, ix_to_char, ix_to_word, dataset, word_index)
		if cf.WORD_LEVEL_WITH_FLAGGER:
			myDataset = MyDatasetWithFlags(data_w, data_x, data_y, data_f)
		else:
			myDataset = MyDataset(data_w, data_x, data_y)

		data_iterator = DataLoader(myDataset, batch_size=cf.BATCH_SIZE, pin_memory=True)
		data_iterators[dataset] = data_iterator
			#for d in data_iterator:
		#		torch.set_printoptions(threshold = 5000000)
	#			print d 
	#			exit()
		logger.info("Loaded %d %s batches.\n" % (len(data_iterator), dataset) +
			"      (%d x %d = ~%d %s total)" % (len(data_iterator), cf.BATCH_SIZE, len(data_iterator) * cf.BATCH_SIZE, "words" if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL] else "sentences"))
		if len(rejected_sents) > 0:
			logger.warning("%d of %d sentences from the %s set were trimmed due to being too long or short." % (len(rejected_sents), len(tagged_sents) + len(rejected_sents), dataset))
		if len(rejected_words) > 0:
			logger.warning("%d words from the %s set were trimmed due to being too long." % (len(rejected_words), dataset))
		if len(rejected_tags) > 0:
			logger.warning("%d labels from the %s set were trimmed due to being too long." % (len(rejected_tags), dataset))
		if len(filtered_words) > 0:
			logger.info("%d words were filtered from the %s set due to beginning with undesirable character sequences." % (len(filtered_words), dataset))
		if len(non_alphabetical_words) > 0:
			logger.info("%d words were filtered from the %s set due to being entirely non-alphabetical." % (len(non_alphabetical_words), dataset))
	return data_iterators

class MyDatasetWithFlags(Dataset):
	def __init__(self, w, x, y, f):
		super(MyDatasetWithFlags, self).__init__()
		self.w = w
		self.x = x
		self.y = y
		self.f = f 

	def __getitem__(self, ids):
		return self.w[ids], self.x[ids], self.y[ids], self.f[ids]

	def __len__(self):
		return self.x.shape[0]


class MyDataset(Dataset):
	def __init__(self, w, x, y):
		super(MyDataset, self).__init__()
		self.w = w
		self.x = x
		self.y = y


	def __getitem__(self, ids):
		return self.w[ids], self.x[ids], self.y[ids], 0

	def __len__(self):
		return self.x.shape[0]

def load_data():
	with open("%s/word_to_ix.pkl" % cf.ASSET_FOLDER, 'rb') as f:
		word_to_ix = pkl.load(f)
	with open("%s/ix_to_word.txt" % cf.ASSET_FOLDER, 'r') as f:
		ix_to_word = [line.strip() for line in f]
	with open("%s/wtag_to_ix.pkl" % cf.ASSET_FOLDER, 'rb') as f:
		wtag_to_ix = pkl.load(f)
	with open("%s/ix_to_wtag.txt" % cf.ASSET_FOLDER, 'r') as f:
		ix_to_wtag = [line.strip() for line in f]
	with open("%s/char_to_ix.pkl" % cf.ASSET_FOLDER, 'rb') as f:
		char_to_ix = pkl.load(f)
	with open("%s/ix_to_char.txt" % cf.ASSET_FOLDER, 'r') as f:
		ix_to_char = [line.strip() for line in f]
	with open("%s/ctag_to_ix.pkl" % cf.ASSET_FOLDER, 'rb') as f:
		ctag_to_ix = pkl.load(f)
	with open("%s/ix_to_ctag.txt" % cf.ASSET_FOLDER, 'r') as f:
		ix_to_ctag = [line.strip() for line in f]

	if cf.MODEL_TYPE == S21 and cf.FLAGGER_MODE:
		ix_to_ctag = { 0: "False", 1: "True" }
		ctag_to_ix = ["False", "True"]
	elif cf.MODEL_TYPE == S2S and cf.FLAGGER_MODE:
		ix_to_wtag = { 0: "False", 1: "True" }
		wtag_to_ix = ["False", "True"]	

	data_iterators = load_datasets(word_to_ix, wtag_to_ix, char_to_ix, ctag_to_ix, ix_to_char, ix_to_word)

	if cf.USE_PRETRAINED_WORD_EMBEDDINGS:
		pretrained_embeddings = get_trimmed_emb_vectors(cf.EMB_TRIMMED_FILENAME)
		logger.info("Loaded %d pretrained word embeddings." % len(pretrained_embeddings)) 
	else:
		pretrained_embeddings = None



	if cf.USE_PRETRAINED_CHAR_EMBEDDINGS:
		pretrained_char_embeddings = get_trimmed_emb_vectors(cf.CHAR_EMB_TRIMMED_FILENAME)
		logger.info("Loaded %d pretrained character embeddings." % len(pretrained_char_embeddings)) 
	else:
		pretrained_char_embeddings = None


	return data_iterators, pretrained_embeddings, pretrained_char_embeddings, word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag
