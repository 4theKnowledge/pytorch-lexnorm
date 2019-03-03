import logging as logger
import codecs
import sys, torch, os
from datetime import datetime
from colorama import Fore, Back, Style
# logger.basicConfig(format=Fore.CYAN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.DEBUG)
# logger.basicConfig(format=Fore.GREEN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGFILE = None

class LoggingFormatter(logger.Formatter):	

	def format(self, record):
		#compute s according to record.levelno
		#for example, by setting self._fmt
		#according to the levelno, then calling
		#the superclass to do the actual formatting
		
		if LOGFILE:
			message = record.msg.replace(Fore.GREEN, "")
			message = message.replace(Fore.RED, "")
			message = message.replace(Fore.YELLOW, "")
			message = message.replace(Style.RESET_ALL, "")
			LOGFILE.write("%s %s %s\n" % (datetime.now().strftime('%d-%m-%Y %H:%M:%S'), record.levelname.ljust(7), message))

			

		if record.levelno == 10:
			return Fore.CYAN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL,  record.msg)
		elif record.levelno == 20:
			return Fore.GREEN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg) 
		elif record.levelno == 30:
			return Fore.YELLOW + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg)
		#return s

hdlr = logger.StreamHandler(sys.stdout)
hdlr.setFormatter(LoggingFormatter())
logger.root.addHandler(hdlr)
logger.root.setLevel(logger.DEBUG)
#logger.setFormatter(LoggingFormatter())

S2S = "Sequence to sequence"
S21 = "Sequence to one"

CHAR_LEVEL = "Character level"
WORD_LEVEL = "Word level"
CHAR_AND_WORD_LEVEL = "Character and word-level"


CF_MODEL = "Word-level"
CF_DATASET = "Twitter"
CF_PRETRAINED = False
CF_FLAGGER = False
CF_EMBEDDING_MODEL = "FastText"


class Config():

	def load_preset(self, model_type, dataset, use_pretrained_word_embeddings = True, flagger = False, embedding_model = "FastText"):
		if model_type is None:
			raise Exception("Model cannot be None")
		if dataset is None:
			raise Exception("Dataset cannot be None")
		self.MODEL_NAME = model_type + " (" + dataset + ")" + ("" if use_pretrained_word_embeddings else " (NO pretrained embeddings)")
		if flagger:
			self.MODEL_NAME = "Flagger GRU (" + dataset + ")"
		if use_pretrained_word_embeddings:
			self.MODEL_NAME += " (" + embedding_model + ")"
		self.GRANULARITY = {
			"Deep Encoding": CHAR_LEVEL,
			"Word-level": WORD_LEVEL,
			"Combined": CHAR_AND_WORD_LEVEL,
			"Word-level + Flagger": WORD_LEVEL,
		}[model_type]
		if model_type == "Word-level + Flagger":
			self.WORD_LEVEL_WITH_FLAGGER = True
		self.MODEL_TYPE = S21 if flagger else S2S
		self.MAX_WORD_LENGTH = {
			"DMP": 40,
			"Twitter": 40,
			"US Acc": 60,
		}[dataset]
		self.MAX_SENT_LENGTH = {
			"DMP": 150,
			"Twitter": 45,
			"US Acc": 95,
		}[dataset]
		self.ignore_non_alphabetical = {
			"DMP": True,
			"Twitter": True,
			"US Acc": False,
		}[dataset]

		self.ignore_words_starting_with = {
			"DMP": [],
			"Twitter": ["#", "@", "http:", "https:"],
			"US Acc": [],
		}[dataset]
		if model_type == "Word-level":
			self.ignore_words_starting_with = []
		
			
		self.DATA_FOLDER = {
			"DMP": "data/datasets/dmp_lexnorm_word",
			"Twitter": "data/datasets/twitter_lexnorm_word",
			"US Acc": "data/datasets/us_accidents",
		}[dataset]
		if self.GRANULARITY == WORD_LEVEL:
			self.DATA_FOLDER += "_self"

		self.USE_PRETRAINED_WORD_EMBEDDINGS = use_pretrained_word_embeddings
		self.EMBEDDINGS_FOLDER = {
			"DMP": "data/dmp_embeddings/dmp_only_fasttext",
			"Twitter": "data/twitter_embeddings",
			"US Acc": "data/us_accidents_embeddings",
		}[dataset]

		self.ELMO_FOLDER = self.EMBEDDINGS_FOLDER + "/elmo"
		self.BERT_FOLDER = self.EMBEDDINGS_FOLDER + "/bert"


		if embedding_model == "Word2Vec":
			self.EMBEDDINGS_FOLDER += "/word2vec"
		if embedding_model == "FastText":
			self.EMBEDDINGS_FOLDER += "/fasttext"
		#elif embedding_model == "Elmo":
		#	self.EMBEDDINGS_FOLDER += "/elmo"
		self.EMBEDDING_MODEL = embedding_model

		self.FLAGGER_MODE = flagger

		
		self.ignored_words_replacement_map = {
			"DMP": {},
			"Twitter": {
				"#": "<HASHTAG>",
				"@": "<ATMENTION>",
				"http:": "<URL>",
				"https:": "<URL>"
			},
			"US Acc": {},
		}[dataset]

		#self.ELMO_FOLDER = {
		#	"DMP": "data/elmo/dmp",
		#	"Twitter": "data/elmo/twitter",
		#	"US Acc": "data/elmo/us_acc"
		#}[dataset]
		if embedding_model == "Elmo":
			self.WORD_EMBEDDING_DIM = 512
		if embedding_model == "Bert":
			self.WORD_EMBEDDING_DIM = 768
			#self.LEARNING_RATE = 0.1
					
		
	def __init__(self, model_type = None, dataset = None, use_pretrained_word_embeddings = True, flagger = False,  embedding_model = "FastText"):

		#self.MODEL_NAME 		= "Deep Encoding GRU (US Acc)"
		#self.GRANULARITY		= CHAR_LEVEL

		#self.FLAGGER_MODE		= False	# Whether to use 'flagger mode', i.e. classify whether a token should be normalised or not.

		#self.MODEL_TYPE 		= S2S

		self.WORD_EMBEDDING_DIM	= 512	# The number of dimensions to use for word embeddings. Usually 300.
		self.CHAR_EMBEDDING_DIM	= 100	# The number of dimensions to use for char embeddings.
		self.HIDDEN_DIM 		= 512	# The number of dimensions of the hidden layer.
		self.BATCH_SIZE 		= 80	# The batch size (larger uses more memory but is faster)
		self.LEARNING_RATE		= 0.1	# The learning rate


		#self.MAX_WORD_LENGTH	= 60
		#self.MIN_SENT_LENGTH 	= 1		# The minimum length of a sentence. Sentences smaller than this will not be trained on.
		#self.MAX_SENT_LENGTH 	= 95	# The maximum length of a sentence. Sentences larger than this will not be trained on.
		self.MAX_EPOCHS 		= 3000	# The maximum number of epochs to run.
		self.EARLY_STOP			= True  # Whether to stop when no progress has been made for the last 50 epochs. (i.e. loss has not improved)

		#self.ignore_words_starting_with = []#"#", "@", "http:", "https:"] # In the char or char_and_word_level model, any words starting with any elements of the list are removed from the dataset by converting them to empty sequences. 
		#self.ignore_non_alphabetical = False	# Remove entirely non-alphabetical words from the training data

		self.WORD_LEVEL_WITH_FLAGGER = False
		
		self.ignored_words_replacement_map = {}

		self.USE_PRETRAINED_CHAR_EMBEDDINGS = False
		#self.USE_PRETRAINED_WORD_EMBEDDINGS = False#True

		self.MIN_SENT_LENGTH = 1
		
		self.UPDATE_PRETRAINED_EMBEDDINGS = True # Whether to update pre-trained emb weights during training

		#if self.FLAGGER_MODE:
			#self.GRANULARITY = CHAR_LEVEL	
		#	self.MODEL_TYPE = S21	

		

		#self.DATA_FOLDER		= 'data/datasets/us_accidents'
		
		#self.TRAIN_FILENAME		= 'train.txt'
		#self.DEV_FILENAME		= 'test.txt'
		#self.TEST_FILENAME		= 'test.txt'

		#self.EMBEDDINGS_FOLDER  = 'data/us_accidents_embeddings'
		

		''' Everything below is based upon previous variables '''

		self.load_preset(model_type, dataset, use_pretrained_word_embeddings, flagger, embedding_model)
		self.TRAIN_FILENAME		= 'train.txt'
		self.TEST_FILENAME		= 'test.txt'

		if self.GRANULARITY == CHAR_AND_WORD_LEVEL:
			self.MAX_SENT_LENGTH = 7 # Hardcoded to a window size of 5 for now
		if self.GRANULARITY == CHAR_LEVEL:
			self.USE_PRETRAINED_WORD_EMBEDDINGS = False

		self.ASSET_FOLDER = "models/%s/asset" % self.MODEL_NAME

		self.EMB_VEC_FILENAME   = '%s/model.vec'	% self.EMBEDDINGS_FOLDER
		self.EMB_BIN_FILENAME   = '%s/model.bin'	% self.EMBEDDINGS_FOLDER
		self.EMB_OOV_FILENAME     = '%s/oov_embeddings.vec' % self.ASSET_FOLDER
		self.EMB_TRIMMED_FILENAME = '%s/embeddings_trimmed.npz' % self.ASSET_FOLDER



		#self.CHAR_EMBEDDINGS_FOLDER = "data/twitter_embeddings_char"
		#self.CHAR_EMB_VEC_FILENAME   = '%s/twitter_embeddings_char.vec'	% self.CHAR_EMBEDDINGS_FOLDER
		#self.CHAR_EMB_BIN_FILENAME   = '%s/twitter_embeddings_char.bin'	% self.CHAR_EMBEDDINGS_FOLDER		
		#self.CHAR_EMB_OOV_FILENAME     = '%s/char_oov_embeddings.vec' % self.ASSET_FOLDER
		#self.CHAR_EMB_TRIMMED_FILENAME = '%s/char_fasttext_trimmed.npz' % self.ASSET_FOLDER

		self.OOV_TOKENS_FILENAME  = '%s/oov_tokens.txt' % self.ASSET_FOLDER
		self.CHAR_OOV_TOKENS_FILENAME = "%s/char_oov_tokens.txt" % self.ASSET_FOLDER
		

		

	
		if not os.path.exists("models/%s" % self.MODEL_NAME):
			os.makedirs("models/%s" % self.MODEL_NAME)
			os.makedirs("models/%s/asset" % self.MODEL_NAME)
			os.makedirs("models/%s/model_trained" % self.MODEL_NAME)
			os.makedirs("models/%s/predictions" % self.MODEL_NAME)

		global LOGFILE
		LOGFILE = codecs.open("models/%s/log.txt" % self.MODEL_NAME, 'w+', 'utf-8')
		self.LOGFILE = LOGFILE
		

# TODO: Use this config object across all files
cf = Config(CF_MODEL, CF_DATASET, CF_PRETRAINED, CF_FLAGGER, CF_EMBEDDING_MODEL)
