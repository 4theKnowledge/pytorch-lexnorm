import logging as logger
import sys, torch
from colorama import Fore, Back, Style
# logger.basicConfig(format=Fore.CYAN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.DEBUG)
# logger.basicConfig(format=Fore.GREEN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoggingFormatter(logger.Formatter):
    def format(self, record):
        #compute s according to record.levelno
        #for example, by setting self._fmt
        #according to the levelno, then calling
        #the superclass to do the actual formatting

        if record.levelno == 10:
        	return Fore.CYAN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL,  record.msg, )
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

class Config():

	def __init__(self):

		self.MODEL_TYPE 		= S2S   # Can be either "S2S" (sequence to sequence) or "S21" (sequence to one).

		self.EMBEDDING_DIM 		= 300	# The number of dimensions to use for embeddings. Usually 300.
		self.HIDDEN_DIM 		= 300	# The number of dimensions of the hidden layer.
		self.BATCH_SIZE 		= 100	# The batch size (larger uses more memory but is faster)
		self.LEARNING_RATE		= 0.1	# The learning rate

		self.MIN_SENT_LENGTH 	= 1		# The minimum length of a sentence. Sentences smaller than this will not be trained on.
		self.MAX_SENT_LENGTH 	= 60	# The maximum length of a sentence. Sentences larger than this will not be trained on.
		self.MAX_EPOCHS 		= 3000	# The maximum number of epochs to run.
		self.EARLY_STOP			= False  # Whether to stop when no progress has been made for the last 10 epochs. (i.e. loss has not improved)

		self.USE_PRETRAINED_EMBEDDINGS = True

		self.CHARACTER_LEVEL 	= False

		# self.DATA_FOLDER		= 'data/datasets/twitter'
		# self.TRAIN_FILENAME		= 'train.tsv'
		# self.DEV_FILENAME		= 'dev.tsv'
		# self.TEST_FILENAME		= 'test.tsv'

		self.DATA_FOLDER		= 'data/datasets/twitter_lexnorm_word'
		self.TRAIN_FILENAME		= 'train.txt'
		self.DEV_FILENAME		= 'test.txt'
		self.TEST_FILENAME		= 'test.txt'

		#self.DATA_FOLDER		= 'data/datasets/dmp'
		#self.TRAIN_FILENAME		= 'train.txt'
		#self.DEV_FILENAME		= 'dev.txt'
		#self.TEST_FILENAME		= 'test.txt'

		self.EMBEDDINGS_FOLDER  = 'data/fasttext'
		
		self.EMB_VEC_FILENAME   = 'data/fasttext/wiki-news-300d-1M-subword.vec'
		self.EMB_BIN_FILENAME   = 'data/fasttext/wiki-news-300d-1M-subword.bin'


		#self.EMB_VEC_FILENAME   = 'data/fasttext/cc.en.300.vec'
		#self.EMB_BIN_FILENAME   = 'data/fasttext/cc.en.300.bin'

		self.EMB_OOV_FILENAME     = 'asset/oov_embeddings.vec'
		self.EMB_TRIMMED_FILENAME = 'asset/fasttext_trimmed.npz'
		self.OOV_TOKENS_FILENAME  = 'asset/oov_tokens.txt'

		#self.DATASET			= self.TRAIN_DATA

		#self.DATA_FILENAME		= 'train.txt'
		# self.SOS_TOKEN = "<SOS>"
		# self.EOS_TOKEN = "<EOS>"


