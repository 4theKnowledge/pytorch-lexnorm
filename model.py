import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from config import *
import time

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class LSTMTagger(nn.Module):

	def __init__(self, model_type, word_embedding_dim, char_embedding_dim, hidden_dim, vocab_size_char, vocab_size_word, tag_size, batch_size, max_word_length, max_sent_length, pretrained_embeddings=None, pretrained_char_embeddings=None):
		super(LSTMTagger, self).__init__()

		self.word_embedding_dim = word_embedding_dim
		self.char_embedding_dim = char_embedding_dim
		self.hidden_dim 	 = hidden_dim
		self.model_type 	 = model_type
		self.batch_size 	 = batch_size
		self.vocab_size_word = vocab_size_word
		self.vocab_size_char = vocab_size_char
		self.tag_size   	 = tag_size
		self.max_word_length = max_word_length
		self.max_sent_length = max_sent_length

		self.char_embeddings = nn.Embedding(vocab_size_char, char_embedding_dim)
		if pretrained_char_embeddings is not None:
			self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrained_char_embeddings))
			self.char_embeddings.weight.requires_grad = False

		print(pretrained_embeddings)
		self.word_embeddings = nn.Embedding(vocab_size_word, word_embedding_dim)
		print(self.word_embeddings.weight)
		if pretrained_embeddings is not None:
			#print "hello"
			self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
			#if not cf.UPDATE_PRETRAINED_EMBEDDINGS:
		self.word_embeddings.weight.requires_grad = False
		print(self.word_embeddings.weight)
		

		self.hidden = self.init_hidden()

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		#self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim * 2, tag_size)

		

		self.setup_model()


	def setup_model(self):
		self.max_x_length = self.max_sent_length
		self.embeddings = self.word_embeddings

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(4, self.batch_size, self.hidden_dim, device=device))#,
				#torch.zeros(4, self.batch_size, self.hidden_dim, device=device))


	def tag_sequence(self, sequence):
		pass


	def forward(self, batch_w, batch_x, batch_w_lengths, batch_x_lengths):

		self.hidden = self.init_hidden()
		
		batch_size, seq_len = batch_x.size()

		# 1. Embed the input
		batch = self.embeddings(batch_x)


		

		# 2. Pack the sequence
		batch = torch.nn.utils.rnn.pack_padded_sequence(batch, [self.max_x_length] * batch_size, batch_first=True)

		# 3. Run through lstm
		batch, self.hidden = self.recurrent_layer(batch, self.hidden)

		# Undo packing
		batch, _ = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first = True)

		batch = batch.contiguous()
		batch = batch.view(-1, batch.shape[2])

		#print batch.size()

		batch = self.hidden2tag(batch)

		#print batch.size()

		batch = F.log_softmax(batch, dim=1)

		Y_hat = batch.view(batch_size, seq_len, self.tag_size)

		#print batch.size()


		#print self.embeddings.weight

		if self.model_type == S2S:
			return Y_hat 
		elif self.model_type == S21:
			return torch.stack([ y[batch_x_lengths[i] - 1] for i, y in enumerate(Y_hat) ])

	# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
	def calculate_loss(self, Y_hat, Y):
		# TRICK 3 ********************************
		# before we calculate the negative log likelihood, we need to mask out the activations
		# this means we don't want to take into account padded items in the output vector
		# simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
		# and calculate the loss on that.

		if(self.model_type == S2S):
			#print Y, Y.size()
			#mask = torch.all(torch.equal(Y, 0), axis = 1)

			ymask = ~torch.all(Y == 0, dim=1)

			Y = Y[ymask]	# Ignore any words or sentences that are completely padding.
			Y_hat = Y_hat[ymask] # Ignore those same rows in Y_hat

			Y = Y.view(-1).to(device)
			Y_hat = Y_hat.view(-1, self.tag_size)
			# create a mask by filtering out all tokens that ARE NOT the padding token
			#tag_pad_token = word_to_ix['<PAD>']



			# In the character-level version, we need to be able to predict the padding token.
			if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
				mask = (Y > -1).float()
			else:
				mask = (Y > 0).float()

			# count how many tokens we have
			nb_tokens = int(torch.sum(mask).item())	

			#print nb_tokens
			# pick the values for the label and zero out the rest with the mask
			Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

			if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
				nb_tokens = Y_hat.shape[0]		

			# compute cross entropy loss which ignores all <PAD> tokens
			ce_loss = -torch.sum(Y_hat) / nb_tokens

		elif self.model_type == S21:
			# We only take the output of the LSTM from the last timestep of each batch, i.e.
			# the last non-padding word in each batch.
			# X_lengths is a list of the batch lengths, so y[X_lengths[i] - 1] gives us the output of the LSTM that corresponds
			# to the last word in the sequence.
			
			#print Y_hat
	
			#ymask = ~torch.all(Y == 0, dim=0)
			#Y = Y[ymask]	# Ignore any words or sentences that are completely padding.
			#Y_hat = Y_hat[ymask] # Ignore those same rows in Y_hat

			#print Y
			#print Y_hat

			Y = Y.view(-1).to(device)

			# Get the probabilties of Y_hat with respect to the indexes of the tags in Y
			Y_hat = Y_hat[range(Y_hat.shape[0]), Y]

			nb_tokens = len(Y_hat)
			ce_loss = -torch.sum(Y_hat) / nb_tokens

		return ce_loss




class CharLSTMTagger(LSTMTagger):
	def setup_model(self):
		self.max_x_length = self.max_word_length
		self.embeddings = self.char_embeddings
		self.recurrent_layer = nn.GRU(self.char_embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)


class WordLSTMTagger(LSTMTagger):
	def setup_model(self):
		self.max_x_length = self.max_sent_length
		self.embeddings = self.word_embeddings
		self.recurrent_layer = nn.GRU(self.word_embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)


class WordTaggerWithFlagger(LSTMTagger):
	def setup_model(self):
		self.max_x_length = self.max_sent_length
		self.embeddings = self.word_embeddings
		self.recurrent_layer = nn.GRU(self.word_embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)
		self.recurrent_layer_f = nn.GRU(self.word_embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)
		self.hidden_f = self.init_hidden()
		self.hidden2flag = nn.Linear(self.hidden_dim * 2, 2) # Two flags (True, False)

	def forward(self, batch_f, batch_x, batch_w_lengths, batch_x_lengths):

		#print batch_x
		#print batch_f
		#print "___"
		#time.sleep(0.5)

		self.hidden = self.init_hidden()
		
		batch_size, seq_len = batch_x.size()

		# 1. Embed the input
		batch = self.embeddings(batch_x)

		# 2. Pack the sequence
		batch_packed = torch.nn.utils.rnn.pack_padded_sequence(batch, [self.max_x_length] * batch_size, batch_first=True)

		# 3. Run through lstm
		batch, self.hidden = self.recurrent_layer(batch_packed, self.hidden)
		batch_f, self.hidden_f = self.recurrent_layer_f(batch_packed, self.hidden)

		# Undo packing
		batch, _ = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first = True)
		batch_f, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_f, batch_first = True)

		batch = batch.contiguous()
		batch = batch.view(-1, batch.shape[2])

		batch_f = batch_f.contiguous()
		batch_f = batch_f.view(-1, batch_f.shape[2])

		batch = self.hidden2tag(batch)
		batch_f = self.hidden2flag(batch_f)


		batch = F.log_softmax(batch, dim=1)
		batch_f = F.log_softmax(batch_f, dim=1)

		Y_hat = batch.view(batch_size, seq_len, self.tag_size)
		Y_hat_f = batch_f.view(batch_size, seq_len, 2)



		return Y_hat, Y_hat_f


	def calculate_loss(self, Y_hat, Y_hat_f, Y, Y_f):
	
		ymask = ~torch.all(Y == 0, dim=1)

		Y = Y[ymask]	# Ignore any sentences that are completely padding.
		Y_hat = Y_hat[ymask] # Ignore those same rows in Y_hat

		Y = Y.view(-1).to(device)
		Y_hat = Y_hat.view(-1, self.tag_size)

		# Flagger portion
		Y_f = Y_f[ymask]
		Y_hat_f = Y_hat_f[ymask]	 # Also ignore all sentences that are completely padding for the flagger as well.

		Y_f = Y_f.view(-1).to(device)
		Y_hat_f = Y_hat_f.view(-1, 2)

		mask = (Y > 0).float()
	
		# count how many tokens we have
		nb_tokens = int(torch.sum(mask).item())	

		#print nb_tokens
		# pick the values for the label and zero out the rest with the mask
		Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

		Y_hat_f = Y_hat_f[range(Y_hat_f.shape[0]), Y_f] * mask


		#print Y_hat
		#print Y_hat_f
		
		# compute cross entropy loss which ignores all <PAD> tokens
		ce_loss_words = -torch.sum(Y_hat) / nb_tokens
		ce_loss_flags = -torch.sum(Y_hat_f) / nb_tokens

		#print ce_loss_words, ce_loss_flags
		#print "---"


		return ce_loss_words + ce_loss_flags

def new_parameter(*size):
	out = nn.Parameter(torch.FloatTensor(*size))
	torch.nn.init.xavier_normal_(out)
	return out






class CombinedLSTMTagger(LSTMTagger):

	#def init_hidden_w(self):
	#	return (torch.zeros(2, self.batch_size, self.char_embedding_dim / 2, device=device))


	def init_hidden_1(self):		
		return (torch.zeros(2, self.batch_size, self.hidden_dim, device=device))	

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(4, self.batch_size, self.hidden_dim, device=device))

	def setup_model(self):
		self.recurrent_layer = nn.GRU(self.char_embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)

		if self.char_embedding_dim % 2 != 0:
			logger.error("Char embedding dim must be even for char and word level model.")
			exit(0)
		
		self.recurrent_layer_word = nn.GRU(self.word_embedding_dim, self.hidden_dim, bidirectional = True, num_layers=2)

		self.recurrent_layer_decode = nn.GRU(self.hidden_dim * 2, self.hidden_dim, bidirectional=True, num_layers=1)

		self.attention = new_parameter(self.batch_size, self.word_embedding_dim, 1)
		pass


	def forward(self, batch_w, batch_x, batch_w_lengths, batch_x_lengths):
		self.hidden = self.init_hidden()
		self.hidden_w = self.init_hidden()
		self.hidden_dec = self.init_hidden_1()
		
		batch_size, seq_len_x = batch_x.size()
		_, seq_len_w 		  = batch_w.size()

		# 1. Embed the input
		batch_w = self.word_embeddings(batch_w)
		batch_x = self.char_embeddings(batch_x)		

		# 1.1 Attention mechanism
		# https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html
		attention_score = torch.matmul(batch_w, self.attention).squeeze()
		attention_score = F.softmax(attention_score, dim=1).view(batch_w.size(0), batch_w.size(1), 1)
		batch_w = batch_w * attention_score

		#batch = torch.cat((batch_w, batch_x), dim=1)

		#print batch_w.size()

		# Pad the word embedding matrix to the same size as the character embedding matrix
		#target = torch.zeros(batch_size, self.max_word_length, self.word_embedding_dim, device=device)
		#target[:, :7, :] = batch_w
		#batch_w = target

		batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x, [self.max_word_length] * batch_size, batch_first=True)
		batch_x, self.hidden = self.recurrent_layer(batch_x, self.hidden)
		batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first = True)


		#print batch_w
		batch_w = torch.nn.utils.rnn.pack_padded_sequence(batch_w, [self.max_sent_length] * batch_size, batch_first=True)
		batch_w, self.hidden_w = self.recurrent_layer_word(batch_w, self.hidden_w)
		batch_w, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_w, batch_first = True)


		batch = torch.cat((batch_w, batch_x), dim=1)

		#print batch.size()

		# 2. Pack the sequence
		batch = torch.nn.utils.rnn.pack_padded_sequence(batch, [self.max_word_length + self.max_sent_length] * batch_size, batch_first=True)



		# 3. Run through lstm
		batch, self.hidden_dec = self.recurrent_layer_decode(batch, self.hidden_dec)

		# Undo packing
		batch, _ = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first = True)

		#batch_w = batch_w.contiguous().view(-1, batch_w.shape[2]).transpose_(0, 1)
		batch = batch[:, self.max_sent_length:, :]
		batch = batch.contiguous().view(-1, batch.shape[2])#.transpose_(0, 1)


		batch = self.hidden2tag(batch)


		#batch = batch.contiguous().view(batch_size, self.max_word_length, self.tag_size)

		# Ignore the first max_sent_length predictions as they correspond to the timesteps of the word embeddings
		#batch = batch[:, self.max_sent_length:, :]	
		#batch = batch.contiguous().view(batch_size * self.max_word_length, self.tag_size)
		batch = F.log_softmax(batch, dim=1)		
		Y_hat = batch.view(batch_size, self.max_word_length, self.tag_size)

		return Y_hat 
			




class FeedForwardBert(nn.Module):

	def __init__(self, model_type, word_embedding_dim, char_embedding_dim, hidden_dim, vocab_size_char, vocab_size_word, tag_size, batch_size, max_word_length, max_sent_length, pretrained_embeddings=None, pretrained_char_embeddings=None):
		super(FeedForwardBert, self).__init__()
		logger.info("Using FeedForwardBert")
		self.hidden_dim = hidden_dim
		self.model_type = model_type

		self.batch_size = batch_size
		self.vocab_size = vocab_size_word
		self.tag_size   = tag_size
		self.max_sent_length = max_sent_length

		self.word_embeddings = nn.Embedding(vocab_size_word, word_embedding_dim)
		
		if cf.USE_PRETRAINED_WORD_EMBEDDINGS:
			self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
			self.word_embeddings.weight.requires_grad = False

		#self.hidden1 = nn.Linear(word_embedding_dim, hidden_dim)
		#self.relu1 = nn.ReLU()
		#self.dropout = nn.Dropout(p = 0.5)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(word_embedding_dim, tag_size)


	def tag_sequence(self, sequence):
		pass

	def forward(self, batch_w, batch_x, batch_w_lengths, batch_x_lengths):	
		
		batch_size, seq_len = batch_x.size()

		batch = self.word_embeddings(batch_x)
		#batch = self.dropout(batch)

		#batch = self.hidden1(batch)
		#batch = self.relu1(batch)
		batch = self.hidden2tag(batch)

		#batch = F.log_softmax(batch, dim=1)

		#Y_hat = batch.view(batch_size, seq_len, self.tag_size)

		#return Y_hat
		return batch

	def calculate_loss(self, Y_hat, Y):

		'''ymask = ~torch.all(Y == 0, dim=1)

		Y = Y[ymask]	# Ignore any words or sentences that are completely padding.
		Y_hat = Y_hat[ymask] # Ignore those same rows in Y_hat

		Y = Y.view(-1).to(device)
		Y_hat = Y_hat.view(-1, self.tag_size)

		# In the character-level version, we need to be able to predict the padding token.
		if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
			mask = (Y > -1).float()
		else:
			mask = (Y > 0).float()

		# count how many tokens we have
		nb_tokens = int(torch.sum(mask).item())	

		# pick the values for the label and zero out the rest with the mask
		Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

		if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
			nb_tokens = Y_hat.shape[0]		

		# compute cross entropy loss which ignores all <PAD> tokens
		ce_loss = -torch.sum(Y_hat) / nb_tokens

		return ce_loss'''


		Y = Y.view(-1).to(device)
		Y_hat = Y_hat.view(-1, self.tag_size)

		
		loss = nn.CrossEntropyLoss()
		return loss(Y_hat, Y)
