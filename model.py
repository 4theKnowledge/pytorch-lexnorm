import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from config import *
cf = Config()

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class LSTMTagger(nn.Module):

	def __init__(self, model_type, embedding_dim, hidden_dim, vocab_size_char, vocab_size_word, tag_size, batch_size, max_word_length, max_sent_length, pretrained_embeddings=None):
		super(LSTMTagger, self).__init__()

		self.embedding_dim 	 = embedding_dim
		self.hidden_dim 	 = hidden_dim
		self.model_type 	 = model_type
		self.batch_size 	 = batch_size
		self.vocab_size_word = vocab_size_word
		self.vocab_size_char = vocab_size_char
		self.tag_size   	 = tag_size
		self.max_word_length = max_word_length
		self.max_sent_length = max_sent_length

		self.char_embeddings = nn.Embedding(vocab_size_char, embedding_dim)

		self.word_embeddings = nn.Embedding(vocab_size_word, embedding_dim)		
		if pretrained_embeddings is not None:
			self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
			self.word_embeddings.weight.requires_grad = False

		self.hidden = self.init_hidden()

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)

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
		return (torch.zeros(4, self.batch_size, self.hidden_dim, device=device),
				torch.zeros(4, self.batch_size, self.hidden_dim, device=device))


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
		batch, self.hidden = self.lstm(batch, self.hidden)

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

			# compute cross entropy loss which ignores all <PAD> tokens
			ce_loss = -torch.sum(Y_hat) / nb_tokens

		elif self.model_type == S21:
			# We only take the output of the LSTM from the last timestep of each batch, i.e.
			# the last non-padding word in each batch.
			# X_lengths is a list of the batch lengths, so y[X_lengths[i] - 1] gives us the output of the LSTM that corresponds
			# to the last word in the sequence.
			

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

	

class WordLSTMTagger(LSTMTagger):
	def setup_model(self):
		self.max_x_length = self.max_sent_length
		self.embeddings = self.word_embeddings


def new_parameter(*size):
	out = nn.Parameter(torch.FloatTensor(*size))
	torch.nn.init.xavier_normal_(out)
	return out


class CombinedLSTMTagger(LSTMTagger):

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(4, self.batch_size, self.hidden_dim, device=device),
				torch.zeros(4, self.batch_size, self.hidden_dim, device=device))

	def setup_model(self):
		self.max_w_length = self.max_sent_length
		self.max_x_length = self.max_word_length

		self.lstm_word = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)
		self.lstm_char = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)
		
		self.hidden_word = self.init_hidden()
		self.hidden_char = self.init_hidden()

		#self.hidden2tag = nn.Bilinear(600, 600, self.tag_size)

		# self.attn = nn.Linear(self.hidden_size * 2, self.max_word_length)

		# self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

		self.attention = new_parameter(self.batch_size, self.embedding_dim, 1)

		self.bil = nn.Bilinear(self.max_sent_length * self.batch_size, self.max_word_length * self.batch_size, self.max_word_length * self.batch_size)#self.hidden_dim)#, self.tag_size)


		#self.wordLstmLinear = nn.Linear(self.hidden_dim * 2, self.batch_size * self.max_word_length)
		self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tag_size)#self.tag_size)
		#self.hidden2tag2 = nn.Linear(self.tag_size, self.max_word_length * self.batch_size)


	def forward(self, batch_w, batch_x, batch_w_lengths, batch_x_lengths):
		self.hidden_word = self.init_hidden()
		self.hidden_char = self.init_hidden()
		
		batch_size, seq_len_x = batch_x.size()
		_, seq_len_w 		  = batch_w.size()

		# 1. Embed the input
		batch_w = self.word_embeddings(batch_w)
		batch_x = self.char_embeddings(batch_x)		


		# 1.1 Attention mechanism
		# https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html
		attention_score = torch.matmul(batch_w, self.attention).squeeze()
		attention_score = F.softmax(attention_score, dim=1).view(batch_w.size(0), batch_w.size(1), 1)
		scored_w = batch_w * attention_score


		# 2. Pack the sequence
		batch_w = torch.nn.utils.rnn.pack_padded_sequence(scored_w, [self.max_w_length] * batch_size, batch_first=True)
		batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x, [self.max_x_length] * batch_size, batch_first=True)
	

		# 3. Run through lstm
		batch_w, self.hidden_word = self.lstm_word(batch_w, self.hidden_word)
		batch_x, self.hidden_char = self.lstm_char(batch_x, self.hidden_char)
	

		# Undo packing
		batch_w, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_w, batch_first = True)
		batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first = True)

		#print batch_w.shape
		#print batch_x.shape
		#print batch_w.shape
		#print batch_x.shape

		batch_w = batch_w.contiguous().view(-1, batch_w.shape[2]).transpose_(0, 1)
		batch_x = batch_x.contiguous().view(-1, batch_x.shape[2]).transpose_(0, 1)#.transpose_(0, 1)


        # now, sum across dim 1 to get the expected feature vector
		#condensed_w = torch.sum(scored_w, dim=1)

		#print "W:", condensed_w.size()

		#print condensed_w

		#scored_w = scored_w.view(-1, scored_w.shape[2])
		#print scored_w.size()
		#print batch_x.size()

		#print batch_w.shape
		#print batch_x.shape

		#batch_w = self.wordLstmLinear(batch_w)

		#attn_weights = F.softmax(
        #    self.attn(torch.cat((batch_w[0], batch_x[0]), 1)), dim=1)

        #attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                         batch_x.unsqueeze(0))

       	#output = torch.cat((embedded[0], attn_applied[0]), 1)
       	#output = self.attn_combine(output).unsqueeze(0)

		#batch = torch.cat((batch_w, batch_x))

		#print batch.shape


		#batch = self.hidden2tag(batch)

		#print batch.shape, "h2t"

		#batch = self.hidden2tag2(batch)

		#print batch.shape

		batch = self.bil(batch_w, batch_x).transpose_(0, 1)

		#print batch.size(), "bil"

		batch = self.hidden2tag(batch)

		#print batch.size(), "h2t"

		#batch_w = self.attn(batch_w)
		


		#batch = self.hidden2tag(batch)
		#print batch.size(), "after hidden2tag"

		batch = F.log_softmax(batch, dim=1)

		#print batch[0:100]

		#print batch.size()

		Y_hat = batch.view(batch_size, seq_len_x, self.tag_size)

		if self.model_type == S2S:
			return Y_hat 
		else:
			raise Exception("Model not yet implemented for S21")
		#elif self.model_type == S21:
		#	return torch.stack([ y[batch_x_lengths[i] - 1] for i, y in enumerate(Y_hat) ])		