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

	def __init__(self, model_type, embedding_dim, hidden_dim, vocab_size, tag_size, batch_size, max_sent_length, glove_embeddings=None):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim
		self.model_type = model_type

		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.tag_size   = tag_size
		self.max_sent_length = max_sent_length
		#print glove_embeddings.size

		#num_embeddings, embedding_dim = glove_embeddings.size

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		
		#self.word_embeddings.load_state_dict({ 'weight': glove_embeddings })
		#self.word_embeddings.weight.requires_grad = False

		self.word_embeddings.weight.data.copy_(torch.from_numpy(glove_embeddings))

		# Ensure the word embeddings layer is not trained
		#self.word_embeddings.weight.requires_grad = False


		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True, num_layers = 2, dropout = 0.5)

		#print "lstm"
		#print self.lstm

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim * 2, tag_size)

		#print "hidden2tag"
		#print self.hidden2tag
		self.hidden = self.init_hidden()

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (torch.zeros(4, self.batch_size, self.hidden_dim, device=device),
				torch.zeros(4, self.batch_size, self.hidden_dim, device=device))


	def tag_sequence(self, sequence):
		pass


	def forward(self, batch, batch_lengths):

		self.hidden = self.init_hidden()
		
		batch_size, seq_len = batch.size()

		# 1. Embed the input
		batch = self.word_embeddings(batch)

		# 2. Pack the sequence
		batch = torch.nn.utils.rnn.pack_padded_sequence(batch, [self.max_sent_length] * batch_size, batch_first=True)

		# 3. Run through lstm
		batch, self.hidden = self.lstm(batch, self.hidden)

		# Undo packing
		batch, _ = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first = True)

		batch = batch.contiguous()
		batch = batch.view(-1, batch.shape[2])

		batch = self.hidden2tag(batch)

		batch = F.log_softmax(batch, dim=1)

		Y_hat = batch.view(batch_size, seq_len, self.tag_size)

		if cf.MODEL_TYPE == S2S:
			return Y_hat 
		elif cf.MODEL_TYPE == S21:
			return torch.stack([ y[batch_lengths[i] - 1] for i, y in enumerate(Y_hat) ])

	# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
	def calculate_loss(self, Y_hat, Y, X_lengths, word_to_ix, tag_to_ix):
		# TRICK 3 ********************************
		# before we calculate the negative log likelihood, we need to mask out the activations
		# this means we don't want to take into account padded items in the output vector
		# simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
		# and calculate the loss on that.

		if(cf.MODEL_TYPE == S2S):
			Y = Y.view(-1).to(device)
			Y_hat = Y_hat.view(-1, len(tag_to_ix))
			# create a mask by filtering out all tokens that ARE NOT the padding token
			#tag_pad_token = word_to_ix['<PAD>']
			mask = (Y > 0).float()

			# count how many tokens we have
			nb_tokens = int(torch.sum(mask).item())

			#print nb_tokens
			# pick the values for the label and zero out the rest with the mask
			Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

			# compute cross entropy loss which ignores all <PAD> tokens
			ce_loss = -torch.sum(Y_hat) / nb_tokens

		elif cf.MODEL_TYPE == S21:
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
