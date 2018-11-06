import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from config import *
cf = Config()

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepEncodingTagger(nn.Module):

	def __init__(self, model_type, embedding_dim, hidden_dim, vocab_size, tag_size, batch_size, max_sent_length, pretrained_embeddings=None):
		super(DeepEncodingTagger, self).__init__()
		self.hidden_dim = hidden_dim
		self.model_type = model_type

		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.tag_size   = tag_size
		self.max_sent_length = max_sent_length

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		
		if cf.USE_PRETRAINED_EMBEDDINGS:
			self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
			self.word_embeddings.weight.requires_grad = False

		self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
		self.relu1 = nn.ReLU()
		#self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
		#self.relu2 = nn.ReLU()
		self.dropout = nn.Dropout(p = 0.5)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim, tag_size)


	def tag_sequence(self, sequence):
		pass

	def forward(self, batch, batch_lengths):
		
		batch_size, seq_len = batch.size()

		batch = self.word_embeddings(batch)
		batch = self.dropout(batch)
		batch = self.hidden1(batch)
		batch = self.relu1(batch)
		# batch = self.dropout(batch)
		# batch = self.hidden2(batch)
		# batch = self.relu2(batch)
		batch = self.dropout(batch)		
		batch = self.hidden2tag(batch)
		#Y_hat = F.log_softmax(batch, dim=1)

		#Y_hat = batch.view(batch_size, seq_len, self.tag_size)

		return batch

	def calculate_loss(self, Y_hat, Y, X_lengths, word_to_ix, tag_to_ix):
		# torch.set_printoptions(threshold=5000)
		# print Y 

		# print Y_hat

		# print Y.size()
		# print Y_hat.size()

		Y = Y.view(-1).to(device)
		Y_hat = Y_hat.view(-1, len(tag_to_ix))

		# print Y 

		# print Y_hat

		# print Y.size()
		# print Y_hat.size()
		# #exit()

		#print Y 
		#print Y.size()
		#print Y_hat 
		#print Y_hat.size()
		#exit()
		loss = nn.CrossEntropyLoss()
		return loss(Y_hat, Y)
