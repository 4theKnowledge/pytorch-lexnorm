from config import *
cf = Config()

from progress_bar import ProgressBar
import time

import sys
from colorama import Fore, Back, Style
from load_data import load_data
from model import LSTMTagger, CharLSTMTagger, WordLSTMTagger, CombinedLSTMTagger
from model_feedforward import DeepEncodingTagger
import torch.optim as optim
import torch
 # TODO: Move to cf
from evaluate import evaluate_model

import numpy as np



def main():
	with open("models/%s/params.txt" % cf.MODEL_NAME, "w") as f:
		f.write("\n".join(["%s : %s" % (k, cf.__dict__[k]) for k in cf.__dict__]))

	progress_bar = ProgressBar()
	data_iterators, word_embeddings, word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag = load_data()
	logger.info("Building model...")

	if cf.GRANULARITY == CHAR_LEVEL:
		model_class = CharLSTMTagger
	elif cf.GRANULARITY == WORD_LEVEL:
		model_class = WordLSTMTagger 
	elif cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
		model_class = CombinedLSTMTagger

	model = model_class(cf.MODEL_TYPE,
					   cf.WORD_EMBEDDING_DIM,
					   cf.CHAR_EMBEDDING_DIM,
					   cf.HIDDEN_DIM,
					   len(char_to_ix),
					   len(word_to_ix),
					   len(wtag_to_ix) if cf.GRANULARITY == WORD_LEVEL else len(ctag_to_ix),
					   cf.BATCH_SIZE,
					   cf.MAX_WORD_LENGTH,
					   cf.MAX_SENT_LENGTH,
					   word_embeddings)
									# Ensure the word embeddings aren't modified during training
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE)
	model.cuda()
	#if(cf.LOAD_PRETRAINED_MODEL):
	#	model.load_state_dict(torch.load('asset/model_trained'))
	#else:
	num_batches = len(data_iterators["train"])
	f1_list = [] # A place to store the f1 history
	loss_list = [] # A place to store the loss history
	for epoch in range(1, cf.MAX_EPOCHS+1):
		epoch_start_time = time.time()
		for (i, (batch_w, batch_x, batch_y)) in enumerate(data_iterators["train"]):

			# Ignore batch if it is not the same size as the others (happens at the end sometimes)
			if len(batch_w) != cf.BATCH_SIZE:
				print len(batch_w)
				logger.warn("A batch did not have the correct number of sentences.")
				continue

			# Ignore batch if it is not the same size as the others (happens at the end sometimes)
			if len(batch_x) != cf.BATCH_SIZE:
				print len(batch_x)
				logger.warn("A batch did not have the correct number of words.")
				continue

			batch_w = batch_w.to(device)
			batch_x = batch_x.to(device)
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			# Also, we need to clear out the hidden state of the LSTM,
			# detaching it from its history on the last instance.
			#model.hidden = model.init_hidden()

			# Step 2. Get our inputs ready for the network, that is, turn them into
			# Tensors of word indices.
			#sentence_in = prepare_sequence(sentence, word_to_ix)
			#target = torch.tensor([word_to_ix[tag]], dtype=torch.long, device=device)

			batch_x_lengths = []
			for x in batch_x:
				batch_x_lengths.append( np.nonzero(x).size(0) )

			batch_w_lengths = []
			for w in batch_w:
				batch_w_lengths.append( np.nonzero(w).size(0) )

			# Step 3. Run our forward pass.
			model.train()
			tag_scores = model(batch_w, batch_x, batch_w_lengths, batch_x_lengths)

			#loss = loss_function(tag_scores, batch_y)

			
			

			loss = model.calculate_loss(tag_scores, batch_y)
			
			loss.backward()
			optimizer.step()
			
			progress_bar.draw_bar(i, epoch, num_batches, cf.MAX_EPOCHS, epoch_start_time)

		#f1 = None
		#if epoch % 5 == 0:
		#	f1 = evaluate_model(model, data_iterators["dev"], ix_to_word, ix_to_tag, tag_to_ix);
		progress_bar.draw_completed_epoch(loss, loss_list, epoch, cf.MAX_EPOCHS, epoch_start_time)
		loss_list.append(loss)

		if epoch % 10 == 0 or epoch == cf.MAX_EPOCHS:
			f1 = evaluate_model(model, data_iterators["test"], word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag, epoch, print_output = True);
			max_f1 = max(f1_list) if len(f1_list) > 0 else 0.0
			f1_list.append(f1)
			if f1 > max_f1:
				logger.info("New best F1 score achieved!")			
				logger.info("Saving model...")
				model_filename = "models/%s/model_trained/epoch_%d" % (cf.MODEL_NAME, epoch)
				torch.save(model.state_dict(), model_filename)
				logger.info("Model saved to %s." % model_filename)

if __name__ == '__main__':
	main()