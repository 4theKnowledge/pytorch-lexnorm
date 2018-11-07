from config import *
cf = Config()

from progress_bar import ProgressBar
import time

import sys
from colorama import Fore, Back, Style
from load_data import load_data
from model import LSTMTagger
from model_feedforward import DeepEncodingTagger
import torch.optim as optim
import torch
 # TODO: Move to cf
from evaluate import evaluate_model

import numpy as np



def main():
	progress_bar = ProgressBar()
	data_iterators, test_dataset, glove_embeddings, word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = load_data()
	logger.info("Building model...")
	model = LSTMTagger(cf.MODEL_TYPE,
					   cf.EMBEDDING_DIM,
					   cf.HIDDEN_DIM,
					   len(word_to_ix),
					   len(tag_to_ix),
					   cf.BATCH_SIZE,
					   cf.MAX_SENT_LENGTH,
					   glove_embeddings)
									# Ensure the word embeddings aren't modified during training
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE)
	model.cuda()
	#if(cf.LOAD_PRETRAINED_MODEL):
	#	model.load_state_dict(torch.load('asset/model_trained'))
	#else:
	num_batches = len(data_iterators["train"])
	f1_list = [] # A place to store the f1 history
	for epoch in range(1, cf.MAX_EPOCHS+1):
		epoch_start_time = time.time()
		for (i, (batch_x, batch_y)) in enumerate(data_iterators["train"]):
			# Ignore batch if it is not the same size as the others (happens at the end sometimes)
			if len(batch_x) != cf.BATCH_SIZE:
				print len(batch_x)
				logger.warn("A batch did not have the correct number of sentences.")
				continue
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
			#batch_x_maxlengths = []
			for x in batch_x:
				batch_x_lengths.append( np.nonzero(x).size(0) )
				#batch_x_maxlengths.append( len(x) )

			# Step 3. Run our forward pass.
			model.train()
			tag_scores = model(batch_x, batch_x_lengths)

			#loss = loss_function(tag_scores, batch_y)
			loss = model.calculate_loss(tag_scores, batch_y)
			
			loss.backward()
			optimizer.step()
			
			progress_bar.draw_bar(i, epoch, num_batches, cf.MAX_EPOCHS, epoch_start_time)

		#f1 = None
		#if epoch % 5 == 0:
		#	f1 = evaluate_model(model, data_iterators["dev"], ix_to_word, ix_to_tag, tag_to_ix);
		progress_bar.draw_completed_epoch(loss, loss_list, epoch, cf.MAX_EPOCHS, epoch_start_time)

		if epoch % 10 == 0 or epoch == cf.MAX_EPOCHS:
			f1 = evaluate_model(model, data_iterators["dev"], ix_to_word, ix_to_tag, tag_to_ix, epoch, print_output = True);
			f1_list.append(f1)			
			if epoch >= 30:
				avg_f1 = sum([f for f in f1_list[(epoch/10)-3:]]) / 3
				logger.info("Average f1 over past 30 epochs: %.6f" % avg_f1)
			if epoch >= 60:
				prev_avg_f1 = sum([l for l in loss_list[(epoch/10)-6:(epoch/10)-9]]) / 3
				if(avg_f1 <= prev_avg_f1 and cf.EARLY_STOP):
					logger.info("Average f1 has not improved over past 60 epochs. Stopping early.")
					evaluate_model(model, data_iterators["dev"], ix_to_word, ix_to_tag, tag_to_ix, epoch, print_output = True);
					break;


	logger.info("Saving model...")
	torch.save(model.state_dict(), "asset/model_trained")
	logger.info("Model saved to %s." % "asset/model_trained")

if __name__ == '__main__':
	main()