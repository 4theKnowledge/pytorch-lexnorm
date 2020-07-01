from config import *


from progress_bar import ProgressBar
import time

import sys
from colorama import Fore, Back, Style
from load_data import load_data
from model import LSTMTagger, CharLSTMTagger, WordLSTMTagger, CombinedLSTMTagger, WordTaggerWithFlagger, FeedForwardBert
#from model_feedforward import DeepEncodingTagger
import torch.optim as optim
import torch
 # TODO: Move to cf
from evaluate import evaluate_model

import numpy as np



def main():
	with open("models/%s/params.txt" % cf.MODEL_NAME, "w") as f:
		f.write("\n".join(["%s : %s" % (k, cf.__dict__[k]) for k in cf.__dict__]))

	progress_bar = ProgressBar()
	data_iterators, word_embeddings, char_embeddings, word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag = load_data()
	logger.info("Building model...")

	if cf.GRANULARITY == CHAR_LEVEL:
		model_class = CharLSTMTagger
	elif cf.GRANULARITY == WORD_LEVEL:
		model_class = WordLSTMTagger 
	elif cf.GRANULARITY == CHAR_AND_WORD_LEVEL:
		model_class = CombinedLSTMTagger
	if cf.GRANULARITY == WORD_LEVEL and cf.WORD_LEVEL_WITH_FLAGGER:
		model_class = WordTaggerWithFlagger
	if cf.GRANULARITY == WORD_LEVEL and cf.EMBEDDING_MODEL == "Bert":
		model_class = FeedForwardBert

	#counter = 0
	#for w in word_embeddings:
	#	if w[0] == 0:
	#		counter+= 1
	#	print w[:5]
	#print counter, len(word_embeddings)
	#exit()

	model = model_class(cf.MODEL_TYPE,
					   cf.WORD_EMBEDDING_DIM,
					   cf.CHAR_EMBEDDING_DIM,
					   cf.HIDDEN_DIM,
					   len(char_to_ix),
					   len(ix_to_word),
					   len(wtag_to_ix) if cf.GRANULARITY == WORD_LEVEL else len(ctag_to_ix),
					   cf.BATCH_SIZE,
					   cf.MAX_WORD_LENGTH,
					   cf.MAX_SENT_LENGTH,
					   word_embeddings,
					   char_embeddings)
									# Ensure the word embeddings aren't modified during training

	epoch_start = 1
	#model.load_state_dict(torch.load('models/%s/model_trained/epoch_90' % cf.MODEL_NAME))
	#epoch_start = 90


	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE, momentum=0.9)
	model.cuda()
	#if(cf.LOAD_PRETRAINED_MODEL):
	#	model.load_state_dict(torch.load('asset/model_trained'))
	#else:
	num_batches = len(data_iterators["train"])
	avg_loss_list = [] # A place to store the loss history
	best_f1 = [0.0, -1] # F1, epoch number
	for epoch in range(epoch_start, cf.MAX_EPOCHS+1):
		epoch_start_time = time.time()
		epoch_losses = []		
		for (i, (batch_w, batch_x, batch_y, batch_f)) in enumerate(data_iterators["train"]):
			#if i > 1:
			#	continue
			# Ignore batch if it is not the same size as the others (happens at the end sometimes)

			if len(batch_w) != cf.BATCH_SIZE:
				print(batch_w)
				print(len(batch_w))
				logger.warn("A batch did not have the correct number of sentences.")
				continue

			# Ignore batch if it is not the same size as the others (happens at the end sometimes)
			if len(batch_x) != cf.BATCH_SIZE:
				print(len(batch_x))
				logger.warn("A batch did not have the correct number of words.")
				continue		

			batch_w = batch_w.to(device)
			batch_x = batch_x.to(device)
			if cf.WORD_LEVEL_WITH_FLAGGER:
				batch_f = batch_f.to(device)
			
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()


			batch_x_lengths = []
			for x in batch_x:
				batch_x_lengths.append( np.nonzero(x).size(0) )

			batch_w_lengths = []
			for w in batch_w:
				batch_w_lengths.append( np.nonzero(w).size(0) )

			#print batch_x
			#print batch_y
			

			# Step 3. Run our forward pass.
			model.train()
			if cf.WORD_LEVEL_WITH_FLAGGER:
				tag_scores, tag_scores_f = model(batch_f, batch_x, batch_w_lengths, batch_x_lengths)	
				loss = model.calculate_loss(tag_scores, tag_scores_f, batch_y, batch_f)	
			else:
				tag_scores = model(batch_w, batch_x, batch_w_lengths, batch_x_lengths)
				loss = model.calculate_loss(tag_scores, batch_y)			

					
			
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss)			
			progress_bar.draw_bar(i, epoch, num_batches, cf.MAX_EPOCHS, epoch_start_time)

		avg_loss = sum(epoch_losses) / float(len(epoch_losses))
		avg_loss_list.append(avg_loss)

		progress_bar.draw_completed_epoch(avg_loss, avg_loss_list, epoch, cf.MAX_EPOCHS, epoch_start_time)
		

		if epoch % 10 == 0 or epoch == cf.MAX_EPOCHS:
			f1 = evaluate_model(model, data_iterators["test"], word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag, epoch, print_output = True);
			if f1 > best_f1[0]:
				best_f1 = [f1, epoch]
				logger.info("New best F1 score achieved!")			
				logger.info("Saving model...")
				model_filename = "models/%s/model_trained/epoch_%d" % (cf.MODEL_NAME, epoch)
				torch.save(model.state_dict(), model_filename)
				logger.info("Model saved to %s." % model_filename)
			elif epoch - best_f1[1] >= 50:
				logger.info("No improvement to F1 score in past 50 epochs. Stopping early.")
				logger.info("Best F1 Score: %.4f" % best_f1[0])
				return

if __name__ == '__main__':
	main()
