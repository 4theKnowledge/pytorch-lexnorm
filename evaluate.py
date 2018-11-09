import torch
from config import *
from load_data import load_data
from model import *
import codecs
from colorama import Fore, Back, Style


def print_sentence():
	pass

def evaluate_model(model, test_iterator, word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag, epoch_number, print_output=False):

	correct_preds = 0.0
	total_preds = 0.0
	total_correctable = 0.0

	max_tag_length = max([len(y) for y in wtag_to_ix.keys()])

	wordlist = []
	predlist = []
	corrlist = []

	with torch.no_grad():
		if print_output:
			print ""
			logger.info("Test set evaluation: ")
		for (bi, (batch_w, batch_x, batch_y)) in enumerate(test_iterator):
			# Ignore batch if it is not the same size as the others (happens at the end sometimes)
			if len(batch_x) != cf.BATCH_SIZE:
				logger.warn("An evaluation batch did not have the correct number of sentences.")
				continue
			batch_x = batch_x.to(device)
			batch_w = batch_w.to(device)
			model.zero_grad()

			batch_x_lengths = []
			for x in batch_x:
				batch_x_lengths.append( np.nonzero(x).size(0) )
			batch_w_lengths = []
			for w in batch_w:
				batch_w_lengths.append( np.nonzero(w).size(0) )

			# TODO: Make this a method of the lstm tagger class instead
			model.eval()
			tag_scores = model(batch_w, batch_x, batch_w_lengths, batch_x_lengths)

			#print tag_scores
		
			for i, sent in enumerate(batch_x):
				s = []
				
				if cf.MODEL_TYPE == S2S:
					if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
						s = [[], [], [], [], []]
					for j, token_ix in enumerate(sent):						
						if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL] or token_ix > 0:			


							pred = tag_scores[i][j]
							v, pi = pred.max(0)
							pi = pi.cpu().numpy()				# predicted index							
							ci = batch_y[i][j].cpu().numpy()	# correct index
						
							word_color = Style.BRIGHT if ci > 0 else Style.DIM
							tag_color = Fore.GREEN if ci == pi	 else Fore.RED
							#if ci == 0:
							#	tag_color = Style.DIM

							if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
								char_ix = token_ix
								s[0].append(tag_color + ix_to_ctag[pi] + Style.RESET_ALL)
								s[1].append(ix_to_char[char_ix])

								s[2].append(ix_to_ctag[pi])	# predicted char
								s[3].append(ix_to_ctag[ci])	# correct char
								s[4].append(ix_to_char[char_ix])	# original char
							else:

								word_ix = token_ix

								original_word = ix_to_word[word_ix]								
								predicted_word = ix_to_wtag[pi]
								correct_word = ix_to_wtag[ci]

								if ci == wtag_to_ix["<SELF>"]:
									correct_word = original_word
								if pi == wtag_to_ix["<SELF>"]:
									predicted_word = original_word

								if correct_word == original_word and predicted_word == correct_word:
									tag_color = Style.DIM

								s.append(word_color + original_word + Style.DIM + (("/" + Style.RESET_ALL + tag_color + ix_to_wtag[pi]) if ci > 0 else "") + Style.RESET_ALL)


								wordlist.append(original_word)
								predlist.append(predicted_word)
								corrlist.append(correct_word)

								if predicted_word != original_word and correct_word == predicted_word:
									correct_preds += 1						  
								if correct_word != original_word:
									total_correctable += 1
								if predicted_word != original_word:
									total_preds += 1

								



					if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:						
						# Calculate the f score with respect to the words, not the characters
						predicted_word = "".join(s[2])
						correct_word   = "".join(s[3])
						original_word  = "".join(s[4])

						if predicted_word != original_word and correct_word == predicted_word:
							correct_preds += 1						  
						if correct_word != original_word:
							total_correctable += 1
						if predicted_word != original_word:
							total_preds += 1

						#print correct_preds, total_correctable, total_preds
				


				if cf.MODEL_TYPE == S21:			
					pred = tag_scores[i]
					v, pi = pred.max(0)
					pi = pi.cpu().numpy()			# predicted index							
					ci = batch_y[i].cpu().numpy()	# correct index

					tag_color = Fore.GREEN if ci == pi	 else Fore.RED
					s.append(tag_color + ix_to_tag[pi].ljust(max_tag_length + 2) + Style.RESET_ALL)
					for word in sent:
						if word > 0:
							s.append(ix_to_word[word])
					
					if ci == 0:
						tag_color = Style.DIM

					# TODO : change '0' to original index
					# note. not sure how f1 score works with S21, might not really make any sense and should use acc instead
					if pi != 0 and ci == pi:
						correct_preds += 1						  
					if ci != 0:
						total_correctable += 1
					if pi != 0:
						total_preds += 1
					

				if cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:
					if print_output and bi < 1: # Print first 10 dev batches only
						print "".join(s[1]).ljust(cf.MAX_SENT_LENGTH).replace("<PAD>", "_"), "".join(s[0]).replace("<PAD>", "_")

					wordlist.append("".join(s[1]).replace("<PAD>", "_"))
					predlist.append("".join(s[2]).replace("<PAD>", "_"))
					corrlist.append("".join(s[3]).replace("<PAD>", "_"))
										
				else:
					if print_output and bi < 1: # Print first 1 dev batches only
						print " ".join(s)		


		if print_output:
			print ""	



		p = correct_preds / total_preds if correct_preds > 0 else 0
		r = correct_preds / total_correctable if correct_preds > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
		
		print "-" * 100
		logger.info("F1 Score: %.4f" % f1)
		print "-" * 100

		predictions_filename = "models/%s/predictions/predictions_%d.txt" % (cf.MODEL_NAME, epoch_number)
		with codecs.open(predictions_filename, "w", 'utf-8') as f:
			f.write("=" * 150)
			f.write("\nResults for Epoch %d\n" % epoch_number)
			f.write("F1 Score: %.4f\n" % f1)
			f.write("Word".ljust(cf.MAX_SENT_LENGTH) + "Pred".ljust(cf.MAX_SENT_LENGTH) + "Correct form\n")
			f.write("=" * 150)
			f.write("\n")
			for word, pred, corr in zip(wordlist, predlist, corrlist):
				f.write(word.ljust(cf.MAX_SENT_LENGTH + 3) + pred.ljust(cf.MAX_SENT_LENGTH + 3) + corr + "\n")
			logger.info("Predictions saved to %s." % predictions_filename)
		return f1

		# logger.info("Generated sentences: ")
		# print " " * 6 + "=" * 60
		# for x in range(10):
		# 	sent = model.generate_sentence(ix_to_word)			
		# 	print " " * 6 + sent
		# print " " * 6 + "=" * 60
		# print ""

def main():
	raise Exception("calling evaluate directly is not yet supported")
	# data_iterator, glove_embeddings, word_to_ix, ix_to_word = load_data()

	# model = LSTMTagger(cf.EMBEDDING_DIM, cf.HIDDEN_DIM, len(word_to_ix), cf.BATCH_SIZE, cf.MAX_SENT_LENGTH, glove_embeddings)
	# model.cuda()
	# model.load_state_dict(torch.load('asset/model_trained'))

	# evaluate_model(model, ix_to_word)

if __name__ == '__main__':
	main()
