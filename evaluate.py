import torch
from config import *
from load_data import load_data
from model import *
import codecs, sys
from colorama import Fore, Back, Style
import time

def print_sentence():
	pass

def calculate_f1(correct_preds, total_preds, total_correctable):
	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correctable if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
	return p, r, f1

def calculate_acc(correct_preds, total_preds):
	return correct_preds / total_preds * 100.0

def evaluate_model(model, test_iterator, word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag, epoch_number, print_output=False):

	correct_preds = 0.0
	total_preds = 0.0
	total_correctable = 0.0

	max_tag_length = max([len(y) for y in wtag_to_ix.keys()])

	wordlist = []
	predlist = []
	corrlist = []

	dd = codecs.open("debug.txt", "w+", "utf-8")

	with torch.no_grad():
		if print_output:
			print("")
			print ("      Test set evaluation: ")	

		num_batches = len(test_iterator)

		for (bi, (batch_w, batch_x, batch_y, batch_f)) in enumerate(test_iterator):			
			if bi > 0:
				sys.stdout.write("\rEvaluating batch %d / %d" % (bi + 1, num_batches))


			# Ignore batch if it is not the same size as the others (happens at the end sometimes)
			if len(batch_x) != cf.BATCH_SIZE:
				logger.warn("An evaluation batch did not have the correct number of sentences.")
				continue
			batch_x = batch_x.to(device)
			batch_w = batch_w.to(device)
			batch_y = batch_y.to(device)
			if cf.WORD_LEVEL_WITH_FLAGGER:
				batch_f = batch_f.to(device)
			model.zero_grad()

			batch_x_lengths = []
			for x in batch_x:
				batch_x_lengths.append( np.nonzero(x).size(0) )
			batch_w_lengths = []
			for w in batch_w:
				batch_w_lengths.append( np.nonzero(w).size(0) )

			# TODO: Make this a method of the lstm tagger class instead
			model.eval()
			if cf.WORD_LEVEL_WITH_FLAGGER:
				tag_scores, tag_scores_f = model(batch_f, batch_x, batch_w_lengths, batch_x_lengths)
				tag_scores_f.to(torch.device("cpu"))
			else:
				tag_scores = model(batch_w, batch_x, batch_w_lengths, batch_x_lengths)

			tag_scores.to(torch.device("cpu"))
			batch_x.to(torch.device("cpu"))
			batch_y.to(torch.device("cpu"))
					
			for i, sent in enumerate(batch_x):
				s = []

				if cf.MODEL_TYPE == S2S and cf.GRANULARITY in [CHAR_LEVEL, CHAR_AND_WORD_LEVEL]:

					filtered_term = False

					original_word_ixs  = batch_x[i]
					correct_word_ixs   = batch_y[i]
					predicted_word_ixs = tag_scores[i].max(1)[1]


					original_word = "".join([ix_to_char[ix] if ix > 0 else "" for ix in original_word_ixs])
					predicted_word = "".join([ix_to_ctag[ix] if ix > 0 else "" for ix in predicted_word_ixs])
					correct_word = "".join([ix_to_ctag[ix] if ix > 0 else "" for ix in correct_word_ixs])

					if any(original_word.startswith(s) for s in cf.ignore_words_starting_with):
						predicted_word = original_word
						filtered_term = True
					if cf.ignore_non_alphabetical and not any(c.isalpha() for c in original_word):
						predicted_word = original_word
						filtered_term = True
			

					#p_same_as_o = torch.all(torch.eq(original_word, predicted_word))
					#c_same_as_p = torch.all(torch.eq(correct_word, predicted_word))
					#c_same_as_o = torch.all(torch.eq(correct_word, original_word))
	
					if predicted_word != original_word and correct_word == predicted_word:
						correct_preds += 1						  
					if correct_word != original_word:
						total_correctable += 1
					if predicted_word != original_word:
						total_preds += 1

					

					wordlist.append(original_word)
					predlist.append(predicted_word)
					corrlist.append(correct_word)
					
					dd.write("%d %d %d | %s %s %s\n" % (correct_preds, total_correctable, total_preds, original_word, predicted_word, correct_word))
					      
					

					# Print words in first batch only, and only the first 50 words
					if bi == 0 and i < 50 and print_output:					
						print(original_word, end="")
						predstr = ""
						longer_wordlen = max(len(correct_word), len(predicted_word))
						pad_p = predicted_word.ljust(longer_wordlen, "_")
						pad_c = correct_word.ljust(longer_wordlen, "_")
						if filtered_term:
							print(Fore.YELLOW + predicted_word + Style.RESET_ALL, end="")
						else:
							for cix, c in enumerate(pad_p):
								char_color = Fore.GREEN if pad_p[cix] == pad_c[cix] else Fore.RED
								predstr += char_color + c
							print(predstr + Style.RESET_ALL, end="")
						print(Style.DIM + correct_word + Style.RESET_ALL)
						
						



				
				if cf.MODEL_TYPE == S2S and cf.GRANULARITY == WORD_LEVEL:				
					for j, token_ix in enumerate(sent):						
						if token_ix > 0:
							pred = tag_scores[i][j]
							
							v, pi = pred.max(0)
							pi = pi.cpu().numpy()				# predicted index							
							ci = batch_y[i][j].cpu().numpy()	# correct index

							if cf.WORD_LEVEL_WITH_FLAGGER:
								pred_f = tag_scores_f[i][j]
								v_f, pi_f = pred_f.max(0)
								pi_f = pi_f.cpu().numpy()
						
							word_color = Style.BRIGHT if ci > 0 else Style.DIM
							tag_color = Fore.GREEN if ci == pi	 else Fore.RED
											

							word_ix = token_ix

							original_word = ix_to_word[word_ix]								
							predicted_word = ix_to_wtag[pi]
							correct_word = ix_to_wtag[ci]

							if cf.ignore_non_alphabetical and not any(c.isalpha() for c in original_word):
								predicted_word = original_word
								filtered_term = True
							#if cf.ignore_non_alphabetical and not any(c.isalpha() for c in original_word):
							#	predicted_word = original_word
							#	filtered_term = True

							if ci == wtag_to_ix["<SELF>"]:
								correct_word = original_word
							if pi == wtag_to_ix["<SELF>"]:
								predicted_word = original_word

							if bi <= 1 and j <= 5 and print_output:
								tag_color = Fore.GREEN if predicted_word == correct_word else Fore.RED
								print(original_word, tag_color, predicted_word, Style.RESET_ALL, correct_word, end="")
								if cf.WORD_LEVEL_WITH_FLAGGER:
									print(" %s" % ["False", "True"][pi_f])
								else:
									print("")

							wordlist.append(original_word)
							predlist.append(predicted_word)
							corrlist.append(correct_word)
							dd.write("%d %d %d | %s %s %s\n" % (correct_preds, total_correctable, total_preds, original_word, predicted_word, correct_word))

							if predicted_word != original_word and correct_word == predicted_word:
								correct_preds += 1						  
							if correct_word != original_word:
								total_correctable += 1
							if predicted_word != original_word:
								total_preds += 1

				if cf.MODEL_TYPE == S21:			
					pred = tag_scores[i]

					v, pi = pred.max(0)
					pi = int(pi.cpu().numpy())			# predicted index							
					ci = int(batch_y[i].cpu().numpy())	# correct index					
					
					if cf.FLAGGER_MODE:
						original_word_ixs  = batch_x[i]
						original_word = "".join([ix_to_char[ix] if ix > 0 else "" for ix in original_word_ixs])

						tag_color = Fore.GREEN if ci == pi	 else Fore.RED

						

						if bi < 1:
							print("%s %s %s" % (original_word, tag_color + ix_to_ctag[pi], Style.RESET_ALL + ix_to_ctag[ci]))	

						wordlist.append(original_word)
						predlist.append(ix_to_ctag[pi])
						corrlist.append(ix_to_ctag[ci])
						if ci == pi:
							correct_preds += 1						  
						total_preds += 1
					else:
						#if ci == 0:	# If correct word is <PAD>, ignore
						#	continue
						
						filtered_term = False
						original_word_ixs  = batch_x[i]

						original_word = "".join([ix_to_char[ix] if ix > 0 else "" for ix in original_word_ixs])
						predicted_word = ix_to_wtag[pi]
						correct_word = ix_to_wtag[ci]

						if any(original_word.startswith(s) for s in cf.ignore_words_starting_with):
							predicted_word = original_word
							filtered_term = True
						if cf.ignore_non_alphabetical and not any(c.isalpha() for c in original_word):
							predicted_word = original_word
							filtered_term = True

						if ci == wtag_to_ix["<SELF>"]:
							correct_word = original_word
						if pi == wtag_to_ix["<SELF>"]:
							predicted_word = original_word

						tag_color = Fore.GREEN if correct_word == predicted_word	 else Fore.RED
						if filtered_term:
							tag_color = Fore.YELLOW

						if bi < 1:	
							print("%s %s %s" % (original_word, tag_color + predicted_word, Style.RESET_ALL + correct_word))
				
						if correct_word != "<PAD>":
							dd.write("%d %d %d | %s %s %s\n" % (correct_preds, total_correctable, total_preds, original_word, predicted_word, correct_word))							

							wordlist.append(original_word)
							predlist.append(predicted_word)
							corrlist.append(correct_word)	
							
							if predicted_word != original_word and correct_word == predicted_word:
								correct_preds += 1						  
							if correct_word != original_word:
								total_correctable += 1
							if predicted_word != original_word:
								total_preds += 1
							
				

					

			if not cf.FLAGGER_MODE and bi == 9:
				p, r, f1 = calculate_f1(correct_preds, total_preds, total_correctable)
				if f1 == 0.0:
					print("")
					logger.info("F1 score of first 10 batches is 0. Not evaluating other batches.")
					print("")
					return f1
		if print_output:
			print("")	

		
		#print correct_preds, total_preds, total_correctable

		if not cf.FLAGGER_MODE:
			p, r, score = calculate_f1(correct_preds, total_preds, total_correctable)
			metric_name = "F1 Score"		


		elif cf.FLAGGER_MODE:
			score = calculate_acc(correct_preds, total_preds)
			metric_name = "Accuracy"

		print("-" * 100)
		logger.info("%s: %.4f" % (metric_name, score))
		print("-" * 100	)		


		predictions_filename = "models/%s/predictions/predictions_%d.txt" % (cf.MODEL_NAME, epoch_number)		
		with codecs.open(predictions_filename, "w", 'utf-8') as f:
			f.write("=" * 150)
			f.write("\nResults for Epoch %d\n" % epoch_number)
			f.write("%s: %.4f\n" % (metric_name, score))
			f.write("Word".ljust(cf.MAX_WORD_LENGTH) + "Pred".ljust(cf.MAX_WORD_LENGTH) + "Correct form\n")
			f.write("=" * 150)
			f.write("\n")
			for word, pred, corr in zip(wordlist, predlist, corrlist):
				f.write(word.ljust(cf.MAX_WORD_LENGTH + 3) + " " + pred.ljust(cf.MAX_WORD_LENGTH + 3) + " " + corr + "\n")
			logger.info("Predictions saved to %s." % predictions_filename)
		return score

		# logger.info("Generated sentences: ")
		# print " " * 6 + "=" * 60
		# for x in range(10):
		# 	sent = model.generate_sentence(ix_to_word)			
		# 	print " " * 6 + sent
		# print " " * 6 + "=" * 60
		# print ""

def main():
	#raise Exception("calling evaluate directly is not yet supported")
	# data_iterator, glove_embeddings, word_to_ix, ix_to_word = load_data()

	# model = LSTMTagger(cf.EMBEDDING_DIM, cf.HIDDEN_DIM, len(word_to_ix), cf.BATCH_SIZE, cf.MAX_SENT_LENGTH, glove_embeddings)
	# model.cuda()
	# model.load_state_dict(torch.load('asset/model_trained'))

	# evaluate_model(model, ix_to_word)


	# TODO: Prevent writing to training log during evaluation
	data_iterators, word_embeddings, char_embeddings, word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag = load_data()
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
					   len(word_to_ix),	# TODO: Change to ix_to_word
					   len(wtag_to_ix) if cf.GRANULARITY == WORD_LEVEL else len(ctag_to_ix),
					   cf.BATCH_SIZE,
					   cf.MAX_WORD_LENGTH,
					   cf.MAX_SENT_LENGTH,
					   word_embeddings)
	model.cuda()
	model.load_state_dict(torch.load('models/%s/model_trained/epoch_180' % cf.MODEL_NAME))
	f1 = evaluate_model(model, data_iterators["test"], word_to_ix, ix_to_word, wtag_to_ix, ix_to_wtag, char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag, 0, print_output = True);
if __name__ == '__main__':
	main()
