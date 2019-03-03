from config import * 
from nltk.corpus.reader import ConllCorpusReader
from build_data import get_char_and_chartag_ids, get_unique_test_tag_set, save_data_to_files
import numpy as np
import jsonlines, codecs, sys

def get_ids_and_bert_embeddings(tagged_sents):

	embedding_dim = cf.WORD_EMBEDDING_DIM

	#word_to_ix = { "<PAD>": 0 }
	ix_to_word = [ "<PAD>" ]

	wtag_to_ix = { "<PAD>" : 0}
	ix_to_wtag = [ "<PAD>" ]

	vocab_len = sum([len(sent) for sent in tagged_sents]) + 1
	logger.info("%d total words in dataset (including padding token)." % vocab_len)
	embedding_vectors = np.zeros([vocab_len, embedding_dim])
	embedding_vectors[0] = np.zeros(embedding_dim)

	word_index = 1	# The index of the current word, regardless of sentence.

	debug_file = "debug_bert_2.txt"
	debug_info = []
	sp = False

	line_clipped = False
	num_clipped = 0

	#current_v = []

	with jsonlines.open(cf.BERT_FOLDER + "/embeddings.jsonl", "r") as f:
		for i, line in enumerate(f):
			line_clipped = False
			print "\r%d / %d sentences processed. (%d lines clipped)" % (i, len(tagged_sents), num_clipped),	
			wi_in_sent = 0 # Word index in current sent
			#current_v = []
			for j, d in enumerate(line["features"]):			


				if wi_in_sent >= len(tagged_sents[i]):
					continue


				bert_token = d["token"]


				if bert_token == "[UNK]":
					print bert_token, [t["token"] for t in d]
				#if wi_in_sent >= cf.MAX_SENT_LENGTH:
					#print " ".join([w[0] for w in tagged_sents[i]])
					#print [w["token"] for w in line["features"]]
				#	if not line_clipped:
				#		num_clipped += 1
				#		line_clipped = True
				#	bert_token = ""		


				# If there are no more word segments, but there are more words in tagged_sents[i], set the embeddings of all remaining words
				# in the sentence to 0s (i.e. skip over them).
				#if j == (len(line["features"]) - 1) and len(tagged_sents[i]) > wi_in_sent:
				if wi_in_sent > cf.MAX_SENT_LENGTH:

					'''print ""
					print [t["token"] for t in line["features"]]
					print bert_token
					print j
					print len(line["features"])
					print tagged_sents[i]
					print len(tagged_sents[i])
					print wi_in_sent
					print "-_-"'''

					sentlen = len(tagged_sents[i])
					
					diff = sentlen - wi_in_sent
					#print "Difference:", diff

					for x in range(diff):
						st = tagged_sents[i][wi_in_sent + x]
						ix_to_word.append(st[0])
						tag = st[1]
						if tag not in wtag_to_ix:
							wtag_to_ix[tag] = len(wtag_to_ix)
							ix_to_wtag.append(tag)
						word_index += 1
						debug_info.append("%s %s | %s" % (st, "(skipped)", "(skipped)"))
					num_clipped += 1
					break					

				if bert_token not in ["[CLS]", "[SEP]"] and not bert_token.startswith("##"):

					v = d["layers"][0]["values"]
					#if bert_token.startswith("##"):
					#	current_v.append(v)
					#	continue

					
					#print bert_token, "<<", wi_in_sent

							

						#print " ".join([w[0] for w in tagged_sents[i]])
						#print [w["token"] for w in line["features"]]
						#print len([w["token"] for w in line["features"]])
						#raise Exception("Error on line %d: BERT embeddings need to be generated with a large max sequence length" % i)


					#if wi_in_sent >= len(tagged_sents[i]):
					#	continue

					sent_token = tagged_sents[i][wi_in_sent][0]


					#print sent_token, word_index

	
					if bert_token != sent_token and not sent_token.startswith(bert_token):
						#print "<<>>>>>>>", sent_token, bert_token
						#print sent_token, bert_token, [t["token"] for t in line["features"]]
						#current_v.append(v)						
						continue

					#if len(current_v) > 1:
						#print sent_token, bert_token, [t["token"] for t in line["features"]]
						#print "<", len(current_v)
					#	embedding_vectors[word_index - 1] = np.average(current_v, axis=0)
						#print np.average(current_v, axis=0)[:5]
						#print word_index - 1
						

		
					#current_v = [v]

					#if len(bert_token) > 0:
						#bert_emb = np.array([d["layers"][0]["values"], d["layers"][1]["values"], d["layers"][2]["values"], d["layers"][3]["values"]])
						#bert_emb = np.average(bert_emb, axis=0)


					#v = d["layers"][0]["values"]#[:512]
					#normalized_v = v / np.sqrt(np.sum(v**2))
										
					embedding_vectors[word_index] = v #normalized_v

					if sp:
						print sent_token, bert_token, wi_in_sent

					debug_info.append("%s %s | %s" % (sent_token, bert_token, str(embedding_vectors[word_index][:5])))
					
					ix_to_word.append(sent_token)

					tag = tagged_sents[i][wi_in_sent][1]
					if tag not in wtag_to_ix:
						wtag_to_ix[tag] = len(wtag_to_ix)
						ix_to_wtag.append(tag)	

					

					wi_in_sent += 1
					word_index += 1
									

	#print ""
	#for i, l in enumerate(embedding_vectors):
	#	print l
	#	if i > 10:
	#		exit()
		
	np.savez_compressed(cf.EMB_TRIMMED_FILENAME, embeddings=embedding_vectors)
	logger.info("Saved %d embedding vectors to %s." % (len(embedding_vectors), cf.EMB_TRIMMED_FILENAME))

	with codecs.open(debug_file, 'w', 'utf-8') as f:
		f.write("\n".join(debug_info))

	return ix_to_word, {}, wtag_to_ix, ix_to_wtag, embedding_vectors
	

def main():

	corpusReader = ConllCorpusReader(cf.DATA_FOLDER, [cf.TRAIN_FILENAME, cf.TEST_FILENAME], ['words', 'pos'])

	tagged_sents = corpusReader.tagged_sents()

	test_unique_wordtags, test_unique_chartags = get_unique_test_tag_set()

	logger.info("%d sentences loaded." % len(tagged_sents))	



	ix_to_word, word_to_ix, wtag_to_ix, ix_to_wtag, embedding_vectors = get_ids_and_bert_embeddings(tagged_sents) 

	char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag = get_char_and_chartag_ids(tagged_sents)

	save_data_to_files(tagged_sents, word_to_ix, wtag_to_ix, ix_to_word, ix_to_wtag, char_to_ix, ctag_to_ix, ix_to_char, ix_to_ctag)

	logger.info("Data building complete.")

if __name__ == "__main__":
	main()
