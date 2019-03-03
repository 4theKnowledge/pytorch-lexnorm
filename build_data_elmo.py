from config import * 
from nltk.corpus.reader import ConllCorpusReader
from build_data import get_char_and_chartag_ids, get_unique_test_tag_set, save_data_to_files
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np

def get_ids_and_elmo_embeddings(tagged_sents):

	embedding_dim = cf.WORD_EMBEDDING_DIM

	#word_to_ix = { "<PAD>": 0 }
	ix_to_word = [ "<PAD>" ]

	wtag_to_ix = { "<PAD>" : 0}
	ix_to_wtag = [ "<PAD>" ]

	options_file = '%s/options.json' % cf.ELMO_FOLDER
	weight_file = '%s/weights.hdf5' % cf.ELMO_FOLDER

	#elmo = Elmo(options_file, weight_file, 2, dropout=0)
	elmoEmbedder = ElmoEmbedder(options_file, weight_file)

	vocab_len = sum([len(sent) for sent in tagged_sents]) + 1
	logger.info("%d total words in dataset (including padding token)." % vocab_len)
	embedding_vectors = np.zeros([vocab_len, embedding_dim])
	embedding_vectors[0] = np.zeros(embedding_dim)

	word_index = 1	# The index of the current word, regardless of sentence.
	for sent_index, sent in enumerate(tagged_sents):
		print("\r%d / %d sentences processed." % (sent_index, len(tagged_sents)), end="")
		sent_tokens = [w[0] for w in sent]
		#character_ids = batch_to_ids(sent_l)
		#embeddings = elmo(character_ids)
		#if word_index < 10:		
		embeddings = elmoEmbedder.embed_sentence(sent_tokens)
		#else:
		#	embeddings = [np.zeros([len(sent), embedding_dim])]
			
		for i, emb in enumerate(embeddings[0]):
			if i < len(sent):
				ix_to_word.append(sent[i][0])
				embedding_vectors[word_index] = emb#.detach().numpy()
			tag = sent[i][1]
			if tag not in wtag_to_ix:
				wtag_to_ix[tag] = len(wtag_to_ix)
				ix_to_wtag.append(tag)
			word_index += 1
	
	#print(ix_to_word)
	#print(ix_to_wtag)
	
	np.savez_compressed(cf.EMB_TRIMMED_FILENAME, embeddings=embedding_vectors)
	logger.info("Saved %d embedding vectors to %s." % (len(embedding_vectors), cf.EMB_TRIMMED_FILENAME))

	return ix_to_word, {}, wtag_to_ix, ix_to_wtag, embedding_vectors


	

def main():

	corpusReader = ConllCorpusReader(cf.DATA_FOLDER, [cf.TRAIN_FILENAME, cf.TEST_FILENAME], ['words', 'pos'])

	tagged_sents = corpusReader.tagged_sents()

	test_unique_wordtags, test_unique_chartags = get_unique_test_tag_set()

	logger.info("%d sentences loaded." % len(tagged_sents))	



	ix_to_word, word_to_ix, wtag_to_ix, ix_to_wtag, embedding_vectors = get_ids_and_elmo_embeddings(tagged_sents) 

	char_to_ix, ix_to_char, ctag_to_ix, ix_to_ctag = get_char_and_chartag_ids(tagged_sents)

	save_data_to_files(tagged_sents, word_to_ix, wtag_to_ix, ix_to_word, ix_to_wtag, char_to_ix, ctag_to_ix, ix_to_char, ix_to_ctag)

	logger.info("Data building complete.")

if __name__ == "__main__":
	main()
