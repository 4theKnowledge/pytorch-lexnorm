from config import *
import codecs

# A corpus reader that reads tabbed data for sequence-to-one models, such as for sentiment analysis. The datasets need to be a tab separated file, with the sentence
# followed by a tab character followed by the label.
class TabbedCorpusReader():
	def __init__(self, data_folder, dataset_filenames):
		self.ts = []

		for df in dataset_filenames:
			self.load_dataset(data_folder + "/" + df)	


	def load_dataset(self, dataset):
		with codecs.open(dataset, 'r', 'utf-8') as f:
			for i, line in enumerate(f):
				if line.count("\t") < 1:
					logger.warn("Sentence %d is missing a tab character. Ignoring..." % i)
				elif line.count("\t") > 1:
					logger.warn("Sentence %d contains more than one tab character. Ignoring..." % i)
				else:
					sent, tag = line.strip().lower().split("\t")
					self.ts.append( (sent.split(), tag ) )

	def tagged_sents(self):
		return self.ts
		#return [(sent, tag) for]