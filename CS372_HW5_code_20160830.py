import nltk
import csv
import re
import torch
import pickle
import traceback

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree

from statistics import mean, stdev

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

data_list = []
results = []

def main():
	# Get data
	train_data = get_train_data()[:100]
	val_data = get_validation_data()
	test_data = get_test_data()

	# Get Parse Trees
	train_tree_list, val_tree_list, test_tree_list = get_trees(train_data, val_data, test_data)
	
	snippet_results = []
	page_results = []

	# For each data, get weighted scores and classify whether each antecedent resolves the coreference.
	for i in range(len(test_data)):
		tree = test_tree_list[i]
		data = test_data[i]

		# score is in the form [freq, syntatic_dist, antecedent_ga, pronoun_ga, constrained]
		A_score = score(data, tree, isA=True)
		B_score = score(data, tree, isA=False)

		# Get weighted scores
		if A_score is None:
			a_score = 10
		else:
			a_score = 50 / (A_score[1]+0.01) + 10 * (A_score[2]) + A_score[0]

		if B_score is None:
			b_score = 10
		else:
			b_score = 50 / (B_score[1]+0.01) + 10 * (B_score[2]) + B_score[0]

		A_res = None
		B_res = None	

		if a_score > b_score:
			A_res = 'TRUE'
			B_res = 'FALSE'
		else:
			A_res = 'FALSE'
			B_res = 'TRUE'		

		if a_score < 2 and b_score < 2:
			A_res = 'FALSE'
			B_res = 'FALSE'

		if A_score is not None and A_score[4] == 1:
			A_res = 'FALSE'
		if B_score is not None and B_score[4] == 1:
			B_res == 'FALSE'

		snippet_results.append([data.id, A_res, B_res])

		if word_in_url(data.A, data.url):
			A_res = 'TRUE'
			B_res = 'FALSE'
		elif word_in_url(data.B, data.url):
			A_res = 'FALSE'
			B_res = 'TRUE'	

		page_results.append([data.id, A_res, B_res])				

	# Write results to tsv
	write_to_tsv(snippet_results, is_snippet=True)
	write_to_tsv(page_results, is_snippet=False)

# Get train data, val data and test data as input. Return the parse trees for each kind of data.
def get_trees(train_data, val_data, test_data):
	try:
		with open('CS372_HW5_local_file_1_20160830', 'rb') as fp:
			train_tree_list = pickle.load(fp)

		print('loaded train trees')

	except:
		try:
			predictor = torch.load('CS372_HW5_local_file_4_20160830')
			print('loaded predictor')

		except:
			predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz')
			torch.save(predictor, 'CS372_HW5_local_file_4_20160830')
			print('saved predictor')

		train_tree_list = parse_all(train_data, predictor)

		with open('CS372_HW5_local_file_1_20160830', 'wb') as fp:
			pickle.dump(train_tree_list, fp)

		print('saved train trees')

	try:
		with open('CS372_HW5_local_file_2_20160830', 'rb') as fp:
			val_tree_list = pickle.load(fp)
		
		print('loaded val trees')


	except: 
		try:
			predictor = torch.load('CS372_HW5_local_file_4_20160830')
			print('loaded predictor')

		except:
			predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz')
			torch.save(predictor, 'CS372_HW5_local_file_4_20160830')
			print('saved predictor')

		val_tree_list = parse_all(val_data, predictor)

		with open('CS372_HW5_local_file_2_20160830', 'wb') as fp:
			pickle.dump(val_tree_list, fp)

		print('saved val trees')

	try:
		with open('CS372_HW5_local_file_3_20160830', 'rb') as fp:
			test_tree_list = pickle.load(fp)
		
		print('loaded test trees')


	except: 
		try:
			predictor = torch.load('CS372_HW5_local_file_4_20160830')
			print('loaded predictor')

		except:
			predictor = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz')
			torch.save(predictor, 'CS372_HW5_local_file_4_20160830')
			print('saved predictor')

		test_tree_list = parse_all(test_data, predictor)

		with open('CS372_HW5_local_file_3_20160830', 'wb') as fp:
			pickle.dump(test_tree_list, fp)

		print('saved test trees')

	return train_tree_list, val_tree_list, test_tree_list


# If the word is in the Wikipedia URL, return True. Otherwise return False.
def word_in_url(word, url):
	url = re.split(r'wiki/', url)[1]
	url = re.split(r'_', url)

	for w in word.split():
		if w in url:
			return True
	return False

# Get the frequency of the antecedent in the parse tree.
def get_frequency(antecedent, tree):
	freq = 0
	for st in tree.subtrees(lambda t: t.label() == 'NP' and t.height() == 3):
		is_same = False
		for w in antecedent.split():
			if w in st.leaves():
				is_same = True
		if is_same:
			freq += 1
	return freq

# Return the tree position of the antecedent given the antecedent, offset, parse tree, and original text as input.
def get_tree_position(word, ofs, tree, text):
	ofs = int(ofs)
	word_count = 0
	word_len = len(word)
	word_idx = None
	for i in range(len(text)-word_len):
		if text[i:i+word_len] == word:
			if i == ofs:			
				word_idx = word_count
			if i == 0:
				if text[i+word_len] in [' ', '.', ',', ';', '-', '\'', '`']:
					word_count += 1
			elif i+word_len == len(text):
				if text[i-1] in [' ', '\'', '`', '-']:
					word_count += 1
			elif text[i-1] in [' ', '\'', '`', '-'] and text[i+word_len] in [' ', '.', ',', ';', '-', '\'', '`']:
				word_count += 1

	if word_idx is None:
		return (None, None)

	word_token = list(filter(lambda e: e is not None and len(e) != 0, re.split(r' |([-,])|(?<=`)(`)|(\')(?=\')', word)))
	word_token_len = len(word_token)
	leaves = tree.leaves()
	potential_tree_pos = []
	found = False
	for i in range(len(leaves)-word_token_len):
		if leaves[i:i+word_token_len] == word_token:
			tree_pos = tree.leaf_treeposition(i)
			potential_tree_pos.append((tree_pos, i))
			found = True

	if found == False:
		return (None, None)

	try:
		return potential_tree_pos[word_idx]
	except:
		return (None, None)
	
# Take the NP position in the parse tree as input. Return the label of the nearest parent of the NP 
# that is not equal to NP
def get_NP_containing_tag(NP_pos, tree):
	tag_pos = NP_pos
	while True:
		tag_pos = tag_pos[:-1]		
		tag = tree[tag_pos].label()
		if tag != 'NP':
			return tag

# Given two tree positions, return the syntatic distance between tree positions.
def get_syntatic_distance(tree_pos1, tree_pos2):
	first_diff_index = None
	for i in range(min(len(tree_pos1), len(tree_pos2))):
		if tree_pos1[i] != tree_pos2[i]:
			first_diff_index = i
			break

	if first_diff_index is None:
		return 0

	dist = len(tree_pos1) + len(tree_pos2) - 2*first_diff_index
	if tree_pos1[first_diff_index] < tree_pos2[first_diff_index]:
		return dist
	else:
		return -dist

# Given the NP position in the parse tree, return the position of the nearest parent in the parse tree
# that is a sentence
def get_sentence_pos(NP_pos, tree):
	tag_pos = NP_pos
	while True:
		tag_pos = tag_pos[:-1]		
		tag = tree[tag_pos].label()
		if tag == 'S' or 'SS':
			return tag_pos

# Given the position of the antecedent in the parse tree, return 1 (subject), 0 (other).
def get_grammatical_argument(full_word_pos, tree):
	if tree[full_word_pos[:-1]].label() == 'PRP$':
		return 0
	NP_containing_tag = get_NP_containing_tag(full_word_pos[:-2], tree)
	if NP_containing_tag == 'S':
		return 1
	if NP_containing_tag == 'VP':
		return 0
	return 0

# Check for conditions where it is highly unlikely for the antecedent to be the actual antecedent of the 
# pronoun. Return True if it is highly unlikely, False otherwise.
def is_constrained(full_antecedent_pos, full_pronoun_pos, tree):
	antecedent_pos = full_antecedent_pos[:-2]
	pronoun_pos = full_pronoun_pos[:-2]

	# constrained because antecedent and pronoun are in the same NP (ex. 'her aunt Mary')
	if antecedent_pos == pronoun_pos:
		return 1

	if '\'s' in tree[antecedent_pos].leaves():
		antecedent_tag = 'NP$'
	else:
		antecedent_tag = 'NP'
	pronoun_tag = tree[full_pronoun_pos[:-1]].label()

	# no constraints because two nouns differ in possessiveness (ex. 'His dog liked Jim' or 'Jim sits on his bed')
	if antecedent_tag[-1] != pronoun_tag[-1]:
		return 0

	# no constraints because two nouns are in separate sentences
	if full_antecedent_pos[0] != full_pronoun_pos[0]:
		return 0

	# no constraints because two nouns are in separate sentences
	antecedent_sentence_pos = get_sentence_pos(antecedent_pos, tree)
	pronoun_sentence_pos = get_sentence_pos(pronoun_pos, tree)
	if antecedent_sentence_pos != pronoun_sentence_pos:
		return 0

	# constrained because both are possessive or both are non-possessive; both nouns are referred in the same sentence as subject-object
	# (ex. 'His car was Jim's possession' or 'She liked Cassidy'), and
	else:
		first_diff_index = None
		for i in range(min(len(antecedent_pos), len(pronoun_pos))):
			if antecedent_pos[i] != pronoun_pos[i]:
				first_diff_index = i
				break

		if antecedent_pos[first_diff_index] < pronoun_pos[first_diff_index]:
			if tree[antecedent_pos[:first_diff_index+1]].label() == 'NP' and tree[pronoun_pos[:first_diff_index+1]].label() == 'VP':
				# the VP has to be in the form of verb, DIRECT OBJECT, others
				if full_pronoun_pos[first_diff_index+1] == 1:
					return 1
		else:
			if tree[pronoun_pos[:first_diff_index+1]].label() == 'NP' and tree[antecedent_pos[:first_diff_index+1]].label() == 'VP':
				# the VP has to be in the form of verb, DIRECT OBJECT, others
				if full_antecedent_pos[first_diff_index+1] == 1:
					return 1

	# Otherwise, inconslusive
	return 0
	
# Return the list of scores for each heuristic
def score(data, tree, isA=True):
	if isA:
		antecedent = data.A
		antecedent_ofs = data.A_ofs
	else:
		antecedent = data.B
		antecedent_ofs = data.B_ofs

	pronoun = data.pronoun
	pronoun_ofs = data.pronoun_ofs

	freq = get_frequency(antecedent, tree)
	(full_antecedent_pos, tap) = get_tree_position(antecedent, antecedent_ofs, tree, data.text)
	(full_pronoun_pos, tpp) = get_tree_position(pronoun, pronoun_ofs, tree, data.text)
	if full_antecedent_pos is None or full_pronoun_pos is None:
		return None
	antecedent_pos = full_antecedent_pos[:-2]
	pronoun_pos = full_pronoun_pos[:-2]
	syntatic_dist = abs(get_syntatic_distance(antecedent_pos, pronoun_pos))
	antecedent_ga = get_grammatical_argument(full_antecedent_pos, tree)
	pronoun_ga = get_grammatical_argument(full_pronoun_pos, tree)
	constrained = is_constrained(full_antecedent_pos, full_pronoun_pos, tree)

	return [freq, syntatic_dist, antecedent_ga, pronoun_ga, constrained, abs(tap-tpp)]

# Remove parenthesis from the given text and return it
def remove_parenthesis(text):
	return re.sub(r'[\(\)]', '', text)

# Preprocess text to modify words that cause errorneous parse trees or introduce errors in the
# scoring process.
def preprocess_text(text):
	text = re.sub(r'\*', 'a', text)
	text = re.sub(r'Mr\.', 'Mr ', text)
	text = re.sub(r'Mrs\.', 'Mrs ', text)
	return text

# Given a list of data, return the list of parse trees that correspond to each data.
def parse_all(datas, predictor):
	res = []
	for i in range(len(datas)):
		data = datas[i]
		tree_list = []
		tree_str = ''
		sents = sent_tokenize(data.text)
		for t in sents:
			sent = remove_parenthesis(t)
			tree_list.append(predictor.predict(sentence=sent)['trees'])

		tree_str += '(SS '
		for tree in tree_list:
			tree_str += tree
		tree_str += ')'
		res.append(Tree.fromstring(tree_str))
		print('Parsed {}/{} data'.format(i+1, len(datas)), end='\r')
		

	return res

# Data Class to facilitate management of data
class Data():
	def __init__(self, row):
		self.id = row[0]
		self.text = preprocess_text(row[1])
		self.pronoun = row[2]
		self.pronoun_ofs = row[3]
		self.A = preprocess_text(row[4])
		self.A_ofs = row[5]
		self.A_coref = row[6]
		self.B = preprocess_text(row[7])
		self.B_ofs = row[8]
		self.B_coref = row[9]
		self.url = row[10]

# Read train data
def get_train_data():
	res = []
	with open("gap-development.tsv") as fd:
	    rd = csv.reader(fd, delimiter="\t", quotechar='"')
	    next(rd)
	    for row in rd:
	    	res.append(Data(row))
	return res

# Read validation data
def get_validation_data():
	res = []
	with open("gap-validation.tsv") as fd:
	    rd = csv.reader(fd, delimiter="\t", quotechar='"')
	    next(rd)
	    for row in rd:
	    	res.append(Data(row))
	return res

# Read test data
def get_test_data():
	res = []
	with open("gap-test.tsv") as fd:
	    rd = csv.reader(fd, delimiter="\t", quotechar='"')
	    next(rd)
	    for row in rd:
	    	res.append(Data(row))
	return res

# Write to snippet output if is_snippet is True. Otherwise, write to page output.
def write_to_tsv(results, is_snippet=True):
	if is_snippet:
		with open('CS372_HW5_snippet_output_20160830.tsv', 'w') as fd:
			wr = csv.writer(fd, delimiter='\t', quotechar='"')
			for row in results:
				wr.writerow(row)
	else:
		with open('CS372_HW5_page_output_20160830.tsv', 'w') as fd:
			wr = csv.writer(fd, delimiter='\t', quotechar='"')
			for row in results:
				wr.writerow(row)

if __name__ == '__main__':
	main()