import nltk
import pickle

def argparser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--reference', type=str, default='../data/reference.pickle', help='Reference pickle file location, file is list of sentences')
    Argparser.add_argument('--hypothesis', type=str, default='../data/hypothesis.pickle', help='Hypothesis pickle file location, file is list of sentences')

    args = Argparser.parse_args()
    return args

def load_pickle(file_name):
    '''
    Load data to pickle format
    '''
    load_doc = open('../data/'+file_name+'.pickle','rb')
    data = pickle.load(load_doc)
    load_doc.close()
    return data

args = argparser()

reference = load_pickle(args.reference)
hypothesis = load_pickle(args.hypothesis)

try:
 	if len(reference) == len(hypothesis):
 		score = 0.
 		for i in range(len(reference)):
    		score += nltk.translate.bleu_score.sentence_bleu([reference[i].strip().split()], hypothesis[i].strip().split())

		score /= len(reference)
		print("The BLEU score is: "+str(score))
 except:
 	raise ValueError('The number of sentences in both files do not match.')
