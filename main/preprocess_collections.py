import re
from textblob import TextBlob
import spacy

def preprocess(context):
    '''
    preprocess context paragraph and return cleaned version of the paragraph

    Arguments:
        context -- context
    Returns:
        context  -- cleaned version of the context
    '''
    # removal of dictionary phonetics
    context = re.sub('\/.*\ˈ.*\/', '', context)

    # removal of japanese characters
    context = re.sub('[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]+', '', context)

    return context

def POStag(context):
    '''
    preprocess context paragraph and return list of words with tags

    Arguments:
        context -- context
    Returns:
        context_tagged  -- list of words with tags
    '''
    context_tagged =  TextBlob(context).tags
    return context_tagged

def NER(context):
    '''
    preprocess context paragraph and return list of entities with labels

    Arguments:
        context -- context
    Returns:
        context_ner -- list of entities with labels
    '''
	nlp = spacy.load('en_core_web_sm')
	entities = nlp(context)
	labels = labels = [x.label_ for x in entities.ents]

	return set(zip(entities.ents,labels))
	