#Code adapted from https://github.com/8horn/seq2seq/blob/master/preprocess_data.py
from textblob import TextBlob
import spacy
import numpy as np
import pandas as pd
import pickle
import re

nlp = spacy.load('en_core_web_sm')

def preprocess(data, remove_stopwords=True, replace_entities=False):
    '''
    preprocess data and subtitute entities to their respective form, removal of stopwords,
    dictionary phonetics, foreign characters, and punctuations. questions are maintained in original form.

    Arguments:
        data -- data contains list of text to be processed
        remove_stopwords -- flag to remove stopwords
        replace_etities -- subtitute entities to their respective form
    Returns:
        preprocessed_data  -- preprocessed data
    '''
    preprocessed_data = []

    for idx in range(len(data)):
        #Parse text
        text = data[idx]

        if replace_entities:
            # Replacing entities to its form
            spacy_text = nlp(text)
            text_ents = [(str(ent), str(ent.label_)) for ent in spacy_text.ents]

            text = text.lower()
            # Replace entities
            for ent in text_ents:
                replacee = str(ent[0].lower())
                replacer = str(ent[1])
                try:
                    text = text.replace(replacee, replacer)
                except:
                    pass
        else:
            text = text.lower()

        # Clean text from stopwords, phonetics, foreign chars and punctuations
        # Addition of EOS tag
        text = nlp(text)
        text_ = ""
        if remove_stopwords:
            for sent in text.sents:
                sent = [str(token.orth_) for token in sent 
                    if not token.is_stop and not token.is_punct]
                sent.append('EOS')
                sent = ' '.join(sent)
                text_ += sent + ' '
        else:
            for sent in text.sents:
                sent = [str(token.orth_) for token in sent if not token.is_punct]
                sent.append('EOS')
                sent = ' '.join(sent)
                text_ += sent + ' '
        text = text_
        
        # Removal phonetics
        text = re.sub('\/.*\ˈ.*\/', '', text)
        # Removal japanese characters
        text = re.sub('[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]+', '', text)

        # Add SOS tag in the beginning
        text = 'SOS ' + text
        preprocessed_data.append(text)

        #Sanity check
        if idx % 2000 == 0:
            try:
                print('Sanity check, current parsed text:')
                print(text)
            except:
                pass

        #Save current preprocessed data
        if idx % 4000 == 0 and idx > 0:
            print('Saving current data...')
            save_pickle(preprocessed_data,'preprocessed_data')

    return preprocessed_data

def count_freq_words(word_freq, data):
    '''
    Count the frequency of words in corpus
    '''
    for text in data:
        for token in text.split():
            if token not in word_freq:
                # Init word
                word_freq[token] = 1
            else:
                word_freq[token] += 1
    return

def create_conversion_dictionaries(word_freq, embed_idx, threshold=10):
    '''
    Clean dataset by removing words appear less than threshold
    Arguments:
        word_freq -- dictionary of word frequencies in corpus
        embed_idx -- dictionary of words with their respective vectors 
        threshold -- limit of words frequencies to be discarded
    Returns:
        vocab2int  -- dictionary to convert vocabulary to integer
        int2vocab -- dictionary to reverse conversion
    '''
    print("Removal of words under threshold")
    missing_words = 0

    # count missing words in dataset from word embedding
    for token, freq in word_freq.items():
        if token not in embed_idx:
            missing_words += 1

    missing_ratio = round(missing_words/len(word_freq), 4) * 100
    print('Number of words mssing from Conceptnet Numberbatch vocabulary:', missing_words)
    print('Percentage of vocabulary missing:', missing_ratio, '%')

    # Dictionary to convert vocab to integer
    vocab2int = {}
    value = 0
    for token, freq in word_freq.items():
        if freq >= threshold or token in embed_idx:
            vocab2int[token] = value
            value += 1

    # Tokens to guide Seq2Seq / Attention RNN model
    vocab2int['UNK'] = len(vocab2int)

    # Create reverse conversion dictionary
    int2vocab = {}
    for token, idx in vocab2int.items():
        int2vocab[idx] = token

    usage_ratio = round(len(vocab2int) / len(word_freq), 4) * 100
    print("Total number of unique words: ", len(word_freq))
    print("Number of words will be used: ",len(vocab2int))
    print("Percent of words will be used: {}%".format(usage_ratio))

    return vocab2int, int2vocab

def create_embed_matrix(vocab2int, embed_idx, embed_dim=300):
    '''
    Create embedding matrix for each token in the corpus. If word vector not available, create random embedding
    Arguments:
        vocab2int  -- dictionary to convert vocabulary to integer
        embed_idx -- dictionary of words with their respective vectors 
        embed_dim -- dimension of word vectors as stated in Conceptnet, 300
    Returns:
        word_embed_matrix = final word vectors for corpus
    '''
    # Total number of words in corpus
    num_words = len(vocab2int)

    # Initial default matrix
    word_embed_matrix = np.zeros((num_words, embed_dim), dtype=np.float32)
    for token, idx in vocab2int.items():
        if token in embed_idx:
            # if token is in Conceptnet vectors
            word_embed_matrix[idx] = embed_idx[token]
        else:
            # random embedding
            new_embed = np.array(np.random.uniform(-1., 1., embed_dim))
            word_embed_matrix[idx] = new_embed
    
    return word_embed_matrix

def data_to_ints(data, vocab2int, word_count, unk_count):
    '''
    Convert words in text/data to its respective integer values
    Arguments:
        data        -- data contains context in dataset
        vocab2int   -- dictionary to convert vocabulary to integer
        word_count  -- integer to count words in dataset
        eos         -- boolean to append eos token at the end
    Returns:
        int_data    -- list of text converted to integer
        word_count  -- updated word count   
    '''
    int_data = []
    for text in data:
        int_text = []
        try:
            # Ensure input consists of tokens
            for token in text.split():
                word_count += 1
                if token in vocab2int:
                    # Convert token to integer
                    int_text.append(vocab2int[token])
                else:
                    # Unknown token
                    int_text.append(vocab2int['UNK'])
                    unk_count += 1
        except:
            # Empty input
            pass

        int_data.append(int_text)

    # Ensure converted data is the same as the original data
    assert len(int_data) == len(data)
    return int_data, word_count, unk_count

def unknown_counter(data, vocab2int):
    '''
    Count UNK token in data
    '''
    unk_count = 0
    for token in data:
        if token == vocab2int['UNK']:
            unk_count += 1
    return unk_count

def filter_data_length(converted_inputs, converted_targets, vocab2int,
                            start_input_length, max_input_length, max_target_length,
                            min_input_length=10, min_target_length=5,
                            unk_input_limit=1, unk_target_limit=0):
    '''
    Learn sequence based on sorted length of the text. Text learned within range length to
    minimize the use of computational resources. Removal of short length sequence as it might
    act as noise for learning.
    '''
    sorted_inputs = []
    sorted_targets = []

    print('Final preprocessing; sorting sequences to within range specified')
    for length in range(start_input_length, max_input_length):
        for idx, words in enumerate(converted_targets):
            if (len(converted_targets[idx]) >= min_target_length and
                len(converted_targets[idx]) <= max_target_length and
                unknown_counter(converted_targets[idx], vocab2int) <= unk_target_limit and
                unknown_counter(converted_inputs[idx], vocab2int) <= unk_input_limit and
                length == len(converted_inputs[idx])
                ):
                sorted_targets.append(converted_targets[idx])
                sorted_inputs.append(converted_inputs[idx])

    # Ensure sorted input and target matched
    assert len(sorted_inputs) == len(sorted_targets)
    print('There are {} inputs and targets pairs'.format(len(sorted_inputs)))
    return sorted_inputs, sorted_targets

# def POStag(context):
#     '''
#     preprocess context paragraph and return list of words with tags

#     Arguments:
#         context -- context
#     Returns:
#         context_tagged  -- list of words with tags
#     '''
#     context_tagged =  TextBlob(context).tags
#     return context_tagged

# def NER(context):
#     '''
#     preprocess context paragraph and return list of entities with labels

#     Arguments:
#         context -- context
#     Returns:
#         context_ner -- list of entities with labels
#     '''
#   nlp = spacy.load('en_core_web_sm')
#   entities = nlp(context)
#   labels = labels = [x.label_ for x in entities.ents]

#   return set(zip(entities.ents,labels))

def save_pickle(data, file_name):
    '''
    Save data to pickle format
    '''
    save_doc = open('../dataset/'+file_name+'.pickle','wb')
    pickle.dump(data, save_doc)
    save_doc.close()

def load_embeddings(embed_idx, file_path):
    '''
    load Numberbatch word embeddings
    '''
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embed = np.asarray(values[1:], dtype='float32')
            embed_idx[word] = embed
    print('Loaded Numberbatch word embeddings:', len(embed_idx))

def tensors_from_pairs(pairs):
    '''
    values are generated from function data_to_int that are saved in pickle format,
    read pickle in other file.
    '''

# Load pairs and preprocess
df_train = pd.read_csv(r'../dataset/train_pairs.csv')
preprocessed_inputs = preprocess(df_train['Paragraph'].tolist(), remove_stopwords=False, replace_entities=False)
save_pickle(preprocessed_inputs, 'preprocessed_inputs')

preprocessed_inputs_ans = preprocess(df_train['Answers'].tolist(), remove_stopwords=False, replace_entities=False)
save_pickle(preprocessed_inputs_ans, 'preprocessed_inputs_ans')

preprocessed_targets = preprocess(df_train['Question'].tolist(), remove_stopwords=False, replace_entities=False)
save_pickle(preprocessed_targets, 'preprocessed_targets')

# Load Numberbatch word embeddings
file_path = '../dataset/numberbatch-en-19.08.txt'
embed_idx = {}
load_embeddings(embed_idx, file_path)

# Calculate word frequency
word_freq = {}
count_freq_words(word_freq, preprocessed_targets)
count_freq_words(word_freq, preprocessed_inputs)
count_freq_words(word_freq, preprocessed_inputs_ans)

# Generate dictionary for conversion of usable vocabulary to integer
vocab2int, int2vocab = create_conversion_dictionaries(word_freq, embed_idx)
save_pickle(vocab2int, 'vocab2int')
save_pickle(int2vocab, 'int2vocab')

# Generate embedding matrix
word_embed_matrix = create_embed_matrix(vocab2int, embed_idx)
del embed_idx
save_pickle(word_embed_matrix, 'word_embed_matrix')

# Convert words to integer and save to pickle data
word_count = 0
unk_count = 0

converted_inputs, word_count, unk_count = data_to_ints(preprocessed_inputs, vocab2int, word_count, unk_count)
converted_targets, word_count, unk_count = data_to_ints(preprocessed_targets, vocab2int, word_count, unk_count)
converted_inputs_ans, word_count, unk_count = data_to_ints(preprocessed_inputs_ans, vocab2int, word_count, unk_count)

# Ensure both input and target converted are on the same length
assert len(converted_inputs) == len(converted_targets)

# Save to pickle format
save_pickle(converted_inputs, 'converted_inputs')
save_pickle(converted_targets, 'converted_targets')
save_pickle(converted_inputs_ans, 'converted_inputs_ans')

# Generate pairs for model input
tokenized_context = converted_inputs
tokenized_questions = converted_targets
tokenized_answers = converted_inputs_ans

# Generate with integers
pairs = []

for context, answer, question in zip(tokenized_context, tokenized_answers, tokenized_questions):
    context.extend(answer)
    pairs.append([context, question])

save_pickle(pairs, 'input_output_pairs_int')

# Generate pairs with string
pairs_str = []

for value in pairs:
    pairs_str.append([[int2vocab[val_in] for val_in in value[0]],[int2vocab[val_out] for val_out in value[1]]])
save_pickle(pairs_str, 'input_output_pairs')

# Saving list of vocabulary only
vocab_list = [str(value) for value in vocab2int]
save_pickle(vocab_list, 'vocab_list')

# Sort inputs and targets to be within the range length
# =========================== Not yet finished ======================
# sorted_inputs, sorted_targets = filter_data_length(converted_inputs, converted_targets, vocab2int, start_input_length)
=======

