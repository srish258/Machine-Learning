import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...

    model = None
    obs_dict = {}
    state_dict = {}
    curr_index = 0
    for tag in tags:
        state_dict[tag] = curr_index
        curr_index += 1

    curr_index = 0
    for line in train_data:
        for word in line.words:
            if word not in obs_dict:
                obs_dict[word] = curr_index
                curr_index += 1

    S = len(state_dict.keys())
    L = len(obs_dict.keys())
    
    ###################################################
    


    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    
    for line in train_data:
        pi[state_dict[line.tags[0]]] += 1
    pi /= np.sum(pi)

    for line in train_data:
        for i in range(len(line.tags)-1):
            A[state_dict[line.tags[i]], state_dict[line.tags[i+1]]] += 1

    for i in range(S):
        A[i, :] /= np.sum(A[i, :])

    for line in train_data:
        for i in range(len(line.words)):
            B[state_dict[line.tags[i]], obs_dict[line.words[i]]] += 1

    for i in range(S):
        B[i, :] /= np.sum(B[i, :])
    ###################################################
    

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, obs_dict, state_dict)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.

    S = len(model.state_dict.keys())

    for line in test_data:
        for word in line.words:
            if word not in model.obs_dict:
                b = np.ones([S, 1]) * 1e-6
                model.B = np.append(model.B, b, axis=1)
                model.obs_dict[word] = len(model.obs_dict.keys())

    for line in test_data:
        tagged_sentence = model.viterbi(line.words)
        tagging.append(tagged_sentence)

    ######################################################################

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words