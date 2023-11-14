import math

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer

def readTrainData(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    texAll = []
    lbAll = []
    voc = []
    for line in lines:
        splitted = line.split('\t')
        lbAll.append(splitted[0])
        texAll.append(splitted[1].split())
        words = splitted[1].split()
        for w in words:
            voc.append(w)
    voc = set(voc)
    cat = set(lbAll)
    return texAll, lbAll, voc, cat

def learn_NB_text():
# go across all categories and return # of times it appears in labels list
#Calculating class priors by Prior(class) = # of sentences in class / # of sentences in training set
    P = [] #We create a list to save the P(Class) for each class.
            #It is calculated by dividing the # of appearance of this class  / # of all appearance of all classes
    for category in cat: #Going across all classes
        P.append(lbAll.count(category) / len(lbAll)) #Add to the list the number of times category appears in lbAll and divide
                                                    # By the number of all classes
    #print(cat)
    current_words_count = []        #This will be used to store then number of words in the current category
    matrix = []                     # Will be used to store data, later will be transformed to Pw using DataFrame

    voc_c = list(voc.copy())      #A copy Vocabolary that contains a column for the unknown words, did not appear in the test.
    voc_c.append('UNKNOWN')

    i = 0
    current_category_words = []     #This list will hold all current words in a specefic category
    for current_category in cat:
        counter = 0
        current_category_words = [' '.join(texAll[i]) for (i, category_pick) in enumerate(lbAll) if category_pick == current_category] #This list contains all the words that appear in the current class
        count = CountVectorizer(vocabulary=list(voc)) # To find the total number of times a word appears in a class. --> CountVectorizer
        X_s = count.fit_transform(current_category_words)
        tdm_s = pd.DataFrame(X_s.toarray(), columns = count.get_feature_names_out()) #Returns the TDM matrix, (FOR TESTING)
        #print(tdm_s)
        word_list_q = count.get_feature_names_out() # List which contains all the words from all categories (No duplicates)
        count_list_q = X_s.toarray().sum(axis = 0) # Number of times each word of word_list_q appeared in the current category

        #Here we apply Laplace smoothing
        words_in_cat = count_list_q.sum() #Number of words in category
        words_p = np.array(count_list_q,dtype=float)
        words_p = count_list_q + 1
        words_p = words_p / (words_in_cat + len(voc))
        # words_p Contains The probabilites of each word to appear in the class after Laplace smoothing

        freq_q = dict(zip(word_list_q,words_p))
        #print(freq_q)
        #print(current_words_count)
        prob_s = []
        current_counter = 0
        for word,count in zip(word_list_q, words_p): #Now that we have frequency of each word in each class, we can
            prob_s.append(count)            #calculate the probability for each word in each class
        dict(zip(word_list_q, prob_s))              #Prob is a list of probability for each word in each class
        matrix.append(prob_s)                       # Appending the list to the matrix.
        matrix[i] = np.append(words_p, 1 / words_in_cat + len(voc))
        i += 1
    #print(current_words_count)
    Pw = pd.DataFrame(matrix, index=tuple(cat), columns=voc_c)
    #print(Pw.to_string())
    return Pw,P



def ClassifyNB_text(Pw, P):
    correct = 0        #Counts number of correct identified sentences.
    max_category = '' #This is used to store the name of the most likely category so far (for the current sentence)
    max_category_prob = -math.inf #This is used to store the probablity of the most likely category so far (for the current sentence)
    sentence_prob_array = []        #This array will hold the probabilites of each word in this sentence
    for i, sentence in enumerate(texAllTest): #Going across all sentences
        max_category_prob = -math.inf
        max_category = ''
        for j, catagory in enumerate(cat):
            sentence_prob_array.clear()
            for word in sentence:       #If word exists in vocabolary, add its P to the lis
                if(word in voc):        #else, add the P of 'UNKNOWN'
                    sentence_prob_array.append(Pw[word][catagory])
                else:
                    sentence_prob_array.append(Pw['UNKNOWN'][catagory])

            log_sentence = np.log(sentence_prob_array) #Summing logs is much faster than
            category_prob = log_sentence.sum()         # Multiplication, so we log every value and sum them.
            category_prob = category_prob + math.log(P[j])  #According to Bayes Theorem

            if category_prob > max_category_prob: #If the current category is more likely to be the sentence's category
                max_category_prob = category_prob   # Then we replace both of those variables
                max_category = catagory

        # count number of correctly classified sentences
        if max_category == lbAllTest[i]: #If the most likely category is actually the real category
            correct += 1                #This sentence belongs to, then we add the counter of
    correct = correct / len(lbAllTest)  # correctly identified sentences by 1.
    return correct                     #The % of correctly identified sentences = correctly identified / all sentences


texAll, lbAll, voc, cat = readTrainData('r8-train-stemmed.txt')
texAllTest, lbAllTest, vocTest, catTest = readTrainData('r8-test-stemmed.txt')

Pw,P = learn_NB_text()
result = ClassifyNB_text(Pw,P)
print(result)

