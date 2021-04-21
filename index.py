#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import sys
import string
import os
import pickle
import math
import linecache
import io
import pandas as pd
from zipfile import ZipFile
from os import listdir
from os.path import join, isfile
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter

# usage
# python3 index.py -i 'dataset.zip' -d dictionary.txt -p postings.txt


def usage():
    print("usage: " +
          sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')
    # This is an empty method
    # Pls implement your code in below

    # Initialisation
    punc = list(string.punctuation)
    punc.append("'")  # Include apostrophe
    punc_dict = Counter(punc)  # Punctuation dictionary for faster access O(1)
    stop_words = stopwords.words('english')
    # Stopword dictionary for faster access O(1)
    stopwords_dict = Counter(stop_words)
    stemmer = PorterStemmer()
    index_dict = {}
    postings_dict = {}
    relevantDocs_dict = {}
    docLengths_dict = {}
    collection_size = 0

    # Load dataset
    df_zf = ZipFile(in_dir)  # Reading from zipped file
    df = pd.read_csv(df_zf.open('dataset.csv'))
    # consider free text from all zones except doc_id
    df = df.drop('date_posted', axis=1)
    df = df.drop('court', axis=1)
    zones = df.drop('document_id',axis = 1).columns # zones = ['title','content']

    # Sort the dataframe by doc_id in ascending order
    df["document_id"] = pd.to_numeric(df['document_id'])
    df = df.sort_values(by=['document_id'])

    # Word processing and tokenisation for each document

    # For each document, we iterate through its zones and populate postings_dict and index_dict
    n = len(df.index)
    for i in range(n):
        record = df.iloc[i]
        docId = record['document_id']
        print('Starting on document: {}'.format(docId))  # Log docId
        docLength = 0
        for zone in zones:
            # Store all observed terms in a list, used to track termFrequency
            termList = []
            # Set data structure is used to store the unique words only
            termSet = set()

            raw_text = record[zone]
            raw_text = raw_text.lower()
            for sentence in nltk.sent_tokenize(raw_text): # Sentence
                for word in nltk.word_tokenize(sentence): # Split the sentence
                    if word.isalpha() and word not in stopwords_dict and word not in punc_dict:
                        termList.append(word)
                        termSet.add(word) 
                
                # clean_text = ''.join(
                    # [word for word in sentence if word not in punc_dict])
                # clean_text_no_sw = ' '.join([word for word in nltk.word_tokenize(
                    # clean_text) if word not in stopwords_dict])  # Remove stopwords
                # if only_alpha:
                    # stemmed = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(
                        # clean_text_no_sw) if word.isalpha()])  # Stem words within the sentence
                # else:
                    # stemmed = ' '.join(
                        # [stemmer.stem(word) for word in nltk.word_tokenize(clean_text_no_sw)]
                # for i in range(1, 2):  # Generate unigrams - No bigrams and trigrams due to lack of memory
                    # gramList = []
                    # gramList = get_ngrams(stemmed, i)
                    # for gram in gramList:
                        # termList.append(gram)
                        # termSet.add(gram)

            # Populate postings_dict
            # postings_dict = {token_zone: {docId: termFrequency}}
            for t in termList:
                if t in postings_dict.keys():
                    if docId in postings_dict[t].keys():
                        termFreq = postings_dict[t][docId]
                        termFreq += 1
                        postings_dict[t][docId] = termFreq
                    else:
                        postings_dict[t][docId] = 1
                else:
                    postings_dict[t] = {}
                    postings_dict[t][docId] = 1

            # Populate index_dict and docLengths_dict
            # index_dict = {token_zone: docFrequency}
            # docLengths_dict = {docId: docLength}
            for t in termSet:
                if t in index_dict.keys():
                    docFreq = index_dict[t]
                    docFreq += 1
                    index_dict[t] = docFreq
                else:
                    index_dict[t] = 1
                # docLength of a document is computed for tf (document) cosine normalization
                docLength += math.pow(1 +
                                      math.log10(postings_dict[t][docId]), 2)

        docLength = math.sqrt(docLength)
        docLengths_dict[docId] = docLength

        # Increment collection size
        collection_size += 1
        print('Indexed: {}/{}'.format(collection_size, n))  # Log indexing

    # Sort index_dict
    sorted_index_dict_array = sorted(index_dict.items())
    sorted_index_dict = {}
    for (term, value) in enumerate(sorted_index_dict_array): 
        docFrequency = value
        sorted_index_dict[term] = [docFrequency]
    # Dictionary is now {term : [termID, docFrequency]}

    # Sort postings_dict
    sorted_postings_dict_array = sorted(postings_dict.items())

    # Output postings file
    postings_out = open(out_postings, 'w')

    print('Starting on postings... ')  # Log postings

    # Generate and write posting strings to postings file
    # Store charOffset and stringLength in sorted_index_dict
    char_offset = 0
    for (term, term_dict) in sorted_postings_dict_array:
        postingStr, strLength = create_postings(term_dict)
        postings_out.write(postingStr)
        termId, docFrequency = sorted_index_dict[term]
        sorted_index_dict[term] = (
            termId, docFrequency, char_offset, strLength)
        char_offset += strLength
    postings_out.close()
    # Final dictionary is now {term : [termID,docFrequency,charOffSet,strLength]}

    print('Postings Done!')
    
    # # Obtain relevant document vectors for pseudo-relevance feedback
    # for docId, docLength in docLengths_dict.items():
    #     # Temporary dictionary to store all tf-idf weights in relevant documents before sorting
    #     temp_relevantDoc_dict = {}
    #     for term in postings_dict.keys():
    #         if docId in postings_dict[term]:
    #             # Calculate tf-wt
    #             termFrequency = postings_dict[term][docId]
    #             d_tf_wt = 1 + math.log10(termFrequency)

    #             # Calculate idf
    #             docFrequency = sorted_index_dict[term][1]
    #             d_idf = math.log10(collection_size/docFrequency)

    #             # tf-idf
    #             d_wt = d_tf_wt * d_idf

    #             # Perform cosine normalization
    #             d_normalize_wt = d_wt/docLength

    #             temp_relevantDoc_dict[term] = d_normalize_wt

    #     # Sort and obtain top-10 tf-idf weights in descending order
    #     # sorted_relevantDoc = {term1: tf-idf1, term2: tf-idf2 ...}
    #     sorted_relevantDoc = sorted(
    #         temp_relevantDoc_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    #     # Store relevant document vector
    #     relevantDocs_dict[docId] = sorted_relevantDoc

    print('Pickling...')

    # Save index, length dictionaries and collection size using pickle
    pickle.dump([sorted_index_dict, docLengths_dict, relevantDocs_dict,
                 collection_size], open(out_dict, "wb"))
    print('Indexing done!')


def get_ngrams(text, n):
    '''
    Returns a list of n-grams from an input string
    '''
    n_grams = ngrams(nltk.word_tokenize(text), n)
    return ['&'.join(grams) for grams in n_grams]


def create_postings(term_dictionary):
    '''
    Returns a constructed posting list in string format
    '''
    result = ''
    for docId, freq in term_dictionary.items():
        result += str(docId)
        result += '^'
        result += str(freq)
        result += ','
    # Output postingStr format: 'docId^termFrequency,docId^termFrequency'
    return result[:-1], len(result[:-1])


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
