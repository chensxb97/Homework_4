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
    list_punc = list(string.punctuation)
    stemmer = PorterStemmer()
    index_dict = {}
    postings_dict = {}
    relevantDocs_dict = {}
    docLengths_dict = {}
    collection_size = 0

    # Load dataset
    df_zf = ZipFile(in_dir)  # Reading from zipped file
    df = pd.read_csv(df_zf.open('dataset.csv'))

    # Sort the dataframe by doc_id in ascending order
    df["document_id"] = pd.to_numeric(df['document_id'])
    df = df.sort_values(by=['document_id'])
    df = df.drop('date_posted', axis=1)
    df = df.drop('court', axis=1)

    # consider text from zones title and content, no doc_id, date_posted and court
    zones = (df.columns).drop('document_id')

    # Word processing and tokenisation for each document
    stop_words = stopwords.words('english')  # English stopword collection
    # Store stopword in dictionary for faster access O(1)
    stopwords_dict = Counter(stop_words)

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
            for sentence in nltk.sent_tokenize(raw_text):
                clean_text = ''
                for c in sentence:
                    if c not in list_punc:
                        clean_text += c  # Remove punctuation
                clean_text = nltk.word_tokenize(clean_text)
                clean_text_no_sw = ' '.join(
                    [word for word in clean_text if word not in stopwords_dict])  # Remove stopwords
                # Stem words within the sentence
                stemmed_words = [stemmer.stem(word)
                                 for word in clean_text_no_sw]
                sentence = ' '.join(stemmed_words)
                for i in range(1, 4):  # Generate unigrams, bigrams and trigrams
                    gramList = []
                    gramList = get_ngrams(sentence, i)
                    for gram in gramList:
                        termList.append(gram)
                        termSet.add(gram)

            # Populate postings_dict
            # postings_dict = {token_zone: {docId: termFrequency}}
            for t in termList:
                # Terms have to be categorised by their zones
                t_zone = t + '_{}'.format(zone)
                if t_zone in postings_dict.keys():
                    if docId in postings_dict[t_zone].keys():
                        termFreq = postings_dict[t_zone][docId]
                        termFreq += 1
                        postings_dict[t_zone][docId] = termFreq
                    else:
                        postings_dict[t_zone][docId] = 1
                else:
                    postings_dict[t_zone] = {}
                    postings_dict[t_zone][docId] = 1

            # Populate index_dict and docLengths_dict
            # index_dict = {token_zone: docFrequency}
            # docLengths_dict = {docId: docLength}
            for t in termSet:
                t_zone = t + '_{}'.format(zone)
                if t_zone in index_dict.keys():
                    docFreq = index_dict[t_zone]
                    docFreq += 1
                    index_dict[t_zone] = docFreq
                else:
                    index_dict[t_zone] = 1
                # docLength of a document is computed for tf (document) cosine normalization
                docLength += math.pow(1 +
                                      math.log10(postings_dict[t_zone][docId]), 2)

        docLength = math.sqrt(docLength)
        docLengths_dict[docId] = docLength

        # Increment collection size
        collection_size += 1
        print('Indexed: {}/{}'.format(collection_size, n))  # Log indexing

    # Sort index_dict
    sorted_index_dict_array = sorted(index_dict.items())
    sorted_index_dict = {}
    for termID, (term, value) in enumerate(sorted_index_dict_array):
        # Addition of 1 ensures that termIDs starts from 1
        termID += 1
        docFrequency = value
        sorted_index_dict[term] = [termID, docFrequency]
    # Dictionary is now {term : [termID, docFrequency]}

    # Sort postings_dict
    sorted_postings_dict_array = sorted(postings_dict.items())

    # Output postings file
    postings_out = open(out_postings, 'w')

    print('Starting on postings... ')  # Log postings

    # Generate and write posting strings to postings file
    # Store charOffset and stringLength in sorted_index_dict
    char_offset = 0
    sorted_posting_len = len(sorted_postings_dict_array)
    for (term, term_dict) in sorted_postings_dict_array:
        # Log Postings.txt creation
        print('Posting: ', term, ' of ', sorted_posting_len)
        postingStr, strLength = create_postings(term_dict)
        postings_out.write(postingStr)
        termId, docFrequency = sorted_index_dict[term]
        sorted_index_dict[term] = (
            termId, docFrequency, char_offset, strLength)
        char_offset += strLength
    postings_out.close()
    # Final dictionary is now {term : [termID,docFrequency,charOffSet,strLength]}

    print('Postings Done!')

    # Obtain relevant document vectors for pseudo-relevance feedback
    for docId, docLength in docLengths_dict.items():
        # Temporary dictionary to store all tf-idf weights in relevant documents before sorting
        temp_relevantDoc_dict = {}
        for term in postings_dict.keys():
            if docId in postings_dict[term]:
                # Calculate tf-wt
                termFrequency = postings_dict[term][docId]
                d_tf_wt = 1 + math.log10(termFrequency)

                # Calculate idf
                docFrequency = sorted_index_dict[term][1]
                d_idf = math.log10(collection_size/docFrequency)

                # tf-idf
                d_wt = d_tf_wt * d_idf

                # Perform cosine normalization
                d_normalize_wt = d_wt/docLength

                temp_relevantDoc_dict[term] = d_normalize_wt

        # Sort and obtain top-20 tf-idf weights in descending order
        # sorted_relevantDoc = {term1: tf-idf1, term2: tf-idf2 ...}
        sorted_relevantDoc = sorted(
            temp_relevantDoc_dict.items(), key=lambda x: x[1], reverse=True)[:20]

        # Store relevant document vector
        relevantDocs_dict[docId] = sorted_relevantDoc

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
