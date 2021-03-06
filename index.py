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
from nltk.corpus import stopwords
from collections import Counter

# usage

# python3 index.py -i dataset.zip -d dictionary.txt -p postings.txt

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
    punc.append("'")  # Include apostrophe as punctuation
    punc_dict = Counter(punc)  # Punctuation dictionary for faster access O(1)
    stop_words = stopwords.words('english')
    stopwords_dict = Counter(stop_words) # Stopword dictionary for faster access O(1)
    stemmer = PorterStemmer() # Stemmer
    index_dict = {}
    postings_dict = {}
    relevantDocs_dict = {}
    docLengths_dict = {}
    collection_size = 0

    # Load dataset
    df_zf = ZipFile(in_dir)  # Reading from zipped file
    df = pd.read_csv(df_zf.open('dataset.csv'))

    # consider free text from all zones except doc_id
    zones = (df.drop(columns=['document_id'])).columns

    # Sort the dataframe by doc_id in ascending order
    df["document_id"] = pd.to_numeric(df['document_id'])
    df = df.sort_values(by=['document_id'])

    # Word processing and tokenisation for each document
    # For each document, we iterate through its zones and populate the dictionaries
    n = len(df.index)
    for i in range(n):
        record = df.iloc[i]
        docId = record['document_id']
        docLength = 0
        for zone in zones:
            # Store all observed terms in a list, used to track termFrequency
            termList = []
            # Set data structure is used to store the unique words only
            termSet = set()

            raw_text = record[zone]
            raw_text = raw_text.lower()
            for sentence in nltk.sent_tokenize(raw_text):
                for word in nltk.word_tokenize(sentence):
                    # Only index alphabetical non-stopword tokens
                    if word.isalpha() and word not in stopwords_dict and word not in punc_dict:
                        stemmed = stemmer.stem(word)
                        termList.append(stemmed)
                        termSet.add(stemmed) 

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

    # Sort index_dict
    sorted_index_dict_array = sorted(index_dict.items())
    sorted_index_dict = {}
    for termID, (term, value) in enumerate(sorted_index_dict_array):
        # Addition of 1 ensures that termIDs starts from 1
        termID += 1
        docFrequency = value
        sorted_index_dict[term] = [termID, docFrequency]
    # Index dictionary is now {term : [termID, docFrequency]}

    # Sort postings_dict
    sorted_postings_dict_array = sorted(postings_dict.items())

    # Output postings file
    postings_out = open(out_postings, 'w')

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
    # Final index dictionary is now {term : [termID,docFrequency,charOffSet,strLength]}

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

        # Sort and obtain top-10 tf-idf weights in descending order
        # sorted_relevantDoc = {term1: tf-idf1, term2: tf-idf2 ...}
        sorted_relevantDoc = sorted(
            temp_relevantDoc_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        # Store relevant document vector
        relevantDocs_dict[docId] = sorted_relevantDoc

    # Save index, length dictionaries and collection size using pickle
    pickle.dump([sorted_index_dict, docLengths_dict, relevantDocs_dict,
                 collection_size], open(out_dict, "wb"))
    print('done!')

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
