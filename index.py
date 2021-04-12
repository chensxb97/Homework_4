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

# usage
# python index.py -i 'dataset.zip' -d dictionary.txt -p postings.txt


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
    docLengths_dict = {}
    collection_size = 0

    # Load dataset
    df_zf = ZipFile(in_dir)  # Reading from a zipped file
    df = pd.read_csv(df_zf.open('dataset.csv'))
    # consider text from all zones except doc_id
    zones = (df.columns).drop('document_id')

    # Sort the dataframe by doc_id in ascending order
    df["document_id"] = pd.to_numeric(df['document_id'])
    df = df.sort_values(by=['document_id'])

    # Test with the first 10 documents
    # Word processing and tokenisation for each document
    # For each record, we iterate through the zones and populate index_dict
    n = 10
    for i in range(n):
        record = df.iloc[i]
        docId = record['document_id']
        docLength = 0
        for zone in zones:
            # Store all observed terms in a list, used to track termFrequency
            termList = []
            # Set data structure is used to store the unique words only
            termSet = set()

            clean_text = ''
            raw_text = record[zone]
            for c in raw_text:
                if c not in list_punc:
                    clean_text += c
            clean_text = clean_text.lower()
            for sentence in nltk.sent_tokenize(clean_text):
                stemmed_words = [stemmer.stem(
                    word) for word in nltk.word_tokenize(sentence)]  # stemming
                sentence = ' '.join(stemmed_words)
                for i in range(1, 4):  # Generate ngrams
                    gramList = []
                    gramList = get_ngrams(sentence, i)
                    for gram in gramList:
                        termList.append(gram)
                        termSet.add(gram)

            # Populate postings_dict
            # postings_dict = {token_zone: {docId: termFrequency}}
            for t in termList:
                # Terms have to be compared within the same zone
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
    # Dictionary is now {term : [termID, docFrequency]}

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
    # Final dictionary is now {term : [termID,docFrequency,charOffSet,strLength]}

    # Save index, length dictionaries and collection size using pickle
    pickle.dump([sorted_index_dict, docLengths_dict,
                 collection_size], open(out_dict, "wb"))
    print('done!')


def get_ngrams(text, n):
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
