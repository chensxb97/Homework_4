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

    df_zf = ZipFile(in_dir) # Reading zip file
    df = pd.read_csv(df_zf.open('dataset.csv')) # Load dataset

    # Test with the first 10 records
    n = 10

    print(df.iloc[0])

    return # Continue from here

    # Word tokenization and stemming
    list_punc = list(string.punctuation)
    stemmer = PorterStemmer()
    index_dict = {}
    postings_dict = {}
    docLengths_dict = {}
    collection_size = 0
    files = [f for f in listdir(in_dir) if isfile(
        join(in_dir, f))]  # all files from directory
    sorted_files = sorted(files, key=lambda f: int(
        os.path.splitext(f)[0]))  # sorted files in ascending order

    # Word processing and tokenisation for each document
    for file in sorted_files:
        file_path = join(in_dir, file)
        f = open(file_path, "r")

        # Store all observed words in a list, used to track termFrequency
        termList = []
        # Set data structure is used to store the unique words only
        termSet = set()
        docLength = 0

        for line in f:
            new_line = ''
            for c in line:
                if c not in list_punc:
                    new_line += c
            new_line = new_line.lower()
            for sentence in nltk.sent_tokenize(new_line):
                for word in nltk.word_tokenize(sentence):
                    word = stemmer.stem(word)
                    termList.append(word)
                    termSet.add(word)

        # Populate postings_dict
        # postings_dict = {token: {docId: termFrequency}}
        for t in termList:
            if t in postings_dict.keys():
                if int(file) in postings_dict[t].keys():
                    termFreq = postings_dict[t][int(file)]
                    termFreq += 1
                    postings_dict[t][int(file)] = termFreq
                else:
                    postings_dict[t][int(file)] = 1
            else:
                postings_dict[t] = {}
                postings_dict[t][int(file)] = 1

        # Populate index_dict and docLengths_dict
        # index_dict = {token: docFrequency}
        # docLengths_dict = {docId: docLength}
        for t in termSet:
            if t in index_dict.keys():
                docFreq = index_dict[t]
                docFreq += 1
                index_dict[t] = docFreq
            else:
                index_dict[t] = 1
            # docLength of a document is computed for tf (document) cosine normalization
            docLength += math.pow(1+math.log10(postings_dict[t][int(file)]), 2)
        docLength = math.sqrt(docLength)
        docLengths_dict[int(file)] = docLength

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
