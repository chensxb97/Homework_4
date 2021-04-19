#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import math
import pickle
import os
import time
import io
from zipfile import ZipFile
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn

# usage
# python3 search.py -d dictionary.txt -p postings.txt  -q queries.zip -o results.txt


def usage():
    print("usage: " +
          sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below

    # Rocchio Configuration Settings
    rocchio_config = {
        "use_rocchio": True,
        "rocchio_alpha": 1,
        "rocchio_beta": 0.2
    }
    wordNet_config = {
        "use_wordNet": True,
        "word_limit": 20,
        "weight": 0.1
    }

    # Initialise stemmer
    stemmer = PorterStemmer()

    # Open and load dictionary
    # Sorted index dictionary is {term : [termID,docFrequency,charOffSet,strLength]}
    # Document length dictionary is {docID: cosine normalized document length}
    # Collection size is the total number of documents, to be used for idf calculation
    in_dict = open(dict_file, 'rb')
    sorted_dict = pickle.load(in_dict)
    sorted_index_dict = sorted_dict[0]
    docLengths_dict = sorted_dict[1]
    relevantDocs_dict = sorted_dict[2]
    collection_size = sorted_dict[3]

    # Open posting lists, but not loaded into memory
    postings = open(postings_file, 'r')

    # Open queries zipped file
    q_zf = ZipFile(queries_file)
    queries = []
    queries_groundtruth_docs_list = []

    # for i in range(1, len(q_zf.namelist())+1):
    #     q_file = io.TextIOWrapper(
    #         q_zf.open("q{}.txt".format(i)), encoding="utf-8")
    #     query = (q_file.readline()).strip()
    #     queries.append(query)

    #     query_groundtruth_docs = q_file.readlines()
    #     query_groundtruth_docs = [x.strip() for x in query_groundtruth_docs]
    #     queries_groundtruth_docs_list.append(query_groundtruth_docs)

    queries = ['government problem', 'illegal racing bet',
               'chinese magistrate petitioners']
    queries_groundtruth_docs_list = [
        [246403, 246427], [246403, 246427], [246403, 246427]]

    # Process each query and store the results in a list
    # query_results = [[result for query1],[result for query 2]...]
    query_results = []
    for query_index, query in enumerate(queries):
        query_groundtruth_docs = queries_groundtruth_docs_list[query_index]
        # Store all normalized query tf-idf weights in query_dict
        query_dict = process_query(
            query, sorted_index_dict, collection_size, stemmer, wordNet_config, rocchio_config, query_groundtruth_docs, relevantDocs_dict)

        # Store all normalized document tf weights in document_dict
        document_dict = process_documents(
            query_dict, sorted_index_dict, postings)

        #print("document_dict", document_dict.keys())

        # Generates the top 10 documents for the query
        scores = process_scores(
            query_dict, document_dict, docLengths_dict)

        query_results.append(scores)

    # Write query results into output results_file
    with open(results_file, 'w') as results_file:
        for result in query_results:
            result_line = ""
            # If result is empty, just write an empty line
            # If result is not empty, write each documentID (starting from highest rank) with a whitespace separating each documentID
            if result is not None:
                for docID, score in result:
                    result_line = result_line + str(docID) + ' '
            # Remove final whitespace in results line
            results_file.write(result_line[:-1])
            # Ensure final result does not print new line in results file
            if result != query_results[-1]:
                results_file.write('\n')
    print('done!')


def process_query(input_query, sorted_index_dict, collection_size, stemmer, wordNet_config, rocchio_config, query_groundtruth_docs, relevantDocs_dict):
    '''
    Processes and extracts terms/phrases from the input query.
    Returns a dictionary containing the normalized tf-idf weights for each term/phrase.
    query_dict = {term: normalized tf-idf-wt}
    '''
    query_dict = {}

    zones = ['title', 'content', 'date_posted', 'court']

    original_query = []

    # Word processing and tokenisation for query terms
    if 'AND' in input_query:
        tokens = input_query.split('AND')  # Separate the phrases/terms
    else:
        tokens = input_query.split(' ')
    for t in tokens:
        if '"' in t:
            t = t.replace('"', '')  # Remove inverted commas
        t = t.strip()  # Remove trailing whitespaces
        t = t.lower()  # lower case-folding
        original_query.append(t)
        if ' ' in t:  # Checks if t is a phrase
            split = t.split(' ')
            stemmed = [stemmer.stem(word)
                       for word in split]  # stem individual terms
            t = '&'.join(split)  # Stemmed phrase with & as the delimiter
        else:
            t = stemmer.stem(t)  # Stem lone term
        for z in zones:
            t_z = t + '_{}'.format(z)  # Transform 'term' to 'term_zone'
            if t_z in query_dict.keys():  # Populate query_dict
                query_dict[t_z] += 1
            else:
                query_dict[t_z] = 1

    print('first one', query_dict)
    print('length', len(query_dict))

    # WORDNET QUERY EXPANSION
    # Set use_wordNet to False if not using query expansion
    if wordNet_config['use_wordNet']:
        limit = wordNet_config['word_limit']
        num_terms = len(original_query)
        partition = ([limit // num_terms + (1 if x < limit %
                                            num_terms else 0) for x in range(num_terms)])
        expanded_query = []
        total_count = 0
        for i in range(num_terms):
            expanded_count = 0
            synsets = wn.synsets(original_query[i])
            for synset in synsets:
                for new_term in synset.lemma_names():
                    if original_query[i] == new_term:
                        continue
                    if expanded_count == partition[i]:
                        break
                    expanded_query.append(new_term)
                    expanded_count += 1
            total_count += expanded_count

        # Expanded_query = ['term1','term2','term3' ...]
        for t in expanded_query:
            if ' ' in t:  # Checks if t is a phrase
                split = t.split(' ')
                stemmed = [stemmer.stem(word)
                           for word in split]  # stem individual terms
                t = '&'.join(split)  # Stemmed phrase with & as the delimiter
            else:
                t = stemmer.stem(t)  # Stem lone term
            for z in zones:
                t_z = t + '_{}'.format(z)  # Transform 'term' to 'term_zone'
                if t_z in query_dict.keys():  # Populate query_dict
                    query_dict[t_z] += 1
                else:
                    query_dict[t_z] = 1

    # Calcualte tf-idf-wt for query terms
    for t_z in query_dict.keys():
        # Calculate tf-wt
        q_tf = query_dict[t_z]
        q_tf_wt = 1 + math.log10(q_tf)

        # Calculate idf
        if t_z in sorted_index_dict.keys():
            q_df = sorted_index_dict[t_z][1]
            q_idf = math.log10(collection_size/q_df)
        else:
            q_idf = 0

        q_wt = q_tf_wt * q_idf

        # Store wt for each term_zone in query back into dictionary
        query_dict[t_z] = q_wt

    print('second one', query_dict)
    print('length', len(query_dict))

    # ROCCHIO FORMULA QUERY REFINEMENT
    # Set use_rocchio to False if not using query refinement
    if rocchio_config['use_rocchio']:
        centroid_dict = {}
        num_groundtruth_doc = len(query_groundtruth_docs)

        for doc_id in query_groundtruth_docs:

            doc_vector = relevantDocs_dict[doc_id]

            for (term, tf_idf) in doc_vector:  # Iterate term weights in the relevant document vector
                if term not in centroid_dict:
                    centroid_dict[term] = tf_idf
                else:
                    centroid_dict[term] += tf_idf

        for term in centroid_dict.keys():
            # Divide the vector sum by the number of relevant documents
            centroid_score = centroid_dict[term] / num_groundtruth_doc
            weighted_centroid_score = centroid_score * \
                rocchio_config['rocchio_beta']  # Weighted centroid score using the rocchio beta weight
            if term in query_dict.keys():
                query_weight = query_dict[term]
                query_weight += weighted_centroid_score
                # equals 1*query_dict[term] + 0.2*weighted_centroid_score
                query_dict[term] = query_weight
            else:
                # equals 0.2*weighted_centroid_score
                query_dict[term] = weighted_centroid_score

    return query_dict


def process_documents(query_dictionary, sorted_index_dict, input_postings):
    '''
    Checks for each term recorded in the input query dictionary with the main index dictionary.
    Returns a document dictionary containing the tf weights for each term found in the main index dictionary.
    document_dict = {document1: {term1: tf1, term2: tf2}, document2:{}...}
    '''
    # Returns None if query dictionary is empty
    if query_dictionary == None or all(value == 0 for value in query_dictionary.values()):
        return None

    document_dict = {}

    # Extract and process posting lists from main index dictionary
    for word in query_dictionary.keys():
        if word in sorted_index_dict.keys():
            charOffset = sorted_index_dict[word][2]
            strLength = sorted_index_dict[word][3]
            input_postings.seek(charOffset, 0)
            posting_str = (input_postings.read(strLength))
            posting_array = posting_str.split(',')
            for p in posting_array:
                documentID = p.split('^')[0]
                tf_raw = p.split('^')[1]
                if documentID not in document_dict.keys():
                    document_dict[documentID] = {}
                # Check if tf_raw can be converted into a valid integer
                if int(tf_raw):
                    document_dict[documentID][word] = int(tf_raw)

    # Calculate tf-wt for each document's terms
    for document in document_dict.keys():
        for word in query_dictionary.keys():
            if word in document_dict[document].keys():
                d_tf = int(document_dict[document][word])
                # Calculate tf-wt
                d_tf_wt = 1 + math.log10(d_tf)
                # Store weight back into dictionary
                document_dict[document][word] = d_tf_wt
            else:
                document_dict[document][word] = 0

    return document_dict


def process_scores(query_dictionary, document_dictionary, docLengths_dict):
    '''
    Computes the cosine-normalized query-document score for all terms for each document.
    Returns a list of the top 10 most relevant documents based on the query-document score. 
    '''

    # Returns empty result if query dictionary is empty
    if query_dictionary == None or all(value == 0 for value in query_dictionary.values()):
        return None

    result = []

    for docID in document_dictionary.keys():
        # Denominator for cosine normalization (docLength)
        normalize_doc = docLengths_dict[int(docID)]
        docScore = 0
        for term in query_dictionary.keys():
            doc_wt = document_dictionary[docID][term]
            term_wt = query_dictionary[term]
            docScore += doc_wt*term_wt
        # Perform cosine normalization for each document's score
        docScore = docScore/normalize_doc
        result.append((docID, docScore))

    # Return results in descending order without top K-threshold
    return sorted(result, key=lambda x: x[1], reverse=True)


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
