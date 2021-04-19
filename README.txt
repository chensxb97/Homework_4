This is the README file for A0228375X-A0228420N-A0230636L-A0229819N's submission 
Email: 
e0673208@u.nus.edu (A0228375X)
e0673253@u.nus.edu (A0228420N)
e0698543@u.nus.edu (A0230636L)
e0684534@u.nus.edu (A0229819N)


== Python Version ==

We're using Python Version 3.7.4 for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

= Indexing =

The index dictionary is built by processing terms from the legal corpus, Intellex’s dataset. 

After case-folding all words to lowercase, stemming, and removing punctuations, terms are stored in both a set(to ensure no duplicates) and a list(to track term frequencies), and are saved in the dictionary, sorted by term in ascending order, with the 
following format: {term: [termID, docFrequency, charOffset, stringLength]}.

- term(string) refers to the processed and stemmed word
- termID(int) is a unique ID associated with each word after the words have all been sorted in ascending order
- docFrequency(int) is the number of unique documents each term exists in
- charOffset(int) are character offset values which point to the start of the posting list in the postings file for that term.
- stringLength(int) states the length of the posting list generated for that term.

In addition to the index dictionary, we also keep track of the collection size by incrementing its value after processing each document. We also pre-compute the normalized
document lengths for each document and store them in a separate document lengths dictionary, with the following format: {docID: docLength}. The lengths are computed by taking 
the square root of the sum of squared (1+ log10(termFrequency)) for all unique terms in each document. These values will be used for cosine normalization of tf-idf weights in search.py.
We also created a postings dictionary that stores the docId-termFrequency pairs for each term using the following format: {term: {docId: termFrequency}}. 

To build the postings file, we iterate through the terms from the postings dictionary and obtain each term's dictionary of docId-termFrequency pairs. We then construct the posting string using the value-pairs, with markers '^' to separate the docId and termFrequency and ',' to separate the pairs. 
After updating the charOffset and stringLength values in the main index dictionary, we write and save the posting strings to the output posting file.

For our query refinement process, we have included an additional step in the indexing phase to get the relevant document vector containing the top 20 tf-idf weights with the pseudo-relevance feedback. For each document ID, we then sort and obtain these weights in descending order and store it in a new dictionary (relevantDocs_dict). We then pickle and store this dictionary to be used in search.py. We will later extract these weights and use it in our scoring system as part of the Rocchio’s Algorithm.

Lastly, we save the finalised index dictionary, document lengths dictionary and collection size value as a list in a pickled file so that they could be easily re-loaded in memory to be used in search.py.


= Searching =

The search algorithm takes in the pickled index dictionary, document lengths dictionary, collection size, postings file and queries as input arguments.
The objective is to process each query and obtain all the documents that are relevant to the query using the vector space model, with the exploration and implementation of query refinement and expansion (which can be activated or deactivated via a boolean flag value).

For the Rocchio Algorithm, the rocchio configuration settings are initialised in the rocchio_config dictionary and contain the boolean flag (use_rocchio) to activate the rocchio algorithm and corresponding alpha and beta values. These alpha and beta values determine the weightage assigned to the original vector space model and the query refinement for the relevant documents respectively. In this implementation, we utilised positive feedback from relevant documents in alignment with many other IR systems which only utilise positive feedback.

Breaking down each query, we extract the terms (by performing the same pre-processing as in index.py) and store them and their individual computed tf-weights
(tf-weight = 1+ log10(tf)) in a dictionary. For every term, we obtain their docFrequencies by accessing the index dictionary and compute their individual idfs (idf = log10(collection_size/docFrequency). 
We compute and store each term's tf-idf weight score back in the dictionary. Lastly, we normalize each weight by the query's length.
The resulting query dictionary has the following format : {term1: tf-idf1/queryLength, term2: tf-idf2/queryLength}. The Rocchio Formula query refinement is then factored in subsequently should the boolean activation flag be turned on. The top 20 most important terms are selected and added into the query vector with a predefined weight applied (rocchio_beta). This is implemented by calculating the centroid and generating a weighted centroid score.

Iterating through the terms in the query dictionary, we extract the posting strings from the posting file using seek(charOffset) and read(strLength). 
We then obtain and compute each document's term's tf-weight as usual. Similarly, we normalize each weight by the pre-computed document's length from the document lengths dictionary.
We then store each document term's normalized tf-weight in a dictionary of dictionaries.
The resulting document dictionary has the following format: {doc1:{term1:tf-wt1/docLength1,term2:tf-wt2/docLength1},doc2:{term1:tf-wt1/docLength2,term2:tf-wt2/docLength2}}

To compute the vector scores, we iterate through both query and document dictionaries and sum up the tf-wt(document)*tf-idf(query) for each document, storing the scores in a list. 

Lastly, we write and save all the relevant document IDs for each query line by line to the output results file.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

> index.py
Builds the index necessary for searching - returns the index dictionary, document length dictionary, collection size and postings file

> search.py
Processes a list of queries and returns all the relevant documents ids

> dictionary.txt
Pickled file containing a list of three objects: 
1) index dictionary: keys(terms) and values(termIDs, termFrequency, charOffsets, stringLengths)
2) document length dictionary: keys(docIDs) and values(docLengths)
3) relevantDocs dictionary: keys(docIDs) and values(top 20 tf-idf weights document vector)
4) collection size: the total number of documents processed

> postings.txt
Returns a single line of all terms' posting lists(arranged in ascending order of terms)

> README.txt
Information file for documentation(this)

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I, A0228375X-A0228420N-A0230636L-A0229819N, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[] I, A0228375X-A0228420N-A0230636L-A0229819N, did not follow the class rules regarding homework
assignment, because of the following reason:

NIL

I suggest that I should be graded as follows:

NIL

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

File operations
https://www.zframez.com/tutorials/python-tutorial-file-operations.html

Using pickle to save multiple dictionaries
https://stackoverflow.com/questions/25318987/working-with-two-dictionaries-in-one-pickle-file

Difference between 'split' and 'tokenize' methods when pre-processing text
https://stackoverflow.com/questions/35345761/python-re-split-vs-nltk-word-tokenize-and-sent-tokenize

Heapq library where we utilised the nlargest function
https://docs.python.org/3/library/heapq.html

Rocchio Algorithm Explaination
https://youtu.be/yPd3vHCG7N4