import numpy as np
import re

# Function to remove punctuation from text
def remove_punctuations(text):
    pattern = r', |\.|\?'
    return re.sub(pattern, " ", text)

# Tokenize the text by splitting on spaces
def tokenise(text):
    text = remove_punctuations(text)
    return text.split()

# Build a sorted vocabulary from a list of texts
def build_vocabulary(text):
    vocab = set()
    for example in text:
        vocab.update(tokenise(example))
    return sorted(vocab)

# Count the occurrences of each term in a list of texts
def count_terms(text):
    text = tokenise(text)
    term_count = {}
    for word in text:
        if word not in term_count:
            term_count[word] = 1
        else:
            term_count[word] +=1
    return term_count

# Count the log-transformed occurrences of each term in a list of texts
def log_count_terms(text):
    term_count = count_terms(text)
    log_tf = {word: np.log(count + 1) for word, count in term_count.items()}
    return log_tf

# Calculate the document frequency of each term in a list of texts
def document_frequency(text, vocab):
    word_doc_count = {}
    for word in vocab:
        for example in text:
            if word in example:
                if word not in word_doc_count:
                    word_doc_count[word] = 1
                else:
                    word_doc_count[word] += 1
    return word_doc_count

# Calculate the inverse document frequency of each term in a list of texts
def inverse_document_frequency(text, vocab):
    df = document_frequency(text, vocab)
    num_documents  = len(text)
    idf = {word: np.log(num_documents / doc_frequency) for word, doc_frequency in df.items()}
    return idf

# Calculate the TF-IDF score for each term in a list of texts
def tf_idf(text, vocab):
    tf_idf_text = {}
    idf = inverse_document_frequency(text, vocab)

    for doc in text:
        tf = log_count_terms(doc.lower())
        total_words = sum(tf.values())

        for word, log_tf_count in tf.items():
            """ Multiplying by (log_tf_count / total_words) scales the TF-IDF score 
            by the relative importance of the term within the document. This ensures
            that longer documents don't dominate in terms of TF-IDF scores."""

            tf_idf_text[word] = log_tf_count * idf[word] * (log_tf_count / total_words)

    return tf_idf_text