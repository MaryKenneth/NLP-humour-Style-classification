import re
import scipy.sparse as sp
from collections import Counter

def generate_ngrams(text,n):
    """
    Generate n-grams from the given text.

    Args:
        text (str): Input text.
        n (int): Size of n-grams.

    Returns:
        list: List of n-grams.
    """
    text = text.lower()
    pattern = re.compile(r'[\S+]+')
    tokens = pattern.findall(text)

    ngrams = []
    for i in range(len(tokens)-n+1):
        pairs = tuple(tokens[i:n+i])
        ngrams.append(pairs)

    return ngrams

def build_ngram_vocab(text,n):
    """
    Build the vocabulary of n-grams from the given text.

    Args:
        text (str): Input text.
        n (int): Size of n-grams.

    Returns:
        set: Set of n-grams.
    """
    ngrams = generate_ngrams(text, n)
    return set(ngrams)

def bag_of_ngrams(text, n, vocab):
    """
    Create a bag-of-n-grams representation for the given text and vocabulary.

    Args:
        text (str): Input text.
        n (int): Size of n-grams.
        vocab (set): Vocabulary of n-grams.

    Returns:
        numpy.ndarray: Dense array representing the bag-of-n-grams.
    """
    ngrams_list  = generate_ngrams(text, n)  # Assuming you have a function n_grams

    # Create a dictionary of n-grams to indices
    ngram_indices  = {ngram: i for i, ngram in enumerate(vocab)}

    # Count the occurrences of each n-gram in the text
    ngram_counts  = Counter(ngrams_list)

    # Create lists to store data, row, and column indices
    data = []
    row = []
    col = []

    # Populate the lists with n-gram counts
    for ngram, count in ngram_counts.items():
        if ngram in vocab:
            col.append(ngram_indices[ngram])
            row.append(0)  # Assuming we are representing a single document
            data.append(count)

    # Create a sparse vector for the text
    sparse_vector = sp.csr_matrix((data, (row, col)), shape=(1, len(vocab)))
    
    # Convert the CSR matrix to a dense array
    dense_array = sparse_vector.toarray()

    return dense_array

# Example usage
text = "This is an example sentence. This sentence is an example."
ngram_size = 2
vocab = build_ngram_vocab(text, ngram_size)
result = bag_of_ngrams(text, ngram_size, vocab)
print(result)