import re
import spacy

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk

nltk.download('punkt')  # Download the punkt tokenizer data
nltk.download('wordnet')  # Download the WordNet data
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load('en_core_web_sm')


def tokens(x):
    """
    Tokenize the input text.
    Args:
        x (str): Input text.
    Returns:
        List of tokens.
    """
    return x.split()

def remove_common(text):
    pattern = r'the | the | a | to | is | an |”|“'  # Remove Common Words
    text = text.lower()
    text = re.sub(pattern," ",text)
    return text

## Lemmatization using NLTK
def lemmatize_text_with_pos(text):
    text = text.lower()                 #Convert to lower Case
    words = word_tokenize(text)         #Apply tokenization
    
    # Perform POS tagging
    pos_tags = pos_tag(words)
    
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word based on its POS tag
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos_tag))
                        for word, pos_tag in pos_tags]
    
    # Join the lemmatized words into a single string
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text

## Get Word Part of Speech
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to 'n' (Noun) if the POS tag is not recognized

def stemming(text):
    ps = PorterStemmer()
    words = word_tokenize(text.lower()) 
    stem_word = [ps.stem(word) for word in words]

    # Join the stemmed words into a single string
    stem_text = ' '.join(stem_word)
    return stem_text

#### Spacy Library
def lemmatize_spacy(text):
    """
    Lemmatize text using Spacy.
    Args:
        text (str): Input text.
    Returns:
        List of lemmatized tokens.
    """
    doc = nlp(text)
    lemms = []
    for token in doc:
        lemms.append(token.lemma_)
    return lemms

def tokenise_spacy(text):
    """
    Tokenize text using Spacy.
    Args:
        text (str): Input text.
    Returns:
        List of tokens.
    """
    doc = nlp(text)
    tokens = []                     # Create an empty list to store tokens
    for token in doc:
        tokens.append(token.text)
    return tokens                   # Return the list of tokens after the loop