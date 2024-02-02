# Humor Style Classification

## Overview
Humour has been identified as therapeutic or detrimental to one’s health and relationships based on the humour styles practised. Consequently, this report tackles the urgent need for an extensive multimodal humour style recognition system that can aid in the improvement of mental health. The study focuses on creating a humour-centric-based machine learning system that can accurately identify the four main humour styles (self-enhancing, self-deprecating, affiliative, and aggressive) in textual, audio, and video content. 

This repository contains implementations and experiments related to humour style classification in the text modality. The project explores various feature engineering techniques and classification models to predict humour styles. The implemented models include Naive Bayes, Multinomial Logistic Regression (both from scratch and using PyTorch), Feedforward Neural Network (FNN), and a Transformer model based on BERT.

## Dataset Details

The dataset used for this project comprises jokes sourced from various websites, covering Neutral, affiliative, self-enhancing, self-deprecating, and aggressive humor styles. A total of 1269 jokes were collected, with each joke associated with a specific humor label. The labeling process involved different approaches:

- **Labeling Based on Website Tags or Titles (1069 jokes):**
  The majority of the jokes (1069) were labeled based on the website #tags or titles. This method leveraged existing categorization provided by the websites hosting the jokes.

- **Labeling by Human Evaluators (200 jokes):**
  For 200 jokes where explicit humor style identification was lacking, three PhD students were involved in the labeling process. Each student provided a label, and the final label for each joke was determined by a majority vote.

This diverse dataset with explicit and implicit humor style labels was used for training and evaluating the humor style classification models.

## Features Explored
- **Feature Engineering Techniques:**
  - Bag-of-ngrams
  - Tf-idf
  - Pos/neg/pronoun count
  - Skip-gram
  - Classification Naive Bayes (from scratch)
  - Multinomial Logistic Regression (from scratch and PyTorch)
  - Feedforward Neural Network (FNN)

- **Text Preprocessing Steps:**
  - Lemmatization
  - Tokenization
  - Stop words and punctuation removal

- **Models:**
  - Naive Bayes
  - Multinomial Logistic Regression
  - Feedforward Neural Network (FNN)
  - BERT-based Transformer Model

## Results
- **Naive Bayes:**
  - Accuracy: 68% ± 3

- **Multinomial Logistic Regression:**
  - Accuracy varies based on feature engineering techniques.

- **Skip-gram:**
  - Accuracy up to 70% for 200 embeddings dimension and 65% for 100 embedding dimension.

- **Pos/neg/pronoun count:**
  - Accuracy of 44% with only four features Positive and negative lexicon count and first/third person pronoun count.

- **FNN with Bag-of-ngram features:**
  - Accuracy up to 64%, F1-score of 63%.
  - Performance worsens with an increasing number of grams.

- **Transformer (BERT) Model:**
  - Best performance with 81% accuracy and 80% F1-score.

## Libraries Used
- numpy
- pandas
- matplotlib
- nltk
- torch
- pyarrow
- datasets
- transformers
- sklearn

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/MaryKenneth/NLP-humour-Style-classification.git
   cd humor-style-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Explore Jupyter notebooks and Python scripts in the `main` directory to understand the implementations and experiments.

## Acknowledgments
Special thanks to the developers of libraries and tools used in this project, including numpy, pandas, nltk, torch, transformers, and datasets.

Feel free to contribute, provide feedback, or use the code for your projects!