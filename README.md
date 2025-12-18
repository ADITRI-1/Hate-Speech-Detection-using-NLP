# Hate-Speech-Detection-using-NLP
## Overview

This project implements a Hate Speech Detection system using Natural Language Processing (NLP) and machine learning techniques. The pipeline includes text preprocessing, feature extraction, model training, evaluation, and prediction generation for unseen test data.

The final output is a solution.csv file containing predicted labels for the test dataset.

### Tech Stack

* Python

* Pandas

* NLTK

* Scikit-learn

### Dataset:- 
https://drive.google.com/drive/folders/1uFLZoH2ORwczc3Q7v0C-4UwbJIniJQDL 

The dataset is loaded using Pandas and follows a predefined file structure.
It contains textual data with binary labels:
* HS = 1 → Hate Speech

* HS = 0 → Non-Hate Speech

The provided directory and file structure are strictly followed to ensure correct execution.

## Project Workflow
#### 1. Data Loading and Exploration

* Loaded the dataset using Pandas.

* Displayed sample rows to understand structure.

* Analyzed class distribution.

* Visualized class imbalance using a bar plot.

#### 2. Text Preprocessing

Performed using NLTK:

* Tokenization: nltk.word_tokenize

* Stop Word Removal: nltk.corpus.stopwords

* Stemming / Lemmatization: Applied to normalize text and reduce vocabulary size

#### 3. Feature Extraction

Used Scikit-learn for text vectorization:

* Bag-of-Words (BoW): CountVectorizer

* TF-IDF: TfidfVectorizer

* Tested different ngram_range values such as (1,1) and (1,2) to analyze impact on performance

#### 4. Model Training and Evaluation

Trained classifiers such as:

* Naive Bayes

* Logistic Regression

* Support Vector Machine (SVM)

* Models were trained using both BoW and TF-IDF features.

Evaluated performance using:

  * Accuracy

  * Precision

  * Recall

  * F1-score

Compared BoW vs. TF-IDF representations.

## How to Run
#### 1. Clone the Repository:
    git clone <repository-url>
    cd hate-speech-detection

#### 2. Set Up the Environment:
   Ensure Python 3.8+ is installed, then install dependencies:

       pip install pandas nltk scikit-learn matplotlib

#### 3. Download NLTK Resources

Run the following once:

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

#### 4. Prepare Dataset

Ensure the dataset follows the required structure:

data/
  train.csv
  test.csv

#### 5. Run the Script
      python hate_speech_detection.py

#### 6. Output
* The script generates solution.csv in the project root.

* This file contains predictions for the test dataset in the required format.

## Project Structure
data/
  train.csv
  test.csv

hate_speech_detection.py
solution.csv
README.md
