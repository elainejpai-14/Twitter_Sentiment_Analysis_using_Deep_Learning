# Twitter Sentiment Analysis using Word2Vec & BiLSTM

## Overview
This project is an implementation of the highest performing Deep Learning model for twitter sentiment analysis mentioned in the IEEE research paper: https://ieeexplore.ieee.org/document/9689241. 
This model performs sentiment analysis on Twitter data using a trained deep learning model which utilizes **Word2Vec** for word embeddings and a **Bi-directional Long Short-Term Memory (BiLSTM)** network for classification.

## Features
- Preprocessing of raw Twitter data
- Word embeddings using **Word2Vec**
- Deep learning-based sentiment classification with **BiLSTM**
- Visualization of sentiment distribution

## Dataset
- The sentiment140 dataset from kaggle contains 1,600,000 tweets extracted using the twitter api.
- The tweets have been annotated as (0 = negative, 4 = positive) and can be used to detect sentiment.
- Data is preprocessed by removing stopwords, punctuation, and performing tokenization.
- Link to dataset: https://www.kaggle.com/datasets/kazanova/sentiment140

## Technologies Used
- Python
- NumPy & Pandas
- NLTK (Natural Language Toolkit)
- Gensim (for Word2Vec)
- TensorFlow/Keras (for BiLSTM)
- Matplotlib & Seaborn (for visualization)

## Installation
Clone the repository and install the required dependencies:
```bash
 git clone https://github.com/yourusername/Twitter_Sentiment_Analysis.git
 cd Twitter_Sentiment_Analysis
 pip install -r requirements.txt
```

## Usage
Run the Jupyter Notebook:
```bash
 jupyter notebook Twitter_Sentiment_Analysis_using_Deep Learning.ipynb
```
Follow the steps in the notebook to preprocess data, train the model, and visualize results.

## Model Architecture
- **Word2Vec Embeddings**: Converts words into vector representations.
- **BiLSTM Layer**: Captures contextual dependencies in both forward and backward directions.
- **Dense Output Layer**: Predicts sentiment categories (positive, negative, neutral).

## Results and Performance
The trained BiLSTM model achieves high accuracy in sentiment classification. The results are visualized using confusion matrices and sentiment distribution plots.
Achieved an accuracy of approximately **85%** on the dataset after training for 12 epochs. Better performance than most of the other Machine Learning models like Naive bayes and Support Vector Machines(SVM) and Deep Learning models like Recurrent Neural Networks and Long-Term Short Memory(LSTM).

