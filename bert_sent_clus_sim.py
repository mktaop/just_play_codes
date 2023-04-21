#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:06:08 2023

@author: avi_patel
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

# Load the dataframe of sentences
df = pd.read_csv('sentences.csv')

# Create an empty list to store the embeddings
embeddings = []

# Loop through the dataframe and get the embedding for each sentence
for i in range(len(df)):
    sentence = df['sentence'][i]
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.pooler_output.squeeze().numpy()
    embeddings.append(embedding)

# Convert the list of embeddings to a numpy array
embeddings = np.array(embeddings)

# Cluster the embeddings using KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
labels = kmeans.labels_

# Create a TfidfVectorizer object to extract keywords
vectorizer = TfidfVectorizer(stop_words='english')

# Loop through each cluster and extract the top 10 keywords
keywords = []
for i in range(5):
    # Get the indices of the embeddings in this cluster
    cluster_indices = np.where(labels == i)[0]
    
    # Get the sentences in this cluster
    cluster_sentences = df.iloc[cluster_indices]['sentence']
    
    # Vectorize the sentences
    X = vectorizer.fit_transform(cluster_sentences)
    
    # Get the top 10 keywords
    feature_names = vectorizer.get_feature_names()
    tfidf_scores = X.sum(axis=0).A1
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:10]]
    keywords.append(top_keywords)

# Create a dataframe to store the keywords
df_keywords = pd.DataFrame(keywords, columns=['Keyword 1', 'Keyword 2', 'Keyword 3', 'Keyword 4', 'Keyword 5', 'Keyword 6', 'Keyword 7', 'Keyword 8', 'Keyword 9', 'Keyword 10'])

# Calculate the cosine distance between each pair of embeddings
cosine_matrix = cosine_distances(embeddings)

# Store the cosine distances in a dataframe
df_cosine = pd.DataFrame(cosine_matrix, columns=df['sentence'], index=df['sentence'])

# Find the 5 nearest neighbors for each sentence
nearest_neighbors = []
for i in range(len(df)):
    distances = cosine_matrix[i]
    sorted_indices = np.argsort(distances)
    nearest_indices = sorted_indices[1:6]
    nearest_sentences = df.iloc[nearest_indices]['sentence'].values.tolist()
    nearest_neighbors.append(nearest_sentences)

# Store the nearest neighbors in a dataframe
df_nearest_neighbors = pd.DataFrame(nearest_neighbors, columns=['nearest_1', 'nearest_2', 'nearest_3', 'nearest_4', 'nearest_5'], index=df['sentence'])


"""
In this example, we use the BERT-base-multilingual-cased model from the Transformers library to get sentence embeddings for the sentences in the dataframe.

First, we load the BERT tokenizer and model using the AutoTokenizer and AutoModel classes from the Transformers library. We then load the dataframe of sentences using the pd.read_csv() function.

We create an empty list called embeddings to store the embeddings for all the sentences.

We use a for loop to loop through the dataframe. In each iteration of the loop, we select a sentence from the dataframe using the indexing syntax df['sentence'][i]. We then tokenize the sentence using the BERT tokenizer and convert the tokens to a PyTorch tensor using the tokenizer() function. We pass return_tensors='pt' to the tokenizer() function to get a PyTorch tensor as output.

We then pass the tensor to the BERT model using the model() function. We wrap the model() function in a with torch.no_grad(): block to disable gradient calculations, as we don't need gradients for inference.

We get the embedding for the sentence by selecting the pooler_output tensor from the output of the BERT model, squeezing it to remove any extra dimensions, and converting it to a numpy array using the numpy() method.

We append the embedding to the embeddings list.

After the loop, we convert the embeddings list to a numpy array using the np.array() function. The resulting embeddings variable is a 2D numpy array with shape (1000, 768), where each row represents the embedding for a sentence.
"""

"""
In this example, we use the KMeans algorithm from the scikit-learn library to cluster the 1000 sentence embeddings into 5 clusters.

After getting the 1000 sentence embeddings as before, we create a KMeans object with 5 clusters using the KMeans() function from scikit-learn. We then fit the KMeans object to the embeddings using the fit() method.

The labels_ attribute of the KMeans object gives the cluster assignments for each of the 1000 embeddings. We store these labels in a variable called labels.

Note that the number of clusters (5 in this example) is arbitrary and can be adjusted to suit your needs. The scikit-learn documentation provides guidance on how to choose an appropriate number of clusters for a given dataset.
"""

"""
In this example, we use the TfidfVectorizer class from scikit-learn to extract the top 10 keywords for each cluster.

After clustering the sentence embeddings using KMeans as before, we create a TfidfVectorizer object with the stop_words argument set to 'english'. This removes common English stopwords from the vectorizer, which are unlikely to be useful keywords.

We then loop through each cluster and perform the following steps:

Get the indices of the embeddings in this cluster
Get the sentences in this cluster
Vectorize the sentences using the fit_transform() method of the TfidfVectorizer object
Get the feature names (i.e. the keywords) using the get_feature_names() method of the vectorizer object
Get the TF-IDF scores for each feature using the sum() and A1 methods of the vectorized sentences
Sort the feature indices in descending order based on their TF-IDF scores
Extract the top 10 feature names as the top 10 keywords for this cluster
We store the top 10 keywords for each cluster in a list called keywords, and then create a Pandas dataframe called df_keywords to store the
"""

"""
In this example, we use the cosine_distances() function from scikit-learn to calculate the cosine distance between each pair of embeddings.

After getting the embeddings for each sentence as before, we convert the list of embeddings to a numpy array and calculate the cosine distance matrix using the cosine_distances() function.

We then create a Pandas dataframe called df_cosine to store the cosine distance matrix. We set the column and row labels of the dataframe to be the sentence strings themselves, so that we can easily look up the cosine distance between any two sentences using their string values as indices.

Note that the cosine distance matrix is a symmetric matrix, meaning that cosine_matrix[i,j] == cosine_matrix[j,i] for any indices i and j. Therefore, the resulting df_cosine dataframe is also symmetric along its main diagonal.
"""

"""
In this example, we loop through each sentence in the dataframe and calculate its cosine distance to every other sentence using the cosine_matrix we calculated earlier.

For each sentence, we sort the cosine distances in ascending order and select the indices of the 5 smallest distances (excluding the distance to itself, which is always zero). We then look up the corresponding sentence strings using their indices in the original dataframe and store them in a list.

After looping through all sentences, we create a new Pandas dataframe called df_nearest_neighbors to store the nearest neighbor information. We set the index of the dataframe to be the sentence strings themselves and the columns to be the top 5 nearest neighbors for each sentence.
"""



