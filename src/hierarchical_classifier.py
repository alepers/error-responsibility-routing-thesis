import os
import sys
import csv
import numpy as np
from pandas import DataFrame
import argparse
from tr_classifier import get_sources, build_data_frame

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

class Node:
    """A node in the hierarchy graph."""
    def __init__(self):
        self.name = ''
        self.classifier = None
        self.probabilities = {}
        self.children = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--path-to-1st', required=True)
    parser.add_argument('-p2', '--path-to-2nd', required=True)
    args = parser.parse_args()

    # Construct the hierarchy graph, starting with the root node
    root_node = Node()
    root_node.name = 'root'

    p1 = args.path_to_1st
    p2 = args.path_to_2nd
    first_level_sources = get_sources(p1)
    second_level_sources = get_sources(p2)

    data = DataFrame({'text': [], 'class': []})

    for path, classification in first_level_sources:
        # Load data from each class into a Panda DataFrame
        data = data.append(build_data_frame(path, classification))

        # Extend the graph by adding children nodes to the root node
        root_node.children.append(Node(name=classification))

    data.reindex(np.random.permutation(data.index))

    pipeline_root = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', LogisticRegression(solver='lbfgs', 
                                          multi_class=('multinomial'),
                                          n_jobs=-1))])

    X_train, X_test, y_train, y_test = train_test_split(data['text'], 
                                                        data['class'], 
                                                        test_size=0.33, 
                                                        random_state=42)

    pipeline_root.fit(X_train.values, y_train.values)
    root_node.classifier = pipeline_root

    #predictions = pipeline.predict(X_test.values)
    #score = accuracy_score(y_test.values, predictions)

    #print('Score:', score)
