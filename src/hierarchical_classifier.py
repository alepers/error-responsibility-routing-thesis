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
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

class Node:
    """A node in the hierarchy graph."""
    def __init__(self, name, data, classifier):
        self.name = name
        self.data = data
        self.classifier = classifier
        self.probabilities = {}
        self.children = []

    def __str__(self, level=0):
        ret = '\t' * level + self.name + '\n'
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<Tree graph representation>'

    def train(self, train_labels):
        if not self.children:
            print('Train: encountered leaf node.')
        elif len(self.children) < 2:
            print('Train: classifier has less than two child classes.')
        else:
            print('Starting training for node ' + self.name + '..')
            self.classifier.fit(self.data.loc[train_labels]['text'].values.astype(str), self.data.loc[train_labels]['class'].values.astype(str))
            for child in self.children:
                child.train(train_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--path-to-1st', required=True)
    parser.add_argument('-p2', '--path-to-2nd', required=True)
    args = parser.parse_args()

    # Construct the hierarchy graph, starting with the root node
    root_node = Node(name='root', data=None, classifier=None)

    p1 = args.path_to_1st
    p2 = args.path_to_2nd
    first_level_sources = get_sources(p1)
    second_level_sources = get_sources(p2)

    data = DataFrame({'text': [], 'class': []})
    class_dict = {}
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())])

    # Use class mapping file to supply corresponding class data
    with open('MHO_Mapping_csv.csv', 'r') as class_mapping:
        reader = csv.reader(class_mapping, delimiter=';')
        class_dict = {rows[0]:rows[1] for rows in reader}
        print('Mapped class relations..')

    for path, classification in first_level_sources:
        print('Fetching data for class ' + classification + ' ..')

        # Load data from each class into a Pandas DataFrame
        group_class_data = build_data_frame(path, classification)
        data = data.append(group_class_data)

        # Extend the graph by adding children nodes to the root node
        print('Appending child node \'' + classification + '\'..')
        data_group = DataFrame({'text': [], 'class': []})
        root_node.children.append(Node(name=classification,
                                       data=data_group,
                                       classifier=clone(pipeline)))

    data = data.reindex(np.random.permutation(data.index))
    root_node.data = data
    root_node.classifier = clone(pipeline)

    for child in root_node.children:
        for path, mho in second_level_sources:
            if mho in class_dict and child.name == class_dict[mho]:
                print('Appending child node \'' + mho + '\' to \'' + child.name + '\'..')
                data_lower = DataFrame({'text': [], 'class': []})
                lower_class_data = build_data_frame(path, mho)
                child.data = child.data.append(lower_class_data)
                child.children.append(Node(name=mho,
                                           data=None,
                                           classifier=None))

    k_fold = KFold(n_splits=10)
    scores = []

    for train_indices, test_indices in k_fold.split(data):
        selected_training_data = data.iloc[train_indices]
        train_labels = list(selected_training_data.index)

        # Train all classifiers recursively
        root_node.train(train_labels)

        # Make predictions
        selected_testing_data = data.iloc[test_indices]
        test_labels = list(selected_testing_data.index)
        #score = root_node.predict(test_labels)
        #scores.append(score)

#('classifier', LogisticRegression(solver='newton-cg', multi_class='multinomial'))]