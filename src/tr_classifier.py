import os
import sys
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

SKIP_FILES = {'cmds'}

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    f = open(file_path, encoding="latin-1")
                    content = f.read().replace('\n', '')
                    f.close()
                    yield file_path, content

def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

def get_sources(path):
    sources = [(os.path.join(os.path.basename(os.path.normpath(path)), name), name) for name in os.listdir(path)]
    return sources

if __name__ == '__main__':
    path = sys.argv[-1]

    data = DataFrame({'text': [], 'class': []})
    sources = get_sources(path)
    for path, classification in sources:
        data = data.append(build_data_frame(path, classification))

    data = data.reindex(np.random.permutation(data.index))

    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('classifier',  LinearSVC()) ])

    k_fold = KFold(n=len(data), n_folds=10)
    scores = []
    confusion = np.zeros((17, 17), dtype=np.int)

    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

    #   confusion += confusion_matrix(test_y, predictions)
        score = accuracy_score(test_y, predictions)
        scores.append(score)

    print('Total TRs classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)
