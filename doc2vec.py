import pandas as pd
import numpy as np
import sys
import gensim
import random
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import logging
from os import listdir


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])

    def doc_perm(self):
        c = list(zip(self.labels_list, self.doc_list))
        random.shuffle(c)
        labels_tuple, doc_tuple = zip(*c)
        self.labels_list = list(labels_tuple)
        self.doc_list = list(doc_tuple)

def load_dictionary(dictionary_type):
    if dictionary_type == "pressman":
        doc_labels = [f for f in listdir('pressman_part') if f.endswith('.txt')]
        doc_data = []
        for doc in doc_labels:
            doc_data.append(open('pressman_part/' + doc, 'r').read())
    elif dictionary_type == "eclipse_bug":
        doc_labels = [f for f in listdir('eclipse_bugs') if f.endswith('.txt')]
        doc_data = []
        for doc in doc_labels:
            doc_data.append(open('eclipse_bugs/' + doc, 'r').read())
    elif dictionary_type == "mozilla_bug":
        doc_labels = [f for f in listdir('mozilla_bugs') if f.endswith('.txt')]
        doc_data = []
        for doc in doc_labels:
            doc_data.append(open('mozilla_bugs/' + doc, 'r').read())
    elif dictionary_type == "openoffice_bug":
        doc_labels = [f for f in listdir('openoffice_bugs') if f.endswith('.txt')]
        doc_data = []
        for doc in doc_labels:
            doc_data.append(open('openoffice_bugs/' + doc, 'r').read())
    return doc_data, doc_labels


def train_doc2vec(dictionary_type):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data, doc_labels = load_dictionary(dictionary_type)
    it = LabeledLineSentence(data, doc_labels)

    model = gensim.models.Doc2Vec(dm=0, vector_size=300, window=15, alpha=0.025,
                                  min_alpha=0.0001, min_count=5, sample=1e-5, negative=5,
                                  dbow_words=1, dm_concat=1, hs=0)

    model.build_vocab(it)
    for epoch in range(20):
        print('epoch: ' + str(epoch))
        it.doc_perm()
        model.train(it, total_examples=model.corpus_count, epochs=model.iter)


    model_name = 'models/'+ dictionary_type +'_doc2vec'
    model.save(model_name)
    print(dictionary_type + '_doc2vec model trained')


def predict_doc2vec(bug_directory, output_file, dictionary_type):
    model = gensim.models.Doc2Vec.load('models/' + dictionary_type + '_doc2vec')
    print(dictionary_type + '_doc2vec model loaded')
    # load all bugs
    bug_list = [f for f in listdir(bug_directory) if f.endswith('.txt')]

    bug_list = bug_list
    # create header to write to csv
    header_num = range(1, 301)
    header_str = ["d" + str(h) for h in header_num]
    df = pd.DataFrame(columns=['bug ID'] + header_str)
    # for each bug, get vector representation
    for idx, bug_file in enumerate(bug_list):
        bug_id = bug_file.replace('.txt', '')
        bug_content = open(bug_directory + bug_file, 'r').read()
        if dictionary_type == "pressman":
            vec = model.infer_vector(bug_content.split(), alpha=0.01, min_alpha=0.0001, steps=500)
        else: # bugs
            vec = model[bug_file]
            #vec = model.infer_vector(bug_content, alpha=0.01, min_alpha=0.0001, steps=500)
        vec_with_id = np.insert(vec, 0, bug_id, axis=0)
        df.loc[idx] = vec_with_id
        if idx % 1000 == 0:
            print(str(idx) + '/' + str(len(bug_list)))
    df.to_csv(df.to_csv(output_file, encoding='utf-8', index=False, float_format='%.6f'))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        is_train = True
        dictonary_type = sys.argv[1]
        # dictionary_type = pressman
        # dictionary_type = eclipse_bug
        # dictionary_type = mozilla_bug
        # dictionary_type = openoffice_bug
        train_doc2vec(dictonary_type)
    elif len(sys.argv) > 1:
        is_train = False
        bug_directory = sys.argv[1]
        # bug_directory = 'eclipse_bugs/'
        # bug_directory = 'mozilla_bugs/'
        # bug_directory = 'openoffice_bugs/'
        output_file = sys.argv[2]
        # output_file = 'doc2vec_tables/eclipse_doc2vec_pressman.csv'
        # output_file = 'doc2vec_tables/mozilla_doc2vec_pressman.csv'
        # output_file = 'doc2vec_tables/openoffice_doc2vec_pressman.csv'
        dictionary_type = sys.argv[3]
        # dictionary_type = pressman
        # dictionary_type = eclipse_bug
        # dictionary_type = mozilla_bug
        # dictionary_type = openoffice_bug
        predict_doc2vec(bug_directory, output_file, dictionary_type)