import pandas as pd
import numpy as np
import sys
import gensim
import os
import random
import nltk.data
from nltk.tokenize import word_tokenize
import logging
from nltk.stem.snowball import SnowballStemmer
import string
import math
import xml.etree.ElementTree as ET
from os import listdir
# nltk.download('punkt')


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in [f for f in listdir(self.dirname) if f.endswith('.txt')]:
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


def train_word2vec(dictionary_type):
    if dictionary_type == "pressman":
        dictionary_folder = 'pressman_part/'
    elif dictionary_type == "mozilla_bug":
        dictionary_folder = 'mozilla_bugs/'
    elif dictionary_type == "eclipse_bug":
        dictionary_folder = 'eclipse_bugs/'
    elif dictionary_type == "openoffice_bug":
        dictionary_folder = 'openoffice_bugs/'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = MySentences(dictionary_folder)  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, min_count=1, size=100, workers=4)
    model.save('models/' + dictionary_type + '_word2vec')



def predict_word2vec(bug_directory, output_file, dictionary_type):
    model = gensim.models.Word2Vec.load('models/' + dictionary_type + '_word2vec')
    print(dictionary_type + '_word2vec model loaded')
    # load all bugs
    bug_list = [f for f in listdir(bug_directory) if f.endswith('.txt')]

    header_num = range(1, 101)
    header_str = ["d" + str(h) for h in header_num]
    df = pd.DataFrame(columns=['bug ID'] + header_str)

    for idx, bug_file in enumerate(bug_list):
        bug_id = bug_file.replace('.txt', '')
        bug_content = open(bug_directory + bug_file, 'r').read()
        bug_content_list = bug_content.split()
        filtered_content_list = list(filter(lambda x: x in model.wv.vocab, bug_content_list))
        if len(filtered_content_list) == 0:
            print(str(bug_id) + " is empty")
            vec_average = [0] * 100
        else:
            vec_list = model[filtered_content_list]
            vec_average = np.mean(vec_list, axis=0)

        vec_with_id = np.insert(vec_average, 0, bug_id, axis=0)
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
        train_word2vec(dictonary_type)
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
        predict_word2vec(bug_directory, output_file, dictionary_type)
