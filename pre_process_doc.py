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


def clean_text(input_text):
    # separate into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(input_text)
    words = [word_tokenize(s) for s in sentences]
    lower_words = [[w.lower() for w in s] for s in words]
    table = str.maketrans('', '', string.punctuation)
    stripped_words = [[w.translate(table) for w in s] for s in lower_words]
    stemmer = SnowballStemmer("english")
    stemmed_words = [[stemmer.stem(w) for w in s] for s in stripped_words]
    for s_idx, s in enumerate(stemmed_words):
        s[:] = [value for value in s if len(value) > 1]
        s = [x for x in s if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]
        stemmed_words[s_idx] = s

    clean_sentences = [' '.join(s) for s in stemmed_words]

    return clean_sentences


def separate_doc(doc):
    fp = open(doc)
    text = fp.read()
    clean_sentences = clean_text(text)

    file_num = math.ceil(len(clean_sentences)/10)
    for part in range(file_num):
        thefile = open('pressman_part/content_' + str(part) + '.txt', 'w')
        begin_sentence = part * 10
        if part == file_num:
            last_sentence = len(clean_sentences)
        else:
            last_sentence = begin_sentence + 10

        for s in clean_sentences[begin_sentence:last_sentence]:
            if len(s) > 0:
                thefile.write("%s\n" % s)


def separate_bugs(project_name):
    if project_name == "eclipse":
        issue_file = "eclipse_issues.xml"
        bugs_directory = "eclipse_bugs/"
    elif project_name == "mozilla":
        issue_file = "mozilla_issues.xml"
        bugs_directory = "mozilla_bugs/"
    elif project_name == "openoffice":
        issue_file = "openoffice_issues.xml"
        bugs_directory = "openoffice_bugs/"
    else:
        issue_file = ""
        bugs_directory = ""
    tree = ET.parse(issue_file)
    root = tree.getroot()
    for bug in root.findall('bug'):
        bug_id = bug.find('bug_id').text
        short_desc_element = bug.find('short_desc')
        description = ""
        if short_desc_element is not None:
            description = description + short_desc_element.text
        for long_desc in bug.findall('long_desc'):
            text = long_desc.find('thetext').text
            description = description + " " + text
        filename = bugs_directory + str(bug_id) + ".txt"
        file = open(filename, "w")

        cleaned_description = clean_text(description)
        for s in cleaned_description:
            if len(s) > 0:
                file.write("%s\n" % s)
        file.close()


if __name__ == '__main__':
    separate_doc("pressman.txt")
    # separate_bugs("eclipse")
    # separate_bugs("mozilla")
    # separate_bugs("openoffice")

