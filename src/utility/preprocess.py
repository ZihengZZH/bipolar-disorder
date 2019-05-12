import os
import json
import string
import numpy as np
import pandas as pd
import xml.etree.ElementTree as etree
from smart_open import smart_open
from gensim.corpora import WikiCorpus
from gensim import utils

from src.utility.io import get_sample, load_label
from src.utility.io import load_baseline_feature


'''
ALL PRE-PROCESS FUNCTIONS
------------------------------------
preproc_baseline_feature(feature_name, verbose=False)
    pre-process the baseline features (LLDs)
preproc_transcript(partition)
    preprocess transcript files to one document
tokenize_tr(content, token_min_len=2, token_max_len=50, lower=True)
    tokenize words in the corpus
process_corpus(verbose=False)
    preprocess Turkish wikimedia cospus to line-based text file
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))


def preproc_baseline_feature(feature_name, verbose=False):
    """pre-process the baseline features (LLDs)
    """
    # para feature_name: which feature to pre-process
    # para verbose: whether or not to output more results
    no_train = data_config['train_len']
    no_dev = data_config['dev_len']
    # keep one instance in every # instances
    keep = data_config['keepinstance']

    def remove_if_exist(filename):
        if os.path.isfile(filename):
            os.remove(filename)

    # load output filenames
    train_data = data_config['baseline_preproc'][feature_name]['train_data']
    train_label = data_config['baseline_preproc'][feature_name]['train_label']
    train_inst = data_config['baseline_preproc'][feature_name]['train_inst']
    dev_data = data_config['baseline_preproc'][feature_name]['dev_data']
    dev_label = data_config['baseline_preproc'][feature_name]['dev_label']
    dev_inst = data_config['baseline_preproc'][feature_name]['dev_inst']

    # remove file if exists
    remove_if_exist(train_data)
    remove_if_exist(train_label)
    remove_if_exist(train_inst)
    remove_if_exist(dev_data)
    remove_if_exist(dev_label)
    remove_if_exist(dev_inst)

    # load the labels
    ymrs_train, ymrs_dev, level_dev, level_train = load_label()

    for partition in ['train', 'dev']:
        index_range = no_train if partition == 'train' else no_dev
        if verbose:
            print("\n----preprocessing on %s, dataset %s----" % (feature_name, partition))
        
        if partition == 'train':
            data_loc, label_loc, inst_loc = train_data, train_label, train_inst
        else:
            data_loc, label_loc, inst_loc = dev_data, dev_label, dev_inst

        dataf = smart_open(data_loc, 'a+', encoding='utf-8')
        labelf = smart_open(label_loc, 'a+', encoding='utf-8')
        instf = smart_open(inst_loc, 'a+', encoding='utf-8')

        for id in range(1, index_range+1):
            sample = get_sample(partition, id)

            if partition == 'train':
                ymrs_sample = ymrs_train[ymrs_train.Instance_name == sample].iat[0,1]
                level_sample = level_train[level_train.Instance_name == sample].iat[0,1]
            else:
                ymrs_sample = ymrs_dev[ymrs_dev.Instance_name == sample].iat[0,1]
                level_sample = level_dev[level_dev.Instance_name == sample].iat[0,1]
            
            if verbose:
                print("YMRS score for %s is %d" %(sample, ymrs_sample))
                print("Mania level for %s is %d" % (sample, level_sample))

            feat = load_baseline_feature(feature_name, partition, id)
            no_frame, _ = feat.shape
            count_nan = 0

            for i in range(0, no_frame, keep):
                if verbose:
                    print("\n----processing no. %d frame----" % i)
                data = feat.iloc[i,:]
                data = data[1:] # remove name
                if data.isnull().values.any():
                    print("----NAN, DROP FEATURE----")
                    count_nan += 1
                    continue

                data_str = data.to_string(header=False, index=False)
                data_str = data_str.replace("\n", ",").replace(" ", "")

                # write baseline features to external file    
                dataf.write(data_str)
                dataf.write("\n")
                # write baseline labels and instance to external file
                if id == 1 and i == 0:
                    labelf.write("%d" % level_sample)
                    instf.write("%d" % id)
                else:
                    labelf.write(",%d" % level_sample)
                    instf.write(",%d" % id)

            if verbose:
                print("\n----next feature----")
        if verbose:
            print("\n----%s partition done----" % partition)
            print("\n----ALL NAN DROPPED %d----" % count_nan)
        
        # close file handles
        dataf.close()
        labelf.close()
        instf.close()


def preproc_transcript(partition):
    """preprocess transcript files to one document
    """
    # para partition: which partition, train/dev/test/ALL
    if partition == 'all':
        len_train = data_config['length']['train']
        len_dev = data_config['length']['dev']
        len_test = data_config['length']['test']
        docs = data_config['transcript_preproc'][partition]

        # open the output IO wrapper
        with smart_open(docs, 'w', encoding='utf-8') as output:
            # loop the train transcripts
            for i in range(len_train):
                filename = get_sample('train', (i+1)) + '.txt'
                # open the input IO wrapper
                with smart_open(os.path.join(data_config['transcript'], filename), 'r', encoding='utf-8') as input:
                    # write the only line 
                    for line in input:
                        output.write(line)
                input.close()
            # loop the dev transcripts
            for j in range(len_dev):
                filename = get_sample('dev', (j+1)) + '.txt'
                # open the input IO wrapper
                with smart_open(os.path.join(data_config['transcript'], filename), 'r', encoding='utf-8') as input:
                    # write the only line 
                    for line in input:
                        output.write(line)
                input.close()
            # loop the test transcripts
            for k in range(len_test):
                filename = get_sample('test', (k+1)) + '.txt'
                # open the input IO wrapper
                with smart_open(os.path.join(data_config['transcript'], filename), 'r', encoding='utf-8') as input:
                    # write the only line 
                    for line in input:
                        output.write(line)
                input.close()
        output.close()

    else:
        length = data_config['length'][partition]
        docs = data_config['transcript_preproc'][partition]
        
        # open the output IO wrapper
        with smart_open(docs, 'w', encoding='utf-8') as output:
            # loop the transcripts
            for i in range(length):
                filename = get_sample(partition, (i+1)) + '.txt'
                print("%s read" % filename)
                # open the input IO wrapper
                with smart_open(os.path.join(data_config['transcript'], filename), 'r', encoding='utf-8') as input:
                    # write the only line 
                    for line in input:
                        output.write(line)
                input.close()
        output.close()


def tokenize_tr(content, token_min_len=2, token_max_len=50, lower=True):
    """tokenize words in the corpus
    """
    if lower:
        lower_map = {ord(u'A'): u'a', 
        ord(u'A'): u'a', ord(u'B'): u'b',
        ord(u'C'): u'c', ord(u'Ç'): u'ç',
        ord(u'D'): u'd', ord(u'E'): u'e',
        ord(u'F'): u'f', ord(u'G'): u'g',
        ord(u'Ğ'): u'ğ', ord(u'H'): u'h',
        ord(u'I'): u'ı', ord(u'İ'): u'i',
        ord(u'J'): u'j', ord(u'K'): u'k',
        ord(u'L'): u'l', ord(u'M'): u'm', 
        ord(u'N'): u'n', ord(u'O'): u'o',
        ord(u'Ö'): u'ö', ord(u'P'): u'p',
        ord(u'R'): u'r', ord(u'S'): u's',
        ord(u'Ş'): u'ş', ord(u'T'): u't',
        ord(u'U'): u'u', ord(u'Ü'): u'ü',
        ord(u'V'): u'v', ord(u'Y'): u'y',
        ord(u'Z'): u'z'}
        content = content.translate(lower_map)
    
    return [utils.to_unicode(token) for token in utils.tokenize(content, lower=False, errors='ignore') if token_min_len <= len(token) <= token_max_len and not token.startswith('_')]


def process_corpus(verbose=False):
    """preprocess Turkish wikimedia cospus to line-based text file
    """
    input_file = data_config['turkish_corpus']
    output_file = data_config['turkish_corpus_proc']
    if os.path.isfile(output_file):
        print("processed file already exist %s" % output_file)
        os.remove(output_file)
    print("\nraw Turkish corpus loaded")

    wiki = WikiCorpus(input_file, lemmatize=False, tokenizer_func=tokenize_tr)
    output = smart_open(output_file, 'w', encoding='utf-8')

    i = 0
    # write to processed file
    for text in wiki.get_texts():
        output.write(" ".join(text)+"\n")
        i += 1
        if (i % 100 == 0) and verbose:
            print("no. %d \tarticle saved." % i)
    output.close()

