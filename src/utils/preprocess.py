import os
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as etree
from collections import Counter
from smart_open import smart_open
from gensim.corpora import WikiCorpus
from gensim import utils

from src.utils.io import get_sample, load_label
from src.utils.io import load_baseline_feature


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
preprocess_AU(verbose=False)
    preprocess Action Units data
preprocess_BOXW(verbose=False)
    preprocess Bags of X Words representations
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


def preprocess_AU(verbose=False):
    """preprocess Action Units data
    """
    raw_dir = data_config['data_path_700']['LLD']['openFace']
    proc_dir = data_config['baseline_preproc']['AU_landmarks']

    length = dict()
    length['train'] = data_config['length']['train']
    length['dev'] = data_config['length']['dev']
    length['test'] = data_config['length']['test']

    time = ['timestamp']
    landmarks = ['%s_%d' % (xy, i) for xy in ['x', 'y'] for i in range(68)]
    gazes = ['gaze_%d_%s' % (no, di) for no in range(2) for di in ['x', 'y', 'z']]
    poses = ['pose_%s' % xyz for xyz in ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']]
    actions = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

    visual = time
    visual.extend(landmarks)
    visual.extend(gazes)
    visual.extend(poses)
    visual.extend(actions)

    if verbose:
        print(time)
        print(landmarks)
        print(gazes)
        print(poses)
        print(actions)

    for partition in ['train', 'dev', 'test']:
        for i in range(length[partition]):
            filename = get_sample(partition, (i+1))
            temp = pd.read_csv(os.path.join(raw_dir, filename + '.csv'))
            temp.columns = temp.columns.str.strip()
            print("file %s loaded" % filename)
            # select specified columns
            temp = temp.loc[:, visual]
            temp.to_csv(os.path.join(proc_dir, filename + '.csv'), index=False)
            print("file %s processing completed & saved" % filename)


def preprocess_BOXW(verbose=False):
    """preprocess Bags of X Words representations
    """
    # load directory from configuration file
    A_input_dir = data_config['data_path_local']['baseline']['BoAW']
    V_input_dir = data_config['data_path_local']['baseline']['BoVW']
    A_output_dir = data_config['baseline_preproc']['BoAW']
    V_output_dir = data_config['baseline_preproc']['BoVW']
    # load length from configuration file
    length = dict()
    length['train'] = data_config['length']['train']
    length['dev'] = data_config['length']['dev']
    length['test'] = data_config['length']['test']
    # load labels from configuration file
    _, _, level_dev, level_train = load_label()
    label_train, label_dev = level_train.values, level_dev.values
    labels = dict()
    labels['train'] = label_train[:, 1]
    labels['dev'] = label_dev[:, 1]

    for partition in ['train', 'dev']:
        # write handle
        A_label_f = smart_open(A_output_dir['%s_label' % partition], 'a+', encoding='utf-8')
        V_label_f = smart_open(V_output_dir['%s_label' % partition], 'a+', encoding='utf-8')
        A_inst_f = smart_open(A_output_dir['%s_inst' % partition], 'a+', encoding='utf-8')
        V_inst_f = smart_open(V_output_dir['%s_inst' % partition], 'a+', encoding='utf-8')
        A_data, V_data = None, None
        label = labels[partition]

        for i in range(length[partition]):

            A_feature = load_baseline_feature('BoAW', partition, (i+1))
            V_feature = load_baseline_feature('BoVW', partition, (i+1))
            A_t, _ = A_feature.shape
            V_t, _ = V_feature.shape
            # ensure timesteps match between Audio and Video
            timestep = A_t if A_t < V_t else V_t
            A_feature = A_feature.iloc[:timestep, 2:]
            V_feature = V_feature.iloc[:timestep, 2:]
            # concatenate features
            A_data = A_feature.copy() if not i else pd.concat([A_data, A_feature])
            V_data = V_feature.copy() if not i else pd.concat([V_data, V_feature])
            # write labels and instances
            A_label_f.write(('%d,' % label[i]) * timestep)
            V_label_f.write(('%d,' % label[i]) * timestep)
            A_inst_f.write(('%d,' % (i+1)) * timestep)
            V_inst_f.write(('%d,' % (i+1)) * timestep)

            if verbose:
                print(A_feature.shape, V_feature.shape)
                print(A_data.shape, V_data.shape)
        
        # save to external files
        A_data.to_csv(A_output_dir['%s_data' % partition], header=None, index=None)
        V_data.to_csv(V_output_dir['%s_data' % partition], header=None, index=None)

        A_label_f.close()
        V_label_f.close()
        A_inst_f.close()
        V_inst_f.close()


def preprocess_align(verbose=False):
    """preprocess MFCC data to align Audio / Video data
    """
    input_dir_A = data_config['data_path_local']['baseline']['MFCC']
    input_dir_V = data_config['baseline_preproc']['AU_landmarks']
    output_dir_A = data_config['baseline_preproc']['MFCC_aligned']

    length = dict()
    length['train'] = data_config['length']['train']
    length['dev'] = data_config['length']['dev']
    length['test'] = data_config['length']['test']

    for partition in ['train', 'dev', 'test']:
        for i in range(length[partition]):
            filename = get_sample(partition, (i+1)) + '.csv'

            if verbose:
                print("file %s loaded." % filename)
            
            temp_A = pd.read_csv(os.path.join(input_dir_A, filename), sep=';', index_col=1)
            temp_A.drop("name", axis=1, inplace=True)
            del temp_A.index.name
            temp_V = pd.read_csv(os.path.join(input_dir_V, filename))
            align_A = pd.DataFrame(np.zeros((len(temp_V), temp_A.shape[1]*3)))
            align_A.index = temp_V.loc[:, 'timestamp']
            del align_A.index.name

            A_list = temp_A.index.tolist()
            V_list = temp_V.loc[:, 'timestamp'].tolist()
            
            for j in range(len(V_list) - 1):
                a_list = []
                for a in A_list:
                    if a > V_list[j] and a < V_list[j+1]:
                        a_list.append(a)
                if len(a_list) == 1:
                    a_list *= 3
                elif len(a_list) == 2:
                    a_list.append(a_list[1])
                elif len(a_list) == 3:
                    a_list = a_list
                else:
                    continue
                
                assert len(a_list) == 3

                align_A.loc[V_list[j], :] = pd.concat([
                    temp_A.loc[a_list[0]], 
                    temp_A.loc[a_list[1]],
                    temp_A.loc[a_list[2]]], 
                    axis=0, sort=False, ignore_index=True)
            
            align_A.to_csv(os.path.join(output_dir_A, filename))

            if verbose:
                print("file %s processed & saved." % filename)


def upsample(X_train, y_train, train_inst, verbose=False):
    # para X_train: pd.DataFrame
    # para y_train: np.ndarray
    # para train_inst: np.ndarray
    stats = Counter(y_train)
    uplimit = max([stats[1], stats[2], stats[3]])
    most_class = [stats[1], stats[2], stats[3]].index(uplimit) + 1

    if verbose:
        print("labels\t", stats)
        print("most class\t%d\nmax number\t%d" % (most_class, uplimit))
    
    for c in [1, 2, 3]:
        if c == most_class:
            continue
        else:
            diff = uplimit - stats[c]
            c_index = np.where(y_train == c)[0]
            np.random.shuffle(c_index)
            while diff > 1:
                diff_index = c_index[:diff]
                X_train = pd.concat((X_train, X_train.iloc[diff_index, :]), axis=0)
                y_train = np.hstack((y_train, y_train[diff_index]))
                if train_inst.any():
                    train_inst = np.hstack((train_inst, train_inst[diff_index]))
                diff -= stats[c]
    
    return X_train, y_train, train_inst

