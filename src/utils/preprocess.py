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


def preprocess_align(eGeMAPS=False, verbose=False):
    """preprocess MFCC / eGeMAPS data to align Audio / Video data
    """
    input_dir_A = data_config['data_path_local']['baseline']['MFCC'] if not eGeMAPS else data_config['data_path_700']['LLD']['eGeMAPS']
    input_dir_V = data_config['baseline_preproc']['AU_landmarks']
    output_dir_A = data_config['baseline_preproc']['MFCC_aligned'] if not eGeMAPS else data_config['baseline_preproc']['eGeMAPS_aligned']

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


def upsample(X_train, y_train, train_inst, regression=False, verbose=False):
    """upsample dataset to balance different classes
    """
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
                if train_inst.any() and not regression:
                    X_train = pd.concat((X_train, X_train.iloc[diff_index, :]), axis=0)
                    y_train = np.hstack((y_train, y_train[diff_index]))
                    train_inst = np.hstack((train_inst, train_inst[diff_index]))
                elif regression:
                    X_train = np.vstack((X_train, X_train[diff_index, :]))
                    y_train = np.hstack((y_train, y_train[diff_index]))
                    train_inst = np.hstack((train_inst, train_inst[diff_index]))
                else:
                    X_train = np.vstack((X_train, X_train[diff_index, :]))
                    y_train = np.hstack((y_train, y_train[diff_index]))
                diff -= stats[c]
            if verbose:
                print("X train shape", X_train.shape)
                print("y train shape", y_train.shape)
    
    return X_train, y_train, train_inst


def get_dynamics(X_0th, time=0.1):
    """compute dynamics for data (1st/2nd derivate)
    """
    X_1st = np.zeros((X_0th.shape[0]-1, X_0th.shape[1]))
    X_2nd = np.zeros((X_0th.shape[0]-2, X_0th.shape[1]))
    for i in range(X_0th.shape[0]-1):
        X_1st[i] = (X_0th[i+1] - X_0th[i]) / time
    for j in range(X_0th.shape[0]-2):
        X_2nd[j] = (X_1st[j+1] - X_1st[j]) / time
    return np.hstack((X_0th[2:], X_1st[1:], X_2nd))


def frame2session(X, y, inst, verbose=False):
    """transfer frame-level data/label/inst to session-level data/label
    """
    # para X: data
    # para y: label
    # para inst: instance
    print(X.shape, y.shape, inst.shape)
    assert X.shape[0] == y.shape[0] == inst.shape[0]
    if y.shape[1] == 1:
        y = y[:,0]
    if inst.shape[1] == 1:
        inst = inst[:,0]
    
    max_inst = int(max(inst))
    min_inst = int(min(inst))
    X_sess, y_sess = [], []
    for i in range(min_inst, max_inst+1):
        idx = np.where(inst == i)[0]
        X_temp = X[idx]
        y_temp = y[idx]
        X_sess.append(X_temp)
        if len(set(y_temp)) == 1:
            y_sess.append(y_temp[0])
        if verbose:
            print("instance %d data shape" % i, X_temp.shape)
    assert max_inst == len(X_sess) == len(y_sess)
    return np.array(X_sess), np.array(y_sess)


def k_fold_cv(length):
    length_cv = length // 10
    all_ids = list(range(length))
    cv_ids = []
    for i in range(10):
        ids_train = all_ids[:length_cv*i] + all_ids[length_cv*(i+1):]
        ids_dev = all_ids[length_cv*i:length_cv*(i+1)]
        cv_ids.append((ids_train, ids_dev))
    return cv_ids


def preprocess_metadata_tensorboard(path, n_kernel):
    X_train = np.load(os.path.join(path, 'X_train_tree_%d.npy' % n_kernel))
    y_train = np.load(os.path.join(path, 'label_train.npy'))
    X_dev = np.load(os.path.join(path, 'X_dev_tree_%d.npy' % n_kernel))
    y_dev = np.load(os.path.join(path, 'label_dev.npy'))

    if X_train.ndim == X_dev.ndim == 3:
        X_train = np.reshape(X_train, (-1, np.prod(X_train.shape[1:])))
        X_dev = np.reshape(X_dev, (-1, np.prod(X_dev.shape[1:])))
    X_train = np.nan_to_num(X_train)
    X_dev = np.nan_to_num(X_dev)

    print("\nsaving FV to metadata file for tensorboard projector visualization")
    with smart_open(os.path.join(path, 'label.tsv'), 'wb', encoding='utf-8') as label_f:
        label_f.write("Index\tLabel\n")
        for i in range(len(y_train)):
            label_f.write("train_%d\t%d\n" % (i+1, y_train[i]))
        for j in range(len(y_dev)):
            label_f.write("dev_%d\t%d\n" % (j+1, y_dev[j]))
    
    with smart_open(os.path.join(path, 'metadata.tsv'), 'wb', encoding='utf-8') as data_f:
        for a in range(len(X_train)):
            for b in range(len(X_train[a])):
                data_f.write("%f\t" % X_train[a][b])
            data_f.write("\n")
        for c in range(len(X_dev)):
            for d in range(len(X_dev[c])):
                data_f.write("%f\t" % X_dev[c][d])
            data_f.write("\n")
    print("\nmetadata processing done\nplease upload the .tsv onto projector.tensorflow.org for visualization")


def preprocess_reconstruction():
    model_path_AV = smart_open('./pre-trained/DDAE/model_list.txt', 'rb', encoding='utf-8')
    model_list_AV = []

    for _, line_AV in enumerate(model_path_AV):
        line_AV = str(line_AV).replace('\n', '')
        model_list_AV.append(line_AV[:-2])

    from src.utils.io import load_aligned_features

    model_path = model_list_AV[6]
    
    landmarks_recon = np.load(os.path.join(model_path, 'decoded_train_1.npy'))
    pose_recon = np.load(os.path.join(model_path, 'decoded_train_3.npy'))
    print(landmarks_recon.shape)
    print(pose_recon.shape)
    y_train, inst_train, _, _ = load_aligned_features(no_data=True, verbose=True)
    assert len(landmarks_recon) == len(pose_recon) == len(y_train) == len(inst_train)
    index, _ = np.where(inst_train == inst_train[0])
    
    landmarks_recon_train01 = landmarks_recon[index]
    pose_recon_train01 = pose_recon[index]
    print(landmarks_recon_train01.shape)
    print(pose_recon_train01.shape)

    np.save(os.path.join(model_path, 'recon_1_train01.npy'), landmarks_recon_train01)
    np.save(os.path.join(model_path, 'recon_3_train01.npy'), pose_recon_train01)

