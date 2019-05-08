import os
import json
import traceback
import statistics
import numpy as np
import pandas as pd
from scipy.io import arff
from smart_open import smart_open


'''
ALL I\O FUNCTIONS
----------------------------------------
get_sample(partition, index)
    retrieve the sample name
load_audio_file(parition, index, gcs=False, verbose=False)
    load audio files for transcribing
save_transcript(partition, index, transcript)
    save transcript to external files
load_label(partition=True, verbose=False)
    load the labels (age, gender, YMRS)
load_LLD(LLD_name, partition, index, verbose=False)
    load LLDs with given feature name, partition, index
load_baseline_feature(feature_name, partition, index, verbose=False)
    load the baseline features with given partition and index
load_proc_baseline_feature(feature_name, matlab=True, verbose=False)
    load the features pre-processed by MATLAB or Python
preproc_baseline_feature(feature_name, verbose=False)
    pre-process the baseline features (LLDs)
save_UAR_results(frame_res, session_res, name, modality)
    save classification results to external files
save_post_probability(prob_dev, model_name, feature_name)
    save posteriors probabilities to external files
load_post_probability(model_name, feature_name)
    load posteriors probabilities from external files
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))


def get_sample(partition, index):
    """retrieve the sample name
    """
    # para partition: which partition, train/dev/test
    # para index: the index of sample
    if index < 0:
        print("\nINCORRECT INDEX INPUT")
        return 
    sample_name = ''
    if partition == 'train':
        if index > 104:
            print("\nSAMPLE NOT EXIST")
        else:
            sample_name = 'train_' + str(index).zfill(3)
    elif partition == 'dev':
        if index > 60:
            print("\nSAMPLE NOT EXIST")
        else:
            sample_name = 'dev_' + str(index).zfill(3)
    elif partition == 'test':
        if index > 54:
            print("\nSAMPLE NOT EXIST")
        else:
            sample_name = 'test_' + str(index).zfill(3)
    else:
        print("\nINCORRECT PARTITION INPUT")
    return sample_name


def load_audio_file(partition, index, gcs=False, verbose=False):
    """load audio files for transcribing
    """
    # para partition: which partition, train/dev/test
    # para index: the index of sample
    # para verbose: whether or not to output more results
    # return: array of audio filenames 
    audio_dir = data_config['data_path_700']['audio'] if not gcs else data_config['data_path_700']['audio_gcs']
    audio_list = []

    if not partition and not index:
        len_train = data_config['length_train']
        len_dev = data_config['length_dev']
        len_test = data_config['length_test']

        for i in range(len_train):
            filename = get_sample('train', (i+1)) + '.wav'
            audio_list.append(os.path.join(audio_dir, filename))
            if verbose:
                print("load audio file:", audio_list[-1])

        for j in range(len_dev):
            filename = get_sample('dev', (j+1)) + '.wav'
            audio_list.append(os.path.join(audio_dir, filename))
            if verbose:
                print("load audio file:", audio_list[-1])

        for k in range(len_test):
            filename = get_sample('test', (k+1)) + '.wav'
            audio_list.append(os.path.join(audio_dir, filename))
            if verbose:
                print("load audio file:", audio_list[-1])

    elif partition and index:
        filename = get_sample(partition, index) + '.wav'
        audio_list.append(os.path.join(audio_dir, filename))
        if verbose:
            print("load audio file:", audio_list[-1])
    
    return audio_list


def save_transcript(partition, index, transcript):
    """save transcript to external files
    """
    # para partition: which partition, train/dev/test
    # para index: the index of sample
    # para transcript: transcript to save
    save_dir = data_config['transcript']
    filename = get_sample(partition, index) + '.txt'
    with smart_open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as output:
        output.write(transcript)
        output.write("\n")
    output.close()


def load_label(partition=True, verbose=False):
    """load the labels (age, gender, YMRS)
    """
    # para partition: whether to partition labels into train/dev sets
    # para verbose: whether or not to output more statistical results
    # return: YMRS score and Mania level for train/dev set
    # return: YMRS score and Mania level for all dataset (if not partition)
    label = pd.read_csv(data_config['data_path_local']['label_metadata'] + 'labels_metadata.csv')
    id_list = label['SubjectID'].tolist()

    id_set = set()
    age_list = list()
    for id in id_list:
        id_set.add(id)
        age_list.extend(label[label.SubjectID == id]['Age'].tolist())

    gender_list = list()
    for sub in id_set:
        gender_list.append(sub[:1])
        if verbose:
            print("%s subject have %d instances" % (sub, id_list.count(sub)))

    if verbose:
        print("All subjects", len(id_set))
        print("Male subjects ", gender_list.count('M'))
        print("Female subjects", gender_list.count('F'))
        print("Age range (%d, %d), Age median %d" % (min(age_list), max(age_list), statistics.median(age_list)))

    ymrs_score = pd.concat([label.iloc[:, 0], label.iloc[:, 4]], axis=1)
    mania_level = pd.concat([label.iloc[:, 0], label.iloc[:, 5]], axis=1)
    if partition:
        ymrs_dev = ymrs_score.iloc[:60, :]
        ymrs_train = ymrs_score.iloc[60:, :]
        level_dev = mania_level.iloc[:60, :]
        level_train = mania_level.iloc[60:, :]
        return ymrs_train, ymrs_dev, level_dev, level_train
    else:
        return ymrs_score, mania_level, 0, 0


def load_LLD(LLD_name, partition, index, verbose=False):
    """load LLDs with given feature name, partition, index
    """
    # para LLD_name: which LLDs, MFCC or eGeMAPS or openFace
    # para partition: which partition, train/dev/test
    # para index: the index of sample
    # para verbose: whether or not to output more results
    if get_sample(partition, index):
        sample = get_sample(partition, index) + '.csv'

    else:
        print("\nWRONG INPUT - PARTITION or INDEX\n")
        return

    if LLD_name == 'MFCC':
        mfcc = pd.read_csv(data_config['data_path_local']['LLD']['MFCC'] + sample, sep=';')
        if verbose:
            print(mfcc.shape)
        return mfcc

    elif LLD_name == 'eGeMAPS':
        egemaps = pd.read_csv(data_config['data_path_local']['LLD']['eGeMAPS'] + sample, sep=';')
        if verbose:
            print(egemaps.shape)
        return egemaps

    elif LLD_name == 'openFace':
        face = pd.read_csv(data_config['data_path_local']['LLD']['openFace'] + sample)
        if verbose:
            print(face.shape)
        return face

    else:
        print("\nWRONG INPUT - LLD NAME\n")
        return


def load_baseline_feature(feature_name, partition, index, verbose=False):
    """load the baseline features with given partition and index
    """
    # para feature_name: which feature, BoAW or eGeMAPS or BoVW
    # para partition: which partition, train/dev/test
    # para index: the index of sample
    # para verbose: whether or not to output more results
    if get_sample(partition, index):
        sample = get_sample(partition, index) + '.csv'

    else:
        print("\nWRONG INPUT - PARTITION or INDEX\n")
        return

    if feature_name == 'BoAW':
        sample = '2_' + sample
        boaw = pd.read_csv(data_config['data_path_local']['baseline']['audio']['BoAW'] + sample, sep=';')
        if verbose:
            print(boaw.shape)
        return boaw

    elif feature_name == 'eGeMAPS':
        sample = sample[:-3] + 'arff'
        egemaps = arff.loadarff(data_config['data_path_local']['baseline']['audio']['eGeMAPS'] + sample)
        egemaps_df = pd.DataFrame(egemaps[0])
        if verbose:
            print(egemaps_df.shape)
        return egemaps_df

    elif feature_name == 'MFCC':
        mfcc = pd.read_csv(data_config['data_path_local']['baseline']['audio']['MFCC'] + sample, sep=';')
        if verbose:
            print(mfcc.shape)
        return mfcc

    elif feature_name == 'Deep':
        deep = pd.read_csv(data_config['data_path_local']['baseline']['audio']['DeepSpectrum'] + sample, sep=';')
        if verbose:
            print(deep.shape)
        return deep

    elif feature_name == 'BoVW':
        sample = '11_' + sample
        bovw = pd.read_csv(data_config['data_path_local']['baseline']['video']['BoVW'] + sample, sep=';')
        if verbose:
            print(bovw.shape)
        return bovw

    elif feature_name == 'AU':
        au = pd.read_csv(data_config['data_path_local']['baseline']['video']['AU'] + sample, sep=';')
        if verbose:
            print(au.shape)
        return au
        
    else:
        print("\nWRONG INPUT - LLD NAME\n")
        return


def load_proc_baseline_feature(feature_name, matlab=True, verbose=False):
    """load the features pre-processed by MATLAB or Python
    """
    # para feature_name: which feature, BoAW or eGeMAPS or BoVW
    # para matlab: whether or not to use MATLAB processed features
    # para verbose: whether or not to output more results
    baseline = 'baseline_MATLAB' if matlab else 'baseline_preproc'

    try:
        if feature_name != 'AU':
            train_inst = pd.read_csv(data_config[baseline][feature_name]['train_inst'], header=None)
            dev_inst = pd.read_csv(data_config[baseline][feature_name]['dev_inst'], header=None)
        else:
            train_inst, dev_inst = None, None
        
        train_data = pd.read_csv(data_config[baseline][feature_name]['train_data'], header=None)
        train_label = pd.read_csv(data_config[baseline][feature_name]['train_label'], header=None)
        dev_data = pd.read_csv(data_config[baseline][feature_name]['dev_data'], header=None)
        dev_label = pd.read_csv(data_config[baseline][feature_name]['dev_label'], header=None)

        if verbose:
            print("--"*20)
            print(feature_name)
            print("--"*20)
            print("Size of training data (extracted from MATLAB)", train_data.shape)
            print("Size of training labels (extracted from MATLAB)", train_label.shape)
            print("Size of dev data (extracted from MATLAB)", dev_data.shape)
            print("Size of dev labels (extracted from MATLAB)", dev_label.shape)

            if feature_name != 'AU':
                print("Size of training instance (extracted from MATLAB)", train_inst.shape)
                print("Size of dev instance (extracted from MATLAB)", dev_inst.shape)
            
            print("--"*20)

    except:
        raise Exception("\nFAILED LOADING PRE-PROCESSED FEATURES")

    return train_data, np.ravel(train_label.T.values), np.ravel(train_inst), dev_data, np.ravel(dev_label.T.values), np.ravel(dev_inst)


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


def save_UAR_results(frame_results, session_results, model_name, feature_name, modality, cv=False):
    """save UAR results to external files
    """
    # para frame_res: classification UAR for frame-level
    # para session_res: classification UAR for session-level
    # para model_name: which model is used
    # para feature_name: which feature is used
    # para modality: either single or multiple
    frame_res = frame_results if not cv else np.mean(frame_results)
    session_res = session_results if not cv else np.mean(session_results)

    if modality == 'single':
        filename = os.path.join(data_config['result_single'], '%s_%s_result.txt' % (model_name, feature_name)) if not cv else os.path.join(data_config['result_single'], 'cv_%s_%s_result.txt' % (model_name, feature_name))

        with smart_open(filename, 'w', encoding='utf-8') as f:
            f.write("UAR on frame-level: %.3f \n" % frame_res)
            f.write("UAR on session-level: %.3f \n" % session_res)
        f.close()
        
    elif modality == 'multiple':
        filename = os.path.join(data_config['result_multi'], '%s_%s_result.txt' % (model_name, feature_name)) if not cv else os.path.join(data_config['result_multi'], 'cv_%s_%s_result.txt' % (model_name, feature_name))

        with smart_open(filename, 'w', encoding='utf-8') as f:
            f.write("UAR on frame-level: %.3f \n" % frame_res)
            f.write("UAR on session-level: %.3f \n" % session_res)
        f.close()

    elif modality == 'baseline':
        filename = os.path.join(data_config['result_baseline'], '%s_%s_result.txt' % (model_name, feature_name)) if not cv else os.path.join(data_config['result_baseline'], 'cv_%s_%s_result.txt' % (model_name, feature_name))

        with smart_open(filename, 'w', encoding='utf-8') as f:
            f.write("UAR on frame-level: %.3f \n" % frame_res)
            f.write("UAR on session-level: %.3f \n" % session_res)
        f.close()
    
    else:
        print("\n-- INVALID INPUT --\n")
        return


def save_post_probability(prob_dev, model_name, feature_name):
    """save posteriors probabilities to external files
    """
    # para prob_dev: posteriors probabilities of development set
    # para model: which model is used
    # para name: which feature is used
    filename = os.path.join(data_config['result_baseline'], '%s_%s_post_prob' % (model_name, feature_name))
    np.save(filename, prob_dev)


def load_post_probability(model_name, feature_name):
    """load posteriors probabilities from external files
    """
    # para model: which model is used
    # para name: which feature is used
    filename = os.path.join(data_config['result_baseline'], '%s_%s_post_prob.npy' % (model_name, feature_name))
    prob_dev = np.load(filename)
    return prob_dev
