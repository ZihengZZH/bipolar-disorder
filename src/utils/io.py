import os
import json
import traceback
import statistics
import numpy as np
import pandas as pd
from scipy.io import arff
from collections import Counter
from smart_open import smart_open


'''
ALL I\O FUNCTIONS
----------------------------------------
get_sample(partition, index)
    retrieve the sample name
load_audio_file(partition, index, gcs=False, verbose=False)
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
save_UAR_results(frame_res, session_res, name, modality)
    save classification results to external files
save_post_probability(prob_dev, model_name, feature_name)
    save posteriors probabilities to external files
load_post_probability(model_name, feature_name)
    load posteriors probabilities from external files
load_facial_landmarks(verbose=False)
    load preprocessed facial landmarks 
load_bags_of_words(verbose=False)
    load preprocessed BoXW (BoAW or BoVW)
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
        len_train = data_config['length']['train']
        len_dev = data_config['length']['dev']
        len_test = data_config['length']['test']

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
    
    classes_stats = Counter(label['ManiaLevel'].tolist())

    if verbose:
        print("All subjects", len(id_set))
        print("Male subjects ", gender_list.count('M'))
        print("Female subjects", gender_list.count('F'))
        print("Age range (%d, %d), Age median %d" % (min(age_list), max(age_list), statistics.median(age_list)))
        print("Class distribution stats", classes_stats)

    ymrs_score = pd.concat([label.iloc[:, 0], label.iloc[:, 4]], axis=1)
    mania_level = pd.concat([label.iloc[:, 0], label.iloc[:, 5]], axis=1)
    if partition:
        ymrs_dev = ymrs_score.iloc[:60, :]
        ymrs_train = ymrs_score.iloc[60:, :]
        level_dev = mania_level.iloc[:60, :]
        level_train = mania_level.iloc[60:, :]
        return ymrs_dev, ymrs_train, level_dev, level_train
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

    if feature_name == 'MFCC':
        sample = sample
    elif feature_name == 'eGeMAPS':
        sample = sample[:-3] + 'arff'
        feature_arff = arff.loadarff(data_config['data_path_local']['baseline'][feature_name] + sample)
        feature = pd.DataFrame(feature_arff[0])
    elif feature_name == 'Deep':
        sample = sample
    elif feature_name == 'BoAW':
        sample = '2_' + sample
    elif feature_name == 'AU':
        sample = sample
    elif feature_name == 'BoVW':
        sample = '11_' + sample
    else:
        print("\nWRONG INPUT - LLD NAME\n")
        return
    
    feature = pd.read_csv(data_config['data_path_local']['baseline'][feature_name] + sample, sep=';', header=None)
    if verbose:
        print("--" * 20)
        print("Feature %s" % feature_name)
        print(feature.shape)
        print("--" * 20)
    return feature


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
            print("Shape of training data (extracted from MATLAB)", train_data.shape)
            print("Shape of training labels (extracted from MATLAB)", train_label.T.shape)
            print("Shape of dev data (extracted from MATLAB)", dev_data.shape)
            print("Shape of dev labels (extracted from MATLAB)", dev_label.T.shape)

            if feature_name != 'AU':
                print("Shape of training instance (extracted from MATLAB)", train_inst.T.shape)
                print("Shape of dev instance (extracted from MATLAB)", dev_inst.T.shape)
            
            print("--"*20)

    except:
        raise Exception("\nFAILED LOADING PRE-PROCESSED FEATURES")

    return train_data, np.ravel(train_label.T.values), np.ravel(train_inst), dev_data, np.ravel(dev_label.T.values), np.ravel(dev_inst)


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


def load_aligned_features(no_data=False, verbose=False):
    """load preprocessed visual and acoustic features 
    """
    visual_dir = data_config['baseline_preproc']['AU_landmarks']
    acoustic_dir = data_config['baseline_preproc']['MFCC_aligned']
    output_dir = data_config['baseline_preproc']['aligned_AV']

    if no_data:
        print("\nprocessed files exist, starting loading (w/o raw data) ...")
        y_train = pd.read_csv(output_dir['train_label'], header=None) 
        inst_train = pd.read_csv(output_dir['train_inst'], header=None) 
        y_dev = pd.read_csv(output_dir['dev_label'], header=None) 
        inst_dev = pd.read_csv(output_dir['dev_inst'], header=None)

        if verbose:
            print("--" * 20)
            print("train label size", y_train.T.shape)
            print("dev label size", y_dev.T.shape)
            print("train inst size", inst_train.T.shape)
            print("dev inst size", inst_dev.T.shape)
            print("--" * 20)
        
        return y_train.T.values, inst_train.T.values, y_dev.T.values, inst_dev.T.values

    elif os.path.isfile(output_dir['test_data_A']) and os.path.isfile(output_dir['test_data_V']):
        print("\nprocessed files exist, starting loading ...")
        X_train_A = pd.read_csv(output_dir['train_data_A'], header=None) 
        X_dev_A = pd.read_csv(output_dir['dev_data_A'], header=None) 
        X_test_A = pd.read_csv(output_dir['test_data_A'], header=None)
        X_train_V = pd.read_csv(output_dir['train_data_V'], header=None, low_memory=False)
        X_dev_V = pd.read_csv(output_dir['dev_data_V'], header=None, low_memory=False) 
        X_test_V = pd.read_csv(output_dir['test_data_V'], header=None, low_memory=False)
        y_train = pd.read_csv(output_dir['train_label'], header=None) 
        inst_train = pd.read_csv(output_dir['train_inst'], header=None) 
        y_dev = pd.read_csv(output_dir['dev_label'], header=None) 
        inst_dev = pd.read_csv(output_dir['dev_inst'], header=None)

        if verbose:
            print("--" * 20)
            print("train data (A) size", X_train_A.shape)
            print("train data (V) size", X_train_V.shape)
            print("dev data (A) size", X_dev_A.shape)
            print("dev data (V) size", X_dev_V.shape)
            print("test data (A) size", X_test_A.shape)
            print("test data (V) size", X_test_V.shape)
            print("--" * 20)
            print("train label size", y_train.T.shape)
            print("dev label size", y_dev.T.shape)
            print("train inst size", inst_train.T.shape)
            print("dev inst size", inst_dev.T.shape)
            print("--" * 20)

        return X_train_A.iloc[:,1:], X_dev_A.iloc[:,1:], X_test_A.iloc[:,1:], X_train_V.iloc[:,1:], X_dev_V.iloc[:,1:], X_test_V.iloc[:,1:], y_train.T.values, inst_train.T.values, y_dev.T.values, inst_dev.T.values
    
    else:
        length = dict()
        length['train'] = data_config['length']['train']
        length['dev'] = data_config['length']['dev']
        length['test'] = data_config['length']['test']

        _, _, level_dev, level_train = load_label()
        label_train, label_dev = level_train.values, level_dev.values
        labels = dict()
        labels['train'] = label_train[:, 1]
        labels['dev'] = label_dev[:, 1]

        dimensionality = dict()
        dimensionality['train'] = 0
        dimensionality['dev'] = 0

        for partition in ['train', 'dev']:
            label_f = smart_open(output_dir['%s_label' % partition], 'w', encoding='utf-8')
            inst_f = smart_open(output_dir['%s_inst' % partition], 'w', encoding='utf-8')
            A_data, V_data = None, None
            label = labels[partition]

            for i in range(length[partition]):
                filename = get_sample(partition, (i+1)) + '.csv'
                A_feature = pd.read_csv(os.path.join(acoustic_dir, filename), low_memory=False)
                V_feature = pd.read_csv(os.path.join(visual_dir, filename), low_memory=False)
                A_t, _ = A_feature.shape
                V_t, _ = V_feature.shape
                assert A_t == V_t
                timestep = A_t
                dimensionality[partition] += timestep

                if verbose:
                    print("file %s loaded with timestep %d" % (filename, timestep), A_feature.shape, V_feature.shape)

                # concatenate features
                A_data = A_feature.copy() if not i else pd.concat([A_data, A_feature])
                V_data = V_feature.copy() if not i else pd.concat([V_data, V_feature])
                # write labels and instances
                label_f.write(('%d,' % label[i]) * timestep)
                inst_f.write(('%d,' % (i+1)) * timestep)
            
            A_data.to_csv(output_dir['%s_data_A' % partition], header=False, index=False)
            V_data.to_csv(output_dir['%s_data_V' % partition], header=False, index=False)
            label_f.close()
            inst_f.close()
            print("partition %s done." % partition)


def load_bags_of_words(modality, verbose=False):
    """load preprocessed BoXW (BoAW or BoVW)
    """
    # para modality: BoAW or BoXW
    if modality == 'BoAW':
        config = data_config['baseline_preproc']['BoAW']
    elif modality == 'BoVW':
        config = data_config['baseline_preproc']['BoVW']
    else:
        return
    
    X_train = pd.read_csv(config['train_data'], header=None) 
    X_dev = pd.read_csv(config['dev_data'], header=None) 
    X_test = pd.read_csv(config['test_data'], header=None)
    y_train = pd.read_csv(config['train_label'], header=None) 
    inst_train = pd.read_csv(config['train_inst'], header=None) 
    y_dev = pd.read_csv(config['dev_label'], header=None) 
    inst_dev = pd.read_csv(config['dev_inst'], header=None) 

    if verbose:
        print("--" * 20)
        print(modality)
        print("--" * 20)
        print("training data size", X_train.shape)
        print("dev data size", X_dev.shape)
        print("test data size", X_test.shape)
        print("--" * 20)
        print("train label size", y_train.T.shape)
        print("dev label size", y_dev.T.shape)
        print("--" * 20)
    
    return X_train, X_dev, X_test, y_train.T.values, inst_train.values, y_dev.T.values, inst_dev.values