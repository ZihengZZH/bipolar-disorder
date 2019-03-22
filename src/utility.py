import os
import json
import statistics
import numpy as np
import pandas as pd
from scipy.io import arff
from smart_open import smart_open


'''
ALL UTILITY FUNCTIONS
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))


# load the labels (age, gender, YMRS)
def load_label(partition=True, verbose=False):
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


# retrieve the sample name
def get_sample(partition, index):
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


# load the audio LLDs
def load_LLD(LLD_name, partition, index, verbose=False):
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


# load the baseline features 
def load_baseline_feature(feature_name, partition, index, verbose=False):
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


# load the features pre-processed by MATLAB or Python as below
def load_proc_baseline_feature(feature_name, matlab=True, verbose=False):
    # para feature_name: which feature, BoAW or eGeMAPS or BoVW
    # para matlab: whether or not to use MATLAB processed features
    # para verbose: whether or not to output more results
    baseline = 'baseline_MATLAB' if matlab else 'baseline_preproc'

    if feature_name == "AU":
        filename = data_config['baseline_MATLAB']['AU']
        featall = pd.read_csv(filename, header=None)
    
    elif feature_name == "BoW":
        filename = data_config['baseline_MATLAB']['BoW']
        featall = pd.read_csv(filename, header=None)

    elif feature_name == "Deep":
        train_data = pd.read_csv(data_config[baseline]['Deep']['train_data'], header=None)
        train_label = pd.read_csv(data_config[baseline]['Deep']['train_label'], header=None)
        train_inst = pd.read_csv(data_config[baseline]['Deep']['train_inst'], header=None)
        dev_data = pd.read_csv(data_config[baseline]['Deep']['dev_data'], header=None)
        dev_label = pd.read_csv(data_config[baseline]['Deep']['dev_label'], header=None)
        dev_inst = pd.read_csv(data_config[baseline]['Deep']['dev_inst'], header=None)

    elif feature_name == "eGeMAPS":
        train_data = pd.read_csv(data_config[baseline]['eGeMAPS']['train_data'], header=None)
        train_label = pd.read_csv(data_config[baseline]['eGeMAPS']['train_label'], header=None)
        train_inst = pd.read_csv(data_config[baseline]['eGeMAPS']['train_inst'], header=None)
        dev_data = pd.read_csv(data_config[baseline]['eGeMAPS']['dev_data'], header=None)
        dev_label = pd.read_csv(data_config[baseline]['eGeMAPS']['dev_label'], header=None)
        dev_inst = pd.read_csv(data_config[baseline]['eGeMAPS']['dev_inst'], header=None)

    elif feature_name == "MFCC":
        train_data = pd.read_csv(data_config[baseline]['MFCC']['train_data'], header=None)
        train_label = pd.read_csv(data_config[baseline]['MFCC']['train_label'], header=None)
        train_inst = pd.read_csv(data_config[baseline]['MFCC']['train_inst'], header=None)
        dev_data = pd.read_csv(data_config[baseline]['MFCC']['dev_data'], header=None)
        dev_label = pd.read_csv(data_config[baseline]['MFCC']['dev_label'], header=None)
        dev_inst = pd.read_csv(data_config[baseline]['MFCC']['dev_inst'], header=None)
    
    if verbose:
        print("Size of training data (extracted from MATLAB)", train_data.shape)
        print("Size of training labels (extracted from MATLAB)", train_label.shape)
        print("Size of training instance (extracted from MATLAB)", train_inst.shape)
        print("Size of dev data (extracted from MATLAB)", dev_data.shape)
        print("Size of dev labels (extracted from MATLAB)", dev_label.shape)
        print("Size of dev instance (extracted from MATLAB)", dev_inst.shape)

    return train_data, train_label, train_inst, dev_data, dev_label, dev_inst


# pre-process the baseline features (LLDs)
def preproc_baseline_feature(feature_name, verbose=False):
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



def save_results(frame_res, session_res, name, modality):
    # para frame_res: classification UAR for frame-level
    # para session_res: classification UAR for session-level
    # para name: which feature is used
    # para modality: either single or multiple
    if modality == 'single':
        filename = os.path.join(data_config['result_single'], '%s_result.txt' % name)
        with smart_open(filename, 'w', encoding='utf-8') as f:
            f.write("UAR on frame-level: %.3f \n" % frame_res)
            f.write("UAR on session-level: %.3f \n" % session_res)
        f.close()
        
    elif modality == 'multi':
        filename = os.path.join(data_config['result_multi'], '%s_result.txt' % name)
        with smart_open(filename, 'w', encoding='utf-8') as f:
            f.write("UAR on frame-level: %.3f \n" % frame_res)
            f.write("UAR on session-level: %.3f \n" % session_res)
        f.close()
    
    else:
        print("\n-- INVALID INPUT --\n")
        return