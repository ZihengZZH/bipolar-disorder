import json
import statistics
import numpy as np
import pandas as pd
from scipy.io import arff


'''
ALL UTILITY FUNCTIONS
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))


# load the labels (age, gender, YMRS)
def load_label(partition=True, verbose=False):
    # para partition: whether to partition labels into train/dev sets
    # para verbose: whether or not to output more statistical results
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

    ymrs_score = pd.concat([label.iloc[:, 0], label.iloc[:,4]], axis=1)
    if partition:
        ymrs_dev = ymrs_score.iloc[:60, :]
        ymrs_train = ymrs_score.iloc[60:, :]
        return ymrs_train, ymrs_dev
    else:
        return ymrs_score, label


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


# load the audio LLDs (MFCC)
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
        print(mfcc.shape)
    elif LLD_name == 'eGeMAPS':
        egemaps = pd.read_csv(data_config['data_path_local']['LLD']['eGeMAPS'] + sample, sep=';')
        print(egemaps.shape)
    elif LLD_name == 'openFace':
        face = pd.read_csv(data_config['data_path_local']['LLD']['openFace'] + sample)
        print(face.shape)
    else:
        print("\nWRONG INPUT - LLD NAME\n")
        return


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
        return boaw
    elif feature_name == 'eGeMAPS':
        sample = sample[:-3] + 'arff'
        egemaps = arff.loadarff(data_config['data_path_local']['baseline']['audio']['eGeMAPS'] + sample)
        egemaps_df = pd.DataFrame(egemaps[0])
        return egemaps_df
    elif feature_name == 'BoVW':
        sample = '11_' + sample
        bovw = pd.read_csv(data_config['data_path_local']['baseline']['video']['BoVW'] + sample, sep=';')
        return bovw
    else:
        print("\nWRONG INPUT - LLD NAME\n")
        return