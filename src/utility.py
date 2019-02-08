import json
import statistics
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


'''
ALL UTILITY FUNCTIONS
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))


# load the label metadata (age, gender, YMRS)
def load_label_metadata(verbose=False):
    # para verbose: whether or not to output more statistical results
    label = pd.read_csv(data_config['data_path']['label_metadata'] + 'labels_metadata.csv')
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

    return label, id_set


# load the audio LLDs (MFCC)
def load_audio_LLD_MFCC(verbose=False):
    # para verbose: whether or not to output more results
    mfcc = pd.read_csv(data_config['data_path']['LLDs_audio_MFCC'] + 'dev_001.csv', sep=';')
    print(mfcc)


# load the audio LLDs (eGeMAPS)
def load_audio_LLD_eGeMAPS(verbose=False):
    # para verbose: whether or not to output more results
    egemaps = pd.read_csv(data_config['data_path']['LLDs_audio_eGeMAPS'] + 'dev_001.csv', sep=';')
    print(egemaps)
