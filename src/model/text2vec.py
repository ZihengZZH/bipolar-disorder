import os
import json
import gensim
import datetime
import subprocess
import numpy as np
import pandas as pd

from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
from smart_open import smart_open
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

from src.utility.io import load_label


class Text2Vec():
    def __init__(self):
        self.model = None
        self.model_name = ''
        self.dm = None
        self.vector_size = None
        self.negative = None
        self.hs = None
        self.min_count = None
        self.sample = None
        self.epochs = None
        self.all_docs = []
        self.save_dir = ''
        self.data_config = json.load(open('./config/data.json', 'r'))
        self.model_config = json.load(open('./config/model.json', 'r'))
        self.interview_transcript = namedtuple('Interview_Transcript', 'words tags sentiment')
        self.prepare_data()
        self.load_config()

    def prepare_data(self):
        labels = dict()
        _, _, level_dev, level_train = load_label()
        labels['train'] = level_train.iloc[:, 1].tolist()
        labels['dev'] = level_dev.iloc[:, 1].tolist()

        for partition in ['train', 'dev', 'test']:
            with smart_open(self.data_config['transcript_preproc'][partition], 'rb', encoding='utf-8') as all_data:
                for line_no, line in enumerate(all_data):
                    tokens = gensim.utils.to_unicode(line).split()
                    words = tokens
                    tags = [line_no]
                    sentiment = [labels[partition][line_no]] if partition != 'test' else [None]
                    self.all_docs.append(self.interview_transcript(words, tags, sentiment))

    def load_config(self):
        self.dm = self.model_config['doc2vec']['dm']
        self.vector_size = self.model_config['doc2vec']['vector_size']
        self.negative = self.model_config['doc2vec']['negative']
        self.hs = self.model_config['doc2vec']['hs']
        self.min_count = self.model_config['doc2vec']['min_count']
        self.sample = self.model_config['doc2vec']['sample']
        self.epochs = self.model_config['doc2vec']['epochs']

    def build_model(self):
        self.model = doc2vec.Doc2Vec(dm=self.dm, 
                                    vector_size=self.vector_size, 
                                    negative=self.negative, 
                                    hs=self.hs, 
                                    min_count=self.min_count, 
                                    sample=self.sample, 
                                    epochs=self.epochs,
                                    workers=8)
        self.model_name = str(self.model).replace('/','-')
        self.save_dir = os.path.join(self.model_config['doc2vec']['save_dir'], self.model_name)
        print("\ndoc2vec %s model initialized." % self.model_name)
        
        print("\nbuilding vocabulary for doc2vec model ...")
        self.model.build_vocab(self.all_docs)
        print("\nvocabulary scanned & built.")

    def train_model(self):
        print("\ntraining doc2vec %s model (with 8 threads) ..." % self.model_name)
        self.model.train(self.all_docs, 
                        total_examples=len(self.all_docs), 
                        epochs=self.model.epochs)
        print("\ntraining completed.")
        self.save_model()

    def infer_embedding(self, partition):
        infer_docs = []

        labels = dict()
        _, _, level_dev, level_train = load_label()
        labels['train'] = level_train.iloc[:, 1].tolist()
        labels['dev'] = level_dev.iloc[:, 1].tolist()
        
        with smart_open(self.data_config['transcript_preproc'][partition], 'rb', encoding='utf-8') as all_data:
                for line_no, line in enumerate(all_data):
                    tokens = gensim.utils.to_unicode(line).split()
                    words = tokens
                    tags = [line_no]
                    sentiment = [labels[partition][line_no]]
                    infer_docs.append(self.interview_transcript(words, tags, sentiment))
        
        infer_vecs = [self.model.infer_vector(doc.words, alpha=.1) for doc in infer_docs]
        infer_labels = [doc.sentiment for doc in infer_docs]

        # save inferred vectors and labels
        print("\nsaving inferred vectors and labels to file")
        if os.path.isdir(self.save_dir):
            np.save(os.path.join(self.save_dir, 'vectors_%s' % partition), infer_vecs)
            np.save(os.path.join(self.save_dir, 'labels_%s' % partition), infer_labels)

    def load_embedding(self, partition):
        if os.path.isdir(self.save_dir):
            infer_vecs = np.load(os.path.join(self.save_dir, 'vectors_%s.npy' % partition))
            infer_labels = np.load(os.path.join(self.save_dir, 'labels_%s.npy' % partition))
        else:
            infer_vecs, infer_labels = [], []
        
        return infer_vecs, infer_labels

    def evaluate_model(self):
        pass

    def save_model(self):
        print("\nsaving doc2vec %s model to file" % self.model_name)
        os.mkdir(self.save_dir)
        self.model.save(os.path.join(self.save_dir, 'doc2vec.model'))
        readme_notes = np.array(["This %s model is trained on %s" % (self.model_name, str(datetime.datetime.now()))])
        np.savetxt(os.path.join(self.save_dir, 'readme.txt'), readme_notes, fmt="%s")

    def load_model(self):
        pass
