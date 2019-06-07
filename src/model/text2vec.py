import os
import json
import gensim
import datetime
import numpy as np
import pandas as pd

from collections import namedtuple
from collections import OrderedDict
from collections import defaultdict
from smart_open import smart_open
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

from src.utils.io import load_label


class Text2Vec():
    """
    Text2Vec: Document embeddings based on transcripts
    ---
    Attributes
    -----------
    model: doc2vec.Doc2Vec()
        doc2vec instance to build on transcripts and corpus
    model_name: str
        name of doc2vec instance
    all_docs: list()
        all documents to train doc2vec model
    save_dir: str
        saving directory for doc2vec model
    data_config: dict()
        configuration file for data
    model_config: dict()
        configuration file for model
    interview_transcript: namedtuple
        data structure used for training
    -----------------------------------------
    Functions
    -----------
    prepare_data(corpus): public
        prepare training data
    load_config(): public
        load configuration for model
    build_model(): public
        build doc2vec model
    train_model(): public
        train doc2vec model
    infer_embedding(partition): public
        infer embeddings given documents using trained model
    load_embedding(partition): public
        load embeddings inferred from trained model
    evaluate_model(given_word): public
        evaluate doc2vec model by finding similar words
    save_model(): public
        save doc2vec model
    load_model(): public
        load doc2vec model
    """
    def __init__(self, build_on_corpus=False):
        # para build_on_corpus: involve Turkish corpus in training or not
        self.model = None
        self.model_name = ''
        self.fitted = False
        self.all_docs = []
        self.save_dir = ''
        self.data_config = json.load(open('./config/data.json', 'r'))
        self.model_config = json.load(open('./config/model.json', 'r'))
        self.interview_transcript = namedtuple('Interview_Transcript', 'words tags sentiment')
        self.prepare_data(build_on_corpus)
        self.load_config()

    def prepare_data(self, corpus):
        """prepared training data
        """
        labels = dict()
        _, _, level_dev, level_train = load_label()
        labels['train'] = level_train
        labels['dev'] = level_dev

        for partition in ['train', 'dev', 'test']:
            with smart_open(self.data_config['transcript_preproc'][partition], 'rb', encoding='utf-8') as all_data:
                for line_no, line in enumerate(all_data):
                    tokens = gensim.utils.to_unicode(line).split()
                    words = tokens
                    tags = [line_no]
                    sentiment = [labels[partition][line_no]] if partition != 'test' else [None]
                    self.all_docs.append(self.interview_transcript(words, tags, sentiment))
        # use addition Turkish corpus for performance boost
        if corpus:
            with smart_open(self.data_config['turkish_corpus_proc'], 'rb', encoding='utf-8') as all_data:
                for line_no, line in enumerate(all_data):
                    tokens = gensim.utils.to_unicode(line).split()
                    words = tokens
                    tags = [line_no]
                    sentiment = [None]
                    self.all_docs.append(self.interview_transcript(words, tags, sentiment))

    def load_config(self):
        """load configuration for model
        """
        self.dm = self.model_config['doc2vec']['dm']
        self.vector_size = self.model_config['doc2vec']['vector_size']
        self.negative = self.model_config['doc2vec']['negative']
        self.hs = self.model_config['doc2vec']['hs']
        self.min_count = self.model_config['doc2vec']['min_count']
        self.sample = self.model_config['doc2vec']['sample']
        self.epochs = self.model_config['doc2vec']['epochs']

    def build_model(self):
        """build doc2vec model
        """
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

        if not os.path.isdir(self.save_dir):
            self.fitted = False
        else:
            self.fitted = True
        
        print("\nbuilding vocabulary for doc2vec model ...")
        self.model.build_vocab(self.all_docs)
        print("\nvocabulary scanned & built.")

    def train_model(self):
        """train doc2vec model
        """
        if self.fitted:
            print("\nmodel already trained ---", self.model_name)
            self.load_model()
            return 
        
        print("\ntraining doc2vec %s model (with 8 threads) ..." % self.model_name)
        self.model.train(self.all_docs, 
                        total_examples=len(self.all_docs), 
                        epochs=self.model.epochs)
        print("\ntraining completed.")
        self.save_model()

    def infer_embedding(self, partition):
        """infer embeddings given documents using trained model
        """
        infer_docs = []

        labels = dict()
        _, _, level_dev, level_train = load_label()
        labels['train'] = level_train
        labels['dev'] = level_dev
        
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
        """load embeddings inferred from trained model
        """
        if os.path.isdir(self.save_dir):
            infer_vecs = np.load(os.path.join(self.save_dir, 'vectors_%s.npy' % partition))
            infer_labels = np.load(os.path.join(self.save_dir, 'labels_%s.npy' % partition))
        else:
            infer_vecs, infer_labels = [], []
        
        return infer_vecs, infer_labels

    def evaluate_model(self, given_word):
        """evaluate doc2vec model by finding similar words
        """
        similar_words = self.model.wv.most_similar(given_word, topn=20)
        print("\nmost similar words to given word %s for doc2vec %s model are as follows" % (given_word, self.model_name))
        print("--" * 20)
        output = smart_open(os.path.join(self.save_dir, 'similar_words_%s.txt' % given_word), 'w', encoding='utf-8')
        for idx, word in enumerate(similar_words):
            print(idx, word)
            output.write("%d %s\n" % (idx, word))
        print("--" * 20)
        output.close()

    def save_model(self):
        """save doc2vec model
        """
        print("\nsaving doc2vec %s model to file" % self.model_name)
        os.mkdir(self.save_dir)
        self.model.save(os.path.join(self.save_dir, 'doc2vec.model'))
        readme_notes = np.array(["This %s model is trained on %s" % (self.model_name, str(datetime.datetime.now()))])
        np.savetxt(os.path.join(self.save_dir, 'readme.txt'), readme_notes, fmt="%s")

    def load_model(self):
        """load doc2vec model
        """
        if os.path.isdir(self.save_dir):
            print("\nloading doc2vec %s model from file" % self.model_name)
            self.model = doc2vec.Doc2Vec.load(os.path.join(self.save_dir, 'doc2vec.model'))
        else:
            print("\n%s model does not exist" % self.model_name)
