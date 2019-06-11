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

        # evaluate without test partition as there is no label
        for partition in ['train', 'dev']:
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
        self.window_size = self.model_config['doc2vec']['window_size']
        self.negative = self.model_config['doc2vec']['negative']
        self.hs = self.model_config['doc2vec']['hs']
        self.min_count = self.model_config['doc2vec']['min_count']
        self.sample = self.model_config['doc2vec']['sample']
        self.epochs = self.model_config['doc2vec']['epochs']

    def build_model(self):
        """build doc2vec model
        """
        assert self.dm == 1
        self.model = doc2vec.Doc2Vec(dm=self.dm, 
                                    vector_size=self.vector_size, 
                                    window=self.window_size,
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

    def evaluate_model(self):
        """evaluate doc2vec model by finding similar words
        """
        given_word = 'iyi'
        given_doc_id = 0
        output = smart_open(os.path.join(self.save_dir, 'evaluation.txt'), 'w', encoding='utf-8')

        similar_words = self.model.wv.most_similar(given_word, topn=20)
        print("--" * 20)
        print("\nmost similar words to given word %s for doc2vec %s model are as follows" % (given_word, self.model_name))
        output.write("--\n")
        output.write("\nmost similar words to given word %s for doc2vec %s model are as follows" % (given_word, self.model_name))
        for idx, word in enumerate(similar_words):
            print(idx, word)
            output.write("%d %s\n" % (idx, word))
            output.write("--\n")
        
        inferred_doc_vec = self.model.infer_vector(self.all_docs[given_doc_id].words)
        print("--" * 20)
        print("\nmost similar transcripts in document embedding space:\n%s:\n%s" % (self.model, self.model.docvecs.most_similar([inferred_doc_vec], topn=3)))
        output.write("\nmost similar transcripts in document embedding space:\n%s:\n%s" % (self.model, self.model.docvecs.most_similar([inferred_doc_vec], topn=3)))
        output.write("--\n")
        
        sims = self.model.docvecs.most_similar(given_doc_id, topn=len(self.all_docs), clip_start=0, clip_end=len(self.all_docs))
        print("--" * 20)
        print("\nTarget: (%d): <<%s>>\n" % (given_doc_id, ' '.join(self.all_docs[given_doc_id].words)))
        output.write("\nTarget: (%d): <<%s>>\n" % (given_doc_id, ' '.join(self.all_docs[given_doc_id].words)))
        output.write("--\n")
        
        for label, index in [('MOST',0), ('MEDIAN',len(sims)//2), ('LEAST',len(sims)-1)]:
            print("\nall the cosine similarity distance\n%s %s: <<%s>>\n" % (label, sims[index], ' '.join(self.all_docs[sims[index][0]].words)))
            output.write("\nall the cosine similarity distance\n%s %s: <<%s>>\n" % (label, sims[index], ' '.join(self.all_docs[sims[index][0]].words)))
            output.write("--\n")
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
    
    def process_metadata_tensorboard(self):
        infer_vector_train, infer_label_train = self.load_embedding('train')
        infer_vector_dev, infer_label_dev = self.load_embedding('dev')
        length_train, length_dev = len(infer_label_train), len(infer_label_dev)

        print("\nsaving embeddings to metadata file for tensorboard projector visualization")
        with smart_open(os.path.join(self.save_dir, 'label.tsv', 'w', encoding='utf-8')) as label_f:
            label_f.write("Index\tLabel\n")
            for i in range(len(length_train)):
                label_f.write("%d\t%d\n" % (i, infer_label_train[i]))
            for j in range(len(length_dev)):
                label_f.write("%d\t%d\n" % (j, infer_label_dev[j]))
        
        with smart_open(os.path.join(self.save_dir, 'metadata.tsv', 'w', encoding='utf-8')) as data_f:
            for a in range(len(infer_vector_train)):
                for b in range(len(infer_vector_train[a])):
                    data_f.write("%f\t" % infer_vector_train[a][b])
                data_f.write("\n")
            for c in range(len(infer_vector_dev)):
                for d in range(len(infer_vector_dev[c])):
                    data_f.write("%f\t" % infer_vector_dev[c][d])
                data_f.write("\n")
        print("\nmetadata processing done\nplease upload the .tsv files onto projector.tensorflow.org to visualize embeddings")

