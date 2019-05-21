import numpy as np
from sklearn.metrics import recall_score, classification_report
from src.utils.io import load_label, save_UAR_results
from src.utils.io import save_post_probability, load_post_probability


def get_UAR(y_pred, y_dev, inst, model_name, feature_name, modality, frame=True, session=True, baseline=False, train_set=False, fusion=False, test=False):
    """
    get UAR metric for both frame-level and session-level
    ---
    # para y_pred: np.array()
        predicted mania level for each frame
    # para y_dev: np.array()
        actual mania level for each frame
    # para inst: np.array()
        session mappings of frames
    # para model_name: str
        given model name
    # para feature_name: str
        given feature name
    # para modality: str
        either single or multiple
    # para frame: bool
        whether to get frame-level UAR or not
    # para session: bool
        whether to get session-level UAR or not
    # para baseline: bool
        whether to get baseline performance or not
    # para train_set: bool
        whether to get UAR on training set or not
    # para fusion: bool
        whether to fuse UAR or not
    # para test: bool
        whether to save UAR results
    """
    frame_res, session_res = 0.0, 0.0
    modality = 'baseline' if baseline else modality

    # UAR for session-level only (AU features)
    if not inst.any():
        # get recalls for three classes
        recall = [0] * 3
        for i in range(3):
            index, = np.where(y_dev == (i+1))
            index_pred, = np.where(y_pred[index] == (i+1))
            recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
        session_res = np.mean(recall)
        if not fusion:
            if train_set:
                print("\nUAR (mean of recalls) using %s feature based on session-level (training set) is %.3f and %.3f (sklearn)" % (feature_name, session_res, recall_score(y_dev, y_pred, average='macro')))
            else:
                print("\nUAR (mean of recalls) using %s feature based on session-level (development set) is %.3f and %.3f (sklearn)" % (feature_name, session_res, recall_score(y_dev, y_pred, average='macro')))
                if not test:
                    session_res = recall_score(y_dev, y_pred, average='macro') 
                    save_UAR_results(frame_res, session_res, model_name, feature_name, modality)
            print(classification_report(y_dev, y_pred, target_names=['depression', 'hypo-mania', 'mania']))
        
        else:
            print("\nUAR (mean of recalls) using fusion based on session-level is %.3f and %.3f" % (session_res, recall_score(y_dev, y_pred, average='macro')))
            if not test:
                session_res = recall_score(y_dev, y_pred, average='macro') 
                save_UAR_results(frame_res, session_res, model_name, 'fusion', modality)

    else:
        # UAR for frame-level
        if frame:
            # get recalls for three classes
            recall = [0] * 3
            for i in range(3):
                index, = np.where(y_dev == (i+1))
                index_pred, = np.where(y_pred[index] == (i+1))
                recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
            frame_res = np.mean(recall)
            if train_set:
                print("\nUAR (mean of recalls) using %s feature based on frame-level (training set) is %.3f" % (feature_name, frame_res))
            else:
                print("\nUAR (mean of recalls) using %s feature based on frame-level (development set) is %.3f" % (feature_name, frame_res))
            print(classification_report(y_dev, y_pred, target_names=['depression', 'hypo-mania', 'mania']))
        
        # UAR for session-level
        if session:
            # get majority-voting for each session
            decision = np.array(([0] * inst.max()))
            for j in range(len(decision)):
                index, = np.where(inst == (j+1))
                count = [0] * 3
                for k in range(3):
                    index_pred, = np.where(y_pred[index] == (k+1))
                    count[k] = len(index_pred)
                decision[j] = np.argmax(count) + 1

            # get recalls for three classes
            recall = [0] * 3
            _, _, level_dev, _ = load_label()
            labels = level_dev.iloc[:, 1].tolist()
            labels = np.array(labels, dtype=np.int8)
            for i in range(3):
                index, = np.where(labels == (i+1))
                index_pred, = np.where(decision[index] == (i+1))
                recall[i] = len(index_pred) / len(index) # TP / (TP + FN)
            session_res = np.mean(recall)
            if train_set:
                print("\nUAR (mean of recalls) using %s feature based on session-level (training set) is %.3f" % (feature_name, session_res))
                
            else:
                print("\nUAR (mean of recalls) using %s feature based on session-level (development set) is %.3f" % (feature_name, session_res))
        
        if not train_set and not test:
            save_UAR_results(frame_res, session_res, model_name, feature_name, modality)

    return frame_res, session_res


def get_post_probability(y_pred, y_dev, inst, session_prob, model_name, feature_name):
    """
    get posteriors probabilities for features
    ---
    # para y_pred: np.array()
        predicted mania level for each frame
    # para y_dev: np.array()
        actual mania level for each frame
    # para inst: np.array()
        session mappings of frames
    # para session_prob: np.array()
        post probabilities on session-level (FAUs only)
    # para model_name: str
        given model name
    # para feature_name: str
        given feature name
    """
    if inst.all():
        len_inst = inst.max()
        prob_dev = np.zeros((3, len_inst))
        # assign values
        for l in range(len_inst):
            index, = np.where(inst == (l+1))
            len_index = len(index)
            for n in range(3):
                index_pred, = np.where(y_pred[index] == (n+1))
                prob_dev[n][l] = len(index_pred) / len_index
    else:
        prob_dev = session_prob.T
    
    save_post_probability(prob_dev, model_name, feature_name)
    print("\nposterior probability for %s model using %s feature saved." % (model_name, feature_name))


def get_late_fusion_UAR(model_name, feature_name_1, feature_name_2, baseline=False):
    """
    apply late fusion strategy on posterior probabilities of two modalities
    ---
    # para model_name: str
        given model name
    # para feature_name_1: str
        given 1st feature name
    # para feature_name_2: str
        given 2nd feature name
    # para baseline: bool
        whether to get baseline performance or not
    """
    prob_dev_1 = load_post_probability(model_name, feature_name_1)
    prob_dev_2 = load_post_probability(model_name, feature_name_2)

    assert prob_dev_1.shape == prob_dev_2.shape        
    # PROB_DEV_1 = (3, 60)
    # PROB_DEV_2 = (3, 60)

    _, _, level_dev, _ = load_label()
    y_dev = level_dev.values[:,1]
    # get the shape
    (_, num_inst) = prob_dev_1.shape
    y_pred = np.array([0] * num_inst)

    for i in range(num_inst):
        prob = prob_dev_1[:,i] + prob_dev_2[:,i]
        # fusion based on majority voting and averaging two modalities
        y_pred[i] = np.argmax(prob) + 1

    get_UAR(y_pred, y_dev, np.array([]), model_name, feature_name_1+feature_name_2, 'multiple', baseline=baseline, fusion=True)