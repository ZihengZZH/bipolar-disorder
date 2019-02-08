# Literature Review

This document records the literature of the project, and it focuses on other papers' contributions and methodological inspiration instead of results or conclusion.

## category


### Bipolar Disorder Recognition with Histogram Features of Arousal and Body Gestures 
#### Yang2018

Some effective features were proposed for the classification.

The audio features include 
* 120 histogram based arousal features (**Audio-Arousal_Hist**);
* 6902 dimensional feature vector for each audio sample (**Audio-LLD_Function**);
* 20 dimensional feature vector about speech pause and speaking rate (**Audio-VAD-SR_Hist**). 

> Among ratings-based features (arousal/valence/dominance), experimental results showed that arousal demonstrated better depression classification results than valence and dominance.

The video features include
* 50 histogram based hands distance features (**Video-HandsDist_Hist**); 
* 400 HDR (Histogram of Displacement) features for each segment (**Video-Body_HDR**);
* 18 dimensional feature vector about action units (**Video-AU_Hist**).

In the data processing, the number of video segments were increased with a sliding window, feature dimensions were decreased with CFS + SFFS + SVM based feature selection + PCC, and features were normalized with z-score normalization.

In the classification, features **Audio-Arousal_Hist** and **Video-HandsDist_Hist** were distinguishable enough to be used for statistical method. While other features were fed into separate DNNs, and then after feature concatenation, random trees were applied to give classification results. In the multistream classification, the weight of **Audio-Arousal_Hist** was increased because of better performance.

The contributions of their work are as follows:
* A histogram based arousal feature, in which the continuous arousal values are estimated from the audio cues by a Long Short-Term Memory Recurrent Neural Network (LSTM-RNN) model (trained on AVEC 2015 affective database). Then the histogram of the arousal values in each segments are used for the classification;
*  A Histogram of Displacement (HDR) based upper body posture feature, which characterizes the displacement and velocity of the key body points in the video segment;
*  A DNN and Random Forest based multi-stream fusion framework, and several DNN models with an ensemble learning strategy in order to mitigate the overfitting because of the limited training data.

> the first to utilize the affective dimension with bipolar depression analysis



### Bipolar Disorder Recognition via Multi-scale Discriminative Audio Temporal Representation
#### Du2018

Bipolar symptoms are episodic, especially with irregular variations among different episodes, making BD very difficult to be diagosed accurately.

16-dimensional Mel-Frequency Spectrum Coefficients (MFCCs) were extracted as Low-Level Descriptors (LLDs). Three kinds of convolution filters with kernel sizes $d\times1$, $d\times3$, $d\times5$ were applied on the temporal feature sequence concurrently ($d$ was the dimension of LLD feature vector). Besides, the $5\times5$ kernel was divided into two $3\times3$ kernels, reducing the parameter number and avoiding overfitting. The LSTM layer takes multi-scale intermediate temporal representation from the Inception module as input to capture dynamic information so as to obtain final high-level description for the whole audio. IncepLSTM was then trained by the joint supervision of the softmax loss with the severity-sensitive loss.

To sum up, for each audio, 16-dimensional normalized feature vectors were formed from MFCCs as the LLD input for IncepLSTM.

The contributions of their work are as follows:
* IncepLSTM, the Inception module joint with an LSTM layer, to achieve multi-scale audio temporal\footnote{Authors of Turkish Audio-Visual BD Corpus merely used statistics across the frame-based features} representation for BD recognition;
* An improved triplet loss\footnote{The triplet loss optimizes the embedding space so that data points with the same identity are closer to each other than those with different identities.} function, called severity sensitive loss, to enhance the supervision of the learning procedure, making BD representation more discriminative;
* The sparse structure to compress the IncepLSTM network, reducing the risk of overfitting and improving the robustness

### Multi-modality Hierarchical Recall based on GBDTs for Bipolar Disorder Classification
#### Xing2018
> possibly the best in AVEC 2018


Authors mentioned that, to eliminate overfitting, feature selection was used further to discard redundant features or select more discriminative features. According to authors, there are two kinds of information integration strategies, namely subject-level and topic-level. Besides, there are two kinds of multi-modality fusion, namely feature-level fusion and decision-level.

One intuition of theirs was to perform different decision strategies for different patients instead of classifying all patients at same layer. They proposed a novel hierarchical recall framework, which was symmetric and took advantages of different modalities in each layer

> the predictions of patients are made layer-by-layer, where patients with high confidence level were first recalled while those with low confidence level were delivered to next layer to perform further judgement.

1002 dimensional audio features were extracted that included $R^{248}$ global features ((23 eGeMAPS + 39 MFCCs) $\times$ 4 statistic functions), $R^{248\times3}$ topic-level features (3 topics in total) and $10$ timing of speech features. 3607 dimensional visual features were extracted, including AUs MHH features $R^{17\times5\times10\times3}$, AUs statistics $R^{(17\times16)\times3}$, eyesight feature $R^{17\times3}$, emotion feature $R^{(11+13\times2+24)\times3}$ and body movement feature $R^7$. 786 dimensional text features were extracted that included 14 linguistic features and 8 kinds of sentiment indices. In total, the dimension of feature in each interview was 5395. The following feature selection was completed by Analysis of Variance algorithm.

The contributions of their work are as follows:
* A novel hierarchical recall model, where patients of different mania level were recalled at multi-layers instead of single layer;
* Topic modelling, where audio, video and text features were generated for each topic segment respectively:
  * detailed information of each topic was retained
  * each topic was characterized by different features


### Automated Screening for Bipolar Disorder from Audio/Visual Modalities
#### Syed2018

> Syed et al. have conducted research in this areas for a long time, and their paper in AVEC 2017 is still worthy reading.

Given that labels are based on YMRS scores, they firstly identify key behavioural characteritics of individuals with mania as per the YMRS. This enabled them to use craft features which can probe for the existence of these characteristics, as opposed to brute-force methods which aim to learn the dataset without using background knowledge. 

The 'turbulence features' (for a wide range of audio/visual features) were the most important in the paper and authors computed the crest factor as the measure of turbulence, the ratio between the absolute maximum value of the signal and its root mean square value. 

> 'turbulence features': the measure of turbulence was computed from the crest factor, which measured the ratio between the absolute maximum value of the signal and its root mean square (RMS) value.

The process of Fisher vector encoding was summarised as follows: ComParE LLDs from each speech recording was concatenated into one matrix and then using a GMM to build a backgound model for the feature space. 

They proposed GEWELMs for classification (the efficacy had been demonstrated), which is based on WELMs and ELMs (Weighted Extreme Learning Machines). They used ELMs as a method to reduce dimensionality and least square regression towards class label prediction. To handle the overfitting of GEWELMs, they trained two sets of GEWELMs, T2D-GEWELMs and D2T-GEWELMs, which are training on the training set and testing on the development set, and training on the development set and testing on the training set. This helped to regularise the selection of WELMs so only those that had acceptable performace for two regimes were selected.

The contributions of the work are as follows:

* The proposed 'turbulence features' to capture suddent, erratic changes in feature contours;
* Fisher vector encoding of Computational Paralinguistics Challenge (ComParE) low level descriptors that were demonstrated to be viable for predicting the severity of mania;
* Four feature sets from OpenSmile toolkit, namely Prosody, IS10-Paralinguistics, ComParE functionals and eGeMAPS features were most useful for automated screening of bipolar disorder;
* Greedy Ensembles of Weighted Extreme Learning Machines (GEWELMs) classifier for the task of predicting the severity of mania.

> GEWELMs are essentially a single layer feed-forward neural network where the hidden layer is assigned randomly generated weights which are not updated during the training process


### The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing
#### Eyben2015

In their research, they mentioned that emotional cues conveyed in the voice have been empirically documented the research have explored a large number of acoustic parameters, including parameters in:
* Time domain (speech rate)
* Frequency domain (fundamental frequency ($F_0$)
* Amplitude domain (intensity or energy)
* spectral distribution domain (relative energy in different frequency bands)

Moreover, large brute-force feature sets reduce the generalisation capabilities to unseen (test) data. Minimalistic parameter sets might reduce this danger and lead to better generalisation in cross-corpus experiments and ultimately in real-world test scenarios.

### paper title
#### bibtex