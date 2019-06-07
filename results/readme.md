# Performance Comparison

All results in this page, unless indicated, are based on UAR.

## single modality

### audio-visual

biSDAE

hidden_ratio: {0.4, 0.5, 0.6}

batch_size: {256, 512}

epochs: {30, 50}

noise: {0.1, 0.2, 0.4}


### text (with Turkish corpus)

doc2vec embeddings 

model: {dm, dbow}

vector_size: {25, 50, 100}

negative: {5, 10}

hs: {0, 1}

epochs: {30, 50}


MOST SIMILAR WORDS (iyi / good)

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --    | - | - | - | - | - | - | - | - |
| dm    | başarılı | azından | ıyi | kötü | az | pahalıdan | gelişmişe | sağlıklısı | basitinden | 
|       | successful | least | good | bad | little | from expensive | advanced to | healthiest | from simple | 
| dbow  | kuyucu | özneden | iğneliyici | oynasaydı | hồng | kavrayışında | kümesinin | kirpiklerden |
|       | kuyucu | the subject | the iğneliyic | NA | NA | in understanding | set of | the lash |

> 9th May test performance (without involvement of Turkish corpus)

## multiple modality

Autoencoder (session-level) {trained on all available data}

* accuracy on training set: 0.744
* accuracy on development set: 0.381

![](../images/models/structure_session.png)

Autoencoder (frame-level) {trained on all available data}

* accuracy on training set: 0.763
* accuracy on development set: 0.402

![](../images/models/structure_frame.png)

## Single Task DNN

The classification results show a better performance on the recall in Mania symptom (label 3, YMRS > 12)

## Multitask DNN


## BASELINE (self-implemented)

| UAR (F\*/S\*) | MFCC        | eGeMAPS     | DeepSpectrum | BoAW        | FAU       | BoVW        |
| --            | --          | --          | --           | --          | --        | --          |
| SVM train     |  /  |  /  | NA           | NA          | NA / | NA          |
| SVM dev       |  /  |  /  | NA           | NA          | NA / | NA          |
| RF train      |  /  |  /  | NA           |  /  | NA /  |  /  |
| RF dev        |  /  |  /  | NA           |  /  | NA /  |  /  |

> F represents frame-level and S represents session-level (FAUs are extracted on session-level)


## BASELINE (AVEC2018)

| Partition | MFCCs | eGeMAPS | BoAW | Deep | FAUs | BoVW | eGeMAPS + FAUs | Deep + FAUs | 
| --        | --    | --      | --   | --   | --   | --   | --             | --          |
| Dev       | 0.495 | 0.550   | 0.550| 0.582| 0.558| 0.558| 0.603          | **0.635**   |
| Test      | NA    | 0.500   | NA   | 0.444| 0.463| NA   | **0.574**      | 0.444       |

> Unweighted Average Recall (%UAR) of the three classes of BD (remission, hypo-mania, and mania) is used as scoring metric

## Challenge results

| Paper         | Features          | UAR (dev / test)      | Acc (dev / test) | 
| --            | --                | --                    | --               |
| Yang2018      | decision fusion   | 0.783 / 0.407         | 0.783 / NA       |
| Yang2018      | model fusion      | 0.714 / **0.574**     | 0.717 / NA       |
| Du2018        | IncepLSTM {32,64} | 0.651 / NA            | 0.650 / NA       |
| Xing2018      | Hierarchical      | **0.868** / **0.574** | NA / NA          |
| Syed2018      | V                 | NA / **0.574**        | NA / NA          |
| Syed2018      | A+V               | NA / 0.518            | NA / NA          |
| **Zhang2019** | SDAE+FV+DNN       | 0.55 / NA             | 0.53 / NA        |
