# results

## single modality

### audio 

### video

LSTM-RNN Autoencoder (w/o audio)

* accuracy on training set: 0.78
* accuracy on development set: 0.35

LSTM-RNN Autoencoder (w/ audio)

* accuracy on training set: 0.75
* accuracy on development set: 0.38

Autoencoder (frame-level)



### text (w/ Turkish corpus)

doc2vec embeddings 
| model name        | acc train | acc dev | 
| --                | --        | --      |
| dm-d100-n5-mc2    | 1.00      | 0.38    |
| dbow-d100-n5-mc2  | 1.00      | 0.45    | 

MOST SIMILAR WORDS (iyi / good)

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --    | - | - | - | - | - | - | - | - |
| dm    | başarılı | azından | ıyi | kötü | az | pahalıdan | gelişmişe | sağlıklısı | basitinden | 
|       | successful | least | good | bad | little | from expensive | advanced to | healthiest | from simple | 
| dbow  | kuyucu | özneden | iğneliyici | oynasaydı | hồng | kavrayışında | kümesinin | kirpiklerden |
|       | kuyucu | the subject | the iğneliyic | NA | NA | in understanding | set of | the lash |

> 9th May test performance (without involvement of Turkish corpus)

## multiple modality


## BASELINE

| UAR (F\*/S\*) | MFCC        | eGeMAPS     | DeepSpectrum | BoAW        | FAU       | BoVW        |
| --            | --          | --          | --           | --          | --        | --          |
| SVM train     | NA          | 0.45 / 0.29 | NA           | NA          | NA / 0.96 | NA          |
| SVM dev       | NA          | 0.35 / 0.33 | NA           | NA          | NA / 0.40 | NA          |
| RF train      | 0.49 / 0.37 | 0.89 / 0.45 | NA           | 0.53 / 0.43 | NA / 0.77 | 0.52 / 0.43 |
| RF dev        | 0.34 / 0.33 | 0.35 / 0.32 | NA           | 0.35 / 0.35 | NA / 0.49 | 0.35 / 0.38 |

> F represents frame-level and S represents session-level (FAUs are extracted on session-level)