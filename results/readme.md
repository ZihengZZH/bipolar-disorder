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



### text

doc2vec embeddings 
| model name        | acc train | acc dev | 
| --                | --        | --      |
| dm-d50-w5-mc2     | 1.00      | 0.35    |
| dbow-d50-n5-mc2   | 0.99      | 0.58    |

MOST SIMILAR WORDS

| model | word | 1 | 2 | 3 | 4 | 5 |
| --    | --            | - | - | - | - | - |
| dm    | iyi | yapma | Hayirdir | gaflette | olayi | kahve |
| dm    | good | making | It is no | in garflet | probable | coffee | 
| dbow  | iyi | birbirimize | tutuldum | ciktiktan | unuttun | iltihap |
| dbow  | good | each other | I kept | after cikti | you forgot | inflammation |

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