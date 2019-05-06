# results log

## single modality classification


## multiple modality classification

## BASELINE

| UAR (F\*/S\*) | MFCC        | eGeMAPS     | DeepSpectrum | BoAW        | FAU       | BoVW        |
| --        | --          | --          | --           | --          | --        | --          |
| SVM train | NA          | NA          | NA           | NA          | NA        | NA          |
| SVM dev   | NA          | NA          | NA           | NA          | NA        | NA          |
| RF train  | 0.49 / 0.37 | 0.89 / 0.45 | NA           | 0.53 / 0.43 | NA / 0.77 | 0.52 / 0.43 |
| RF dev    | 0.34 / 0.33 | 0.35 / 0.32 | NA           | 0.35 / 0.35 | NA / 0.49 | 0.35 / 0.38 |

> F represents frame-level and S represents session-level (FAUs are extracted on session-level)