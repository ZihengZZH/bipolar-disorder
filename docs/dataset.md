# ABOUT THE DATASET

## basics about the dataset

[The Turkish Audio-Visual Bipolar Disorder Corpus](https://www.cmpe.boun.edu.tr/~salah/ciftci18aciiasia.pdf) is a new dataset for the affective computing and psychiatric communities. The corpus is annotated for BD state, as well as Young Mania Rating Scale (YMRS) by psychiatrists. The investigated features include __functionals of appearance descriptors__ extracted from fine-tuned Deep Convolutional Neural Networks (DCNN), __geometric features__ obtained using tracked facial landmarks, as well as __acoustic features__ extracted via openSMILE tool. Furthermore, acoustics based emotion models are trained on a Turkish emotional dataset and emotion predictions are cast on the utterances of the BD corpus. The affective scores/predictions are investigated with linear regression and correlation analysis against YMRS declines to give insights about BD, which is directly linked with emotional lability, i.e. quick changes in affect.

The core aim of the efforts on the corpus is to find **biological markers/predictors** of treatment response via signal processing and machine learning techniques to reduce treatment resistance.

## recordings in the dataset

After reviewing some clips in the recordings, some feelings are listed as below:
1. In some video clips, the audio quality is quite poor partly because of the low quality recoding device (the background noise is loud)
2. On the other hand, the video/image quality is relatively good
3. When the subject's sentiment shifts, the change is more clear in the voice than in the video/image [dev_038]()


## details about the dataset

The BD corpus used for the AVEC 2018 BDS includes *audiovisual recordings* of structured interviews performed by 46 Turkish speaking subjects. Participants of the BD corpus were asked to complete seven tasks, e.g. explaining the reason to participate the activity, describing happy and sad memories, counting up to thirty, and explaining two emotion eliciting pictures. During hospitalisation, in every follow up day (0th, 3rd, 7th, 14th, 28th day) and after discharge on the 3rd month, the presence of depressive and manic features were evaluated using YMRS. The dataset can be summarized with following data:

* audio (.wav), video recording (.mp4) of each subject
* labels (level of mania and YMRS) and metadata (age, gender)
* baseline features

More specifically, the tree structure of the BD corpus:

```tree
├── recordings
│   ├── recordings_audio (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.wav)
|   ├── recordings_video (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.mp4)
├── LLDs_audio_SMILE
|   ├── LLDs_audio_eGeMAPS (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv)
|   ├── LLDs_audio_MFCCs (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv)
├── LLDs_video_openFace
|   ├── LLDs_video_openFace (#436)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv + .hog)
├── sound_separator
|   ├── sound_separator (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv)
├── VAD_turns
|   ├── VAD_turns (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv)
├── baseline_features
|   ├── features_audio_BoAW_A20_C1000 (#218)
|   |   ├── 2_train_001 ... 2_train_104
|   |   ├── 2_dev_001 ... 2_dev_060
|   |   ├── 2_test_001 ... 2_test_054
|   |   ├── (.csv)
|   ├── features_audio_eGeMAPS_turns (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.arff)
|   ├── features_video_BoVW_A20_C1000 (#218)
|   |   ├── 11_train_001 ... 11_train_104
|   |   ├── 11_dev_001 ... 11_dev_060
|   |   ├── 11_test_001 ... 11_test_054
|   |   ├── (.csv)
|   ├── LLDs_audio_DeepSpectrum_turns (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv)
|   ├── LLDs_audio_opensmile_MFCCs_turns (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv)
|   ├── LLDs_video_openFace_AUs (#218)
|   |   ├── train_001 ... train_104
|   |   ├── dev_001 ... dev_060
|   |   ├── test_001 ... test_054
|   |   ├── (.csv)
├── labels_metadata.csv
├── BD_corpus_intro.pdf
├── readme.txt
```

## labels_metadata (only train & dev)
In this csv file, each instance has been labelled with SubjectID, Age, Gender, Total_YMRS, and ManiaLevel. Note that each subjects produces 3, 4, 5, or 6 instances and labels for test data are not given.

| # subjects | # female | # male | age range  | median age |
| --         | --       | --     | --         | --         |
| 34         | 11       | 23     | 18 ~ 53    | 36         |

## audio LLDs and features

### MFCC
In speech processing, the __mel-frequency cepstrum (MFC)__ is a representation of the short-term power spectrum of a sound based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. __Mel-frequency cepstral coefficients (MFCC)__ are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip. 

MFCCs are commonly derived as follows:
1. Take the Fourier transform of (a windowed excerpt of) a signal.
2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
3. Take the logs of the powers at each of the mel frequencies.
4. Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
5. The MFCCs are the amplitudes of the resulting spectrum.

Extracted MFCCs features (#5931) have 40 dimensions:
* frameTime
* pcm_fftMag_mfcc[0:12]
* pcm_fftMag_mfcc_de[0:12]
* pcm_fftMag_mfcc_de_de[0:12]

> MFCCs from 25ms audio frames (sampled at a rate of 10ms). It computes 13 MFCC from 26 Mel-Frequency bands, and applies a cepstral liftering filter with a weight parameter of 22. 13 delta and 13 acceleration coefficients are appended to the MFCC.

### eGeMAPS

GeMAPS stands for the Geneva Minimalistic Acoustic Parameter Set and it is a basic standard acoustic parameter set for various areas of automatic voice analysis. These parameters are selected based on a) the potential to index effective physiological changes in voice production, b) the proven value in former studies as well as the automatic extractability, and c) the theoretical significance.

Extracted eGeMAPS features (#5928) have 24 dimensions:
* frameTime
* **Loudness**_sma3 ------ estimate of perceived signal intensity from an auditory spectrum
* **alphaRatio**_sma3 ------ ratio of the summed energy from 50-1000Hz and 1-5kHz
* **hammarbergIndex**_sma3 ------ ratio of the strongest energy peak in the 0-2kHz region to the strongest peak in the 2-5kHz region
* **slope0-500**_sma3 ------ linear regression slope of the logarithmic power spectrum within band
* **slope500-1500**_sma3 ------ linear regression slope of the logarithmic power spectrum within band
* **spectralFlux**_sma3
* **mfcc[1:4]**_sma3
* **F0semitoneFrom27.5Hz**_sma3nz
* **jitterLocal**_sma3nz ------ deviations in individual consecutive $F_0$ period lengths
* **shimmerLocaldB**_sma3nz ------ difference of the peak amplitudes of consecutive $F_0$ periods 
* **HNRdBACF**_sma3nz ------ Harmonics-to-Noise Ratio, relation of energy in harmonic components to energy in noise-like components
* **logRelF0-H1-[H1;A3]**_sma3nz
* **F1[frequency;bandwidth;amplitudeLogF0]**_sma3nz 
* **F2[frequency;amplitudeLogRelF0]**_sma3nz
* **F3[frequency;amplitudeLogRelF0]**_sma3nz

## video LLDs and features
