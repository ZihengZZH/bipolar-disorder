Tree structure of Bipolar Disorder project

```tree
|__ baseline
|__ config
|   |__ data.json
|   |__ model.json
|   |__ baseline
|__ dataset
|__ docs
|__ images
|__ materials
|__ pre-trained
|__ results
|   |__ baseline
|   |__ multi_modality
|   |__ single_modality
|__ src
|   |__ init.py
|   |__ baseline.py
|   |__ experiment.py
|   |__ metric
|   |   |__ init.py
|   |   |__ uar.py
|   |__ model
|   |   |__ init.py
|   |   |__ autoencoder.py
|   |   |__ autoencoder_bimodal.py
|   |   |__ autoencoder_lstm.py
|   |   |__ dnn_classifier.py
|   |   |__ fisher_encoder.py
|   |   |__ linear_svm.py
|   |   |__ random_forest.py
|   |   |__ speech2text.py
|   |   |__ text2vec.py
|   |__ utils
|   |   |__ io.py
|   |   |__ preprocess.py
|   |   |__ vis.py
|__ test
|   |__ all_test.py
|   |__ autoencoder_test.py
|   |__ baseline_test.py
|   |__ fisher_encoder_test.py
|   |__ linear_svm_test.py
|   |__ random_forest_test.py
|   |__ speech2text_test.py
|   |__ text2vec_test.py
|   |__ utility_test.py
|__ tools
|__ LICENSE
|__ main.py
|__ README.md
```