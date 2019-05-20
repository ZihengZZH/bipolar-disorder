## GPU_VM

### upload: local -> remote
```
scp <local_file> user@remote_host:<remote_file>
```
e.g.
```
scp ~/Downloads/dataset.zip zz362@dev-gpu-0.cl.cam.ac.uk:~/workspace
```

### download: remote -> local
```
scp user@remote_host:<remote_file> <local_file>
```
e.g.
```
scp zz362@dev-gpu-0.cl.cam.ac.uk:~/workspace/bipolar-disorder/pre-trained/SDAE/xxx.zip ~/Downloads/
```

### unzip
```
unzip <filename>.zip -d <destination_folder>
```

## Hyperparameter tuning

Optimizing hyperparameters is considered to be the trickiest part of building machine learning models. It is nearly impossible to predict the optimal parameters while building a model, at least in the first few attempts. That is why we always go by playing with the hyperparameters to optimize them. However, this is not scalable for high dimensional data because the number of the increase in iterations, which in turn expands the training time.

Hyperparameter tuning refers to the shaping of the model architecture from the available space. Two of the most widely-used parameter optimizer techniques: **Grid Search** vs. **Random Search**.


### Grid Search


### Random Search