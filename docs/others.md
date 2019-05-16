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