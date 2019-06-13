import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smart_open import smart_open

save_dir = '/media/zzh/Ziheng-700G/Dataset/bipolar-disorder/pre-trained/DDAE'

with smart_open(os.path.join(save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
    for line_no, line in enumerate(model_path):
        line = str(line).replace('\n', '')
        print(line_no, '\t', line[65:])
        df = pd.read_csv(os.path.join(line, 'logger.csv'))
        loss = df['loss'].tolist()
        val_loss = df['val_loss'].tolist()
        
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'dev'], loc='upper left')
        plt.savefig(os.path.join('images', 'loss', '%s.png' % line[65:]))
        plt.clf()