import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smart_open import smart_open

save_dir = './pre-trained/DDAE'

with smart_open(os.path.join(save_dir, 'model_list.txt'), 'rb', encoding='utf-8') as model_path:
    for line_no, line in enumerate(model_path):
        line = str(line).replace('\n', '')
        print(line_no, '\t', line[19:])
        df = pd.read_csv(os.path.join(line, 'logger.csv'))
        if 'unimodal' in line[19:]:
            loss = df['loss'].tolist()
            plt.plot(loss[1:])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['recon loss'], loc='upper right')

        elif 'bimodal' in line[19:]:
            loss = df['loss'].tolist()
            loss_A = df['audio_recon_loss'].tolist()
            loss_V = df['video_recon_loss'].tolist()
            plt.plot(loss[1:])
            plt.plot(loss_A[1:])
            plt.plot(loss_V[1:])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['recon loss', 'audio recon loss', 'video recon loss'], loc='upper right')

        elif 'multimodal' in line[19:]:
            loss = df['loss'].tolist()
            loss_A = df['audio_recon_loss'].tolist()
            loss_V1 = df['facial_recon_loss'].tolist()
            loss_V2 = df['gaze_recon_loss'].tolist()
            loss_V3 = df['pose_recon_loss'].tolist()
            loss_V4 = df['action_recon_loss'].tolist()

            # loss_A = np.log(loss_A)
            # loss_V1 = np.log(loss_V1)
            # loss_V2 = np.log(loss_V2)
            # loss_V3 = np.log(loss_V3)
            # loss_V4 = np.log(loss_V4)

            plt.plot(loss[1:])
            plt.plot(loss_A[1:])
            plt.plot(loss_V1[1:])
            plt.plot(loss_V2[1:])
            plt.plot(loss_V3[10:])
            plt.plot(loss_V4[1:])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['recon loss', 'audio recon loss', 'facial recon loss', 'gaze recon loss', 'pose recon loss', 'action recon loss'], loc='upper right') 
        
        plt.savefig(os.path.join('images', 'loss', '%s.png' % line[19:]))
        plt.clf()