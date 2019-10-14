from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import json
import os
import sys
from dataset import Drive360Loader
from model import FullDataModel

config = json.load(open('./config.json'))

normalize_targets = config['target']['normalize']
target_mean = config['target']['mean']
target_std = config['target']['std']

def train_nn(train_loader, validation_loader, model, optimizer, 
             criterion, epochs=5, validation=True, load_path='', save_path='', print_freq = 200):
    '''Training the model
    Args:
        validation: boolean, whether process validation
        load_path: string, the model weights file to load
        save_path: string, the model weights file to save
    Returns:
    '''
    
    if torch.cuda.is_available():
        model = model.cuda()
        # model = nn.DataParallel(model)
        
    if os.path.exists(load_path):
        print('='*10 + 'loading weights from ' + load_path + '='*10)
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('='*10 + 'loading finished' + '='*10)
    else:
        print('load_path does not exist!')
        
    
    train_metrics = {
        'angle_loss': {}
        # 'speed_loss': {}
    }
    validation_metrics = {
        'angle_loss': {}
        # 'speed_loss': {}
    }
    for epoch in range(epochs):
        print('='*10 + 'start ' + str(epoch+1) + ' epoch' + '='*10)
        model.train()
        angle_loss = 0.0
        # speed_loss = 0.0
        since = datetime.now()
        cnt = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction['canSteering'], target['canSteering'].cuda())
            # loss2 = criterion(prediction['canSpeed'], target['canSpeed'].cuda())
            # loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            
            # print statistics
            angle_loss += loss.item()
            # speed_loss += loss2.item()
            cnt += 1
#             if batch_idx > 100:
#                 break
            if (batch_idx+1) % print_freq == 0:
                if normalize_targets:
                    angle_loss = (angle_loss * target_std['canSteering']**2) / cnt
                    # speed_loss = (speed_loss * target_std['canSpeed']**2) / cnt
                else:
                    angle_loss /= cnt
                    # speed_loss /= cnt
                train_metrics['angle_loss'].setdefault(str(epoch), []).append(angle_loss)
                # train_metrics['speed_loss'].setdefault(str(epoch), []).append(speed_loss)
                print('[epoch: %d, batch: %5d] time: %.2f angle_loss: %.2f' %
                      (epoch + 1, batch_idx + 1, (datetime.now() - since).total_seconds(), angle_loss))
                angle_loss = 0.0
                # speed_loss = 0.0
                since = datetime.now()
                cnt = 0
        print('='*10 + 'saving the model to' + save_path + '='*10)
        torch.save({
            "model_state_dict":model.state_dict(),
            "angle_loss": angle_loss,
            # "speed_loss": speed_loss,
            "optimizer_state_dict":optimizer.state_dict(),
            "epoch":epoch
            }, save_path)
        print('saving success!')
        if validation:
            print('='*10 + 'starting validation' + '='*10)
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tqdm(validation_loader)):
                    if torch.cuda.is_available():
                        target['canSteering'].cuda()
                    prediction = model(data)
                    mse1 = (np.square(prediction['canSteering'].cpu() - target['canSteering'].cpu())).mean()
                    # mse2 = (np.square(prediction['canSpeed'].cpu() - target['canSpeed'].cpu())).mean()
                    if normalize_targets:
                        mse1 = mse1 * target_std['canSteering'] ** 2
                        # mse2 = mse2 * target_std['canSpeed'] ** 2
                    validation_metrics['angle_loss'].setdefault(str(epoch), []).append(mse1)
                    # validation_metrics['speed_loss'].setdefault(str(epoch), []).append(mse2)
            print('angle_loss: %.2f' % (np.mean(validation_metrics['angle_loss'][str(epoch)])))
            print('='*10 + 'validation finished' + '='*10)
    return train_metrics, validation_metrics

if __name__ == '__main__':
    train_loader = Drive360Loader(config, 'train')
    validation_loader = Drive360Loader(config, 'validation')
    test_loader = Drive360Loader(config, 'test')

    # print the data (keys) available for use. See full 
    # description of each data type in the documents.
    print('Loaded train loader with the following data available as a dict.')
    print(train_loader.drive360.dataframe.keys())
    NOW = datetime.now().strftime("%m-%d-%H-%M")
    MODEL_NAME = 'full_data_angle'

    if not os.path.isdir(os.path.join('../output', MODEL_NAME)):
        os.mkdir(os.path.join('../output', MODEL_NAME))

    LOAD_PATH = os.path.join('../output', MODEL_NAME, '.pth')
    SAVE_PATH = os.path.join('../output', MODEL_NAME, NOW + '_' + MODEL_NAME + '.pth')

    model = FullDataModel()
    criterion =nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    result = train_nn(train_loader, validation_loader, model, optimizer, criterion, epochs=5, 
                    validation=True, load_path=LOAD_PATH, save_path=SAVE_PATH, print_freq=200)
