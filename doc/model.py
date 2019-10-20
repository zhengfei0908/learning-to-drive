from torchvision import models
import torch.nn as nn
import torch
import sys
sys.path.append('../lib')

from Non_local_pytorch.lib.non_local_embedded_gaussian import NONLocalBlock2D

class FullDataModel(nn.Module):
    def __init__(self):
        super(FullDataModel, self).__init__()
        self.final_concat_size = 0
        self.directions = ['front', 'right']
        
        # CNN for images
        self.cnn = nn.ModuleDict()
        for k in self.directions:
            self.cnn[k] = models.resnet34(pretrained=True)

        for k in self.directions:
            for i, layer in enumerate(self.cnn[k].children()):
                if i <= 6:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.features = nn.ModuleDict()
        for k in self.directions:
            self.features[k] = nn.Sequential(
                *list(self.cnn[k].children())[0:4],
                NONLocalBlock2D(64),
                list(self.cnn[k].children())[4],
                NONLocalBlock2D(64),
                list(self.cnn[k].children())[5],
                NONLocalBlock2D(128),
                list(self.cnn[k].children())[6],
                NONLocalBlock2D(256),
                *list(self.cnn[k].children())[7:9]
            )
        
        self.intermediate = nn.ModuleDict()
        for k in self.directions:
            self.intermediate[k] = nn.Sequential(
                nn.Linear(self.cnn[k].fc.in_features, 128),
                nn.ReLU()
            )
            
        self.final_concat_size += 128*len(self.directions)

        # Main LSTM
        self.lstm = nn.LSTM(input_size=128*len(self.directions),
                            hidden_size=128,
                            num_layers=2,
                            batch_first=False)
        self.final_concat_size += 128
        
        # Angle Regressor
        self.control_angle = nn.Sequential(
            nn.Linear(self.final_concat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Speed Regressor
        self.control_speed = nn.Sequential(
            nn.Linear(self.final_concat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        module_outputs = []
        lstm_i = []
        # Loop through temporal sequence of
        # front facing camera images and pass 
        # through the cnn.
        for k, imgF, imgR in zip(range(len(data)),
                                            data['cameraFront'].values(),
                                            # data['cameraRear'].values(),
                                            data['cameraRight'].values()
                                            # data['cameraLeft'].values()
                                ):
            if torch.cuda.is_available():
                imgF = imgF.cuda()
#                 imgB = imgB.cuda()
                imgR = imgR.cuda()
#                 imgL = imgL.cuda()

            imgF = self.features['front'](imgF)
            imgF = imgF.view(imgF.size(0), -1)
            imgF = self.intermediate['front'](imgF)
            
#             imgB = self.features['rear'](imgB)
#             imgB = imgB.view(imgB.size(0), -1)
#             imgB = self.intermediate['rear'](imgB)
            
            imgR = self.features['right'](imgR)
            imgR = imgR.view(imgR.size(0), -1)
            imgR = self.intermediate['right'](imgR)
            
#             imgL = self.features['left'](imgL)
#             imgL = imgL.view(imgL.size(0), -1)
#             imgL = self.intermediate['left'](imgL)
            
            img_out = torch.cat([imgF, imgR], dim=-1)
            lstm_i.append(img_out)
            if k == 0:
                module_outputs.append(img_out)

        # Feed temporal outputs of CNN into LSTM
        # self.gru.flatten_parameters()
        i_lstm, _ = self.lstm(torch.stack(lstm_i))
        module_outputs.append(i_lstm[-1])
        
        # Concatenate current image CNN output 
        # and LSTM output.
        x_cat = torch.cat(module_outputs, dim=-1)
        
        # Feed concatenated outputs into the 
        # regession networks.
        prediction = {'canSteering': torch.squeeze(self.control_angle(x_cat)),
                      'canSpeed': torch.squeeze(self.control_speed(x_cat))
                     }
        return prediction