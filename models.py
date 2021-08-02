import torch
import torch.nn as nn
from collections import namedtuple


def calc_mean_std(x, eps=1e-10):
        """
        calculating channel-wise instance mean and standard variance
        x: shape of NCWH
        """
        mean = torch.mean(x, (2,3), keepdim=True) # size of (N, C, 1, 1)
        std = torch.std(x, (2,3), keepdim=True) + eps # size of (N, C, 1, 1)
        
        return mean, std

class Encoder(nn.Module):
    def __init__(self, vgg_path):
        super().__init__()

        #pretrained vgg19
        vgg19 = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        vgg19.load_state_dict(torch.load(vgg_path))

        vgg19_layers = list(vgg19.children())
        self.relu1_1 = nn.Sequential(*vgg19_layers[:4])
        self.relu2_1 = nn.Sequential(*vgg19_layers[4:11])
        self.relu3_1 = nn.Sequential(*vgg19_layers[11:18])
        self.relu4_1 = nn.Sequential(*vgg19_layers[18:31])

        #fix parameters
        for layer in [self.relu1_1, self.relu2_1, self.relu3_1, self.relu4_1]:
            for parameter in layer.parameters():
                parameter.requires_grad = False


    def forward(self, x):
        _output = namedtuple('output', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        relu1_1 = self.relu1_1(x)
        relu2_1 = self.relu2_1(relu1_1)
        relu3_1 = self.relu3_1(relu2_1)
        relu4_1 = self.relu4_1(relu3_1)
        output = _output(relu1_1, relu2_1, relu3_1, relu4_1)

        return output

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # relu4-1
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-4
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-3
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-2
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),# relu3-1
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu2-2
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),# relu2-1
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu1-2
            nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect'),
            # nn.Conv2d(3, 3, 1),
            nn.ReLU(), # relu1-1
        )

    def forward(self, x):
        return self.layers(x)

class Adain(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y): 
        """
        x: content_features;  y: style_features
        x, y: shape of NCWH
        """
        x_mean, x_std = calc_mean_std(x)
        y_mean, y_std = calc_mean_std(y)

        normalized_x = (x-x_mean) / x_std
        
        return normalized_x * y_std + y_mean


class Network(nn.Module):
    def __init__(self, vgg_path):
        super().__init__()
        self.encoder = Encoder(vgg_path)
        self.decoder = Decoder()
        self.adain = Adain()
        self.MSEloss = nn.MSELoss()

    def cal_style_loss(self, x, y):
        """
        calculating style loss in one layer
        """
        x_mean, x_std = calc_mean_std(x)
        y_mean, y_std = calc_mean_std(y)
        
        return self.MSEloss(x_mean, y_mean) + self.MSEloss(x_std, y_std)

    def style_transfer(self, x, y):
        """
        x, y, out_image are tensors

        x, y of shape 1, C, W, H
        out_image of shape 1, c, w, h
        """
        assert x.size()[0] == 1 and y.size()[0] == 1
        # encode
        x_enc = self.encoder(x)
        y_enc = self.encoder(y)
        
        # adain
        transformed_x = self.adain(x_enc.relu4_1, y_enc.relu4_1)

        # decode
        out_image = self.decoder(transformed_x)

        return out_image

    def forward(self, x, y):
        # encode
        x_enc = self.encoder(x)
        y_enc = self.encoder(y)
        
        # adain
        transformed_x = self.adain(x_enc.relu4_1, y_enc.relu4_1)

        # decode
        out_image = self.decoder(transformed_x)

        # compute loss
        out_image_enc = self.encoder(out_image)
        
        content_loss = self.MSEloss(transformed_x, out_image_enc.relu4_1)
        style_loss = 0.
        for x, y in zip(y_enc, out_image_enc):
            style_loss += self.cal_style_loss(x, y)

        return content_loss, style_loss


