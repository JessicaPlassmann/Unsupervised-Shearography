import torch

"""
ATTENTION CODE TAKEN FROM https://github.com/donggong1/memae-anomaly
-detection/blob/master/models/ae_3dconv.py
Code uses MIT Licence, thus if this is published it has to do so as well
https://github.com/donggong1/memae-anomaly-detection/blob/master/LICENSE
Code was modified for 2D conv instead of 3d conv
"""


class AutoEncoderConv2DModel(torch.nn.Module):
    def __init__(self, chnum_in: int = 1):
        super(AutoEncoderConv2DModel, self).__init__()
        self.chnum_in = chnum_in  # input channel number is 1;
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(self.chnum_in, feature_num_2, (3, 3),
                            stride=(1, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(feature_num_2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(feature_num_2, feature_num, (3, 3),
                            stride=(2, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(feature_num),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(feature_num, feature_num_x2, (3, 3),
                            stride=(2, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(feature_num_x2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(feature_num_x2, feature_num_x2, (3, 3),
                            stride=(2, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(feature_num_x2),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(feature_num_x2, feature_num_x2, (3, 3),
                                     stride=(2, 2), padding=(1, 1),
                                     output_padding=(1, 1)),
            torch.nn.BatchNorm2d(feature_num_x2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(feature_num_x2, feature_num, (3, 3),
                                     stride=(2, 2), padding=(1, 1),
                                     output_padding=(1, 1)),
            torch.nn.BatchNorm2d(feature_num),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(feature_num, feature_num_2, (3, 3),
                                     stride=(2, 2), padding=(1, 1),
                                     output_padding=(1, 1)),
            torch.nn.BatchNorm2d(feature_num_2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(feature_num_2, self.chnum_in, (3, 3),
                                     stride=(1, 2), padding=(1, 1),
                                     output_padding=(0, 1))
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out
