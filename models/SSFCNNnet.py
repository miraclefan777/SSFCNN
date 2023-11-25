import torch
import torch.nn as nn


class SSFCNNnet(nn.Module):
    def __init__(self, num_spectral=31, scale_factor=8, pdconv=False):
        super(SSFCNNnet, self).__init__()
        self.scale_factor = scale_factor
        self.pdconv = pdconv

        self.Upsample = nn.Upsample(mode='bicubic', scale_factor=self.scale_factor)

        self.conv1 = nn.Conv2d(num_spectral + 3, 64, kernel_size=3, padding="same")
        if pdconv:
            self.conv2 = nn.Conv2d(64 + 3, 32, kernel_size=3, padding="same")
            self.conv3 = nn.Conv2d(32 + 3, num_spectral, kernel_size=5, padding="same")
        else:
            self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding="same")
            self.conv3 = nn.Conv2d(32, num_spectral, kernel_size=5, padding="same")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, lr_hs, hr_ms):
        """
            :param lr_hs:LR-HSI低分辨率的高光谱图像
            :param hr_ms:高分辨率的多光谱图像
            :return:
        """
        # 对LR-HSI低分辨率图像进行上采样，让其分辨率更高
        lr_hs_up = self.Upsample(lr_hs)
        # 将上采样后的LR-HSI低分辨率图像与高分辨率的多光谱图像进行拼接
        x = torch.cat((lr_hs_up, hr_ms), dim=1)

        x = self.relu(self.conv1(x))
        if self.pdconv:
            x = torch.cat((x, hr_ms), dim=1)
            x = self.relu(self.conv2(x))
            x = torch.cat((x, hr_ms), dim=1)
        else:
            x = self.relu(self.conv2(x))

        out = self.conv3(x)
        return out
