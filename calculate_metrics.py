import numpy as np
import torch
from torch import nn



# AverageMeter 类用于计算平均值。
# Loss_PSNR 类定义了计算峰值信噪比（PSNR）的损失函数。
# forward 方法接受真实图像（im_true）和生成图像（im_fake）作为输入。
# data_range 参数用于指定数据范围，默认为255。
# 函数首先将输入图像缩放到指定数据范围，然后计算图像之间的均方差误差（err）。
# 最后，通过计算 PSNR 并返回平均 PSNR 值。
class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        im_fake = im_fake.squeeze(0)
        im_true = im_true.squeeze(0)
        Itrue = np.clip(im_true, 0., 1.) * data_range
        Ifake = np.clip(im_fake, 0., 1.) * data_range
        err = torch.tensor(Itrue - Ifake)
        err = torch.pow(err, 2)
        err = torch.mean(err, dim=0)


        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr = torch.mean(psnr)
        return psnr


# Loss_RMSE 类定义了均方根误差（RMSE）的损失函数。
# forward 方法接受输出（outputs）和标签（label）作为输入。
# 函数首先确保输出和标签具有相同的形状，然后计算它们之间的均方根误差（RMSE）。

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        outputs = outputs.squeeze(0)
        label = label.squeeze(0)
        assert outputs.shape == label.shape
        error = np.clip(outputs, 0., 1.) * 255 - np.clip(label, 0., 1.) * 255

        sqrt_error = torch.tensor(np.power(error, 2))
        temp = sqrt_error.contiguous()
        temp = temp.view(-1)
        rmse = torch.sqrt(torch.mean(temp))
        return rmse




# Loss_SAM 类定义了结构相似性（SAM）的损失函数。
# forward 方法接受两个输入图像（im1 和 im2）作为输入。
# 函数首先确保两个图像具有相同的形状，然后进行一系列操作以计算结构相似性（SAM）。
# 结构相似性度量被计算为两个输入图像之间的平均值，并作为输出返回。
# 请注意，此实现中使用了 NumPy 而不是 PyTorch 张量，您可能需要将其转换为 PyTorch 操作以使其适用于深度学习模型的训练。

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps = 2.2204e-16

    def forward(self, im1, im2):
        im1 = im1.squeeze(0)
        im2 = im2.squeeze(0)
        assert im1.shape == im2.shape
        H, W, C = im1.shape

        im1 = np.reshape(im1, (H * W, C))
        im2 = np.reshape(im2, (H * W, C))

        core = np.multiply(im1, im2)
        mole = np.sum(core, axis=1)


        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)

        sam = np.rad2deg(np.arccos(((mole + self.eps) / (deno + self.eps)).clip(-1, 1)))
        return np.mean(sam)



