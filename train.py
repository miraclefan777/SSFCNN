import argparse
from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR
from models.SSFCNNnet import SSFCNNnet
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from train_dataloader import CAVEHSIDATAprocess
from utils import create_F, fspecial,AverageMeter
import os
import copy
import torch
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="SSFCNNnet")
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--resume', type=str, default=False)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    assert args.model in ['SSFCNNnet', 'PDcon_SSF']

    outputs_dir = os.path.join(args.outputs_dir, '{}'.format(args.model))
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    # 训练参数
    # loss_func = nn.L1Loss(reduction='mean').cuda()
    criterion = nn.MSELoss()


    #################数据集处理#################
    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    downsample_factor = 8
    training_size = 64
    stride = 32
    stride1 = 32

    train_dataset = CAVEHSIDATAprocess(args.train_file, R, training_size, stride, downsample_factor, PSF, 20)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    eval_dataset = CAVEHSIDATAprocess(args.eval_file, R, training_size, stride, downsample_factor, PSF, 12)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    #################数据集处理#################

    # 模型
    if args.model == 'SSFCNNnet':
        model = SSFCNNnet().cuda()
    else:
        model = SSFCNNnet(pdconv=True).cuda()

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # 模型初始化
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume)  # 加载断点
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        psnr_optimal = checkpoint['psnr_optimal']
        rmse_optimal = checkpoint['rmse_optimal']

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                label, lr_hs, hr_ms = data

                label = label.to(device)
                lr_hs = lr_hs.to(device)
                hr_ms = hr_ms.to(device)
                lr = optimizer.param_groups[0]['lr']
                pred = model(hr_ms, lr_hs)
                loss = criterion(pred, label)

                epoch_losses.update(loss.item(), len(label))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), lr='{0:1.8f}'.format(lr))
                t.update(len(label))

        # torch.save(model.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))

        if epoch % 5 == 0:
            model.eval()
            val_loss = AverageMeter()

            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()

            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            for data in eval_dataloader:
                label, lr_hs, hr_ms = data
                lr_hs = lr_hs.to(device)
                hr_ms = hr_ms.to(device)
                label = label.cpu().numpy()

                with torch.no_grad():
                    preds = model(hr_ms, lr_hs).cpu().numpy()

                sam.update(SAM(preds, label), len(label))
                rmse.update(RMSE(preds, label), len(label))
                psnr.update(PSNR(preds, label), len(label))

            if psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

            print('eval psnr: {:.2f}  RMSE: {:.2f}  SAM: {:.2f} '.format(psnr.avg, rmse.avg, sam.avg))
