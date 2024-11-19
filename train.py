import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import albumentations as albu
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import Model
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

ARCH_NAMES = Model.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""

指定参数：
--dataset 
--arch NestedUNet

"""
# 神经网络训练超参数参数设置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    # 神经网络迭代次数
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    # Batch_size
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='VGG_UNet', #default='Resnet50_Unet' default='VGG_UNet'
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: Resnet50_Unet)')
    # 在训练时不剪枝
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    # 输入通道数
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 分割类别数
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    # 训练图片的宽
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    # 训练图片的高
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # 设置损失函数
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # 训练数据集
    parser.add_argument('--dataset', default='data/train',
                        help='dataset name')
    # 原图像格式
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    # masks图像格式
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # 设置神经网络优化器optimizer为Adam,效果在道路裂痕分割中收敛效果比SGD好
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    # 自适应的学习率的调整策略，起始为0.001
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    #动量
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    #权重衰减
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    #是否使用Nesterov动量
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    # 学习率调度器(余弦学习率衰减)
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    #最小学习率
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    #学习率调度器的因子
    parser.add_argument('--factor', default=0.1, type=float)
    #学习率调度器的耐心值
    parser.add_argument('--patience', default=2, type=int)
    #学习率调度器的里程碑列表
    parser.add_argument('--milestones', default='1,2', type=str)
    #学习率调度器的衰减因子
    parser.add_argument('--gamma', default=2/3, type=float)
    # 是否提前停止，防止过拟合
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    #数据加载时的并发数，默认为0，表示不使用多线程加载数据
    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer):
    # 创建一个字典avg_meters，用于存储平均损失和平均交并比。AverageMeter是一个辅助类，用于计算平均值。
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    #将模型设置为训练模式，启用Batch Normalization和Dropout等训练特定的操作。
    model.train()
    #创建一个进度条，用于显示训练进度。
    pbar = tqdm(total=len(train_loader))
    #遍历训练数据集，将输入数据和目标数据移至GPU。
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        # 计算输出
        if config['deep_supervision']:
            outputs = model(input) # 神经网络输出
            loss = 0
            for output in outputs:
                loss += criterion(output, target) # 计算损失函数值
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target) # 计算交并比
        else:
            output = model(input)
            loss = criterion(output, target) # 计算损失函数值
            iou = iou_score(output, target) # 计算交并比

        # 计算梯度并执行优化步骤
        # 进行反向传播，更新神经网络参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()


            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()# WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])

    model = Model.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                            config['deep_supervision'])#VGG

    # model = Model.__dict__[config['arch']](config['input_channels'])#resnet
    model = model.cuda()


    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 切分训练集和验证集
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 数据增强：
    train_transform = Compose([
        albu.RandomRotate90(),# 随机旋转
        albu.HorizontalFlip(),# 随机翻转
        OneOf([
            transforms.HueSaturationValue(),# 色调饱和度值
            transforms.RandomBrightnessContrast(),# 随机亮度对比度
        ], p=1), # 按照归一化的概率选择执行哪一个
        albu.Resize(config['input_h'], config['input_w']),# 调整大小
        transforms.Normalize(),# 归一化
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True) # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 记录训练日志
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # 一个epoch的训练
        train_log = train(config, train_loader, model, criterion, optimizer)
        # 在验证集上评估
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        # 打印训练和验证的损失和IOU评价指标
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        # 记录训练过程中的指标值
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        # 将指标记录保存到日志文件中
        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        # 模型保存策略，这里选择了最大交并比的模型，也可以选择最小loss
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
