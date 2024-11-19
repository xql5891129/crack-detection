import argparse
import csv
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import albumentations as albu
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import Model
from dataset import Dataset
from metrics import iou_score
from metrics import dice_coef
from utils import AverageMeter

# 定义阈值
threshold=0.5

def parse_args():
    parser = argparse.ArgumentParser()#创建一个解析器——创建 ArgumentParser() 对象

    # 模型选择
    parser.add_argument('--name', default="data/train_Resnet50_UNet_woDS",   #data/train_VGG_UNet_woDS  data/train_Resnet50_UNet_woDS
                        help='model name')
    #添加参数——调用 add_argument() 方法添加参数
    args = parser.parse_args()#解析参数——使用 parse_args() 解析添加的参数
    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = Model.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])

    model = Model.__dict__[config['arch']](config['input_channels'])
    model = model.cuda()



    # Data loading code
    # 测试集文件地址
    #img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = glob(os.path.join('inputs/data/test/images',  '*' + 'jpg'))
    # 将每张图像的名字取出
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]


    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # 测试参数
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs/data/test/images'),
        mask_dir=os.path.join('inputs/data/test/masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    iou_scores = []
    dice_scores = []
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()


            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            dice=dice_coef(output, target)
            avg_meter.update(iou, input.size(0))

            # 将每个IoU,dice分数添加到列表中
            iou_scores.append(iou.item())
            dice_scores.append(dice.item())

            output = torch.sigmoid(output).cpu().numpy()
            prob_pred = np.copy(output)
            output[prob_pred < threshold] = 0
            output[prob_pred > threshold] = 1

            for i in range(len(output)):
                for c in range(config['num_classes']):

                    # 将输出图片输出到文件夹
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))


    print('IoU: %.4f' % avg_meter.avg)
    print(iou_scores)

    # 将IoU分数保存为CSV文件
    csv_file = 'iou_scores.csv'
    header = ['ImageID', 'IoU']
    rows = []
    for i, score in enumerate(iou_scores):
        image_id = val_img_ids[i]  # 图像ID
        row = [image_id, score]
        rows.append(row)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # 将dice分数保存为CSV文件
    csv_file = 'dice_scores.csv'
    header = ['ImageID', 'dice']
    rows = []
    for i, score in enumerate(dice_scores):
        image_id = val_img_ids[i]  # 图像ID
        row = [image_id, score]
        rows.append(row)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


    print(f"IoU,dice分数已保存为CSV文件：{csv_file}")
    # 作sample图
    plot_examples(input, target, model,num_examples=3)
    
    torch.cuda.empty_cache()

def plot_examples(datax, datay, model,num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
