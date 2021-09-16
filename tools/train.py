# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.data import Dataset   # add by xp
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.function import train
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import read_split_data

class Hrnet_16bit(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path       # 训练集所有图片的路径列表。
        self.images_class = images_class     # 训练样本的标签信息。
        self.transform = transform

    def __len__(self):
        return len(self.images_path)     # 返回训练集下所有的数据个数。

    def __getitem__(self, item):         # 传入的item为索引。
        img4 = Image.open(self.images_path[item])   # self.images_path获得索引对应图片的路径。
        # RGB为彩色图片，L为灰度图片
        if img4.mode != 'I;16':
            raise ValueError("image: {} isn't I;16 mode.".format(self.images_path[item]))
        img3 = img4.convert("I")
        trans1=transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])
        img2 = trans1(img3)
        img = torch.div(img2.float(), 65535.0)
        label = self.images_class[item]   # 将item传入得到数字索引。

        return img, label   # 返回图片以及标签。

    @staticmethod   # @staticmethod为装饰器，表明该方法为静态方法。
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))  #通过非关键字参数的形式传入zip方法，这样图片和图片在一起，标签和标签在一起。
                                             #每个image对应的维度是[3,224,224]

        images = torch.stack(images, dim=0)  # 将images通过torch.stack方法进行拼接，stack方法会增加新的维度，即bath的数量，指定新的维度在dim=0的位置上。
                                             # stack后的数据维度是[8,3,224,224]
        labels = torch.as_tensor(labels)     # 将labels也转化成tensor的形式。
        return images, labels


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

#     dump_input = torch.rand(
#         (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
#     )                                                                       # commented by xp and changed as below
    dump_input = torch.rand(
        (1, 1, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    
    logger.info(get_model_summary(model, dump_input))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True
            
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )

    # Data loading code
#     traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
#     valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5],
                                     std=[0.5])


    traindir = '/xiaopeng/data10/train/'
    valdir = '/xiaopeng/data10/val/'

    
#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     )
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(traindir)
    train_dataset = Hrnet_16bit(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=None)
    valid_dataset = Hrnet_16bit(images_path=val_images_path,
                              images_class=val_images_label,
                              transform=None)
    #以下为增加的代码，上面几行是原有的代码
    #print(train_dataset.classes)  #根据分的文件夹的名字来确定的类别
    with open("class.txt","w") as f1:
        for classname in train_dataset.classes:
            f1.write(classname + "\n")
 
    #print(train_dataset.class_to_idx) #按顺序为这些类别定义索引为0,1...
    with open("classToIndex.txt", "w") as f2:
        for key, value in train_dataset.class_to_idx.items():
            f2.write(str(key) + " " + str(value) + '\n')
 
    #print(train_dataset.imgs) #返回从所有文件夹中得到的图片的路径以及其类别
 
 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

#     valid_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#             transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
#             transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
#             transforms.ToTensor(),
#             normalize,
#         ])),
#         batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
#         shuffle=False,
#         num_workers=config.WORKERS,
#         pin_memory=True
#     )                                                                 # commented by xp and changed as below

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    final_pth_file = os.path.join(final_output_dir,  'HRNet.pth')#增加的代码
    print("final_pth_file:", final_pth_file)#增加的代码
    torch.save(model.module, final_pth_file)#增加的代码
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

# # ------------------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT License.
# # Written by Bin Xiao (Bin.Xiao@microsoft.com)
# # Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# # ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import argparse
# import os
# import pprint
# import shutil
# import sys

# import torch
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter

# import _init_paths
# import models
# from config import config
# from config import update_config
# from core.function import train
# from core.function import validate
# from utils.modelsummary import get_model_summary
# from utils.utils import get_optimizer
# from utils.utils import save_checkpoint
# from utils.utils import create_logger


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train classification network')
    
#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         required=True,
#                         type=str)

#     parser.add_argument('--modelDir',
#                         help='model directory',
#                         type=str,
#                         default='')
#     parser.add_argument('--logDir',
#                         help='log directory',
#                         type=str,
#                         default='')
#     parser.add_argument('--dataDir',
#                         help='data directory',
#                         type=str,
#                         default='')
#     parser.add_argument('--testModel',
#                         help='testModel',
#                         type=str,
#                         default='')

#     args = parser.parse_args()
#     update_config(config, args)

#     return args

# def main():
#     args = parse_args()

#     logger, final_output_dir, tb_log_dir = create_logger(
#         config, args.cfg, 'train')

#     logger.info(pprint.pformat(args))
#     logger.info(pprint.pformat(config))

#     # cudnn related setting
#     cudnn.benchmark = config.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = config.CUDNN.ENABLED

#     model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
#         config)

#     dump_input = torch.rand(
#         (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
#     )
#     logger.info(get_model_summary(model, dump_input))

#     # copy model file
#     this_dir = os.path.dirname(__file__)
#     models_dst_dir = os.path.join(final_output_dir, 'models')
#     if os.path.exists(models_dst_dir):
#         shutil.rmtree(models_dst_dir)
#     shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

#     writer_dict = {
#         'writer': SummaryWriter(log_dir=tb_log_dir),
#         'train_global_steps': 0,
#         'valid_global_steps': 0,
#     }

#     gpus = list(config.GPUS)
#     model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

#     # define loss function (criterion) and optimizer
#     criterion = torch.nn.CrossEntropyLoss().cuda()

#     optimizer = get_optimizer(config, model)

#     best_perf = 0.0
#     best_model = False
#     last_epoch = config.TRAIN.BEGIN_EPOCH
#     if config.TRAIN.RESUME:
#         model_state_file = os.path.join(final_output_dir,
#                                         'checkpoint.pth.tar')
#         if os.path.isfile(model_state_file):
#             checkpoint = torch.load(model_state_file)
#             last_epoch = checkpoint['epoch']
#             best_perf = checkpoint['perf']
#             model.module.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             logger.info("=> loaded checkpoint (epoch {})"
#                         .format(checkpoint['epoch']))
#             best_model = True
            
#     if isinstance(config.TRAIN.LR_STEP, list):
#         lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
#             last_epoch-1
#         )
#     else:
#         lr_scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
#             last_epoch-1
#         )

#     # Data loading code
#     traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
#     valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)

#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
#         shuffle=True,
#         num_workers=config.WORKERS,
#         pin_memory=True
#     )

#     valid_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#             transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
#             transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
#             transforms.ToTensor(),
#             normalize,
#         ])),
#         batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
#         shuffle=False,
#         num_workers=config.WORKERS,
#         pin_memory=True
#     )

#     for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
#         lr_scheduler.step()
#         # train for one epoch
#         train(config, train_loader, model, criterion, optimizer, epoch,
#               final_output_dir, tb_log_dir, writer_dict)
#         # evaluate on validation set
#         perf_indicator = validate(config, valid_loader, model, criterion,
#                                   final_output_dir, tb_log_dir, writer_dict)

#         if perf_indicator > best_perf:
#             best_perf = perf_indicator
#             best_model = True
#         else:
#             best_model = False

#         logger.info('=> saving checkpoint to {}'.format(final_output_dir))
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'model': config.MODEL.NAME,
#             'state_dict': model.module.state_dict(),
#             'perf': perf_indicator,
#             'optimizer': optimizer.state_dict(),
#         }, best_model, final_output_dir, filename='checkpoint.pth.tar')

#     final_model_state_file = os.path.join(final_output_dir,
#                                           'final_state.pth.tar')
#     logger.info('saving final model state to {}'.format(
#         final_model_state_file))
#     torch.save(model.module.state_dict(), final_model_state_file)
#     writer_dict['writer'].close()


# if __name__ == '__main__':
#     main()
