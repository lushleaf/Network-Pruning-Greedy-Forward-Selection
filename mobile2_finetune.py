import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
import shutil
import math

#from tensorboardX import SummaryWriter

from lib.utils import accuracy, AverageMeter
from lib.data import get_dataset_ft
from lib.net_measure import measure_model

from models.mobilenet_v3 import MobileNetV3_mask, eps, Mask, mb3_prune_ratio

from torch.autograd import Variable

from tqdm import tqdm
import time

import copy

# import setGPU


def parse_args():
    parser = argparse.ArgumentParser(description='finetune for mbv3')

    # model and data
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')

    # seed
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')

    # intermediate finetune schedule
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate for intermediate finetune')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')  # default 128
    parser.add_argument('--lr_type', default='cos', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--n_gpu', default=2, type=int, help='number of GPUs to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')

    # load and save
    parser.add_argument('--load_path', default='./checkpoint', type=str,
                        help='pretrain model path to prune')
    parser.add_argument('--save_path', default='./checkpoint', type=str, help='path the save the prunde model')
    parser.add_argument('--resume', action='store_true',
                    help='always load the previous saved model')

    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')

    return parser.parse_args()


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_model(path, n_class):
    layer = -1
    from models.mobilenet_v2 import MobileNetV2
    fullnet = MobileNetV2(num_classes=1000)
    net = MobileNetV3_prescreen(fullnet)
    checkpoint = torch.load(args.load_path, map_location='cpu')
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k[0:6] == 'module':
            name = k[7:]  # remove module.
        else:
            name = k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(net, device_ids=gpu_list)
        model = model.to(device)
    else:
        model = net.to(device)

    return model


def train(train_loader, model, criterion):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 200 ==0:
            print('[{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            batch_idx, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    global best_accd
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs= model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 ==0:
                print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_idx, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    return top1.avg



def adjust_learning_rate(optimizer, epoch):
    if args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':
    global args, best_prec1
    args = parse_args()
    best_prec1 = 0
    #global args, best_prec1
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    if args.n_gpu > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_tmp')
        memory_gpu = [int(x.split()[2]) for x in open('gpu_tmp', 'r').readlines()]
        memory_gpu = np.array(memory_gpu)
        print(memory_gpu)
        gpu_list = list(memory_gpu.argsort()[-args.n_gpu:][::-1])
        print(gpu_list)

        gpu_list = [int(idx) for idx in gpu_list]
        gpu_list_ = ",".join(str(i) for i in gpu_list)

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_

        device = torch.device('cuda', int(gpu_list[0]))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)


    batch_size =  args.batch_size
    print('setting Batch Size, LR, N_GPU', batch_size, args.lr, args.n_gpu)

    print('=> Preparing data..')
    train_loader,  val_loader, n_class = get_dataset_ft(args.dataset, batch_size, args.n_worker,
                                                                       data_root=args.data_root,
                                                                       istrick2=0)

    model = get_model(args.load_path, n_class)

    criterion = CrossEntropyLabelSmooth(n_class).cuda()

    no_wd_params, wd_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ".bn" in name or '.bias' in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)
    no_wd_params = nn.ParameterList(no_wd_params)
    wd_params = nn.ParameterList(wd_params)

    optimizer = torch.optim.SGD([
                                {'params': no_wd_params, 'weight_decay':0.},
                                {'params': wd_params, 'weight_decay': args.wd},
                            ], args.lr, momentum=args.momentum, nesterov=True)

    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))

    if args.resume:
        if os.path.isfile(args.load_path):
            print("=> loading checkpoint '{}'".format(args.load_path))
            print("=> saving checkpoint '{}'".format(args.save_path))
            checkpoint = torch.load(args.load_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('checkpoint loaded')
        else:
            pass
    
    print('Start EVAL')    
    prec1 = validate(val_loader, model, criterion)
    fullflops, pruneflops, fullparams, pruneparams = mb2_prune_ratio(model)
    print("Full Flops, Prune Flops, Full Params, Prune Params")
    print(fullflops, pruneflops, fullparams, pruneparams)
    for epoch in range(start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)

        lr = adjust_learning_rate(optimizer, epoch)
        print('learning rate', lr)
        train(train_loader, model, criterion)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print('current epoch: ', epoch)
        print('current top1: ', prec1)
        print('current best top1: ', best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint_dir = args.save_path)
    

    train_loader,  val_loader, n_class = get_dataset_ft(args.dataset, batch_size, args.n_worker,
                                                                       data_root=args.data_root,
                                                                       istrick2=1)
    
    for epoch in range(10):
        lr = 5e-4
        train(train_loader, model, criterion)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print('current epoch: ', epoch)
        print('current top1: ', prec1)
        print('current best top1: ', best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint_dir = args.save_path)

