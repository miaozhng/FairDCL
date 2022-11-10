# MoCo related codes are adapted from https://github.com/facebookresearch/moco
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

import moco.loader
import moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--sensitive_id1', default='', type=str, metavar='PATH')
parser.add_argument('--sensitive_id2', default='', type=str, metavar='PATH')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=2048, type=int,
                    help='queue size; number of negative keys (default: 2048)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


WIDTHS = [128, 64, 32, 16]
HEIGHTS = [128, 64, 32, 16]
CHANNELS = [2050, 1026, 514, 258]
REP_NUM = 4
INNER_ROUND = 20
ALPHA = 0.5


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def compute_cat(reps, ids, sensitive1, sensitive2):
    rep_num = len(reps)
    new_batch_joint = []*rep_num
    new_batch_marginal = []*rep_num
    for batch in range(reps[0].shape[0]):
        embeddings = []
        if(ids[batch] in sensitive1):
            for rep_index in range(rep_num):
                embeddings.append(torch.cat((torch.ones([1, WIDTHS[rep_index], HEIGHTS[rep_index]]), torch.zeros([1, WIDTHS[rep_index], HEIGHTS[rep_index]])), 0))
        elif(ids[batch] in sensitive2):
            for rep_index in range(rep_num):
                embeddings.append(torch.cat((torch.zeros([1, WIDTHS[rep_index], HEIGHTS[rep_index]]), torch.ones([1, WIDTHS[rep_index], HEIGHTS[rep_index]])), 0))
        joint = [torch.cat((reps[i][batch], embeddings[i].cuda()), 0) for i in range(rep_num)]

        shuffle_embeddings = []
        for rep_index in range(rep_num):
                shuffle_embeddings.append(torch.permute(F.one_hot(torch.randint(low=0, high=2, size=(WIDTHS[rep_index], HEIGHTS[rep_index])), num_classes=2), (2,0,1)))
        marginal = [torch.cat((reps[i][batch], shuffle_embeddings[i].type(torch.float32).cuda()), 0) for i in range(rep_num)]

        for rep_index in range(rep_num):
            new_batch_joint[rep_index].append(torch.unsqueeze(joint[rep_index], 0))
            new_batch_marginal[rep_index].append(torch.unsqueeze(marginal[rep_index], 0))

    for i in range(rep_num):
        new_batch_joint[i] = torch.cat(new_batch_joint[i], 0)
        new_batch_marginal[i] = torch.cat(new_batch_marginal[i], 0)

    new_batch = [torch.cat([new_batch_joint[i], new_batch_marginal[i]], 0) for i in range(rep_num)]
    return new_batch
    


def main_worker(gpu, ngpus_per_node, args):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # # initialize the process group
    # dist.init_process_group("gloo", rank=0, world_size=1)


    # create model
    print("=> creating model '{}'".format(args.arch))

    resnet50 = models.__dict__[args.arch](pretrained=True)
    resnet50.fc = nn.Linear(2048, args.moco_dim)
    resnet50.train()

    resnet50_2 = models.__dict__[args.arch](pretrained=True)
    resnet50_2.fc = nn.Linear(2048, args.moco_dim)
    resnet50_2.train()

    model = moco.builder.MoCo(
        resnet50, resnet50_2, 
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)


    sensitive1 = set(np.load(arg.sensitive_id1))
    sensitive2 = set(np.load(arg.sensitive_id2))


    discriminators = []
    for i in range(REP_NUM):
        discriminators.append(moco.builder.Discriminator(input_size=CHANNELS[i], hidden_size=int(CHANNELS[i]/10)).cuda())



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    optimizer_D = [torch.optim.Adam(discriminators[i].parameters(), args.lr,
                                weight_decay=args.weight_decay) for i in range(REP_NUM)]


    

    # Data loading code
    traindir = os.path.join(args.data, '')
    normalize = transforms.Normalize(mean=[123.675, 116.28, 103.53],
                                     std=[58.395, 57.12, 57.375])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            # transforms.RandomResizedCrop(1024),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            # transforms.RandomResizedCrop(512),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]


    train_dataset = moco.loader.TwoCropsTransform(traindir, transforms.Compose(augmentation))
    train_dataset_d= moco.loader.TwoCropsTransform(traindir, transforms.Compose(augmentation))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)

    train_loader_d= torch.utils.data.DataLoader(
        train_dataset_mine, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, train_loader_d train_dataset_d, model, discriminators, criterion, optimizer_D, epoch, sensitive1, sensitive2, args)

        # Saving checkpoints for encoder and discriminators
        if epoch % 1 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
            for i in range(REP_NUM):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': discriminators[i].state_dict(),
                    'optimizer' : optimizer_D[i].state_dict(),
                }, is_best=False, filename='checkpoint_discriminator{}_{:04d}.pth.tar'.format(i, epoch))
    

def train(train_loader, train_loader_d train_dataset_d, model, discriminators, criterion, optimizer_D, epoch, sensitive1, sensitive2, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    discriminator_train = []
    discriminator_infer= []
    moco_loss = []

    # switch to train mode
    model.train()
    for i in range(len(REP_NUM)):
        discriminators[i].train()

    end = time.time()
    for i, (images, image_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)


        running_loss, running_mst = 0, 0
        for j, (images_d, image_ids_d) in enumerate(train_loader_d):
        
            images_d[0] = images_d[0].cuda(args.gpu, non_blocking=True)
            images_d[1] = images_d[1].cuda(args.gpu, non_blocking=True)
            if (j > INNER_ROUND):
                train_loader_d = torch.utils.data.DataLoader(train_dataset_d, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
                break
            q, q_l1, q_l2, q_l3, q_l4 = model(im_q=images_d[0], im_k=images_d[1], training_discriminator=True)
            
            new_batch = compute_cat([q_l1, q_l2, q_l3, q_l4], image_ids_d, sensitive1, sensitive2)

            
            for i in range(len(REP_NUM)):
                loss_m, mst_m = discriminators[i](new_batch[i])
                running_loss += loss_m.item()
                running_mst += mst_m.item()

                optimizer_D.zero_grad()
                loss_m.backward()
                optimizer_D.step()
            

        discriminator_train.append(running_mst/INNER_ROUND)
        print("MI mst from training is: " + str(running_mst/INNER_ROUND))


        # compute output
        output, target, backbone_feature, q_l1, q_l2, q_l3, q_l4 = model(im_q=images[0], im_k=images[1], training_discriminator=False)

        new_batch = compute_cat([q_l1, q_l2, q_l3, q_l4], image_ids_d, sensitive1, sensitive2)

        loss_M = []
        for i in range(len(REP_NUM)):
            loss_m, mst_m = discriminators[i](new_batch[i])
            loss_M.append(loss_m)
            discriminator_infer += mst_m.item()

        print("MI mst from inference is: " + str(discriminator_infer[-1]))

        loss = criterion(output, target)
        moco_loss.append(loss.item())
        print("Moco loss is: " + str(loss.item()))
        loss = loss - ALPHA*sum(loss_M)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    # Record mutual information measured during training
    np.save(str(epoch)+'_discriminator_train.npy', discriminator_train)
    np.save(str(epoch)+'_discriminator_infer.npy', discriminator_infer)
    np.save(str(epoch)+'_moco_loss.npy', moco_loss)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
