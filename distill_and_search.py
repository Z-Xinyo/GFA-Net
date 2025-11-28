import argparse
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from gfa_net.st_encoder_teacher import PretrainingEncoder
from gfa_net.st_encoder_downstream import DownstreamEncoder
from gfa_net.builder_distill import SEED

from torch.utils.tensorboard import SummaryWriter

from dataset import get_distill_set_intra
from dataset import get_distill_training_set
from dataset import get_distill_validation_set

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
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
parser.add_argument('--lr', '--learning-rate', default=2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[350], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--schedule-test', default=[50, 70], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--checkpoint-path', default='./checkpoints', type=str)
parser.add_argument('--skeleton-representation', type=str,
                    help='input skeleton-representation  for self supervised training (joint or motion or bone)')
parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='traiining protocol cross_view/cross_subject/cross_setup')

parser.add_argument('--hico-dim', default=2048, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--hico-k', default=2048, type=int,
                    help='queue size; number of negative keys (default: 16384)')
parser.add_argument('--hico-t', default=0.07, type=float,
                    help='student temp')
parser.add_argument('--hico-temp', default=1e-3, type=float,
                    help='teacher softmax temperature')


# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('Weight initial finished!')


def load_moco_encoder_q(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("message", msg)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def load_moco_encoder_student(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_student') and not k.startswith('encoder_student.fc'):
                # remove prefix
                state_dict[k[len("encoder_student."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("message", msg)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))



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

    # create model

    # training dataset
    from options import options_distill as options
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'pku_part1' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part1_cross_subject()
    elif args.pre_dataset == 'pku_part2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part2_cross_subject()

    opts.train_feeder_args['input_representation'] = args.skeleton_representation
    opts.train_feeder_args_test['input_representation'] = args.skeleton_representation
    opts.test_feeder_args['input_representation'] = args.skeleton_representation

    initial_individual = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 0, 1, 1, 2, 2]
    #initial_individual = np.repeat(np.arange(4), 16)
    cooperative_points = {0: [1, 20], 1: [0, 20], 20: [0, 1], 2: [3], 3: [2],
                          4: [5], 5: [4], 6: [7], 7: [6], 8: [9], 9: [8], 10: [11], 11: [10],
                          12: [13, 14, 15], 13: [12, 14, 15], 14: [12, 13, 15], 15: [12, 13, 14],
                          16: [17, 18, 19], 17: [16, 18, 19], 18: [16, 17, 19], 19: [16, 17, 18],
                          21: [22], 22: [21], 23: [24], 24: [23]}
    new_number = 10

    split = SpatialGroupMutator(initial_individual, cooperative_points, new_number)
    #split = TemporalGroupMutator(initial_individual, new_number)
    group_list = split.generate_mutations()
    print(group_list)

    teacher_model = PretrainingEncoder(**opts.encoder_args_teacher)
    load_moco_encoder_q(teacher_model, args.pretrained)

    model_list = []
    for group in group_list:
        net = SEED(teacher_model, opts.encoder_args, group, args.hico_dim, args.hico_k, args.hico_t, args.hico_temp)
        model_list.append(net)
    #model = SEED(opts.encoder_args_teacher, opts.encoder_args, args.pretrained, args.hico_dim, args.hico_k,args.hico_t,args.hico_temp)

    # 将列表中的每个模型移动到 GPU
    # model_list = [model.cuda() for model in model_list]
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizers = [torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
                  for model in model_list]
    #if True:
    #    for parm in optimizer.param_groups:
    #        print("optimize parameters lr", parm['lr'])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    ## Data loading code

    train_dataset = get_distill_set_intra(opts)
    #train_dataset_2 = get_distill_training_set(opts)
    #val_dataset = get_distill_validation_set(opts)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    #train_loader_2 = torch.utils.data.DataLoader(
    #    train_dataset_2, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #    num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    #val_loader = torch.utils.data.DataLoader(val_dataset,
    #    batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=True, drop_last=False)

    writer = SummaryWriter(args.checkpoint_path)

    #acc_list = []
    for i, (model, optimizer, group) in enumerate(zip(model_list, optimizers, group_list)):
      best_acc1 = 0
      for epoch in range(args.start_epoch, args.epochs):

          model = model.cuda()

          adjust_learning_rate(optimizer, epoch, args)

          # train for one epoch
          loss = train(train_loader, model, criterion, optimizer, epoch, args)
          writer.add_scalar('train_loss', loss.avg, global_step=epoch)

          if epoch % 450 == 0 and epoch != 0:
              save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
              }, is_best=False, filename=args.checkpoint_path + '/No.{:04d}checkpoint_{:04d}.pth.tar'.format(i, epoch))

              #model_test = DownstreamEncoder(group, **opts.encoder_args)
              #model_test = model_test.cuda()
              #optimizer_test = torch.optim.SGD(model_test.parameters(), 0.1,
              #                                   momentum=0.9,
              #                                   weight_decay=0.0001)

              #for name, param in model_test.named_parameters():
              #    if name not in ['fc.weight', 'fc.bias']:
              #        param.requires_grad = False
              #    else:
              #        print('params', name)
              # init the fc layer
              #model_test.fc.weight.data.normal_(mean=0.0, std=0.01)
              #model_test.fc.bias.data.zero_()

              #load_moco_encoder_student(model_test,
              #                          args.checkpoint_path + '/No.{:04d}checkpoint_{:04d}.pth.tar'.format(i, epoch))

              #for epoch in range(60):

              #    adjust_learning_rate_class(optimizer_test, epoch, args)

                  # train for one epoch
              #    train_test(train_loader_2, model_test, criterion, optimizer_test, epoch, args)

                  # evaluate on validation set
              #    acc1 = validate(val_loader, model_test, criterion, args)

                  # remember best acc@1 and save checkpoint
              #    is_best = acc1 > best_acc1
              #    if is_best:
              #        print("found new best accuracy:= ", acc1)
              #        best_acc1 = max(acc1, best_acc1)

              #print("group", group)
              #print("Final  best accuracy", best_acc1)
      #acc_list.append(best_acc1)
      #print(acc_list)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch, optimizer.param_groups[0]['lr']))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.train()

    end = time.time()
    for i, (x_teacher, x) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_teacher = x_teacher.float().cuda(non_blocking=True)
        x = x.float().cuda(non_blocking=True)

        # compute output
        output1, output2, output3, output4, target1, target2, target3, target4 = model(x_teacher, x)

        loss = criterion(output1, target1) + criterion(output2, target2) + \
                criterion(output3, target3) + criterion(output4, target4)

        losses.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses


def train_test(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = x.float().cuda(non_blocking=True)
        # qp_input = qp_input.float().cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(x)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, target) in enumerate(val_loader):

            x = x.float().cuda(non_blocking=True)
            #qp_input = qp_input.float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(x)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def mse(x, y):
    loss = (x - y) ** 2
    loss = loss.sum(0).mean(0)

    return loss

def kl(student_output, teacher_output, temperature=1.0):
    student_log_prob = F.log_softmax(student_output / temperature, dim=1)
    teacher_prob = F.softmax(teacher_output / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='sum') * (temperature * temperature)

def cross(student_output, teacher_output, temperature=1.0):
    teacher_prob = F.softmax(teacher_output / temperature, dim=1)
    student_log_prob = F.log_softmax(student_output / temperature, dim=1)
    return -(teacher_prob * student_log_prob).sum(dim=1).mean()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check_encoder_q(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder_q.' + k
        # k_pre = 'encoder_q.' + k[len('module.'):] \
        #     if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

class SpatialGroupMutator:
    def __init__(self, initial_individual, cooperative_points, new_number,  num_points=25, max_groups=5):
        self.initial_individual = initial_individual
        self.cooperative_points = cooperative_points
        self.num_points = num_points
        self.max_groups = max_groups
        self.new_number = new_number

    def mutate_individual_with_cooperation(self, individual):
        new_individual = individual.copy()

        # 随机选择一个要变异的骨骼点
        index = random.randint(0, self.num_points - 1)
        skeleton_group = new_individual[index]
        new_group = random.randint(0, self.max_groups - 1)
        while new_group == skeleton_group:
            new_group = random.randint(0, self.max_groups - 1)

        # 变异选中的骨骼点
        new_individual[index] = new_group

        # 对协同运动的骨骼点进行相同的变异
        if index in self.cooperative_points:
            for coop_index in self.cooperative_points[index]:
                new_individual[coop_index] = new_group

        return new_individual

    def generate_mutations(self):
        group_list = []  # 首先存储初始个体
        for _ in range(self.new_number):  # 生成10个新的变异个体
            mutated_individual = self.mutate_individual_with_cooperation(self.initial_individual)
            group_list.append(mutated_individual)

        return group_list

class TemporalGroupMutator:
    def __init__(self, initial_individual, new_number, array_length=64, group_size=4):
        self.array_length = array_length
        self.group_size = group_size
        self.new_number = new_number
        self.initial_individual = initial_individual

    def mutate_individual_with_cooperation(self, individual):

        new_individual = individual.copy()

        # 随机选择一个要
        frame_index = random.randint(0, self.array_length - 1)

        # 确定该索引所在的weizhi
        group_index = frame_index // self.group_size
        start_index = group_index * 4

        skeleton_group = new_individual[frame_index]
        new_group = random.randint(0, self.group_size - 1)
        while new_group == skeleton_group:
            new_group = random.randint(0, self.group_size - 1)

        # 更新该组的所有值
        new_individual[start_index:start_index + self.group_size] = new_group

        #print(f"Changed group from index {group_start_index} to {group_end_index-1} to {new_value}")
        return new_individual

    def generate_mutations(self):
        # 生成多个不同的数组
        group_list = [self.initial_individual.copy()]  # 首先存储初始个体
        for _ in range(self.new_number):  # 生成10个新的变异个体
            mutated_individual = self.mutate_individual_with_cooperation(self.initial_individual)
            group_list.append(mutated_individual)

        return group_list


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
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_class(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule_test:
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
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
