import argparse
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fixmatch.dataset.cifar import get_cifar10, get_cifar100
from fixmatch.models.confucius_model import *
from fixmatch.utils import AverageMeter, accuracy
from logger import Logger

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--set', default='default', type=str)
    parser.add_argument('--version', default='0', type=str)
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--epochs', default=256, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--k-img', default=65536, type=int,
                        help='number of labeled examples')
    # parser.add_argument('--out', default=str(logger.exp_dir / 'result'),
    #                     help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true', default=True,
                        help="don't use progress bar")
    parser.add_argument('--no-semi-confucius', action='store_true',
                        help="train confucius with unlabelled data")
    parser.add_argument('--expose', default=2, help="control level of confucius expose")

    args = parser.parse_args()
    best_acc = 0
    logger = Logger(args.set, args.version)
    args.out = str(logger.exp_dir / 'result')

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes,
                                            expose=args.expose)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)

        logger.log_print("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))

        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        if args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 10
        if args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    logger.log_print(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.log_print(dict(args._get_kwargs()))

    if args.seed != -1:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        logger.make_writer()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        '../../data/cifarfm', args.num_labeled, args.k_img, args.k_img * args.mu)

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    if args.expose == 1:
        confucius = Confucius(10, 128, 32)
    elif args.expose == 2:
        confucius = WideResConfucius(model)
    else:
        raise Exception()
    confucius.to(args.device)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.iteration = args.k_img // args.batch_size // args.world_size
    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup * args.iteration, args.total_steps)

    confucius_optim = optim.Adam(confucius.parameters())
    # no scheduler. not sure if SGD is the matter, Adam needs no scheduler # TODO ablation

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay, device)

    start_epoch = 0

    if args.resume:
        logger.log_print("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    #
    # if args.amp:
    #     from apex import amp
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.log_print("***** Running training *****")
    logger.log_print(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.log_print(f"  Num Epochs = {args.epochs}")
    logger.log_print(f"  Batch size per GPU = {args.batch_size}")
    logger.log_print(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.log_print(f"  Total optimization steps = {args.total_steps}")

    test_accs = []
    model.zero_grad()
    model = nn.DataParallel(model)
    confucius = nn.DataParallel(confucius)

    if args.use_ema:
        test_model = ema_model.ema
    else:
        test_model = model

    # logger.log_print("Dry run")
    # test_loss, test_acc = ttest(args, test_loader, test_model, epoch, logger)

    for epoch in range(start_epoch, args.epochs):

        train_loss, train_loss_x, train_loss_u, mask_prob, epoch_time = train_one_epoch(
            args, labeled_trainloader, unlabeled_trainloader,
            model, optimizer, confucius, confucius_optim, ema_model, scheduler, epoch, logger)

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc, test_acc_5 = ttest(args, test_loader, test_model, 0, logger)

        # if args.local_rank in [-1, 0]:
        #     logger.auto_log("train loss", epoch=epoch, iter=len(labeled_trainloader),
        #                     loss=train_loss, loss_x=train_loss_x,
        #                     loss_u=train_loss_u, mask_prob=mask_prob, epoch_time=epoch_time)
        #     logger.auto_log("test loss", epoch=epoch,
        #                     accu=test_acc, loss=test_loss)
        #
        # writer.add_scalar('train/1.train_loss', train_loss, epoch)
        # writer.add_scalar('train/2.train_loss_x', train_loss_x, epoch)
        # writer.add_scalar('train/3.train_loss_u', train_loss_u, epoch)
        # writer.add_scalar('train/4.mask', mask_prob, epoch)
        # writer.add_scalar('test/1.test_acc', test_acc, epoch)
        # writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if args.local_rank in [-1, 0]:
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

        test_accs.append(test_acc)
        if args.local_rank in [-1, 0]:
            logger.auto_log("test", tensorboard=True, epoch=epoch,
                            loss=test_loss, top_1_acc=test_acc, top_5_acc=test_acc_5,
                            best_top_1=best_acc, mean_top_1=np.mean(test_accs[-20:]))
            logger.writer.add_scalar('test/1.test_acc', float(test_acc), epoch)
            logger.writer.add_scalar('test/2.test_loss', float(test_loss), epoch)

        # test_accs.append(test_acc)
        # logger.auto_log("test accuracy", epoch=epoch, best_top_1=best_acc,
        #                 mean_top_1=np.mean(test_accs[-20:]))

    if args.local_rank in [-1, 0]:
        logger.writer.close()


def train_one_epoch(args, labeled_trainloader, unlabeled_trainloader,
                    model, optimizer, confucius, confucius_optim, ema_model, scheduler, epoch, logger):
    if args.amp:
        from apex import amp
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    avg_mask = AverageMeter()
    avg_mask_conf = AverageMeter()
    avg_conf_loss = AverageMeter()
    avg_true_confidence, avg_false_confidence, avg_true_semi_confidence, avg_false_semi_confidence = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    epoch_start = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration),
                     disable=args.local_rank not in [-1, 0])

    train_loader = zip(labeled_trainloader, unlabeled_trainloader)
    model.train()
    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        inputs_x, targets_x = data_x
        (inputs_u_w, inputs_u_s), _ = data_u
        data_time.update(time.time() - end)
        batch_size = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
        targets_x = targets_x.to(args.device)
        if args.expose ==2:
            logits, e1, e3 = model(inputs)
            confidence_logits = confucius(logits.detach(), e1.detach(), e3.detach())
        else:
            logits, e = model(inputs)
            confidence_logits = confucius(logits.detach(), e.detach())
        conf_logits_x = confidence_logits[:batch_size]
        uconf = confidence_logits[batch_size:]
        conf_logits_u_w, conf_logits_u_s = uconf.chunk(2)

        conf_x, conf_u_w, conf_u_s = torch.sigmoid(conf_logits_x), torch.sigmoid(conf_logits_u_w), \
                                     torch.sigmoid(conf_logits_u_s)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

        # compute the labelled loss
        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        # generate the pseudo labels
        pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
        max_probs, pseudo_label = torch.max(pseudo_label, dim=-1)
        # use confidence instead to mask the pseudo-label below threshold
        # I'm guessing that the mask does not have gradient, the masked has
        mask = conf_u_w.ge(args.threshold).float()
        # mask = max_probs.ge(args.threshold).float()
        avg_mask_conf.update(conf_u_w.mean().item())
        Lu = (F.cross_entropy(logits_u_s, pseudo_label,
                              reduction='none') * mask).mean()

        loss = Lx + args.lambda_u * Lu

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())

        # mem, util=report_mem_and_util()
        # gpu_mem.update(mem)
        # gpu_util.update(util)

        optimizer.step()
        scheduler.step()
        if args.use_ema:
            ema_model.update(model)
        model.zero_grad()
        # I'm guessing that the
        confucius.zero_grad()

        ################# main model finishes, start confidence model ################
        # generate label for confucius, we use precision as usual
        # labelled prediction
        x_pred = torch.softmax(logits_x.detach(), dim=-1)
        x_pred_prob, x_pred = x_pred.max(dim=1, keepdim=True)
        # TODO high bar prediction
        # the label: x_pred needs to be correct
        # x_pred_prob also needs to be more than 0.5
        x_accu = x_pred.eq(targets_x.view_as(x_pred))
        x_semi_target=x_pred_prob.ge(args.threshold)*x_accu
        true_confidence = (conf_x * x_semi_target).mean().item()
        false_confidence = (conf_x * ~x_semi_target).mean().item()

        x_conf_loss = F.binary_cross_entropy_with_logits(conf_logits_x, x_semi_target.float())

        if args.no_semi_confucius:
            u_s_conf_loss = 0
        else:
            # will the model predict the strongly augmented labels correctly as well?
            u_s_pred = torch.softmax(logits_u_s.detach(), dim=-1)
            u_s_pred_prob, u_s_pred = u_s_pred.max(dim=1, keepdim=True)
            u_s_accu = u_s_pred.eq(pseudo_label.view_as(u_s_pred))
            u_s_semi_target = u_s_pred_prob.ge(args.threshold)*u_s_accu
            if mask.sum()==0:
                true_semi_confidence, false_semi_confidence=0,0
            else:
                true_semi_confidence = (((u_s_semi_target * conf_u_s) * mask).sum() / mask.sum()).item()
                false_semi_confidence = (((~u_s_semi_target * conf_u_s) * mask).sum() / mask.sum()).item()
            u_s_conf_loss = (F.binary_cross_entropy_with_logits(
                conf_u_s, u_s_semi_target.float(), reduction="none") * mask).mean()

        conf_loss = x_conf_loss + args.lambda_u * u_s_conf_loss

        conf_loss.backward()
        avg_conf_loss.update(conf_loss.item())
        avg_true_confidence.update(true_confidence)
        avg_false_confidence.update(false_confidence)
        avg_true_semi_confidence.update(true_semi_confidence)
        avg_false_semi_confidence.update(false_semi_confidence)

        confucius_optim.step()
        model.zero_grad()
        confucius.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        avg_mask.update(mask.mean().item())

        if not args.no_progress:
            p_bar.set_description(
                f"Train Epoch: {epoch}/{args.epochs:4}. Iter: {batch_idx:4}/{args.iteration:4}. "
                f"LR: {scheduler.get_last_lr()[0]:.6f}. Data: {data_time:.3f}s. "
                f"Batch: {batch_time:.3f}s. Loss: {losses:.4f}. Loss_x: {losses_x:.4f}. \n"
                f"Loss_u: {losses_u:.4f}. Mask prob: {avg_mask:.4f}. Mask conf: {avg_mask_conf:.4f}."
                f"Conf loss:{avg_conf_loss:.4f}. True labelled confidence: {avg_true_confidence:4f}. "
                f"False labelled confidence: {avg_false_confidence:.4f}. True semi confidence: {avg_true_semi_confidence:4.f}."
                f"False semi confidence: {avg_false_semi_confidence:.4f}.")
            p_bar.update()
        else:
            print_intvl = int(len(labeled_trainloader) / 10)
            if batch_idx % print_intvl == 0 or batch_idx == len(labeled_trainloader) - 1:
                logger.auto_log("train", tensorboard=True, epoch=epoch, iter=batch_idx,
                                loss=losses, loss_x=losses_x,
                                loss_u=losses_u, mask_prob=avg_mask,
                                mask_conf=avg_mask_conf, conf_loss=avg_conf_loss,
                                true_confidence=avg_true_confidence,
                                false_confidence=avg_false_confidence, true_semi_conf=avg_true_semi_confidence,
                                false_semi_conf=avg_false_semi_confidence)
    if not args.no_progress:
        p_bar.close()

    if args.local_rank in [-1, 0]:
        logger.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        logger.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        logger.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        logger.writer.add_scalar('train/4.mask', avg_mask.avg, epoch)

    return losses.avg, losses_x.avg, losses_u.avg, avg_mask.avg, time.time() - epoch_start


def ttest(args, test_loader, model, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            if args.expose == 2:
                outputs, _, _ = model(inputs)
            else:
                outputs, _ = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. "
                    "top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()
    # logger.auto_log("top accuracy", epoch=epoch,
    #                 top_1_accu=top1.avg, top_5_acc=top5.avg)
    # logger.log_print("top-1 acc: {:.2f}".format(top1.avg))
    # logger.log_print("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg, top5.avg


class ModelEMA(object):
    def __init__(self, args, model, decay, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        self.wd = args.lr * args.wdecay
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        assert isinstance(checkpoint, dict)
        if 'ema_state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['ema_state_dict'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                # weight decay
                if 'bn' not in k:
                    msd[k] = msd[k] * (1. - self.wd)


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
