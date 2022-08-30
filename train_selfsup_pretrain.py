# Copyright (c) Markus Hiller and Rongkai Ma -- 2022
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
The code in this file is heavily based on the original iBOT code for self-supervised pretraining, available at
https://github.com/bytedance/ibot/blob/main/main_ibot.py ;
Changes include:
Replaced dataloader with several new ones for FSL tasks, as well as appropriate sampling methods;
Added flexibility for choice of dataset via cmdline argument (parser);
Added support for different image sizes via cmdline argument (parser); Likewise for local crop size;
Added wandb for logging, removed tensorboard; Added several additional logging methods, arguments to txt file, etc.;
Modifications of checkpoint save logic / file management;
Modified multi-gpu settings (also see utils file)
Added hashed name generation, ...
"""
#
import os
import argparse
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import transforms
from models.head import iBOTHead
from loader import DatasetMask

####################################################################
USE_WANDB = False

if USE_WANDB:
    import wandb
    # Note: Make sure to specify your username for correct logging
    WANDB_USER = 'username'
####################################################################


def get_args_parser():
    parser = argparse.ArgumentParser('pretrain_FewTURE', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'deit_tiny', 'deit_small', 'swin_tiny'],
        help="""Name of architecture to train. We use vit_small and swin_tiny as default architectures in our paper.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values could lead to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base), but is used in swin to compute the 
        predictor size (8*patch_size vs. pred_size = patch_size in ViT) -- so be wary!. 
        If <16, disabling mixed precision training is recommended (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transformer and is ignored for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Whether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Default: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this parameter to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly chosen for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be identical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs and Swin.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--image_size', type=int, default=224,
        help="""Size of the squared input images, 224 for imagenet-style.
        Note that the --local_crops_size must be chosen appropriately considering this image size!""")
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--local_crops_size', type=int, default=96,
        help="""Crop region size of local views to generate. MUST be chosen wrt image size! Default for ImageNet-based 
        data is 96 (input image 224), but has to be adapted for other FSL datasets of input size 84!""")

    # Dataset related parameters
    parser.add_argument('--dataset', default='miniimagenet', type=str,
                        choices=['miniimagenet', 'tieredimagenet', 'fc100', 'cifar_fs'],
                        help='Please specify the name of the dataset to be used for training.')
    parser.add_argument('--data_path', default=None, type=str,
                        help='Please specify path to the root folder containing the training dataset(s). If dataset '
                             'cannot be loaded, check naming convention of the folder in the corresponding dataloader.')

    # Misc
    parser.add_argument('--output_dir', default="", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser


def pretrain_fewture(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    if utils.is_main_process():
        if args.use_wandb:  # Log main process stats
            run = wandb.init(config=args, project="pretrain_FewTURE", entity=WANDB_USER)
        else:
            run = None
        with (Path(args.output_dir) / "args.txt").open("w") as f:
            f.write(json.dumps(args.__dict__, indent=4))
    else:
        run = None

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # Run benchmarking to find the fastest kernels for computation:
    # (trade-off between benchmarking-based increase in time and decreased run/train time afterwards)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # Get transforms for data augmentation for FewTURE pretraining (slightly extended compared to original iBOT)
    transform = DataAugmentationFewTURE(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
        args.local_crops_size,
        args.image_size,
        args.dataset
    )

    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size

    DataSetObj = set_up_dataset(args)
    dataset = DatasetMask(
        DataSetCl=DataSetObj,
        setname='train', args=args,
        train_augmentation=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Using {args.dataset} to run training for this experiment.")
    print(f"Data successfully loaded: There are {len(dataset)} training images available.")

    # ============ Building student and teacher networks ... ============
    # deit and vit architectures are identical: change to 'vit' in case 'deit' is used as input arguments (easier)
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierarchical nature (i.e. swin_tiny, ...)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, ...)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    # otherwise, we could check if the architecture is in torchvision models;
    # Note: Not used in our current paper, thus not supported (yet)
    # elif args.arch in torchvision_models.__dict__.keys():
    #     student = torchvision_models.__dict__[args.arch]()
    #     teacher = torchvision_models.__dict__[args.arch]()
    #     embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False) if \
            'swin' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False) if \
        'swin' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher have been built: they are both {args.arch} networks.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    # ============ preparing optimiser ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # Default used in our paper with both ViT and Swin
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler -- Not used in paper
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # Not used in our paper
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule using batch size
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print(f"Loss, optimiser and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print(f"Starting self-supervised pretraining from epoch {start_epoch}!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of self-supervised pretraining using MIM ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, run)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if (args.saveckp_freq and (epoch % args.saveckp_freq == 0)) or (epoch == args.epochs - 1):
            torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch + 1}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # Log the averaged log_stats also to wandb, can later be plotted over epochs
            if args.use_wandb:
                run.log(log_stats)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Store final model(s) / save_dict as artifact on wandb if active
    if utils.is_main_process() and args.use_wandb:
        name_str = args.dataset + f'_{args.image_size}-' + args.arch + f'-outdim_{args.out_dim}' + \
                   f'-bs_total_{args.batch_size_total}'
        model_config = args.__dict__
        mdl_art = wandb.Artifact(name=name_str, type="model",
                                 description="Models after finishing the self-supervised training",
                                 metadata=model_config)
        try:
            # In case we reach this point in 'normal' training, we will have the 'save_dict' available
            with mdl_art.new_file(f'checkpoint_ep{args.epochs}.pth', 'wb') as file:
                torch.save(save_dict, file)
        except:
            print("Training had already finished for specified number of epochs. "
                  "Loading most-recent model from disk and uploading model to wandb.")
            mdl_final = torch.load(os.path.join(args.output_dir, f'checkpoint.pth'))
            ep = mdl_final['epoch']
            with mdl_art.new_file(f'checkpoint_ep{ep}.pth', 'wb') as file:
                torch.save(mdl_final, file)
        with (Path(args.output_dir) / "log.txt").open("r") as f:
            with mdl_art.new_file('logs.txt', 'w') as file:
                file.write(f.read())
        time_str = f'Total training time spent for {args.epochs} epochs: {total_time_str}!'
        with mdl_art.new_file('time_info.txt', 'w') as file:
            file.write(time_str)
        # Actually upload to wandb
        run.log_artifact(mdl_art)
        print("Artifact uploaded to wandb server.")

        run.finish()
        print("Wrapping up wandb logging on main process -- Finished run.")


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, run=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch + 1, args.epochs)
    real_labels, pred_labels = [], []
    for it, (images, labels, masks) in enumerate(metric_logger.log_every(data_loader, 10, header, epoch, run)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(images[:args.global_crops_number])
            student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])
            
            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop('loss')

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            names_q, params_q, names_k, params_k = [], [], [], []
            for name_q, param_q in student.module.named_parameters():
                names_q.append(name_q)
                params_q.append(param_q)
            for name_k, param_k in teacher_without_ddp.named_parameters():
                names_k.append(name_k)
                params_k.append(param_k)
            names_common = list(set(names_q) & set(names_k))
            params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
            params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore = utils.eval_pred(real_labels, pred_labels)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("NMI: {}, ARI: {}, F: {}".format(nmi, ari, fscore))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore})
    return return_dict


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Apply a warm-up for the teacher temperature to avoid instabilities
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)


class DataAugmentationFewTURE(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number,
                 local_crops_size, image_size, dataset):
        assert image_size > local_crops_size, "Error: local crops for student are larger then input image! " \
                                              "Please check."
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        # Regular case for miniImagenet, tieredImagenet etc.
        if not dataset == 'cifar_fs':
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:  # common for cifar_fs in other works
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
            ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops (small)
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def set_up_dataset(args):
    # Datasets and corresponding number of classes
    if args.dataset == 'miniimagenet':
        # (Vinyals et al., 2016), (Ravi & Larochelle, 2017)
        # train num_class = 64
        from datasets.dataloaders.miniimagenet.miniimagenet import MiniImageNet as dataset
    elif args.dataset == 'tieredimagenet':
        # (Ren et al., 2018)
        # train num_class = 351
        from datasets.dataloaders.tieredimagenet.tieredimagenet import tieredImageNet as dataset
    elif args.dataset == 'fc100':
        # (Oreshkin et al., 2018) Fewshot-CIFAR 100 -- orig. images 32x32
        # train num_class = 60
        from datasets.dataloaders.fc100.fc100 import DatasetLoader as dataset
    elif args.dataset == 'cifar_fs':
        # (Bertinetto et al., 2018) CIFAR-FS (100) -- orig. images 32x32
        # train num_class = 64
        from datasets.dataloaders.cifar_fs.cifar_fs import DatasetLoader as dataset
    else:
        raise ValueError('Unknown dataset. Please check your selection!')
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pretrain_FewTURE', parents=[get_args_parser()])
    args = parser.parse_args()
    # Set potential debug arguments
    # if dbg_args is not None:
    #     set_debug_settings(args, dbg_args)

    if args.data_path is None:
        raise ValueError("No path to dataset provided. Please do so to run experiments!")

    args.__dict__.update({'use_wandb': True} if USE_WANDB else {'use_wandb': False})

    if args.output_dir == '':
        args.output_dir = os.path.join(utils.get_base_path(), 'fewture_pretrain_saves')

    try:
        args.__dict__.update({'batch_size_total': int(os.environ['WORLD_SIZE']) * args.batch_size_per_gpu})
    except:
        args.__dict__.update({'batch_size_total': args.batch_size_per_gpu * utils.get_world_size()})

    # Creating hash to uniquely identify parameter setting for run, but w/o elements that are non-essential and
    # might change due to moving the dataset, using different server, etc.
    non_essential_keys = ['local_rank', 'dist_url', 'num_workers', 'saveckp_freq', 'output_dir', 'data_path']
    exp_hash = utils.get_hash_from_args(args, non_essential_keys)
    args.output_dir = os.path.join(args.output_dir, args.dataset + f'_{args.image_size}', args.arch,
                                   f'outdim_{args.out_dim}', f'bs_{args.batch_size_total}', f'ep{args.epochs}',
                                   exp_hash)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Start pretraining
    pretrain_fewture(args)

    print('Self-supervised pretraining has finished! Happy meta fine-tuning or evaluating!')
    print("If you found this helpful, consider giving us a star on github (https://github.com/mrkshllr/FewTURE)"
          " and cite our work: https://arxiv.org/pdf/2206.07267.pdf")
