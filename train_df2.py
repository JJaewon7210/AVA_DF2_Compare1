# Standard library imports
import argparse
import logging
import os
import random
from copy import deepcopy
from threading import Thread
from pathlib import Path

# Related third party imports
import yaml
import torch
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
from timm.optim import create_optimizer_v2

# Local application/library specific imports

from model.YOWO import YOWO_CUSTOM as Model
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    get_latest_run, check_img_size, colorstr, ConfigObject, non_max_suppression
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.torch_utils import intersect_dicts, is_parallel
from utils.plots import read_labelmap, un_normalized_images, plot_batch_image_from_preds
from utils.loss_ava import ComputeLoss
from datasets.yolo_datasets import LoadImagesAndLabels, InfiniteDataLoader
from test_df2 import test_df2

logger = logging.getLogger(__name__)

def main(hyp, opt, device, tb_writer):
    '''
    YOLOv7 Style Trainer
    '''
    save_dir, epochs, batch_size, weights = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights


    # 1. Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'


    # 2. Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)


    # 3. Configure
    plots = True
    cuda = device.type != 'cpu'
    init_seeds(1)
    opt_dict = opt.to_dict()  # data dict
    labelmap_ava, _ = read_labelmap("D:/Data/AVA/annotations/ava_action_list_v2.2.pbtxt")
    labelmap_df2, _ = read_labelmap("D:/Data/DeepFashion2/df2_list.pbtxt")

    # 4. Logging- Doing this before checking the dataset. Might update opt_dict
    opt.hyp = hyp  # add hyperparameters
    run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
    wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, opt_dict)
    opt_dict = wandb_logger.data_dict
    if wandb_logger.wandb:
        weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming


    # 5. Model 
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg=opt).to(device)
        exclude = ['anchor'] if (hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(cfg=opt).to(device)
    
    # 6. Dataset
    imgsz, imgsz_test = [check_img_size(x, 32) for x in opt.img_size]  # verify imgsz are gs-multiples
    dataset_df2 = LoadImagesAndLabels(path=opt.train, img_size=imgsz, batch_size=opt.batch_size, 
                                             augment=True, hyp=hyp, rect=False, image_weights=opt.image_weights,
                                             cache_images=opt.cache_images, single_cls=opt.single_cls, 
                                             stride=32, pad=0.0, prefix='train: ')
    # 6. Dataloader
    loader = torch.utils.data.DataLoader if opt.image_weights else InfiniteDataLoader
    dataloader = loader(dataset_df2, batch_size=opt.batch_size, pin_memory=True, num_workers=opt.workers, collate_fn=LoadImagesAndLabels.collate_fn)
    num_batch = len(dataloader)  # number of batches
    
    # 6. Test dataset, dataloader
    if not opt.notest:
        logger.info('\n====> (For test) Loading LoadImagesAndLabels Dataset')
        testset_df2 = LoadImagesAndLabels(path=opt.val, img_size=imgsz_test, batch_size=opt.batch_size_test, 
                                                augment=False, hyp=opt.hyp, rect=False, image_weights=opt.image_weights,
                                                cache_images=opt.cache_images, single_cls=opt.single_cls, 
                                                stride=32, pad=0.0, prefix='val: ')

        # test loader
        loader = torch.utils.data.DataLoader if opt.image_weights else InfiniteDataLoader
        testloader_df2 = loader(testset_df2, batch_size=opt.batch_size_test, num_workers=opt.workers, drop_last=True,
                                collate_fn=LoadImagesAndLabels.collate_fn)

    else:
        logger.info('\n====> No test section during training model.')
        
    # 7. Optimizer, LR scheduler
    accumulation_steps = 8
    optimizer = create_optimizer_v2(model.parameters(), opt='adam', lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hyp['lrmax'], total_steps=int(epochs * num_batch / accumulation_steps), div_factor=int(hyp['lrmax'] / hyp['lr0']))
    
    # 8. Resume
    start_epoch, best_fitness = 0, 1e+9
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        
        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt
        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict


    # Start training ----------------------------------------------------------------------------------------------------
    scheduler.last_epoch = start_epoch - 1  # do not move
    torch.save(model, wdir / 'init.pt')
    scaler = amp.GradScaler(enabled=True)
    logger.info(f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    LOSS = ComputeLoss(detector_head=model.heads, hyp=hyp)

    # Start epoch ------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):  
        logger.info(('\n' + '%10s' * 6) % ('gpu_mem', 'box', 'obj', 'clo', 'total', 'lr'))
        model.train()
        model._freeze_modules()
        optimizer.zero_grad()
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=num_batch)  # progress bar
        mloss = torch.zeros(3, device='cpu')  # mean losses
        mtotal_loss = torch.zeros(1, device='cpu') # mean total_loss (sum of losses)
        
        # Update image weights
        if opt.image_weights:
            maps = np.zeros(opt.nc)
            cw = labels_to_class_weights(dataset_df2.labels, opt.nc)
            cw = cw.cpu().numpy() * (1 - maps) ** 2 / opt.nc
            iw = labels_to_image_weights(dataset_df2.labels, nc=opt.nc, class_weights=cw)
            dataset_df2.indices = random.choices(range(dataset_df2.n), weights=iw, k=dataset_df2.n)

        # Start batch ----------------------------------------------------------------------------------------------------
        for i, (imgs, labels, paths, _shapes) in pbar:
            # Batch-01. Input data
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            '''
             Explanation of variables in item1_batch:
               - imgs (torch.uint8): Image data with shape [B, 3, H, W] which has the scale of 0~255
               - labels (torch.float32): Label data with shape [num, 6], where num is the number of objects in the image.
                                        Each row in the labels contains: [Batch_num, class_num, x, y, h, w].
               - paths (tuple[str]): List of image paths with length B.
               - _shapes (tuple[tuple]): List of tuples, each containing (h0, w0), ((h / h0, w / w0), pad).
               - features (torch.float32): Features data with shape [B, 15, 7, 7], representing 3 (num_anchor) x 5 (bbox xywh + confidence score).
            '''

            # Concatenate 'imgs_duplicated' and 'clips' along the first dimension, which has the shape of [2B, 3, T, H, W]
            imgs_duplicated = imgs.unsqueeze(2).repeat((1, 1, opt.DATA.NUM_FRAMES, 1, 1))
            img = imgs_duplicated[:, :, -1, :, :] # keyframe
            
            # Batch-02. Forward
            with amp.autocast(enabled=True):
                out_bboxs, out_clos, out_acts = model(imgs_duplicated)
                
                out_bbox_infer, out_bbox_features = out_bboxs[0], out_bboxs[1]
                out_clo_infer, out_clo_features = out_clos[0], out_clos[1]
                out_act_infer, out_act_features = out_acts[0], out_acts[1]

                total_loss, losses = LOSS.forward_df2(
                    p_cls = out_clo_features, 
                    p_bbox = out_bbox_features, 
                    targets = labels)
                
                _lbox, _lobj, lclo, _ = torch.split(losses, 1)
                
            # Batch-03. Backward
            scaler.scale(lclo).backward()
            
            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                # if ema: ema.update(model)

            # Batch-04. Print
            loss_item = torch.tensor([_lbox, _lobj, lclo], device='cpu')
            
            if torch.all(torch.isfinite(loss_item)):
                mloss = (mloss * i + loss_item) / (i + 1)  # update mean losses
                mtotal_loss = (mtotal_loss * i + total_loss.detach().cpu()) / (i+1) # update mean total_loss
                
            gpu_memory_usage_gb = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            lr = [x['lr'] for x in optimizer.param_groups]
            output_string = '%g/%g' % (epoch, epochs - 1)  # Display current epoch / total epochs
            output_string += ' ' + gpu_memory_usage_gb
            for loss_idx in range(len(mloss)):
                output_string += ' ' + '%10.4g' % mloss[loss_idx]
            output_string += ' ' + '%10.4g' % mtotal_loss
            output_string += ' ' + '%10.6g' % lr[0]
            pbar.set_description(output_string)
            
            cur_step = epoch * num_batch + i
            if (cur_step % opt.log_step) == 0:
                tags = ['train/box_loss', 'train/obj_loss', 'train/clo_loss', 'train/lr']
                for x, tag in zip(list(mloss) + lr, tags):
                    wandb_logger.log({tag: x})
                    
            # Batch-05. Plot
            if (plots) and (cur_step % opt.log_step == 0):
                plot_i = (cur_step // opt.log_step) % 4
                f_clo = save_dir / f'train_batch_clo{plot_i}.jpg'  # filename
                f_act = save_dir / f'train_batch_act{plot_i}.jpg'  # filename
                
                img = un_normalized_images(img)
                
                preds_clo = torch.cat((out_bbox_infer, out_clo_infer), dim=2)
                preds_clo = non_max_suppression(preds_clo, conf_thres=0.3, iou_thres=0.5)
                Thread(target=plot_batch_image_from_preds, args=(img.copy(), preds_clo, str(f_clo), labelmap_df2), daemon=True).start()
                
                preds_act = torch.cat((out_bbox_infer, out_act_infer), dim=2)
                preds_act = non_max_suppression(preds_act, conf_thres=0.5, iou_thres=0.5)
                Thread(target=plot_batch_image_from_preds, args=(img.copy(), preds_act,str(f_act), labelmap_ava), daemon=True).start()
            
            elif plots and i == 5 and wandb_logger.wandb:
                wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                save_dir.glob('train*.jpg') if x.exists()]})
                
            # End batch ----------------------------------------------------------------------------------------------------

        # End epoch --------------------------------------------------------------------------------------------------------
    
        # Start write ------------------------------------------------------------------------------------------------------
        final_epoch = epoch + 1 == epochs
        wandb_logger.current_epoch = epoch + 1        
        
        if not opt.notest or final_epoch:

            results_df2, maps_df2, times_df2 = test_df2(opt,
                                        batch_size=opt.batch_size_test,
                                        imgsz=imgsz_test,
                                        model=model,
                                        single_cls=opt.single_cls,
                                        dataloader=testloader_df2,
                                        save_dir=save_dir,
                                        verbose=final_epoch,
                                        plots=plots,
                                        wandb_logger=wandb_logger,
                                        compute_loss=False,
                                        is_coco=False,
                                        v5_metric=opt.v5_metric)
        
        # Log
        tags = [
                'metrics/precision(DF2)', 'metrics/recall(DF2)', 'metrics/mAP_0.5(DF2)', 'metrics/mAP_0.5:0.95(DF2)',
                'val/box_loss(DF2)', 'val/obj_loss(DF2)', 'val/clo_loss(DF2)',  # val loss (default is False)
                ]  # params
        
        # Write
        simple_tags = ['gpu_mem', 'box', 'obj', 'clo', 'total', 'lr']+[tag.split('/')[-1] for tag in tags]
        formatted_tags = [f"{tag:<{width}}" for tag, width in zip(simple_tags, [10]* (6+len(simple_tags)))] 
        header_line = " ".join(formatted_tags) + '\n'

        if not os.path.exists(results_file) or os.path.getsize(results_file) == 0: # Check if file is empty or does not exist
            with open(results_file, 'w') as f:
                f.write(header_line)

        with open(results_file, 'a') as f:
            f.write(output_string  + '%10.4g' * 7 % results_df2 +'\n')  # append metrics, val_loss
        
        for x, tag in zip(list(results_df2), tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            if wandb_logger.wandb:
                wandb_logger.log({tag: x})  # W&B
                
        # Update best score (Define best_fitness as minimum loss)
        fi = mtotal_loss
        if fi < best_fitness:
            best_fitness = fi
        wandb_logger.end_epoch(best_result=best_fitness == fi)
        
        # Save model
        if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': None}

            # Save last, best and delete
            last = wdir / f'epoch{str(epoch)}.pt'
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)

        # End write --------------------------------------------------------------------------------------------------------
    # End training -----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Import configuration files
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open('cfg/deepfashion2.yaml', 'r') as f:
        _dict_df2 = yaml.safe_load(f)
        opt_df2 = ConfigObject(_dict_df2)
    
    with open('cfg/ava.yaml', 'r') as f:
        _dict_ava = yaml.safe_load(f)
        opt_ava = ConfigObject(_dict_ava)
        
    with open('cfg/model.yaml', 'r') as f:
        _dict_model = yaml.safe_load(f)
        opt_model = ConfigObject(_dict_model)
    
    with open('cfg/hyp.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    
    logger.info(colorstr('Hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt = ConfigObject({})
    opt.merge(opt_df2)
    opt.merge(opt_ava)
    opt.merge(opt_model)
    
    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'

        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            print(Path(ckpt).parent.parent/'opt.yaml')
            _dict_resume = yaml.safe_load(f)
            opt = ConfigObject(_dict_resume) # replace
        opt.weights, opt.resume, opt.batch_size = ckpt, True, opt.batch_size  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    
    main(hyp, opt, device = torch.device('cuda:0'), tb_writer = None)