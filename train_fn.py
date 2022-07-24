import numpy as np
import torch
import os
from torch import nn, optim

bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).to(torch.device('cuda:0')))
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

def masked_softmax(x, m, dim, tau=1):
    exp_x = torch.exp(tau*x)
    return exp_x / (exp_x*m).sum(dim, keepdim=True)

def weighted_mse_loss(x, y, w):
    return (w.view(-1)*((x-y).view(-1))**2).mean()


def model_loss(ho_autoencoder, data):
    corr_mask_gt, corr_pts_gt, corr_dist_gt = data[..., 11], data[..., 12: 15], data[..., 15]
    dec_cond = data[..., 16:]
    data = data[..., :11]
    corr_mask_pred, corr_pts_pred, corr_dist_pred = ho_autoencoder(data, dec_cond)
    output = [corr_mask_pred, corr_pts_pred, corr_dist_pred]
    corr_mask_bool = corr_mask_gt.bool()

    mask_cls_loss = bce_loss(corr_mask_pred.view(-1), corr_mask_gt.view(-1))
    corr_pts_loss = mse_loss(corr_pts_pred[corr_mask_bool], corr_pts_gt[corr_mask_bool])

    dist_weight = masked_softmax(-torch.abs(10*corr_dist_gt), corr_mask_gt, dim=2) * \
        corr_mask_gt.sum(dim=2, keepdim=True)
    corr_dist_loss = weighted_mse_loss(corr_dist_pred[corr_mask_bool],
        10*corr_dist_gt[corr_mask_bool], dist_weight[corr_mask_bool])

    return output, mask_cls_loss, corr_pts_loss, corr_dist_loss


def train_fn_iter(ho_autoencoder, data, opt, epoch=None):
    opt.zero_grad()

    output, mask_cls_loss, corr_pts_loss, corr_dist_loss = model_loss(ho_autoencoder, data)

    loss = mask_cls_loss + corr_pts_loss*20 + corr_dist_loss*5
    loss.backward()
    opt.step()

    return mask_cls_loss.item(), corr_pts_loss.item(), corr_dist_loss.item()


def eval_fn_iter(ho_autoencoder, data):
    with torch.no_grad():
        _, mask_cls_loss, corr_pts_loss, corr_dist_loss = model_loss(ho_autoencoder, data)
    
    return mask_cls_loss.item(), corr_pts_loss.item(), corr_dist_loss.item()


def train_model(ho_autoencoder, train_loader, vald_loader, opt, device, args):
    num_epochs = args.num_epoch
    save_path = args.ckpt_path
    save_int = args.save_int_ckpt

    print('Training started', flush=True)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, eta_min=1e-4)

    for epoch in range(1, num_epochs+1):
        ho_autoencoder.train()

        train_loss_dict = {'mask_cls_loss': 0, 'corr_pts_loss': 0, 'corr_dist_loss': 0}
        vald_loss_dict = {'mask_cls_loss': 0, 'corr_pts_loss': 0, 'corr_dist_loss': 0}

        train_iters = len(train_loader)
        for i, data in enumerate(train_loader):
            data = data.to(device)
            losses = train_fn_iter(ho_autoencoder, data, opt, epoch)
            train_loss_dict['mask_cls_loss'] += losses[0]
            train_loss_dict['corr_pts_loss'] += losses[1]
            train_loss_dict['corr_dist_loss'] += losses[2]

            lr_scheduler.step(epoch-1+(i/train_iters))

        # validation
        ho_autoencoder.eval()

        for data in vald_loader:
            data = data.to(device)
            losses = eval_fn_iter(ho_autoencoder, data)
            vald_loss_dict['mask_cls_loss'] += losses[0]
            vald_loss_dict['corr_pts_loss'] += losses[1]
            vald_loss_dict['corr_dist_loss'] += losses[2]

        print('====> Epoch {}/{}: Training'.format(epoch, num_epochs), flush=True)
        
        for term in train_loss_dict:
            print('\t{} {:.5f}'.format(term, train_loss_dict[term] / len(train_loader)), flush=True)

        print('                   Validation', flush=True)

        for term in vald_loss_dict:
            print('\t{} {:.5f}'.format(term, vald_loss_dict[term] / len(vald_loader)), flush=True)
     
        if save_int or epoch == num_epochs:
            checkpoint = {
                'model': ho_autoencoder.module.state_dict(),
                'opt': opt.state_dict()
            }   
            torch.save(checkpoint, os.path.join(save_path, '{}.pth'.format(epoch)))
