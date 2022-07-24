import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os, glob
from utils import random_rotate_np

class GRAB_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, split, window_size=30, step_size=15, num_points=8000):
        self.clips = []
        self.len = 0
        self.window_size = window_size
        self.step_size = step_size
        self.num_points = num_points
        self.split = split

        files_clean = glob.glob(os.path.join(data_folder, split, '*.npy'))
        for f in files_clean:
            clip_clean = np.load(f)
            clip_pert = np.load(os.path.join(data_folder, split+'_pert', os.path.basename(f)))
            clip_len = (len(clip_clean) - window_size) // step_size + 1
            self.clips.append((self.len, self.len+clip_len, clip_pert,
                [clip_clean['f9'], clip_clean['f11'], clip_clean['f10']]))
            self.len += clip_len
        self.clips.sort(key=lambda x: x[0])
            
    def __getitem__(self, index):
        for c in self.clips:
            if index < c[1]:
                break
        start_idx = (index - c[0]) * self.step_size
        data = c[2][start_idx:start_idx+self.window_size]
        corr_mask_gt, corr_pts_gt, corr_dist_gt = c[3][0], c[3][1], c[3][2]
        corr_mask_gt = corr_mask_gt[start_idx:start_idx+self.window_size]
        corr_pts_gt = corr_pts_gt[start_idx:start_idx+self.window_size]
        corr_dist_gt = corr_dist_gt[start_idx:start_idx+self.window_size]

        samp_ind = np.random.choice(list(range(self.num_points)), 4000, replace=False)

        object_pc = data['f3'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_normal = data['f4'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_pc, R = random_rotate_np(object_pc)

        object_pc = object_pc / np.max(np.sqrt(np.sum(object_pc[0]**2, axis=1)))

        object_normal = np.matmul(object_normal, R)
        object_corr_mask = data['f9'].reshape(self.window_size, -1, 1).astype(np.float32)
        object_corr_pts = data['f11'].reshape(self.window_size, -1, 3).astype(np.float32)
        object_corr_dist = data['f10'].reshape(self.window_size, -1, 1).astype(np.float32)

        corr_mask_gt = corr_mask_gt.reshape(self.window_size, -1, 1).astype(np.float32)
        corr_pts_gt = corr_pts_gt.reshape(self.window_size, -1, 3).astype(np.float32)
        corr_dist_gt = corr_dist_gt.reshape(self.window_size, -1, 1).astype(np.float32)

        # distance thresholding
        corr_mask_gt[corr_dist_gt>0.1] = 0
        object_corr_mask[object_corr_dist>0.1] = 0

        window_feat = np.concatenate([object_pc, object_corr_mask, object_corr_pts,
                object_corr_dist, object_normal], axis=2)[:, samp_ind[:2000]]

        object_pc_dec = object_pc[:, samp_ind[2000:]]
        object_normal_dec = object_normal[:, samp_ind[2000:]]
        corr_mask_gt = corr_mask_gt[:, samp_ind[2000:]]
        corr_pts_gt = corr_pts_gt[:, samp_ind[2000:]]
        corr_dist_gt = corr_dist_gt[:, samp_ind[2000:]]
        
        dec_cond = np.concatenate([object_pc_dec, object_normal_dec], axis=2)

        if np.random.uniform() > 0.5:
            window_feat = np.flip(window_feat, axis=0).copy()
            corr_mask_gt = np.flip(corr_mask_gt, axis=0).copy()
            corr_pts_gt = np.flip(corr_pts_gt, axis=0).copy()
            corr_dist_gt = np.flip(corr_dist_gt, axis=0).copy()
            dec_cond = np.flip(dec_cond, axis=0).copy()

        return np.concatenate([window_feat, corr_mask_gt, corr_pts_gt, corr_dist_gt, dec_cond], axis=2)

    def __len__(self):
        return self.len


class GRAB_Single_Frame(torch.utils.data.Dataset):
    def __init__(self, seq_path):
        self.data = np.load(seq_path)

    def __getitem__(self, index):
        num_points = self.data[index]['f3'].shape[0] // 3
        samp_ind = np.random.choice(list(range(num_points)), 2000)

        rhand_pc = self.data[index]['f0'].reshape(778, 3)
        object_pc = self.data[index]['f3'].reshape(-1, 3)[samp_ind]
        object_vn = self.data[index]['f4'].reshape(-1, 3)[samp_ind]
        object_corr_mask = self.data[index]['f9'].reshape(-1)[samp_ind]
        object_corr_pts = self.data[index]['f11'].reshape(-1, 3)[samp_ind]
        object_corr_dist = self.data[index]['f10'].reshape(-1)[samp_ind]

        object_global_orient = self.data[index]['f5']
        object_transl = self.data[index]['f6']
        object_id = self.data[index]['f7']

        is_left = self.data[index]['f12']

        return rhand_pc, object_pc, object_corr_mask, object_corr_pts, \
               object_corr_dist, object_global_orient, object_transl, object_id, \
               object_vn, is_left

    def __len__(self):
        return len(self.data)
