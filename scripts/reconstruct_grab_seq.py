import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import os, argparse, copy, json
import pickle as pkl
from scipy.spatial.transform import Rotation as R
from psbody.mesh import Mesh
from manopth.manolayer import ManoLayer
from dataloading import GRAB_Single_Frame
from model import TemporalPointAE

# TODO:
# saving

def seal(mesh_to_seal):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.v[circle_v_id, :]).mean(0)

    sealed_mesh = copy.copy(mesh_to_seal)
    sealed_mesh.v = np.vstack([mesh_to_seal.v, center])
    center_v_id = sealed_mesh.v.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        sealed_mesh.f = np.vstack([sealed_mesh.f, new_faces])
    return sealed_mesh


def joint_acc_loss(x, J_regressor):
    args.num_init, num_frames = x.size()[:2]
    joints = torch.matmul(J_regressor, x.permute(2, 3, 0, 1).contiguous()[:-1].view(778, -1))
    joints = joints.view(16, 3, args.num_init, -1).permute(2, 3, 0, 1).contiguous()
    joints_acc = torch.sum((joints[:, 2:] - 2*joints[:, 1:num_frames-1] + joints[:, :num_frames-2])**2,
        dim=-1)
    return torch.mean(joints_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grab_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--mano_path', type=str)    
    parser.add_argument('--seq_path', type=str) 
    parser.add_argument('--out_path', default='./recon_results', type=str)
    parser.add_argument('--num_coarse_iter', default=100, type=int)
    parser.add_argument('--num_fine_iter', default=2000, type=int)
    parser.add_argument('--coarse_lr', default=0.1, type=float)
    parser.add_argument('--fine_lr', default=0.1, type=float)
    parser.add_argument('--latent_size', default=64, type=int)
    parser.add_argument('--window_size', default=30, type=int)
    # number of random initialization of MANO parameters
    parser.add_argument('--num_init', default=1, type=int) 
    args = parser.parse_args()

    device = torch.device('cpu')

    #load model
    ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu'))

    ho_autoencoder = TemporalPointAE(input_dim=11, latent_dim=args.latent_size,
        window_size=args.window_size)
    ho_autoencoder.load_state_dict(ckpt['model'])
    ho_autoencoder.eval()

    # load MANO
    with open(os.path.join(args.mano_path, 'MANO_RIGHT.pkl'), 'rb') as f:
        mano = pkl.load(f, encoding="latin-1")
        mano_mesh = Mesh(mano['v_template'], mano['f'])
        J_regressor = torch.tensor(mano['J_regressor'].todense()).float()
    with open('../data/grab/scale_center.pkl', 'rb') as f:
        scale, center = pkl.load(f)
        mano_mesh.v = mano_mesh.v * scale + center
        mano_mesh = seal(mano_mesh)

    # load data
    obj_mesh_path = os.path.join(args.grab_path, 'tools/object_meshes/contact_meshes')
    id2objmesh = []
    obj_meshes = sorted(os.listdir(obj_mesh_path))
    for i, fn in enumerate(obj_meshes):
        id2objmesh.append(os.path.join(obj_mesh_path, fn))

    seq = GRAB_Single_Frame(args.seq_path)
    num_frames = len(seq)
    os.makedirs(args.out_path, exist_ok=True)

    input_rhand_pcs = []
    object_verts = []
    input_features = []
    Rs = []
    ts = []
    
    for i in range(num_frames):
        input_rhand_pc, obj_pc, obj_corr_mask, obj_corr_pts, obj_corr_dist, \
            obj_rot, obj_transl, obj_id, obj_vn, is_left = test_set[i]
        obj_corr_mask[obj_corr_dist > 0.1] = 0
        obj_mesh = Mesh(filename=id2objmesh[obj_id])
        obj_verts = np.dot(obj_mesh.v, R.from_rotvec(obj_rot).as_matrix()) + obj_transl.reshape(1, 3)
        obj_verts -= obj_verts.mean(axis=0, keepdims=True)
        
        if i == 0:
            obj_pc_variance = np.max(np.sqrt(np.sum(obj_pc**2, axis=1)))
        obj_pc = obj_pc / obj_pc_variance
        object_pc = torch.tensor(obj_pc, dtype=torch.float32)

        object_vn = torch.tensor(obj_vn, dtype=torch.float32)
        object_corr_mask = torch.tensor(obj_corr_mask, dtype=torch.float).unsqueeze(-1)
        object_corr_pts = torch.tensor(obj_corr_pts, dtype=torch.float)
        object_corr_dist = torch.tensor(obj_corr_dist, dtype=torch.float).unsqueeze(-1)

        if is_left:
            input_rhand_pc[..., 0] = -input_rhand_pc[..., 0]
        input_rhand_pcs.append(input_rhand_pc)
        object_verts.append(obj_verts)

        Rs.append(R.from_rotvec(obj_rot).as_matrix())
        ts.append(obj_transl.reshape(1, 3))

        input_features.append(torch.cat([object_pc, object_corr_mask, object_corr_pts, object_corr_dist, object_vn], dim=1))

    # organize frames into batches
    window_mid = args.window_size // 2
    batched_input = []
    window_ind = list(range(0, num_frames, window_mid))[:-1]
    for i in window_ind:
        if i+args.window_size >= num_frames:
            batched_input.append(torch.stack(input_features[num_frames-args.window_size:num_frames], dim=0))
        else:
            batched_input.append(torch.stack(input_features[i:i+args.window_size], dim=0))
    batched_input = torch.stack(batched_input, dim=0)

    with torch.no_grad():
        corr_mask_output, corr_pts_output, corr_dist_output = ho_autoencoder(batched_input)

    data_collection = []
    object_contact_pts = []
    dist_weights = []
    for b in range(batched_input.size(0)):
        corr_mask_pred = (torch.sigmoid(corr_mask_output[b]).numpy() > 0.5)
        corr_pts_pred = corr_pts_output[b].numpy()

        if b == 0:
            start_idx = 0
        elif b == batched_input.size(0) - 1:
            start_idx = window_mid if num_frames%window_mid == 0 else \
                args.window_size - (num_frames % window_mid)
        else:
            start_idx = window_mid
        for i in range(start_idx, args.window_size):
            obj_corr_pts = corr_pts_pred[i]
            obj_corr_mask = corr_mask_pred[i]

            object_pc = batched_input[b][i, :, :3] * obj_pc_variance
            object_vn = batched_input[b][i, :, -3:]
            object_corr_dist = corr_dist_output[b][i] * 0.1
            object_corr_mask = torch.from_numpy(obj_corr_mask)

            closest_face, closest_points = mano_mesh.closest_faces_and_points(obj_corr_pts[obj_corr_mask])
            vert_ids, bary_coords = mano_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
            vert_ids = torch.from_numpy(vert_ids.astype(np.int64)).view(-1)
            bary_coords = torch.from_numpy(bary_coords).float()
            obj_contact_pts = (object_pc + object_corr_dist.unsqueeze(1)*object_vn)[object_corr_mask]
            data_collection.append((vert_ids, bary_coords))
            object_contact_pts.append(obj_contact_pts)
    object_contact_pts = torch.cat(object_contact_pts, dim=0).unsqueeze(0).repeat(args.num_init, 1, 1)
        
    # setup MANO layer
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side='right',
        mano_root=args.mano_path,
        ncomps=24,
        use_pca=True,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    ).to(device)

    circle_v_id = torch.tensor([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype=torch.long)    

    # initialize variables
    beta_var = torch.randn([args.num_init, 10]).to(device)
    # first 3 global orientation
    rot_var = torch.randn([args.num_init*num_frames, 3]).to(device)
    theta_var = torch.randn([args.num_init*num_frames, 24]).to(device)
    transl_var = torch.randn([args.num_init*num_frames, 3]).to(device)
    beta_var.requires_grad_()
    rot_var.requires_grad_()
    theta_var.requires_grad_()
    transl_var.requires_grad_()

    # coarse optimization loop
    num_iters = args.num_coarse_iter
    opt = optim.Adam([rot_var, transl_var], lr=args.coarse_lr)
    for i in range(num_iters):
        opt.zero_grad()
        hand_verts, _ = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, num_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(args.num_init, num_frames, 778, 3) * 0.001

        center = (hand_verts[:, :, circle_v_id, :]).mean(2, keepdim=True)
        hand_verts = torch.cat([hand_verts, center], dim=2)

        pred_contact_pts = []
        for j in range(args.num_init):
            for k in range(num_frames):
                vert_ids = data_collection[k][0]
                bary_coords = data_collection[k][1]
                pred_contact_pts.append((hand_verts[j, k, vert_ids].view(-1, 3, 3) * bary_coords[..., np.newaxis]).sum(dim=1))
        pred_contact_pts = torch.cat(pred_contact_pts, dim=0).view(args.num_init, -1, 3)

        corr_loss = F.mse_loss(pred_contact_pts, object_contact_pts)

        loss = corr_loss
        loss.backward()
        opt.step()

        print('Iter {}: {}'.format(i, loss.item()))
        print('\tCorrespondence Loss: {}'.format(corr_loss.item()))

    # fine optimization loop
    num_iters = args.num_fine_iter
    opt = optim.Adam([beta_var, rot_var, theta_var, transl_var], lr=args.fine_lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    for i in range(num_iters):
        opt.zero_grad()
        hand_verts, _ = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, num_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(args.num_init, num_frames, 778, 3) * 0.001

        center = (hand_verts[:, :, circle_v_id, :]).mean(2, keepdim=True)
        hand_verts = torch.cat([hand_verts, center], dim=2)

        shape_prior_loss = torch.mean(beta_var**2)
        pose_prior_loss = torch.mean(theta_var**2)

        pred_contact_pts = []
        for j in range(args.num_init):
            for k in range(num_frames):
                vert_ids = data_collection[k][0]
                bary_coords = data_collection[k][1]
                pred_contact_pts.append((hand_verts[j, k, vert_ids].view(-1, 3, 3) * bary_coords[..., np.newaxis]).sum(dim=1))
        pred_contact_pts = torch.cat(pred_contact_pts, dim=0).view(args.num_init, -1, 3)

        corr_loss = F.mse_loss(pred_contact_pts, object_contact_pts)

        pose_smoothness_loss = F.mse_loss(theta_var.view(args.num_init, num_frames, -1)[:, 1:], theta_var.view(args.num_init, num_frames, -1)[:, :-1])
        joints_smoothness_loss = joint_acc_loss(hand_verts, J_regressor)

        loss = corr_loss*20 + pose_smoothness_loss*0.05 + joints_smoothness_loss +\
            shape_prior_loss*0.001 + pose_prior_loss*0.0001
        loss.backward()
        opt.step()
        scheduler.step()

        print('Iter {}: {}'.format(i, loss.item()), flush=True)
        print('\tShape Prior Loss: {}'.format(shape_prior_loss.item()))
        print('\tPose Prior Loss: {}'.format(pose_prior_loss.item()))
        print('\tCorrespondence Loss: {}'.format(corr_loss.item()))
        print('\tPose Smoothness Loss: {}'.format(pose_smoothness_loss.item()))
        print('\tJoints Smoothness Loss: {}'.format(joints_smoothness_loss.item()))

    # find best initialization
    with torch.no_grad():
        hand_verts, _ = mano_layer(torch.cat([rot_var, theta_var], dim=-1),
            beta_var.unsqueeze(1).repeat(1, num_frames, 1).view(-1, 10), transl_var)
        hand_verts = hand_verts.view(args.num_init, num_frames, 778, 3) * 0.001
        center = (hand_verts[:, :, circle_v_id, :]).mean(2, keepdim=True)
        hand_verts = torch.cat([hand_verts, center], dim=2)

        pred_contact_pts = []
        for j in range(args.num_init):
            for k in range(num_frames):
                vert_ids = data_collection[k][0]
                bary_coords = data_collection[k][1]
                pred_contact_pts.append((hand_verts[j, k, vert_ids].view(-1, 3, 3) * bary_coords[..., np.newaxis]).sum(dim=1))
        pred_contact_pts = torch.cat(pred_contact_pts, dim=0).view(args.num_init, -1, 3)

        corr_loss = torch.sum((pred_contact_pts-object_contact_pts)**2, dim=-1).mean(dim=1)
        min_id = torch.argmin(corr_loss)
        hand_verts = hand_verts[min_id]

    if is_left:
        mano_mesh.f = mano_mesh.f[..., [2, 1, 0]]
        mano['f'] = mano['f'][..., [2, 1, 0]]

    hand_verts = hand_verts.cpu().numpy()
    if is_left:
        hand_verts[..., 0] = -hand_verts[..., 0]

    for i in range(num_frames):
        hand_mesh = Mesh(v=hand_verts[i], f=mano_mesh.f)
        hand_mesh_input = seal(Mesh(v=input_rhand_pcs[i], f=mano['f']))
        object_mesh = Mesh(v=object_verts[i], f=obj_mesh.f)
        hand_mesh.write_ply(os.path.join(args.out_path, 'hand_{}.ply'.format(start_frame+i)))
        hand_mesh_input.write_ply(os.path.join(args.out_path, 'input_hand_{}.ply'.format(start_frame+i)))
        object_mesh.write_ply(os.path.join(args.out_path, 'object_{}.ply'.format(start_frame+i)))
