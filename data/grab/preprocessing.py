# The code is adapted from the GRAB repository by Omid Taheri.

import numpy as np
import os, glob, pickle, argparse
import smplx
import trimesh

from tqdm import tqdm
from psbody.mesh import Mesh
from tools.objectmodel import ObjectModel
from tools.utils import makepath, parse_npz, params2torch, prepare_params, to_cpu, append2dict


class GRABDataSet(object):
    def __init__(self, args, grab_splits):
        self.grab_path = args.grab_path
        self.out_path = args.out_path
        makepath(self.out_path)

        print('Starting data preprocessing !')

        # convert object names to ids
        self.objname2id = {}
        obj_meshes = sorted(os.listdir(os.path.join(self.grab_path, 'tools/object_meshes/contact_meshes')))
        for i, fn in enumerate(obj_meshes):
            obj_name = fn.split('.')[0]
            self.objname2id[obj_name] = i

        if grab_splits is None:
            self.splits = {'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                            'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                            'train': []}
        else:
            assert isinstance(grab_splits, dict)
            self.splits = grab_splits
            
        self.all_seqs = glob.glob(os.path.join(self.grab_path, 'grab/*/*.npz'))

        self.split_seqs = {'test': [],
                           'val': [],
                           'train': []
                           }

        self.process_sequences()

        self.data_preprocessing(args)

        print('Total sequences: %d' % len(self.all_seqs))
        print('Number of sequences in each data split : train: %d , test: %d , val: %d'
                         %(len(self.split_seqs['train']), len(self.split_seqs['test']), len(self.split_seqs['val'])))
        print('Number of objects in each data split : train: %d , test: %d , val: %d'
                         % (len(self.splits['train']), len(self.splits['test']), len(self.splits['val'])))


    def process_sequences(self):
        for sequence in self.all_seqs:
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

            # split train, val, and test sequences
            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)

        
    def data_preprocessing(self,args):
        # load MANO model
        mano_path = os.path.join(args.mano_path, 'MANO_RIGHT.pkl')
        with open(mano_path, 'rb') as f:
            mano_model = pickle.load(f, encoding='latin1')
        
        # normalize MANO template
        mano_template = Mesh(v=mano_model['v_template'], f=mano_model['f'])
        with open('data/grab/scale_center.pkl', 'rb') as f:
            scale, center = pickle.load(f)
        mano_template.v = mano_template.v * scale + center

        self.obj_info = {}

        for split in self.split_seqs.keys():
            print('Processing data for %s split.' % (split))

            object_data = {'verts': [], 'vn': [], 'center': [], 'global_orient': [],
                           'transl': [], 'object_id': [], 'frame_id': []}
            rhand_data = {'verts': [], 'global_orient': [], 'hand_pose': [],
                          'transl': [], 'fullpose': [], 'is_left': []}
            rhand_pert_data = {'verts': []}


            starting_frame_id = 0

            for sequence in tqdm(self.split_seqs[split]):

                seq_data = parse_npz(sequence)

                obj_name = seq_data.obj_name
                n_comps  = seq_data.n_comps

                ds_mask = self.downsample_frames(seq_data, rate=args.ds_rate)
                hand_contact_mask = self.filter_contact_frames(seq_data, args)
                frame_mask = ds_mask & hand_contact_mask
                #frame_mask = ds_mask
                if args.hand == 'right':
                    rhand_data['is_left'].append(np.zeros(frame_mask.sum()).astype(bool))
                else:
                    rhand_data['is_left'].append(np.ones(frame_mask.sum()).astype(bool))

                frame_ids = np.arange(starting_frame_id, starting_frame_id+int(seq_data.n_frames))[frame_mask]
                object_data['frame_id'].append(frame_ids)
                starting_frame_id += int(seq_data.n_frames) + 100

                # total selectd frames
                T = frame_mask.sum()
                if T < 1:
                    continue # if no frame is selected continue to the next sequence

                if args.hand == 'right':
                    hand_params  = prepare_params(seq_data.rhand.params, frame_mask)
                else:
                    hand_params  = prepare_params(seq_data.lhand.params, frame_mask)
                obj_params = prepare_params(seq_data.object.params, frame_mask)

                append2dict(rhand_data, hand_params)
                append2dict(object_data, obj_params)

                object_data['object_id'].extend([self.objname2id[obj_name]]*T)

                if args.hand == 'right':
                    hand_mesh = os.path.join(args.grab_path, seq_data.rhand.vtemp)
                else:
                    hand_mesh = os.path.join(args.grab_path, seq_data.lhand.vtemp)
                hand_vtemp = np.array(Mesh(filename=hand_mesh).v)

                if args.hand == 'right':
                    rh_m = smplx.create(model_path=args.smplx_path,
                                        model_type='mano',
                                        is_rhand=True,
                                        v_template=hand_vtemp,
                                        num_pca_comps=n_comps,
                                        flat_hand_mean=True,
                                        batch_size=T)
                else:
                    rh_m = smplx.create(model_path=args.smplx_path,
                                        model_type='mano',
                                        is_rhand=False,
                                        v_template=hand_vtemp,
                                        num_pca_comps=n_comps,
                                        flat_hand_mean=True,
                                        batch_size=T)

                hand_parms = params2torch(hand_params)
                verts_rh = to_cpu(rh_m(**hand_parms).vertices)
                if args.hand == 'left':
                    verts_rh[..., 0] = -verts_rh[..., 0]
                rhand_data['verts'].append(verts_rh)


                # perturb hand pose
                aug_t = np.random.randn(T, 3) * args.aug_trans
                aug_o = np.random.randn(T, 3) * args.aug_rot
                aug_p = np.random.randn(T, 24) * args.aug_pose
                hand_params['global_orient'] += aug_o
                hand_params['hand_pose'] += aug_p
                hand_params['transl'] += aug_t
                hand_pert_parms = params2torch(hand_params)

                verts_rh_pert = to_cpu(rh_m(**hand_pert_parms).vertices)
                if args.hand == 'left':
                    verts_rh_pert[..., 0] = -verts_rh_pert[..., 0]
                rhand_pert_data['verts'].append(verts_rh_pert)


                ### for objects
                obj_info = self.load_obj_verts(obj_name, seq_data, args.num_points)

                obj_m = ObjectModel(v_template=obj_info['verts_sample'],
                                    batch_size=T,
                                    center=obj_info['center'],
                                    vn=obj_info['vn_sample'])
                obj_parms = params2torch(obj_params)
                obj_output = obj_m(**obj_parms)
                verts_obj = to_cpu(obj_output.vertices)
                vn_obj = to_cpu(obj_output.vn)
                center_obj = to_cpu(obj_output.center)
                if args.hand == 'left':
                    verts_obj[..., 0] = -verts_obj[..., 0]
                    center_obj[..., 0] = -center_obj[..., 0]
                    vn_obj[..., 0] = -vn_obj[..., 0]
                object_data['verts'].append(verts_obj)
                object_data['vn'].append(vn_obj)
                object_data['center'].append(center_obj)

            object_data['verts'] = np.concatenate(object_data['verts'])
            object_data['center'] = np.concatenate(object_data['center'])
            rhand_data['verts'] = np.concatenate(rhand_data['verts'])
            rhand_pert_data['verts'] = np.concatenate(rhand_pert_data['verts'])
            
            # center all verts
            object_verts = object_data['verts']
            object_center = object_data['center']
            object_verts -= object_center
            rhand_verts = rhand_data['verts'] - object_center
            rhand_pert_verts = rhand_pert_data['verts'] - object_center

            # save data
            # rhand_verts: 778*3
            # rhand_global_orient: 3
            # rhand_hand_pose: 24
            # object_verts: V*3
            # object_vn: V*3
            # object_global_orient: 3
            # object_transl: 3
            # object_id: 1
            # frame_id: 1
            # is_left: 1
            np_dtype = np.dtype('(2334)f4, (3)f4, (24)f4, ({})f4, ({})f4, (3)f4, (3)f4, i4, i4, ?'.format(
                args.num_points*3, args.num_points*3
            ))
            
            is_left = np.concatenate(rhand_data['is_left'])
            rhand_verts = rhand_verts.reshape(-1, 778*3)
            rhand_global_orient = np.concatenate(rhand_data['global_orient'])
            rhand_pose = np.concatenate(rhand_data['hand_pose'])
            object_verts = object_verts.reshape(-1, args.num_points*3)
            object_vn = np.concatenate(object_data['vn']).reshape(-1, args.num_points*3)
            object_global_orient = np.concatenate(object_data['global_orient'])
            object_transl = np.concatenate(object_data['transl'])
            object_id = np.array(object_data['object_id'])
            frame_id = np.concatenate(object_data['frame_id'])

            np_data = list(zip(rhand_verts, rhand_global_orient, rhand_pose, object_verts, object_vn, 
                object_global_orient, object_transl, object_id, frame_id, is_left))
            np_data = np.array(np_data, dtype=np_dtype)
            np.save(os.path.join(self.out_path, '{}_{}.npy'.format(split, args.hand)), np_data)
            

            rhand_pert_verts = rhand_pert_verts.reshape(-1, 778*3)

            np_pert_data = list(zip(rhand_pert_verts, rhand_global_orient, rhand_pose, object_verts,
                object_vn, object_global_orient, object_transl, object_id, frame_id, is_left))
            np_pert_data = np.array(np_pert_data, dtype=np_dtype)
            np.save(os.path.join(self.out_path, '{}_{}_pert.npy'.format(split, args.hand)), np_pert_data)

            print('Processing for %s split finished' % split)
            print('Total number of frames for %s split is:%d' % (split, len(np_data)))


    def downsample_frames(self, seq_data, rate=2):
        num_frames = int(seq_data.n_frames)
        ones = np.ones(num_frames//rate+1).astype(bool)
        zeros = [np.zeros(num_frames//rate+1).astype(bool) for _ in range(rate-1)]
        mask = np.vstack((ones, *zeros)).reshape((-1,), order='F')[:num_frames]
        return mask

    def filter_contact_frames(self, seq_data, args):
        '''
        left hand not in contact
        '''
        obj_contact = seq_data['contact']['object']

        if args.hand == 'right':
            # left hand not in contact
            frame_mask = ~(((obj_contact == 21) | ((obj_contact >= 26) & (obj_contact <= 40))).any(axis=1))
        else:
            # right hand not in contact
            frame_mask = ~(((obj_contact == 22) | ((obj_contact >= 41) & (obj_contact <= 55))).any(axis=1))
        
        return frame_mask

    def load_obj_verts(self, obj_name, seq_data, n_points_sample=2000):
        mesh_path = os.path.join(self.grab_path, seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            obj_mesh = trimesh.load(mesh_path, process=False)
            obj_mesh.remove_degenerate_faces(height=1e-06)

            verts_obj = np.array(obj_mesh.vertices)
            faces_obj = np.array(obj_mesh.faces)

            verts_sampled, face_ind = obj_mesh.sample(n_points_sample, return_index=True)
            vn_sampled = obj_mesh.face_normals[face_ind]

            assert vn_sampled.shape == verts_sampled.shape
            assert np.isclose((vn_sampled**2).sum(axis=-1), 0).sum() == 0, mesh_path
            self.obj_info[obj_name] = {'verts': verts_obj,
                                       'faces': faces_obj,
                                       'center': verts_obj.mean(axis=0, keepdims=True),
                                       'verts_sample': verts_sampled,
                                       'vn_sample': vn_sampled,
                                       'obj_mesh_file': mesh_path}

        return self.obj_info[obj_name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grab_path', type=str)
    parser.add_argument('--smplx_path', type=str)
    parser.add_argument('--mano_path', type=str)
    parser.add_argument('--out_path', type=str)
    # from 'all', 'use' , 'pass', 'lift' , 'offhand'
    parser.add_argument('--intent', default='all', type=str)
    # from 'left', 'right', 'both'
    parser.add_argument('--hand', default='both', type=str)
    # number of points sampled for each object
    parser.add_argument('--num_points', default=8000, type=int)
    # sequence downsample rate
    parser.add_argument('--ds_rate', default=4, type=int)
    # sequence minimum window size
    parser.add_argument('--window_size', default=30, type=int)
    # noise levels
    parser.add_argument('--aug_trans', default=0.01, type=float)
    parser.add_argument('--aug_rot', default=0.05, type=float)
    parser.add_argument('--aug_pose', default=0.3, type=float)           
    args = parser.parse_args()

    # split the dataset based on the objects
    grab_splits = { 'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                    'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                    'train': []
                }

    if args.hand == 'both':
        args.hand = 'right'
        GRABDataSet(args, grab_splits)
        args.hand = 'left'
        GRABDataSet(args, grab_splits)
        args.hand = 'both'
    else:
        GRABDataSet(args, grab_splits)

    for split in grab_splits.keys():
        split_path_right = os.path.join(args.out_path, '{}_right.npy'.format(split))
        split_path_left = os.path.join(args.out_path, '{}_left.npy'.format(split))
        if args.hand == 'both':
            np_data_right = np.load(split_path_right)
            np_data_left = np.load(split_path_left)
            np_data = np.concatenate([np_data_right, np_data_left], axis=0)
        elif args.hand == 'right':
            np_data = np.load(split_path_right)
        elif args.hand == 'left':
            np_data = np.load(split_path_left)

        ids_to_remove = []
        for i in range(len(np_data)):
            rhand_pc = np_data[i]['f0'].reshape(778, 3)
            object_pc = np_data[i]['f3'].reshape(args.num_points, 3)
            
            circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
            center = np.broadcast_to((rhand_pc[circle_v_id, :]).mean(axis=0, keepdims=True), object_pc.shape)

            wrist2obj_dist = np.min(np.sqrt(((center-object_pc)**2).sum(axis=-1)))
            if wrist2obj_dist > 0.15:
                ids_to_remove.append(i)

        mask = np.ones(len(np_data)).astype(bool)
        mask[ids_to_remove] = 0
        np_data = np_data[mask]

        # correct frame ids
        np_data['f8'] = np_data['f8'] // args.ds_rate

        print('{} set: {} frames remaining'.format(split, mask.sum()))

        makepath(os.path.join(args.out_path, split))

        max_clip_len = 0
        min_clip_len = 100000
        total_clip_len = 0

        file_id = 0
        start_idx = 0

        while start_idx <= len(np_data) - args.window_size:
            for i in range(start_idx+1, len(np_data)):
                if (np_data[i]['f8']-1 > np_data[i-1]['f8']) or (np_data[i]['f9'] != np_data[i-1]['f9']):
                    clip_len = i - start_idx
                    if clip_len >= args.window_size:
                        np.save(os.path.join(args.out_path, split, '{}.npy'.format(file_id)), np_data[start_idx:i])
                        file_id += 1
                        total_clip_len += clip_len
                        if clip_len > max_clip_len:
                            max_clip_len = clip_len
                        if clip_len < min_clip_len:
                            min_clip_len = clip_len

                    start_idx = i
                    break
            if i == len(np_data) - 1:
                break

        print('Number of clips for {} set: {}'.format(split, file_id))
        print('Mean clip length for {} set: {} frames'.format(split, total_clip_len // file_id))
        print('Max clip length for {} set: {} frames'.format(split, max_clip_len))
        print('Min clip length for {} set: {} frames'.format(split, min_clip_len))


        makepath(os.path.join(args.out_path, '{}_pert'.format(split)))

        pert_path_right = os.path.join(args.out_path, '{}_right_pert.npy'.format(split))
        pert_path_left = os.path.join(args.out_path, '{}_left_pert.npy'.format(split))
        if args.hand == 'both':
            pert_data_right = np.load(pert_path_right)
            pert_data_left = np.load(pert_path_left)
            pert_data = np.concatenate([pert_data_right, pert_data_left], axis=0)
        elif args.hand == 'right':
            pert_data = np.load(pert_path_right)
        elif args.hand == 'left':
            pert_data = np.load(pert_path_left)

        pert_data['f8'] = pert_data['f8'] // args.ds_rate
        print('{} pert length:'.format(split), len(pert_data))

        unpert_clips = glob.glob(os.path.join(args.out_path, split, '*.npy'))
        for c in unpert_clips:
            d = np.load(c)
            start_frame_id = d[0]['f8']
            start_frame_hand = d[0]['f9']
            
            for i in range(len(pert_data)):
                if pert_data[i]['f8'] == start_frame_id and pert_data[i]['f9'] == start_frame_hand:
                    break

            assert i < len(pert_data)

            np.save(os.path.join(args.out_path, '{}_pert'.format(split), '{}'.format(os.path.basename(c))),
                pert_data[i:i+len(d)])
