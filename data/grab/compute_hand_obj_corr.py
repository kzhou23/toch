import numpy as np
import trimesh
import os, glob, pickle, argparse
from multiprocessing import Pool
from itertools import repeat
from scipy.spatial.transform import Rotation as R

def seal(mesh_to_seal, rh=True):
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    sealed_mesh = mesh_to_seal.copy()
    sealed_mesh.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = sealed_mesh.vertices.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        if rh:
            new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id]
        else:
            new_faces = [center_v_id, circle_v_id[i], circle_v_id[i-1]]
        sealed_mesh.faces = np.vstack([sealed_mesh.faces, new_faces])
    return sealed_mesh

def get_object_mesh(d, id2objmesh):
    object_id, object_rot, object_transl = d['f7'], d['f5'], d['f6']
    is_left = d['f9']
    object_mesh = trimesh.load_mesh(id2objmesh[object_id], process=False)
    object_mesh.vertices = np.dot(object_mesh.vertices, R.from_rotvec(object_rot).as_matrix()) + object_transl.reshape(1, 3) 
    if is_left:
        object_mesh.vertices[..., 0] = -object_mesh.vertices[..., 0]
        object_mesh.faces = object_mesh.faces[..., [2, 1, 0]]    
    return object_mesh    

def compute_corr(c, mano_template, id2objmesh):
    d = np.load(c)
            
    mano_template_sealed = seal(mano_template)
    num_points = d[0]['f3'].shape[0] // 3
    np_dtype = np.dtype('(2334)f4, (3)f4, (24)f4, ({})f4, ({})f4, (3)f4, (3)f4, i4, i4, ({})?, ({})f4, ({})f4, ?'.format(
        num_points*3, num_points*3, num_points, num_points, num_points*3
    ))
    obj_corr_mask = []
    obj_corr_dist = []
    obj_corr_pts = []

    for i in range(len(d)):
        rhand_verts = d[i]['f0'].reshape(-1, 3)
        obj_verts = d[i]['f3'].reshape(-1, 3)
        obj_vn = d[i]['f4'].reshape(-1, 3)
        obj_mesh = get_object_mesh(d[i], id2objmesh)

        corr_mask = np.zeros(obj_verts.shape[0]).astype(bool)
        corr_dist = np.zeros(obj_verts.shape[0]).astype(float)
        corr_pts = np.zeros((obj_verts.shape[0], 3)).astype(float)

        rhand_mesh = trimesh.Trimesh(vertices=rhand_verts, faces=mano_template.faces,
            process=False, use_embree=True)
        rhand_mesh = seal(rhand_mesh)
        verts_inside_hand = rhand_mesh.contains(obj_verts)
        obj_vn_new = obj_vn.copy()
        obj_vn_new[verts_inside_hand] = -obj_vn[verts_inside_hand]

        for normal in obj_vn_new:
            if np.allclose(normal, np.zeros(3)):
                print('degenerate normal found:', normal)
        locations, index_ray, index_tri = rhand_mesh.ray.intersects_location(
            ray_origins=obj_verts,
            ray_directions=obj_vn_new,
            multiple_hits=False)

        # check for concave hits
        if len(index_ray) > 0:
            locations2, index_ray2, _ = obj_mesh.ray.intersects_location(
                ray_origins=obj_verts[index_ray] + 1e-4*obj_vn_new[index_ray],
                ray_directions=obj_vn_new[index_ray],
                multiple_hits=False)
            if len(index_ray2) > 0:
                dist_to_self = np.sum((locations2-obj_verts[index_ray][index_ray2])**2, axis=-1)
                dist_to_hand = np.sum((locations[index_ray2]-obj_verts[index_ray][index_ray2])**2, axis=-1)
                non_concave_hits = np.ones(len(index_ray)).astype(bool)
                non_concave_hits[index_ray2[dist_to_self < dist_to_hand]] = 0
                #print('{}/{} concave hits found!'.format((dist_to_self < dist_to_hand).sum(), non_concave_hits.sum()))
                index_ray = np.array(index_ray, dtype=np.int32)[non_concave_hits]
                locations = np.array(locations)[non_concave_hits]
                index_tri = np.array(index_tri, dtype=np.int32)[non_concave_hits]

            corr_mask[index_ray] = True
            corr_dist[index_ray] = np.sqrt(np.sum((locations - obj_verts[index_ray])**2, axis=-1))
            corr_dist[verts_inside_hand] = -corr_dist[verts_inside_hand]

            tri_hit = rhand_mesh.vertices[rhand_mesh.faces[index_tri]]
            assert tuple(tri_hit.shape[1:]) == (3, 3), tri_hit.shape
            bary_coords = trimesh.triangles.points_to_barycentric(tri_hit, locations, method='cross')
            tri_hit_template = mano_template_sealed.vertices[rhand_mesh.faces[index_tri]]
            corr_pts[index_ray] = trimesh.triangles.barycentric_to_points(tri_hit_template, bary_coords)

        obj_corr_mask.append(corr_mask)
        obj_corr_dist.append(corr_dist)
        obj_corr_pts.append(corr_pts.reshape(-1))

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
    # obj_corr_mask: V
    # obj_corr_dist: V
    # obj_corr_pts: V*3
    # is_left: 1
    np_data = list(zip(d['f0'], d['f1'], d['f2'], d['f3'], d['f4'], d['f5'], d['f6'],
        d['f7'], d['f8'], obj_corr_mask, obj_corr_dist, obj_corr_pts, d['f9']))
    np_data = np.array(np_data, dtype=np_dtype)
    np.save(c, np_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grab_path', type=str)
    # preprocessed sequences
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--mano_path', type=str)
    parser.add_argument('--num_proc', default=1, type=int)
    args = parser.parse_args()

    # create process pool
    pool = Pool(args.num_proc)

    # load MANO model
    mano_path = os.path.join(args.mano_path, 'MANO_RIGHT.pkl')
    with open(mano_path, 'rb') as f:
        mano_model = pickle.load(f, encoding='latin1')

    # normalize MANO template
    mano_template = trimesh.Trimesh(vertices=mano_model['v_template'], faces=mano_model['f'],
        process=False)
    with open('data/grab/scale_center.pkl', 'rb') as f:
        scale, center = pickle.load(f)
    mano_template.vertices = mano_template.vertices * scale + center

    id2objmesh = []
    obj_meshes = sorted(os.listdir(os.path.join(args.grab_path, 
        'tools/object_meshes/contact_meshes')))
    for i, fn in enumerate(obj_meshes):
        id2objmesh.append(os.path.join(args.grab_path, 'tools/object_meshes/contact_meshes', fn))

    for split in ['train', 'train_pert', 'val', 'val_pert', 'test', 'test_pert']:
        clips = glob.glob(os.path.join(args.data_path, split, '*.npy'))
        clips = sorted(clips)
        num_clips = len(clips)
        print('Start processing {} split: {} clips found!'.format(split, num_clips))

        pool.starmap(compute_corr, zip(clips, repeat(mano_template), repeat(id2objmesh)))

