import numpy as np
import os
import torch
#from smplx import SMPL
from lib.models.smpl import SMPL
from tqdm import tqdm
from pathlib import Path
from lib.vis.traj import *
import pickle
from quat import from_xform, to_scaled_angle_axis


def convert_smpl_output_to_pkl(smpl_output, locations, output_path):
    global_orient = smpl_output.global_orient.cpu().numpy()  # (N, 1, 3, 3)
    body_pose = smpl_output.body_pose.cpu().numpy()        # (N, 24, 3, 3)
    num_frames = global_orient.shape[0]

    smpl_poses = np.zeros((num_frames, 72)) #注意这里是72

    for i in range(num_frames):
        global_orient_matrix = global_orient[i, 0]  # (3, 3)
        global_orient_quat = from_xform(global_orient_matrix[None,:,:])  # (1, 4)
        global_orient_scaled_aa = to_scaled_angle_axis(global_orient_quat).flatten()  # (3,)
        smpl_poses[i, :3] = global_orient_scaled_aa

        for j in range(23):
            body_pose_matrix = body_pose[i, j]  # (3, 3)
            body_pose_quat = from_xform(body_pose_matrix[None,:,:])   # (1, 4)
            body_pose_scaled_aa = to_scaled_angle_axis(body_pose_quat).flatten()  # (3,)
            smpl_poses[i, 3 + j * 3:3 + (j + 1) * 3] = body_pose_scaled_aa

    smpl_scaling = np.array([1.0])

    data = {
        "smpl_poses": smpl_poses,
        "smpl_scaling": smpl_scaling,
        "smpl_trans": locations,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)



def save_obj(vertices, faces, file_path):
    """
    将顶点和面信息保存为 .obj 文件格式
    """

    with open(file_path, 'w') as f:
        # 写入顶点信息
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # 写入面信息（索引从1开始）
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_smpl_frames_to_objs(hps_folder="results/example_video/hps", output_folder="results", smpl_model_path="data/smpl/SMPL_NEUTRAL.pkl", gender='neutral'):
    """
    将每一帧 SMPL 模型保存为 .obj 文件
    """
    # 初始化 SMPL 模型
    smpl = SMPL(model_path=smpl_model_path, gender=gender)
    faces = smpl.faces  # 获取模型的面数据

    # 获取所有 .npy 文件（每个文件对应一组帧数据）
    hps_files = sorted([os.path.join(hps_folder, f) for f in os.listdir(hps_folder) if f.endswith('.npy')])

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    pred_cam = np.load(f'results/example_video/camera.npy', allow_pickle=True).item()
    img_focal = pred_cam['img_focal'].item()
    pred_cam_R = torch.tensor(pred_cam['pred_cam_R'])
    pred_cam_T = torch.tensor(pred_cam['pred_cam_T'])

    for hps_file in tqdm(hps_files, desc="Processing frames"):
        stem = Path(hps_file).stem
        # 加载帧数据
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        #print(pred_smpl)

        pred_rotmat = pred_smpl['pred_rotmat']  # 旋转矩阵
        pred_shape = pred_smpl['pred_shape']    # 形状参数
        pred_trans = pred_smpl['pred_trans']    # 平移
        frame_indices = pred_smpl['frame']  # 帧索引

        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        pred = smpl(body_pose=pred_rotmat[:,1:],
                    global_orient=pred_rotmat[:,[0]],
                    betas=pred_shape,
                    transl=pred_trans.squeeze(),
                    pose2rot=False,
                    default_smpl=True)
       # print(pred)

        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]

        cam_r = pred_cam_R[frame_indices]
        cam_t = pred_cam_T[frame_indices]

        pred_vert_w = torch.einsum('bij,bnj->bni', cam_r, pred_vert) + cam_t[:,None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', cam_r, pred_j3d) + cam_t[:,None]
        pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
        locations = pred_j3d_w.mean(1).cpu().numpy()

        convert_smpl_output_to_pkl(pred, locations,f'{stem}_smplx_for_bvh.pkl')

        '''
        # 逐帧生成顶点并保存为 .obj 文件
        for i, frame_idx in enumerate(frame_indices):
            # 获取当前帧的姿态、形状和位移
            vertices = pred_vert[i]

            print(output_folder)
            # 保存为 .obj 文件
            obj_filename = f"results/{stem}_frame_{frame_idx:04d}.obj"
            print(obj_filename)
            save_obj(vertices, faces, obj_filename)
        '''

if __name__ == '__main__':
    save_smpl_frames_to_objs()
