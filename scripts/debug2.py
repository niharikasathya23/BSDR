import numpy as np
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import cv2
from utils.calc import HostSpatialsCalc
import depthai as dai
import open3d as o3d
from models import *
from datasets import *
import argparse
from scipy.signal import medfilt
import cmapy
from utils.losses import get_loss

def gt_callback(key):
    global gt_cloud, gt_there

    if gt_there:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        vis.remove_geometry(gt_cloud)
        ctr.convert_from_pinhole_camera_parameters(params)
        gt_there = False
    else:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        vis.add_geometry(gt_cloud)
        ctr.convert_from_pinhole_camera_parameters(params)
        gt_there = True

def pred_callback(key):
    global pred_cloud, pred_there

    if pred_there:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        vis.remove_geometry(pred_cloud)
        ctr.convert_from_pinhole_camera_parameters(params)
        pred_there = False
    else:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        vis.add_geometry(pred_cloud)
        ctr.convert_from_pinhole_camera_parameters(params)
        pred_there = True

def disp_callback(key):
    global disp_cloud, disp_there

    if disp_there:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        vis.remove_geometry(disp_cloud)
        ctr.convert_from_pinhole_camera_parameters(params)
        disp_there = False
    else:
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        vis.add_geometry(disp_cloud)
        ctr.convert_from_pinhole_camera_parameters(params)
        disp_there = True

def calc_fov_D_H_V(f, w, h):
    return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))

def get_point_cloud(slc, gray, disp, mask, focal, baseline):
    # get point cloud
    vs, us = np.where(mask == 1)
    ds = disp[vs, us]
    zs = -(focal*baseline)/ds # -Z to follow right hand rule
    zs[ds==0] = 0
    zs[zs<-2000] = 0
    spatials = slc.calc_point_spatials(us, vs, zs)/1000 # div by 1000 to convert mm to m
    print(np.unique(spatials))

    channel_intensity = gray[vs, us]/255
    colors = np.zeros((channel_intensity.shape[0], 3))
    colors[:,0] = channel_intensity
    colors[:,1] = channel_intensity
    colors[:,2] = channel_intensity
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(spatials)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main(cfg, save_dir):
    # set up model
    device = torch.device(cfg['DEVICE'])
    palette = eval(cfg['DATASET']['NAME']).PALETTE
    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(palette))
    model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
    model = model.to(device)
    model.eval()

    global gt_cloud
    global gt_there
    global pred_cloud
    global pred_there
    global disp_cloud
    global disp_there
    global vis

    width, height = 640, 400
    if 'CALIB' in cfg['DATASET'].keys():
        calib_path = cfg['DATASET']['CALIB']
        calibData = dai.CalibrationHandler(calib_path)
        K = np.array(calibData.getCameraIntrinsics(calibData.getStereoLeftCameraId(), width, height))
        focal = K[0,0]
        _, hfov, _ = calc_fov_D_H_V(focal, width, height)
        # baseline = 75
        baseline = 40
        slc = HostSpatialsCalc(hfov, focal, width, height)

    # dataloader
    dataset_cfg = cfg['DATASET']
    testset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', dataset_cfg)
    testloader = DataLoader(testset, batch_size=1, num_workers=1, pin_memory=True, shuffle=True)

    loss_fn = get_loss(cfg['LOSS']['NAME'], testset.ignore_label, None)

    for img, lbl, gt_disp, gt_err in testloader:
        img = img.to(device)
        seg, ref = model(img)

        # loss = loss_fn((seg.to('cpu'), ref.to('cpu')), lbl, gt_err)
        # print(loss)

        im = np.squeeze(img.detach().to('cpu').numpy())
        im = np.transpose(im, (1,2,0))
        gray = im[...,0]
        if cfg['DATASET']['NORM_IMG']: gray = gray*255
        gray = gray.astype(np.uint8)
        lbl = lbl[0].detach().to('cpu').numpy().astype(np.uint8)
        key = lbl.copy()
        lbl = (lbl>0).astype(np.uint8)
        lbl_show = np.zeros((key.shape[0],key.shape[1],3)).astype(np.uint8)
        lbl_show[key==1] = [255,0,0]
        lbl_show[key==2] = [0,0,255]

        # post-process seg map
        seg = seg.softmax(dim=1).argmax(dim=1).to(int)
        key = seg[0].detach().to('cpu').numpy().astype(np.uint8)
        # person_mask = person_mask == 0
        seg = np.zeros((key.shape[0],key.shape[1],3)).astype(np.uint8)
        seg[key==1] = [255,0,0]
        seg[key==2] = [0,0,255]
        person_mask = key > 0

        disp = np.squeeze(img[:,1,:,:].detach().to('cpu').numpy())
        disp_show = disp*lbl
        # disp_show = disp
        disp_show = ((disp_show)*255).astype(np.uint8)
        disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        med_disp = medfilt(disp, 7)
        med_disp_show = med_disp*lbl
        med_disp_show = ((med_disp_show)*255).astype(np.uint8)
        med_disp_show = cv2.applyColorMap(med_disp_show, cv2.COLORMAP_JET)

        # post-process disparity
        ref = np.squeeze(ref.detach().to('cpu').numpy())
        if cfg['MODEL']['NAME'] == 'BiSeNetv1Disp3':
            ref = ref.reshape(disp.shape)
            err = np.zeros_like(disp)
        else:
            err = ref.copy()
            ref = ref+disp
        ref_show = ref*person_mask
        if cfg['MODEL']['NAME'] == 'BiSeNetv1Disp3': ref_show = ref_show/cfg['DATASET']['MAX_DISP']
        ref_show = (ref_show*255).astype(np.uint8)
        ref_show = cv2.applyColorMap(ref_show, cv2.COLORMAP_JET)
        disp_values = ref[person_mask]
        # print('Pred:', disp_values)

        gt_disp = gt_disp[0].detach().to('cpu').numpy()
        # print(gt_disp.min(), gt_disp.max())
        gt_disp_show = gt_disp*person_mask
        if not cfg['DATASET']['NORM_GT_DISP']: gt_disp_show = gt_disp_show/cfg['DATASET']['MAX_DISP']
        gt_disp_show = (gt_disp_show*255).astype(np.uint8)
        gt_disp_show = cv2.applyColorMap(gt_disp_show, cv2.COLORMAP_JET)
        disp_values = gt_disp[person_mask]
        # print('GT:', disp_values)

        ref_show_2 = ref*lbl
        # ref_show_2 = ref
        if cfg['MODEL']['NAME'] == 'BiSeNetv1Disp3': ref_show_2 = ref_show_2/cfg['DATASET']['MAX_DISP']
        ref_show_2 = (ref_show_2*255).astype(np.uint8)
        ref_show_2 = cv2.applyColorMap(ref_show_2, cv2.COLORMAP_JET)
        gt_disp_show_2 = gt_disp*lbl
        # gt_disp_show_2 = gt_disp
        if not cfg['DATASET']['NORM_GT_DISP']: gt_disp_show_2 = gt_disp_show_2/cfg['DATASET']['MAX_DISP']
        gt_disp_show_2 = (gt_disp_show_2*255).astype(np.uint8)
        gt_disp_show_2 = cv2.applyColorMap(gt_disp_show_2, cv2.COLORMAP_JET)

        gt_err = np.squeeze(gt_err.detach().to('cpu').numpy())

        # err_show = ((err*lbl+1)/2*255).astype(np.uint8)
        err_show = ((err+1)/2*255).astype(np.uint8)
        err_show = cv2.applyColorMap(err_show, cmapy.cmap('seismic'))
        # gt_err_show = ((gt_err*lbl+1)/2*255).astype(np.uint8)
        gt_err_show = ((gt_err+1)/2*255).astype(np.uint8)
        gt_err_show = cv2.applyColorMap(gt_err_show, cmapy.cmap('seismic'))

        overlay = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.5, seg, 0.5, 0)

        # cv2.imshow('gray', gray)
        # cv2.imshow('seg', seg)
        # cv2.imshow('label', lbl_show)
        cv2.imshow('mask', overlay)
        # cv2.imshow('ref + pred mask', ref_show)
        # cv2.imshow('GT + pred mask', gt_disp_show)

        cv2.imshow('ref + lbl', ref_show_2)
        cv2.imshow('GT + lbl', gt_disp_show_2)
        cv2.imshow('disp + lbl', disp_show)
        # cv2.imshow('median disp + lbl', med_disp_show)

        # cv2.imshow('pred err', err_show)
        # cv2.imshow('gt err', gt_err_show)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

        if args.vis_3d:

            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            gt_cloud = get_point_cloud(slc, gray, gt_disp, lbl, focal, baseline)
            pred_cloud = get_point_cloud(slc, gray, ref, lbl, focal, baseline)
            disp = disp*256
            print(gt_disp.min(), gt_disp.max())
            print(disp.min(),disp.max())
            disp_cloud = get_point_cloud(slc, gray, disp, lbl, focal, baseline)

            # o3d.visualization.draw_geometries([gt_cloud, axes])
            # o3d.visualization.draw_geometries([pred_cloud, axes])

            # combined
            # gt_colors = np.ones_like(gt_cloud.points)*[0,0,1]
            gt_colors = np.array(gt_cloud.colors)+[0,0,0.2]
            gt_colors = np.clip(gt_colors, 0, 1)
            gt_cloud.colors = o3d.utility.Vector3dVector(gt_colors)
            # pred_colors = np.ones_like(pred_cloud.points)*[1,0,0]
            pred_colors = np.array(pred_cloud.colors)+[0.2,0,0]
            pred_colors = np.clip(pred_colors, 0, 1)
            pred_cloud.colors = o3d.utility.Vector3dVector(pred_colors)
            disp_colors = np.array(disp_cloud.colors)+[0,0.2,0]
            disp_colors = np.clip(disp_colors, 0, 1)
            disp_cloud.colors = o3d.utility.Vector3dVector(disp_colors)

            pred_cloud, ind = pred_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                          std_ratio=2.0)

            # o3d.visualization.draw_geometries([gt_cloud, axes])
            # o3d.visualization.draw_geometries([disp_cloud, axes])
            # o3d.visualization.draw_geometries([pred_cloud, axes])
            # o3d.visualization.draw_geometries([gt_cloud, pred_cloud, disp_cloud, axes])

            gt_there, pred_there, disp_there = True, False, False
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            vis.register_key_callback(73, gt_callback)
            vis.register_key_callback(79, pred_callback)
            vis.register_key_callback(80, disp_callback)
            vis.add_geometry(axes)
            vis.add_geometry(gt_cloud)
            vis.run()
            vis.destroy_window()

            # gt_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(gt_cloud, voxel_size=0.05)
            # pred_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pred_cloud, voxel_size=0.05)
            # o3d.visualization.draw_geometries([gt_voxel, pred_voxel, axes])

            # o3d.io.write_point_cloud('gt.ply', gt_cloud)
            # o3d.io.write_point_cloud('pred.ply', pred_cloud)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--vis_3d', action='store_true')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)

    main(cfg, save_dir)
