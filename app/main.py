import argparse
import depthai as dai
import numpy as np
import cv2
import time
from depthai_sdk.fps import FPSHandler
import open3d as o3d
import logging
from typing import Union, Tuple

from utils import RosBagLogger, calc_fov_D_H_V, create_xyz

def create_point_cloud_from_disparity(
    disp: np.ndarray,
    focal: float,
    baseline: Union[float, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a point cloud from a depth map

    disp (np.ndarray): a 1-channel disparity image. It can be from the sensor or NN output
    focal (int): approximate f_x or f_y of the camera
    baseline (Union[float, int]): the physical distance between left and right sensors in mm

    Returns:
        xyzrgb: a Nx6 point cloud of coordinates and colors
        cloud: a XYZ Nx3 point cloud of coordinates only
        colors: an RBG Nx3 point cloud of colors only
    """

    depth = (focal*baseline)/disp
    depth[disp==0] = 0 # filter out occlusions 
    depth[depth>1500] = 0 # filter out point > 1.5 meters
    depth[seg==0] = 0
    cloud = xyz*np.expand_dims(depth, -1)
    cloud = cloud/1000
    cloud = (cloud*np.array([1,-1,-1])).reshape((-1,3))
    colors = np.stack([right, right, right], -1)
    colors[seg==1, 2] = 255
    colors[seg==2, 0] = 255
    cv2.imshow('overlay', cv2.cvtColor(colors.copy(), cv2.COLOR_BGR2RGB))
    colors = colors.reshape(-1, 3)/255
    xyzrbg = np.concatenate([cloud, colors], axis=-1).astype(np.float32)
    return xyzrbg, cloud, colors

def getShortRangePipeline(blob_path: str, use_nn: bool = True, subpixel: bool = False) -> dai.Pipeline:
    """Returns the DepthAI pipeline to run the BSDR neural network

    blob_path (str): path to the exported blob binary file of the neural network
    subpixel (bool): whether to use subpixel disparity
    """

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)
    if subpixel: stereo.setSubpixel(True)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    if use_nn:

        neuralNetwork = pipeline.create(dai.node.NeuralNetwork)
        neuralNetwork.setBlobPath(blob_path)
        neuralNetwork.setNumInferenceThreads(2)

        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutNN.setStreamName("nn")
        neuralNetwork.out.link(xoutNN.input)

        stereo.rectifiedLeft.link(neuralNetwork.inputs['left'])
        stereo.rectifiedRight.link(neuralNetwork.inputs['right'])
        stereo.disparity.link(neuralNetwork.inputs['disp'])

    else:

        xOutLeft = pipeline.create(dai.node.XLinkOut)
        xOutLeft.setStreamName("left")
        stereo.rectifiedLeft.link(xOutLeft.input)

        xOutRight = pipeline.create(dai.node.XLinkOut)
        xOutRight.setStreamName("right")
        stereo.rectifiedRight.link(xOutRight.input)

        xOutDisparity = pipeline.create(dai.node.XLinkOut)
        xOutDisparity.setStreamName("disp")
        stereo.disparity.link(xOutDisparity.input)

    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=None, type=str, help="Path to the blob file")
    parser.add_argument('-l', '--log', default=None, type=str, help="Optional path in which to save the ROS bag")
    parser.add_argument('-b', '--baseline', default=40, type=str, help="Physical baseline of device in mm")
    args = parser.parse_args()

    fps_logger = logging.getLogger(__name__)

    is_inference = args.model is not None
    pipeline = getShortRangePipeline(
        args.model,
        use_nn=is_inference,
        subpixel=True
    )
    if args.log:
        logger = RosBagLogger(args.log, inference=is_inference, subpixel=True)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    with dai.Device(pipeline) as device:

        width, height = 640, 400
        calibData = device.readCalibration()
        K = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), width, height))
        focal = K[0,0]
        _, hfov, _ = calc_fov_D_H_V(focal, width, height)

        base_ts = time.monotonic()
        simulated_fps = 30
        input_frame_shape = (640, 384)
        counter = 0
        startTime = time.monotonic()

        fps = FPSHandler()

        if args.model:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            xyz = create_xyz(width, height, hfov, K)
            q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=False)
        else:
            q_left = device.getOutputQueue(name="left", maxSize=1, blocking=False)
            q_right = device.getOutputQueue(name="right", maxSize=1, blocking=False)
            q_disp = device.getOutputQueue(name="disp", maxSize=1, blocking=False)

        while True:

            fps.tick("tock")
            fps_logger.debug(f"FPS: {fps.tickFps('tock')}")

            if args.model:

                nnData = q_nn.get()

                ref_layer, seg_layer, left_layer, right_layer, depth_layer = nnData.getAllLayerNames()[:5]
                seg = np.array(nnData.getLayerFp16(seg_layer)).reshape((384,640))
                disp = np.array(nnData.getLayerFp16(depth_layer)).reshape((384,640))
                right = np.array(nnData.getLayerFp16(right_layer)).reshape((384,640)).astype(np.uint8)
                ref = np.array(nnData.getLayerFp16(ref_layer)).reshape((384,640))
                left = np.array(nnData.getLayerFp16(left_layer)).reshape((384,640)).astype(np.uint8)

                disp_cloud, _, _ = create_point_cloud_from_disparity(disp, focal, args.baseline)

                ref_cloud, xyz_cloud, rgb_cloud = create_point_cloud_from_disparity(ref, focal, args.baseline)

            else:

                left = q_left.get().getCvFrame()
                right = q_right.get().getCvFrame()
                disp = q_disp.get().getCvFrame()/8.

            logging_dict = {
                'rect_left': left,
                'rect_right': right,
                'disparity': (disp*8).astype(np.uint16)
            }
            if args.model:
                logging_dict.update({
                    'refined': (ref*8).astype(np.uint16),
                    'mask': (seg*255).astype(np.uint8),
                    'cloud': disp_cloud,
                    'cloud_refined': ref_cloud
                })

            if args.log:
                logger.write(logging_dict)

            disp_show = (disp/190*255).astype(np.uint8)
            if args.model: disp_show = disp_show*(seg>0)
            disp_show = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
            cv2.imshow('left', left)
            cv2.imshow('right', right)
            cv2.imshow('disp', disp_show)
            if args.model:
                ref_show = (ref/190*255).astype(np.uint8)
                ref_show = ref_show*(seg>0)
                ref_show = cv2.applyColorMap(ref_show, cv2.COLORMAP_JET)
                cv2.imshow('ref', ref_show)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s') and args.model:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_cloud)
                pcd.colors = o3d.utility.Vector3dVector(rgb_cloud)
                o3d.visualization.draw_geometries([pcd, axes])

        if args.log:
            logger.writer.close()
