import cv2
import depthai as dai
import numpy as np
import torch
import torch.nn.functional as F
import sys

# Assuming crestereo.py is in the specified directory, adjust if necessary
sys.path.append("/home/nataliya/bsdr/scripts/utils/crestereo/nets/crestereo.py")
from nets.crestereo import CREStereo as Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(left, right, model, n_iter=20):
    # Ensure 'device' is correctly defined as a string or torch.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(left, np.ndarray):
        imgL = torch.tensor(left.transpose(2, 0, 1).astype("float32")).to(device)
        imgR = torch.tensor(right.transpose(2, 0, 1).astype("float32")).to(device)
    elif isinstance(left, torch.Tensor):
        imgL = left.to(device)
        imgR = right.to(device)
    else:
        raise TypeError("Input images must be either numpy arrays or PyTorch tensors.")

    imgL = imgL.unsqueeze(0) if imgL.ndim == 3 else imgL
    imgR = imgR.unsqueeze(0) if imgR.ndim == 3 else imgR

    imgL_dw2 = F.interpolate(imgL, size=(imgL.shape[2] // 2, imgL.shape[3] // 2), mode="bilinear", align_corners=True)
    imgR_dw2 = F.interpolate(imgR, size=(imgR.shape[2] // 2, imgR.shape[3] // 2), mode="bilinear", align_corners=True)

    with torch.no_grad():
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    
    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().numpy()
    return pred_disp


import torch.nn as nn
class ModelWrapper(nn.Module):

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, imgL, imgR, iters=20, flow_init=None):

        imgL_dw2 = F.interpolate(
            imgL,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        imgR_dw2 = F.interpolate(
            imgR,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )

        with torch.inference_mode():
            pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=20, flow_init=None)

            pred_flow = model(imgL, imgR, iters=20, flow_init=pred_flow_dw2)
        
        return self.model(imgL, imgR, iters=iters, flow_init=flow_init)


def create_pipeline():
    pipeline = dai.Pipeline()
    cam_left = pipeline.createMonoCamera()
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam_right = pipeline.createMonoCamera()
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    xout_left = pipeline.createXLinkOut()
    xout_left.setStreamName("left")
    cam_left.out.link(xout_left.input)

    xout_right = pipeline.createXLinkOut()
    xout_right.setStreamName("right")
    cam_right.out.link(xout_right.input)

    return pipeline

if __name__ == '__main__':
    model_path = "/home/nataliya/bsdr/scripts/utils/crestereo/models/crestereo_eth3d.pth"
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    model_wrapper = ModelWrapper(model)

    pipeline = create_pipeline()
    with dai.Device(pipeline) as device:
        leftQueue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        rightQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)

        while True:
            inLeft = leftQueue.tryGet()
            inRight = rightQueue.tryGet()

            if inLeft and inRight:
                frameLeft = np.array(inLeft.getCvFrame())
                frameRight = np.array(inRight.getCvFrame())

                imgL = cv2.cvtColor(frameLeft, cv2.COLOR_GRAY2BGR)
                imgR = cv2.cvtColor(frameRight, cv2.COLOR_GRAY2BGR)
                imgL = cv2.resize(imgL, (640, 400))  # Resize to match model input
                imgR = cv2.resize(imgR, (640, 400))

                pred_disp = inference(imgL, imgR, model_wrapper, n_iter=20)

                # Visualization
                disp_vis = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min()) * 255.0
                disp_vis = disp_vis.astype("uint8")
                disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                cv2.imshow("Disparity", disp_vis)

                if cv2.waitKey(1) == ord('q'):
                    break

    cv2.destroyAllWindows()
