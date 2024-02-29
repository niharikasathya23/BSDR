import torch
import torch.nn.functional as F
import numpy as np
import cv2
import imread_from_url

from nets.crestereo import CREStereo as Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

    # print("Model Forwarding...")
    # imgL = left.transpose(2, 0, 1)
    # imgR = right.transpose(2, 0, 1)
    # imgL = np.ascontiguousarray(imgL[None, :, :, :])
    # imgR = np.ascontiguousarray(imgR[None, :, :, :])

    # imgL = torch.tensor(imgL.astype("float32")).to(device)
    # imgR = torch.tensor(imgR.astype("float32")).to(device)
    # If left and right are already PyTorch tensors, use permute
    if isinstance(left, np.ndarray):
        imgL = torch.tensor(left.transpose(2, 0, 1).astype("float32")).to(device)
        imgR = torch.tensor(right.transpose(2, 0, 1).astype("float32")).to(device)
    elif isinstance(left, torch.Tensor):
        imgL = left
        imgR = right
    else:
        raise TypeError("Input images must be either numpy arrays or PyTorch tensors.")

    # Ensure the tensors have a batch dimension
    imgL = imgL.unsqueeze(0) if imgL.ndim == 3 else imgL
    imgR = imgR.unsqueeze(0) if imgR.ndim == 3 else imgR

    
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
    # print(imgR_dw2.shape)
    with torch.inference_mode():
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

    return pred_disp

import torch.nn as nn
class ModelWrapper(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, imgL, imgR):

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


if __name__ == '__main__':

    left_img = imread_from_url.imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
    right_img = imread_from_url.imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

    # left_img = cv2.imread('/Users/niharika/Desktop/hiro_is/bsdr/scripts/april_6_capture_frames/rect_left/april_6_1_rect_left_50.png')
    # right_img = cv2.imread('/Users/niharika/Desktop/hiro_is/bsdr/scripts/april_6_capture_frames/rect_right/april_6_1_rect_right_50.png')

    in_h, in_w = left_img.shape[:2]

    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h,in_w)
    assert eval_h%8 == 0, "input height should be divisible by 8"
    assert eval_w%8 == 0, "input width should be divisible by 8"

    imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    model_path = "/Users/rithik/Desktop/bsdr/scripts/utils/crestereo/models/crestereo_eth3d.pth"

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)

    # imgL = cv2.resize(imgL, (640, 384))
    # imgR = cv2.resize(imgR, (640, 384))
    # imgL = np.transpose(imgL, (2,0,1))
    # imgR = np.transpose(imgR, (2,0,1))
    # imgL = torch.tensor(imgL).unsqueeze(0)
    # imgR = torch.tensor(imgR).unsqueeze(0)
    imgL = imgL.transpose(2, 0, 1)
    imgR = imgR.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])
    imgL = torch.tensor(imgL.astype("float32"))#.to(device)
    imgR = torch.tensor(imgR.astype("float32"))#.to(device)
    model2 = ModelWrapper(model)
    print(type(imgL), type(imgR))
    from fvcore.nn import flop_count, parameter_count
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model2, (imgL, imgR))
    # print(flops.by_module())
    # flops = flop_count(model, (imgL, imgR))
    print("GFLOPS", np.sum(list(flops.by_module().values()))/1e9)
    mparams = parameter_count(model2)
    print("MParams", np.sum(list(mparams.values()))/1e6)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()

    pred = inference(imgL, imgR, model, n_iter=20)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    combined_img = np.hstack((left_img,right_img, disp_vis))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", combined_img)
    cv2.imwrite("output.jpg", disp_vis)
    cv2.waitKey(0)
