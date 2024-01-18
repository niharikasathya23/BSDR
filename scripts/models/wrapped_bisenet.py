import torch
import torch.nn as nn

class WrappedBiSeNetRGB(nn.Module):

    def __init__(self, model):
        super(WrappedBiSeNet, self).__init__()
        self.model = model

    def forward(self, x):
        x = x[:,:,8:-8,:]
        x = x.repeat(1,3,1,1)
        return self.model(x)

class WrappedBiSeNet(nn.Module):

    def __init__(self, model):
        super(WrappedBiSeNet, self).__init__()
        self.model = model

    # def forward(self, xyz, right, disp):
    #     baseline = 40 # constant, mm
    #     focal = 399.0516357421875 # constant to simplify static vars in pipeline
    #     right = right[:,:,8:-8,:]
    #     disp = disp[:,:,8:-8,:] #/ 8
    #     disp = 256.0 * disp[:,:,:,1::2] + disp[:,:,:,::2]
    #     disp = disp/8
    #     depth = (focal*baseline)/disp
    #     # depth = (focal*baseline)/(disp+1e-16)
    #     # depth[disp==0] = 0
    #     depth = depth.permute(0, 2, 3, 1)
    #     full_cloud = xyz * depth
    #     full_cloud.permute(0, 3, 1, 2)
    #     input = torch.cat([right, disp/190.], dim=1)
    #     seg, ref = self.model(input)
    #
    #     mask = torch.argmax(seg, dim=1).float()
    #     # cloud = full_cloud[mask==1]
    #
    #     # cloud = cloud/1000
    #     # cloud = cloud*torch.tensor([1,-1,-1])
    #
    #     return mask, disp, full_cloud

    def forward(self, left, right, disp):
        baseline = 40 # constant, mm
        focal = 399.0516357421875 # constant to simplify static vars in pipeline
        left = left[:,:,8:-8,:]
        right = right[:,:,8:-8,:]
        disp = disp[:,:,8:-8,:]
        disp = 256.0 * disp[:,:,:,1::2] + disp[:,:,:,::2]
        disp = disp/8
        # depth = (focal*baseline)/disp

        input = torch.cat([right, disp/190.], dim=1)
        seg, err = self.model(input)
        mask = torch.argmax(seg, dim=1).float()
        ref = disp+(err*190.)

        return mask, disp, ref, left, right
