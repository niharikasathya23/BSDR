import math
import numpy as np

class HostSpatialsCalc:
    # We need device object to get calibration data
    def __init__(self, fov, focal, width, height):
        # Required information for calculating spatial coordinates on the host
        self.monoHFOV = np.deg2rad(fov) # final FOV we use
        self.focal = focal
        self.width = width
        self.height = height

    def getBaseline(self):
        return 75. # mm

    def getFocal(self):
        if self.focal:
            return self.focal

    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low
    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low

    def _calc_angle(self, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (self.width / 2.0))

    def calc_point_spatials(self, us, vs, zs):
        # shape is (width, height)

        midW = int(self.width / 2) # middle of the depth img width
        midH = int(self.height / 2) # middle of the depth img height

        spatials = np.zeros((len(zs), 3))

        for i, (u, v, z) in enumerate(list(zip(us, vs, zs))):
            bb_x_pos = u - midW
            bb_y_pos = v - midH

            angle_x = self._calc_angle(bb_x_pos)
            angle_y = self._calc_angle(bb_y_pos)
            spatials[i,0] = -z * math.tan(angle_x)
            spatials[i,1] = z * math.tan(angle_y)
            spatials[i,2] = z

        return spatials
