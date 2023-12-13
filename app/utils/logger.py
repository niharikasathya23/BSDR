import time
from typing import Dict
import numpy as np

from rosbags.rosbag2 import Writer
from rosbags.serde import serialize_cdr
from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2
from rosbags.typesys.types import sensor_msgs__msg__Image as Image
from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
from rosbags.typesys.types import std_msgs__msg__Header as Header
from std_msgs.msg import Header as HeaderPointCloud2

from rclpy.serialization import serialize_message

from .calc import HostSpatialsCalc
from .projector_3d import PointCloudVisualizer
from .point_cloud2 import create_cloud_xyzrgb32

class RosBagLogger():

    def __init__(self, path: str, inference: bool = True, subpixel: bool = False) -> None:
        """Class to save BSDR inputs and inference to a ROS bag.

        path (str): path in which to save ROS bag
        subpixel (bool): whether to use subpixel disparity
        """

        self.start_nanos = 0
        self.path = path
        self.subpixel = subpixel
        self.writer = Writer(self.path)
        self.writer.open()
        self.inference = inference
        self.connections = {
            'rect_left': self.writer.add_connection('/rect_left', Image.__msgtype__),
            'rect_right': self.writer.add_connection('/rect_right', Image.__msgtype__),
            'disparity': self.writer.add_connection('/disparity', Image.__msgtype__)
        }
        if self.inference:
            self.connections.update({
                'refined': self.writer.add_connection('/refined', Image.__msgtype__),
                'mask': self.writer.add_connection('/mask', Image.__msgtype__),
                'cloud': self.writer.add_connection('/cloud', PointCloud2.__msgtype__),
                'cloud_refined': self.writer.add_connection('/cloud_refined', PointCloud2.__msgtype__)
            })

        self.name_to_encoding = {
            'rect_left': 'mono8',
            'rect_right': 'mono8',
            'disparity': 'mono16' if self.subpixel else 'mono8',
            'refined': 'mono16' if self.subpixel else 'mono8',
            'mask': 'mono8',
            'cloud': 'float32',
            'cloud_refined': 'float32'
        }

    def write(self, data: Dict[str, np.ndarray]) -> None:
        """Writes data at a given timestamp to ROS bag.

        data (Dict[str, np.ndarray]): a dictionary mapping stream names to the image
        """
        if self.start_nanos == 0: self.start_nanos = time.time_ns()

        for name in data:

            frame = data[name]
            if name in ['rect_left', 'rect_right', 'disparity', 'refined', 'mask']:

                img = Image(header=self._get_current_header(),
                    height=frame.shape[0],
                    width=frame.shape[1],
                    encoding=self.name_to_encoding[name],
                    is_bigendian=0,
                    step=frame.shape[1]*2,
                    data=frame.flatten().view(dtype=np.int8))
                type = Image.__msgtype__

                self.writer.write(self.connections[name], time.time_ns() - self.start_nanos, serialize_cdr(img, type))

            elif name in ['cloud', 'cloud_refined']:
                points = frame
                header = HeaderPointCloud2()
                header.frame_id = "odom"
                pc2 = create_cloud_xyzrgb32(header, points)
                ser_msg = serialize_message(pc2)
                self.writer.write(self.connections[name], time.time_ns() - self.start_nanos, ser_msg)

    def _get_current_header(self):
        t_str_arr = ("%.9f" % time.time()).split('.')
        t = Time(sec=int(t_str_arr[0]), nanosec=int(t_str_arr[1]))
        return Header(stamp=t, frame_id='0')