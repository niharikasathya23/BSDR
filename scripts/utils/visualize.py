import numpy as np
import open3d as o3d

class Open3DVisualizer:

    def __init__(self):
        self.started = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.vis.add_geometry(origin)

        self.geoms = [None]

    def publish(self, geoms):
        if not self.started:
            for gi, geom in enumerate(geoms):
                self.geoms[gi] = geom
                # if vox:
                #
                # else:
                self.vis.add_geometry(self.geoms[gi])
            self.started = True
        else:
            for gi, geom in enumerate(geoms):
                # self.geoms[gi].voxels = geom.voxels
                self.geoms[gi].points = geom.points
                self.geoms[gi].colors = geom.colors
                self.vis.update_geometry(self.geoms[gi])

        self.vis.poll_events()
        self.vis.update_renderer()
