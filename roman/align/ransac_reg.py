import numpy as np
from typing import List

import open3d as o3d

from roman.align.object_registration import ObjectRegistration
from roman.object.pointcloud_object import PointCloudObject

class RansacReg(ObjectRegistration):
    def __init__(self, edge_len=0.95, dim=3, max_iteration=int(1e6)): # We are not using these
        assert dim == 3, "Only 3D points supported for RANSAC registration."
        self.edge_len = edge_len
        self.max_iteration = max_iteration
        super().__init__(0.0, 0.0, 0.0, dim)
    
    def register(self, map1: List[PointCloudObject], map2: List[PointCloudObject]):
        # For RANSAC, we take the center of each object's pointcloud.
        pcd_ransac_1 = [seg.center.reshape(-1) for seg in map1]
        pcd_ransac_2 = [seg.center.reshape(-1) for seg in map2]

        # for seg in map1:
        #     pcd_ransac_1.append(seg.center)

        # for seg in map2:
        #     pcd_ransac_2.append(seg.center)
        
        corres = []
        for i in range(len(pcd_ransac_1)):
            for j in range(len(pcd_ransac_2)):
                corres.append([i, j])

        corres = o3d.utility.Vector2iVector(corres)

        try:
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(np.asarray(pcd_ransac_1))

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd_ransac_2))
        except:
            return np.array([[]])

        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source = pcd1,
            target = pcd2,
            corres = corres,
            max_correspondence_distance=0.5,
            ransac_n = 3,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.edge_len)],
            criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=self.max_iteration)
        )
        
        return np.asarray(result.correspondence_set)

        