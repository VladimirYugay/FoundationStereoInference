import argparse
import gzip
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from pytorch3d.implicitron.dataset.types import \
    FrameAnnotation as ImplicitronFrameAnnotation
from pytorch3d.implicitron.dataset.types import load_dataclass
from pytorch3d.renderer.cameras import PerspectiveCameras
from PIL import Image
import numpy as np
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection
import open3d as o3d


def plot_ptcloud(point_clouds: Union[List, o3d.geometry.PointCloud], show_frame: bool = True):
    """ Visualizes one or more point clouds, optionally showing the coordinate frame.
    Args:
        point_clouds: A single point cloud or a list of point clouds to be visualized.
        show_frame: If True, displays the coordinate frame in the visualization. Defaults to True.
    """
    # rotate down up
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]
    if show_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        point_clouds = point_clouds + [mesh_frame]
    o3d.visualization.draw_geometries(point_clouds)


def rgbd2ptcloud(img, depth, intrinsics, pose=np.eye(4)):
    """converts rgbd image to point cloud
    Args:
        img (ndarray): rgb image
        depth (fcndarray): depth map
        intrinsics (ndarray): intrinsics matrix
    Returns:
        (PointCloud): resulting point cloud
    """
    height, width, _ = img.shape
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(img)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=100,
    )
    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_img, intrinsics, extrinsic=pose, project_valid_depth_only=True)


@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    camera_name: Optional[str] = None


def get_pytorch3d_camera(entry_viewpoint, image_size, scale: float = 1.0) -> PerspectiveCameras:
    principal_point = torch.tensor(
        entry_viewpoint.principal_point, dtype=torch.float
    )
    focal_length = torch.tensor(
        entry_viewpoint.focal_length, dtype=torch.float)
    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )

    format = entry_viewpoint.intrinsics_format
    if format.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig
    elif format.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {format}")

    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    # now, convert from pixels to PyTorch3D v0.5+ NDC convention
    # if self.image_height is None or self.image_width is None:
    out_size = list(reversed(image_size))

    half_image_size_output = torch.tensor(
        out_size, dtype=torch.float) / 2.0
    half_min_image_size_output = half_image_size_output.min()

    # rescaled principal point and focal length in ndc
    principal_point = (
        half_image_size_output - principal_point_px * scale
    ) / half_min_image_size_output
    focal_length = focal_length_px * scale / half_min_image_size_output
    return PerspectiveCameras(
        focal_length=focal_length[None],
        principal_point=principal_point[None],
        R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
        T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
    )


class DynamicReplicaDataset(torch.utils.data.Dataset):

    def __init__(self, path, split, sequence_names):
        super().__init__()
        self.path = Path(path)
        self.split = split
        self.sequence_names = sequence_names

        frame_annotations_file = f'frame_annotations_{split}.jgz'
        with gzip.open(self.path / split / frame_annotations_file, "rt", encoding="utf8") as zipfile:
            self.frame_annots_list = load_dataclass(
                zipfile, List[DynamicReplicaFrameAnnotation])
        self._create_samples()

    def _create_samples(self):
        self.samples = []
        for seq_name in self.sequence_names:
            seq_samples = [
                annot for annot in self.frame_annots_list if annot.sequence_name == seq_name
            ]
            left_samples = [annot for annot in seq_samples]
            right_samples = [annot for annot in seq_samples]
            for left_sample, right_sample in zip(left_samples, right_samples):
                sample = {
                    "seq_name": seq_name,
                    "left_sample": left_sample,
                    "right_sample": right_sample,
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def _load_sample_data(self, sample):

        camera = get_pytorch3d_camera(sample.viewpoint, sample.image.size)
        R, t, K = opencv_from_cameras_projection(camera, torch.tensor(sample.image.size)[None])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R[0].numpy()
        T[:3, 3] = t[0].numpy()

        image_path = self.path / self.split / sample.image.path
        image = np.array(Image.open(image_path).convert("RGB"))

        depth_path = self.path / self.split / sample.depth.path
        with Image.open(depth_path) as depth_pil:
            depth = (
                np.frombuffer(
                    np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )

        mask_path = self.path / self.split / sample.mask.path
        mask = np.array(Image.open(mask_path)) / 255.0
        return {
            "image": image,
            "depth": depth,
            "T": T,
            "K": K[0].numpy(),
            "mask": mask,
        }

    def __getitem__(self, index):
        sample = self.samples[index]
        left_data = self._load_sample_data(sample["left_sample"])
        right_data = self._load_sample_data(sample["right_sample"])
        return {
            "seq_name": sample["seq_name"],
            "left": left_data,
            "right": right_data,
        }


if __name__ == "__main__":
    dataset = DynamicReplicaDataset(
        path="data/datasets/dynamic_replica_data",
        split="valid",
        sequence_names=["f14caa-3_obj"]
    )
    for sample in dataset:
        left_cloud = rgbd2ptcloud(
            sample["left"]["image"],
            sample["left"]["depth"],
            sample["left"]["K"],
            sample["left"]["T"],
        )
        right_cloud = rgbd2ptcloud(
            sample["right"]["image"],
            sample["right"]["depth"],
            sample["right"]["K"],
            sample["right"]["T"],
        )
        plot_ptcloud([left_cloud, right_cloud], show_frame=True)
