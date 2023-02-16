import torch
import argparse
import glob
import time

from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import os, numpy as np, time, sys, argparse
from tracking_modules.utils import Config, get_subfolder_seq, createFolder
from tracking_modules.io import (
    load_detection,
    get_saving_dir,
    get_frame_det,
    save_results,
    save_affinity,
)
import numpy as np
from tracking_modules.model import AB3DMOT


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="cfgs/kitti_models/pv_rcnn.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="demo_data",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/pv_rcnn_8369.pth",
        type=str,
        help="specify the pretrained model",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="specify the extension of your point cloud data file",
    )
    parser.add_argument(
        "--tracking_config",
        type=str,
        default="./tracking_modules/configs/config.yml",
        help="tracking config file path",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="confidence threshold of detection",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    tracking_cfg = Config(args.tracking_config)
    return args, cfg, tracking_cfg


class DemoDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        training=True,
        root_path=None,
        logger=None,
        ext=".bin",
    ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = (
            glob.glob(str(root_path / f"*{self.ext}"))
            if self.root_path.is_dir()
            else [self.root_path]
        )

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == ".bin":
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32
            ).reshape(-1, 4)
        elif self.ext == ".npy":
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            "points": points,
            "frame_id": index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def apply_seperate(pred_dicts, tracking_info_data, num_label, thres):
    bbox = {}
    score = {}
    for idx in range(num_label):
        bbox[str(idx + 1)] = []
        score[str(idx + 1)] = []

    for idx, pred_bbox in enumerate(pred_dicts[0]["pred_boxes"]):
        if thres > pred_dicts[0]["pred_scores"][idx]:
            continue

        label = str(pred_dicts[0]["pred_labels"][idx].item())
        bbox[label].append(pred_bbox.tolist())
        score[label].append(pred_dicts[0]["pred_scores"][idx].tolist())

    for idx in range(num_label):
        tracking_info_data["bbox"][str(idx + 1)].append(bbox[str(idx + 1)])
        tracking_info_data["score"][str(idx + 1)].append(score[str(idx + 1)])


def main():
    args, detection_cfg, tracking_cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Superb 3D CAL-------------------------")
    logger.info(f"cuda available : {torch.cuda.is_available()}")
    demo_dataset = DemoDataset(
        dataset_cfg=detection_cfg.DATA_CONFIG,
        class_names=detection_cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.data_path),
        ext=args.ext,
        logger=logger,
    )
    logger.info(f"Total number of samples: \t{len(demo_dataset)}")

    model = build_network(
        model_cfg=detection_cfg.MODEL,
        num_class=len(detection_cfg.CLASS_NAMES),
        dataset=demo_dataset,
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    # args.confidence
    tracking_info_data = {}
    tracking_info_data["pcd"] = []
    tracking_info_data["bbox"] = {}
    tracking_info_data["score"] = {}

    for class_idx in range(len(detection_cfg.CLASS_NAMES)):
        tracking_info_data["bbox"][str(class_idx + 1)] = []
        tracking_info_data["score"][str(class_idx + 1)] = []

    # TODO : set batch size
    total_time = time.time()
    inference_time = time.time()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            tracking_info_data["pcd"].append(data_dict["points"][:, 1:])
            apply_seperate(
                pred_dicts,
                tracking_info_data,
                len(detection_cfg.CLASS_NAMES),
                args.confidence,
            )

    logger.info(f"detection inference time : {time.time()-inference_time}")
    # Tracking

    tracking_time = time.time()
    tracking_results = []
    print(len(tracking_info_data["bbox"][str(0 + 1)]))
    for class_idx, class_name in enumerate(detection_cfg.CLASS_NAMES):
        for pred_bbox in tracking_info_data["bbox"][str(class_idx + 1)]:
            # print(pred_bbox)
            ID_start = 1
            tracker = AB3DMOT(tracking_cfg, class_name, ID_init=ID_start)
            tracking_result, affi = tracker.track(pred_bbox)
            tracking_result = np.squeeze(tracking_result)
        tracking_results.append(tracking_result.tolist())

    logger.info(f"tracking time : { time.time()-tracking_time}")
    logger.info(f"total time : {time.time()-total_time}")
    logger.info("========= Finish =========")


if __name__ == "__main__":
    main()
