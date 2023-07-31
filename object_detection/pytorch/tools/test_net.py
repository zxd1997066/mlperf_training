# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # for oob
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--precision', type=str, default='float32', help='precision')
    parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_iter', type=int, default=-1, help='num_iter')
    parser.add_argument('--num_warmup', type=int, default=-1, help='num_warmup')
    parser.add_argument('--profile', dest='profile', action='store_true', help='profile')
    parser.add_argument('--quantized_engine', type=str, default=None, help='quantized_engine')
    parser.add_argument('--ipex', dest='ipex', action='store_true', help='ipex')
    parser.add_argument('--jit', dest='jit', action='store_true', help='jit')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # for oob
    cfg.PER_EPOCH_EVAL = True
    cfg.MODEL.DEVICE = args.device
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.num_iter
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    # NHWC
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- Use NHWC model")

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        if args.precision == "bfloat16":
            print('---- Enable AMP bfloat16')
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                inference(
                    model,
                    data_loader_val,
                    dataset_name=dataset_name,
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                    args=args,
                )
        elif args.precision == "float16":
            print('---- Enable AMP float16')
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.half):
                inference(
                    model,
                    data_loader_val,
                    dataset_name=dataset_name,
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                    args=args,
                )
        else:
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                args=args,
            )
        synchronize()


if __name__ == "__main__":
    main()
