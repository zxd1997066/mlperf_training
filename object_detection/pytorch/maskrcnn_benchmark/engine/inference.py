# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize


def compute_on_dataset(model, data_loader, device, args=None):
    model.eval()
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})
    results_dict = {}
    cpu_device = torch.device("cpu")
    total_time = 0.0
    total_sample = 0
    print(args)
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        elapsed = time.time()
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.time() - elapsed
        if args.profile:
            args.p.step()
        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
        if i >= args.num_warmup:
            total_time += elapsed
            total_sample += args.batch_size

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    throughput = total_sample / total_time
    latency = total_time / total_sample * 1000
    print('inference latency: %.3f ms' % latency)
    print('inference Throughput: %f images/s' % throughput)

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        args=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cpu_time_total")
        print(output)
        import pathlib
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            try:
                os.makedirs(timeline_dir)
            except:
                pass
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    'maskrcnn-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
        p.export_chrome_trace(timeline_file)

    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int(args.num_iter/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            args.p = p
            predictions = compute_on_dataset(model, data_loader, device, args=args)
    else:
        predictions = compute_on_dataset(model, data_loader, device, args=args)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    exit(0)
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
