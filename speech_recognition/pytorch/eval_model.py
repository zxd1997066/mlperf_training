import json
import os
import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

import torch.nn.functional as F

import sys
import time
### Import Data Utils ###
sys.path.append('../')

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from params import cuda

def eval_model(model, test_loader, decoder, args, device="cpu", batch_time=None):
    start_iter = 0  # Reset start iteration for next epoch
    total_cer, total_wer = 0, 0
    model.eval()
    batch_time_list = []
    for i, (data) in enumerate(test_loader):  # test
        if args.eval_iter != 0 and i > args.eval_iter:
            break
        inputs, targets, input_percentages, target_sizes = data
        print("inputs.size ", inputs.shape)

        # channels last format
        if args.channels_last:
            oob_inputs = inputs
            oob_inputs = oob_inputs.to(memory_format=torch.channels_last)
            inputs = oob_inputs

        inputs = Variable(inputs, volatile=True)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        # if cuda:
        #     inputs = inputs.cuda()

        inputs = inputs.to(device)
        tic = time.time()
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                out = model(inputs)
            profile_iter = args.eval_iter/2 if args.eval_iter > 0 else len(test_loader)/2
            if i == int(profile_iter):
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    os.makedirs(timeline_dir)
                timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                            args.arch + str(i) + '-' + str(os.getpid()) + '.json'
                print(timeline_file)
                prof.export_chrome_trace(timeline_file)
                table_res = prof.key_averages().table(sort_by="cpu_time_total")
                print(table_res)
                # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
        else:
            out = model(inputs)

        #out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()

        decoded_output = decoder.decode(out.data, sizes)
        target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
        toc = time.time()
        if i >= args.eval_warmup and batch_time is not None:
            batch_time_list.append((toc - tic) * 1000)
            batch_time.update(toc - tic)
        print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)

        wer, cer = 0, 0
        for x in range(len(target_strings)):
            wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
            cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
        total_cer += cer
        total_wer += wer

        if device == "cuda":
            torch.cuda.synchronize()
        del out
    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)
    wer *= 100
    cer *= 100
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))

    return wer, cer

def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()

