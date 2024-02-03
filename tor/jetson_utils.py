# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import time
from typing import List, Tuple, Union

import torch
from tqdm import tqdm

import numpy as np
from jtop import jtop
import multiprocessing
from multiprocessing import Process, Value

power_sample_period = 0.0005


def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0

    #############
    manager = multiprocessing.Manager()
    power_samples = manager.list()
    inference_done = Value('i', 1)

    def poll_power():
        jetson = jtop()
        jetson.start()

        while inference_done.value == 0:
            power_samples.append(jetson.power["rail"]["VDD_CPU_GPU_CV"]["power"])
            time.sleep(power_sample_period) #e.g. 0.0005 is 0.5 ms -> power sampling rate
        jetson.close()

    #################


    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    ###########################################
                    inference_done.value = 0
                    power_process = Process(target=poll_power)
                    power_process.start()
                    ###########################################
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    ##########################################################
    inference_done.value = 1
    ##########################################################
    elapsed = end - start

    throughput = total / elapsed

    ##############################################
    latency = elapsed * batch_size / total #per batch

    avg_power = np.mean(power_samples) #avg power consumption in mW
    avg_power /= 1000 #avg power consumption in W
    #avg_power = round(avg_power, 2) # W, 2 dp
    #print("Average power consumed (W):", avg_power)
    energy = avg_power * latency
    ###################################

    if verbose:
        print(f"Latency: {latency:.2f} s, Throughput: {throughput:.2f} im/s, Power: {avg_power:.2f} W, Energy: {energy:.2f} J")

    return latency, throughput, avg_power, energy


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

def parse_keep_rate(num_layers: int, keep_rate: int, drop_loc: List[int]) -> List[int]:
    keep_rate_list = [1] * num_layers
    for i in drop_loc:
        keep_rate_list[i] = keep_rate
    return keep_rate_list

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
