# Copyright ETRI. All Rights Reserved.
import os
import logging
import sys
from typing import List, Tuple
import numpy as np
import ast
import argparse
import torch
import cv2
from termcolor import colored
from torch.nn.parallel._functions import Gather


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def setup_logger(name, save_dir, distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    #formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=name,
            )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
            super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def printDict(mydict, name):
    logger.info("\n", "="*10, "< ", name, " >", "="*10)
    for i in range(len(mydict)):
        logger.info(i, " : ", mydict[i])


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumetTypeError("Argument is not a list")
    return v


def get_final_res(lenList: int, col_num: int, c_pix: int, r_pix: int) -> Tuple:
    if lenList <= col_num:
        return (col_num*c_pix, r_pix)
    #elif lenList <= col_num*2:
    #    return (7*c_pix, 2*r_pix)
    else:
        return (col_num*c_pix, (lenList//col_num + 1)*r_pix)


def to_matrixView(pic_list: list, col: int, res: Tuple) -> np.ndarray:
    row = []
    full = []
    count = 0
    for pic in pic_list:
        # logger.info("pic.shape ==> ", pic.shape)
        row.append(pic)
        count = count + 1
        if count == col:
            # logger.info("======================")
            full.append(np.concatenate(row, 1))
            row = []
            count = 0
    while not count==col:
        count = count + 1
        black = np.zeros(pic.shape, dtype=np.uint8)
        row.append(black)
    full.append(np.concatenate(row, 1))

    return cv2.resize(np.concatenate(full, 0), res)


def endProcess(proc):
    proc.terminate()
    logger.info ('Process terminated:', proc, proc.is_alive())
    proc.join()
    logger.info ('Process joined:', proc, proc.is_alive())
    logger.info ('Process exit code:', proc.exitcode)


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.

    try:
        #res = gather_map(outputs)
        res = []
        for output in outputs:
            res+=output
    finally:
        gather_map = None
    return res


def create_reconnectDict(args):
    ## Create reconnect_dict
    '''
    reconnect_dict = dict{
        0 : [A, B, C]
        1 : [A, B, C]
        ...
    }
    - A (Boolean) : Is it necessary to reconnect this channel? (default = False)
    - B (time.time()) : The most recent time when event sent successfully (default = 1)
    - C (dict()) : if A == True, event data to resend (default = None)
    - possible values :
        1. [False, 1, None] : Default
        2. [False, time.time(), None] : send_msg Process sent event successfully
        3. [True, 1, info] : send_msg Process failed to send event
        4. [True, 0, None] : Thread reconnected rtsp and put resending event into queue
    '''
    reconnect_dict = dict()
    for idx in range(len(args.rtsp_channels)):
        reconnect_dict[idx] = [False, 1, None]

    return reconnect_dict


