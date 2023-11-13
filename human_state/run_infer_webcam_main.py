from run_infer_webcam_data import LoadStreams

import threading
import os
import logging
import cv2
import torch
import time

logger = logging.getLogger(__name__)


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availablity
    cuda = False if cpu_request else torch.cuda.is_available()

    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, f'batch-size {batch_size} not multiple of GPU count {ng}'
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i, d in enumerate((device or '0').split(',')):
            if i == 1:
                s = ' ' * len(s)
            logger.info(f"{s}CUDA:{d} ({x[i].name}, {x[i].total_memory / c}MB)")
    else:
        logger.info(f'Using torch {torch.__version__} CPU')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def detect(rtsp_url):
    dataset = LoadStreams(rtsp_url)
    device = select_device('1')
    count = 0
    view_img = True
    imgsz = 640
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    try:
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):  # for every frame
            count += 1
            print(im0s[0])
            im0 = im0s[0].copy()
            if view_img:
                cv2.imshow(str(path), im0)
                # if cv2.waitKey(1) == ord('q'):  # q to quit
                #     raise StopIteration
    except:
        print("finish execption")
        dataset.stop()
    return "good"


if __name__ == '__main__':
    rtsp_url = "rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/2/media.smp"
    while True:
        for thread in threading.enumerate():
            print("thread.name: ", thread.name)
        print(detect(rtsp_url))

