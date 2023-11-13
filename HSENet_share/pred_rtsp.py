from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

import torch
from PIL import Image
import fvcore
import os
import glob
import tqdm
import time

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, KEYPOINT_CONNECTION_RULES
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

import datetime
import requests, json
import numpy as np

# HSENet_example
from hsenet.utils.visualizer import Visualizer

'''
New Import
'''
import argparse
import ast
import detectron2.data.transforms as T
import logging
import detectron2.data.transforms as T

from hsenet.modeling.meta_arch.build import build_model
from detectron2.checkpoint import DetectionCheckpointer
from camera_read_v3 import VideoScreenshot
from torch.multiprocessing import Process, Queue, Manager
from hsenet.config.config import add_posture_config
from hsenet.modeling.backbone import build_vovnet_fpn_backbone
from hsenet.modeling.meta_arch.rcnn import HSENetRCNN
from hsenet.modeling.roi_heads.roi_heads import HSENetROIHeads
import multiprocessing
from open_cam_rtsp import camera_id_return, open_cam_login
from utils import setup_logger, arg_as_list, to_matrixView, get_final_res, gather, create_reconnectDict

# Turn opencv log off
# os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
import cv2

dataset_root = './datasets/'

meta = {
    "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
    "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
    "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
}

register_coco_instances("MPHBE2020_train", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_train2020.json", dataset_root + "MPHBE2020/train2020")
register_coco_instances("MPHBE2020_test", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_test2020.json", dataset_root + "MPHBE2020/test2020")

data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])
data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_colors=[(0, 0, 0), (0, 0, 0), (0, 0, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

logger = setup_logger('CCTV', 'results/logs/')

# Disable detectron2 logger
detectron2_logger = logging.getLogger('detectron2')
detectron2_logger.disabled = True
detectron2_logger = logging.getLogger('detectron2.utils.env')
detectron2_logger.disabled = True
detectron2_logger = logging.getLogger('fvcore.common.checkpoint')
detectron2_logger.disabled = True


class CustomQueue():
    def __init__(self, max_size):
        self.max_size = max_size
        manager = Manager()
        self.queue = manager.list()
        self.currIndex = manager.Value('i', 0)

        thread = Process(target=self.pop, args=())
        thread.start()

    def put(self, x):
        if (len(self.queue) - self.currIndex.value) < self.max_size:
            self.queue.append(x)

    def pop(self):
        while 1:
            for _ in range(self.currIndex.value):
                del self.queue[0]
                self.currIndex.value -= 1
            time.sleep(0.03)

    def get(self):
        while 1:
            if (len(self.queue) - self.currIndex.value) > 0:
                # print(self.currIndex.value)
                x = self.queue[0]
                # del self.queue[0]
                self.currIndex.value += 1
                # print(x, len(self.queue))
                return x
                # return self.queue.pop(0)
            else:
                time.sleep(0.01)

    def get_nowait(self):
        # print(len(self.queue), self.currIndex.value)
        if (len(self.queue) - self.currIndex.value) > 0:
            x = self.queue[0]
            # del self.queue[0]
            self.currIndex.value += 1
            # print(x, len(self.queue))
            return x
            # return self.queue.pop(0)
        else:
            # pass
            return False, None, None
            # time.sleep(0.01)

    def empty(self):
        if len(self.queue) == 0:
            return True
        else:
            return False

    def full(self):
        if len(self.queue) == self.max_size:
            # time.sleep(0.3)
            return True
        else:
            return False

    def qsize(self):
        return len(self.queue)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_posture_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def multi_gpu_rtsp(gpu, ngpus_per_node, args, multi_rtsp_queue, multi_event_queue, multi_vis_queue, multi_batch_queue, skipped_frame_dict, func_time):
    ## Inference with a multitask model
    cfg = setup(args)
    # filter out low-scored bounding boxes predicted by the Fast R-CNN
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # set threshold for this model

    save_path_org = args.save_path_org
    save_path_result = args.save_path_result

    play_video = True
    video_res = args.video_res
    ret = True
    # force_reconnect = False
    max_count = 10
    counter = 0
    count_thres = 5
    recent_sents = [None for _ in range(len(args.rtsp_channels))]
    time_count = np.zeros(max_count)
    sleep_time = 60

    fail_count = 1

    rtsp_channel_div = np.array_split(args.rtsp_channels, ngpus_per_node)[gpu]
    rtsp_channel_div = [camera_id_return(x, args.rtsp_region, args.server_number) for x in rtsp_channel_div]

    event_counts = np.zeros((len(rtsp_channel_div), max_count))

    # Initialize event_sended time
    event_sended = dict()
    # for i in range(len(rtsp_channel_div)):
    for i in rtsp_channel_div:
        event_sended[i] = -1

    '''
    View by channel & Final view setting
    '''
    if args.show_video:
        if gpu == 0:
            cv2.namedWindow('test')
        black_view = np.zeros([video_res[1], video_res[0], 3], dtype=np.uint8)
        views = [black_view for _ in range(len(args.rtsp_channels))]

        garo = 455 // 2
        saero = 350 // 2
        if len(views) < 4:
            final_viewCol = len(views)
        else:
            final_viewCol = 7
        final_res = get_final_res(len(views), final_viewCol, garo, saero)
        while len(views) % final_viewCol != 0:
            views.append(black_view)

    if args.debug:
        logger.info("Command Line Args:", args)

    '''
    Model Configuration
    '''
    predictor = build_model(cfg)
    DetectionCheckpointer(predictor).load(cfg.MODEL.WEIGHTS)
    predictor.train(False)

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:13456',
        world_size=ngpus_per_node,
        rank=gpu)
    torch.cuda.set_device(gpu)
    predictor = torch.nn.parallel.DistributedDataParallel(predictor.half().cuda(gpu), device_ids=[gpu])
    predictor.eval()
    predictor.gather = gather

    if args.debug:
        logger.info(fvcore.nn.parameter_count_table(predictor.model))

    if gpu == 0:
        t_bar = tqdm.auto.tqdm()

    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

    # time.sleep(10)
    frame_skipped = 0
    while 1:
        t = time.time()

        if multi_batch_queue[gpu].empty():
            # frame_skipped+=len(args.rtsp_channels)
            continue

        frame_skipped = 0
        for skip_gpu_key in skipped_frame_dict.keys():
            status = skipped_frame_dict[skip_gpu_key]
            frame_skipped += status

        status, inputs, ori_img = multi_batch_queue[gpu].get()

        image_grab_time = time.time()

        with torch.no_grad():
            outputs = predictor(inputs.cuda())

        model_inference_time = time.time()

        if args.show_video:
            if play_video:
                for output_i, view_i in enumerate(range(len(ori_img))):
                    # h, w, c = im_q[view_i].shape
                    h, w, c = ori_img[view_i].shape
                    v = Visualizer(ori_img[output_i], data_meta, scale=1.2)
                    # logger.info(outputs[output_i]["instances"].pred_classes)
                    out = v.draw_instance_predictions(outputs[output_i]["instances"].to("cpu"))
                    views[view_i] = cv2.resize(out.get_image()[:, :, ::-1], (w, h))
                    view_string = 'Channel ' + str(rtsp_channel_div[output_i])
                    cv2.putText(views[view_i], view_string, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    multi_vis_queue[rtsp_channel_div[output_i]].put(views[view_i])

                if gpu == 0:
                    multi_frame_gather = [multi_vis_queue[x].get() for x in multi_vis_queue.keys()]
                    # cv2.imshow('test', to_matrixView(views, final_viewCol, final_res))
                    cv2.imshow('test', to_matrixView(multi_frame_gather, final_viewCol, final_res))
            if gpu == 0:
                ret = cv2.waitKey(1)
            if ret > 0 and gpu == 0:
                play_video = not play_video

        if args.send_events:
            ## Event detection (including send_events and save_frame)
            event_detect = np.array([(output["instances"].pred_classes == 2).any().cpu() for output in outputs])
            event_counts[:, counter % max_count] = event_detect

            if np.abs(5 / (time_count[np.nonzero(time_count)].mean()) - max_count) > 5:
                max_count = np.floor(5 / (time_count[np.nonzero(time_count)].mean())).astype(int) + 1
                count_thres = max_count // 2

                event_counts = np.zeros((len(rtsp_channel_div), max_count))
                time_count = np.zeros(max_count)
                logger.info("max_count changed to {}".format(max_count))

            if counter > max_count:
                counter = 0

            event_send = event_counts.sum(1) > count_thres
            cur_time = time.time()

            event_flag = [[rtsp_channel_div[i], send] for i, send in enumerate(event_send)]  # if ((cur_time - event_sended[rtsp_channel_div[i]]) > sleep_time) and send]

            event_time = str(datetime.datetime.now())[:-3]
            for i, [channel, send] in enumerate(event_flag):
                # print(cur_time - event_sended[channel])
                if send and cur_time - event_sended[channel] > sleep_time:
                    multi_event_queue.put([channel, event_time, 0, ori_img[i], outputs[i]["instances"].to("cpu")])
                    event_counts[i, :] = 0
                    event_sended[channel] = time.time()

        postprocess_time = time.time()

        inf_t = time.time() - t
        time_count[counter % max_count] = inf_t
        counter += 1

        func_time['Total'][gpu] = inf_t
        func_time['Input'][gpu] = image_grab_time - t
        func_time['Inference'][gpu] = model_inference_time - image_grab_time
        func_time['Post'][gpu] = postprocess_time - model_inference_time

        if gpu == 0:
            func_list = []
            for kk in ['Total', 'Input', 'Inference', 'Post', 'update', 'preprocess_inputs', 'batch_inputs']:
                func_list.append(sum([func_time[kk][key] for key in func_time[kk].keys()]) / len(func_time[kk].keys()))
            # Print total time including all processes.
            t_bar.set_description("Total: {:.1f}ms, Input: {:.1f}ms, Inference: {:.1f}ms, Post: {:.1f}ms, SkipFrame: {}, Update: {:.1f}ms, Preprocess: {:.1f}ms, Batch: {:.1f}ms".format(
                1000. * func_list[0],
                1000. * func_list[1],
                1000. * func_list[2],
                1000. * func_list[3],
                frame_skipped,
                1000. * func_list[4],
                1000. * func_list[5],
                1000. * func_list[6]
            )
            )
            t_bar.refresh()


if __name__ == "__main__":
    ## Arguments
    args = default_argument_parser()
    args.add_argument('--rtsp_region', default=0, type=int)
    args.add_argument('--rtsp_mode', default=0, type=int)  # channel for list
    args.add_argument('--show_video', action='store_true', default=False)
    args.add_argument('--debug', action='store_true', default=False)
    args.add_argument('--send_events', action='store_true', default=False)
    args.add_argument('--save_frame', action='store_true', default=False)
    args.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    args.add_argument('--save_path_org', type=str, default='results/test_frame')
    args.add_argument('--save_path_result', type=str, default='results/test_results')
    args.add_argument('--server_number', type=int, default=0)
    '''
    Getting multi-channels by argument
    '''
    args.add_argument('--rtsp_channels', default=[], type=arg_as_list)
    args = args.parse_args()

    ## Parameters
    # video_res = None
    # video_res = (1280, 720)
    # video_res = (640, 480)
    args.video_res = (1067, 800)

    ## Create log paths
    save_path_org = args.save_path_org
    if not os.path.exists(save_path_org):
        os.makedirs(save_path_org)

    save_path_result = args.save_path_result
    if not os.path.exists(save_path_result):
        os.makedirs(save_path_result)

    ## Create reconnect_dict
    '''
    Getting url for each rtsp
    '''
    rtsp_infos = dict()
    rtsp_infos['send_msg'] = Manager().dict()
    rtsp_infos['send_msg']['camera_id'] = 'send_msg'
    rtsp_infos['send_msg']['api_serial'] = None
    rtsp_infos['send_msg']['auth_token'] = None
    rtsp_infos['send_msg']['api_serial'] = None

    rtsp_infos['global'] = Manager().dict()
    rtsp_infos['global']['camera_id'] = 'global'
    rtsp_infos['global']['api_serial'] = None
    rtsp_infos['global']['auth_token'] = None
    rtsp_infos['global']['api_serial'] = None
    for idx, channel in enumerate(args.rtsp_channels):
        camera_id = camera_id_return(channel, args.rtsp_region, args.server_number)
        idx = camera_id
        rtsp_infos[idx] = Manager().dict()
        rtsp_infos[idx]['rtspUrl'] = None
        rtsp_infos[idx]['auth_token'] = None
        rtsp_infos[idx]['api_serial'] = None
        rtsp_infos[idx]['target_ip'] = None
        rtsp_infos[idx]['camera_id'] = camera_id
        rtsp_infos[idx]['userInput'] = channel

    ## Multiprocessing Queue
    manager = Manager()
    manager.Queue = CustomQueue

    # RTSP raw input queue
    multi_raw_inputs = dict()
    for idx, channel in enumerate(args.rtsp_channels):
        camera_id = camera_id_return(channel, args.rtsp_region, args.server_number)
        multi_raw_inputs[camera_id] = manager.Queue(5)

    # RTSP input queue
    multi_rtsp_queue = dict()
    for idx, channel in enumerate(args.rtsp_channels):
        camera_id = camera_id_return(channel, args.rtsp_region, args.server_number)
        multi_rtsp_queue[camera_id] = manager.Queue(5)
    # multi_rtsp_queue = manager.List(len(args.rtsp_channels)*[None])

    # Event queue
    multi_event_queue = manager.Queue(20)

    # Visualization queue
    multi_vis_queue = dict()
    for idx, channel in enumerate(args.rtsp_channels):
        camera_id = camera_id_return(channel, args.rtsp_region, args.server_number)
        multi_vis_queue[camera_id] = manager.Queue(5)

    # Preprocess to batch
    ngpus_per_node = args.num_gpus
    multi_batch_queue = dict()
    for i in range(ngpus_per_node):
        multi_batch_queue[i] = manager.Queue(5)

    # Misc queuess
    skipped_frame_dict = manager.dict()
    func_time = dict()
    func_time['update'] = manager.dict()
    func_time['preprocess_inputs'] = manager.dict()
    func_time['batch_inputs'] = manager.dict()
    func_time['Total'] = manager.dict()
    func_time['Input'] = manager.dict()
    func_time['Inference'] = manager.dict()
    func_time['Post'] = manager.dict()

    ## Generating threads for each rtsp
    rtsp_q = VideoScreenshot(args, rtsp_infos, ngpus_per_node, args.rtsp_region, multi_raw_inputs, multi_rtsp_queue, multi_event_queue, multi_batch_queue, skipped_frame_dict, func_time)

    ## Spawn multiprocessing
    # ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node

    torch.multiprocessing.spawn(multi_gpu_rtsp, nprocs=ngpus_per_node,
                                args=(ngpus_per_node, args, multi_rtsp_queue, multi_event_queue, multi_vis_queue, multi_batch_queue, skipped_frame_dict, func_time))
