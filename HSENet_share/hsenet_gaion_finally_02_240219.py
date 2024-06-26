import datetime
import os.path
import cv2
import time

import pafy
import torch
import multiprocessing
import numpy as np
import requests
import json
import logging
import warnings

from PIL import Image
from threading import Thread
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, \
    KEYPOINT_CONNECTION_RULES
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor

""" This is the most recent file, updated on 2024.02.20 """
""" The file is the same as the file in the WTC. """

# 경고 메세지 끄기
warnings.filterwarnings(action='ignore')

# # Check GPU allocation information
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# print('Current cuda device name:', torch.cuda.get_device_name(0))
# print('Current cuda device:', torch.cuda.current_device())
# print('Count of using GPUs:', torch.cuda.device_count())

dataset_root = './datasets/'

meta = {
    "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
    "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
    "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
}

register_coco_instances("MPHBE2020_train", meta,
                        dataset_root + "MPHBE2020/annotations/instances_mphbE_train2020.json",
                        dataset_root + "MPHBE2020/train2020")
register_coco_instances("MPHBE2020_test", meta,
                        dataset_root + "MPHBE2020/annotations/instances_mphbE_test2020.json",
                        dataset_root + "MPHBE2020/test2020")
register_coco_instances("MPHBE2021_train", meta,
                        dataset_root + "MPHBE2021/annotations/instances_mphbE_train2021.json",
                        dataset_root + "MPHBE2021/train2021")
register_coco_instances("MPHBE2021_test", meta,
                        dataset_root + "MPHBE2021/annotations/instances_mphbE_test2021.json",
                        dataset_root + "MPHBE2021/test2021")
register_coco_instances("keypoints_coco_2017_train_mphbe", meta,
                        dataset_root + "coco/annotations/person_keypoints_train2017.json",
                        dataset_root + "coco/train2017")
register_coco_instances("keypoints_coco_2017_val_mphbe", meta,
                        dataset_root + "coco/annotations/person_keypoints_val2017.json",
                        dataset_root + "coco/val2017")

data_meta = MetadataCatalog.get("MPHBE2020_test").set(
    thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])
data_meta = MetadataCatalog.get("MPHBE2020_test").set(
    thing_colors=[(0, 0, 0), (0, 0, 0), (0, 0, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

cfg = get_cfg()
cfg.merge_from_file("./configs/mphbe_hsenet_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "./checkpoints/hsenet_R_50_FPN_3x/model_0269999.pth"
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)

logging.basicConfig(filename="./saved_results/system_log.txt",
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING)


def producer(stream_index, rtsp_info, frame_queue):
    """
     read RTSP and input frame into queue

    Args:
        stream_id: camera channel id
        rtsp_info: dictionary of rtsp id, name and url info
        frame_queue: queue with frame

    Returns: input frame into frame queue
    """
    rtsp_id = "".join(rtsp_info.get("cameraId"))
    rtsp_name = "".join(rtsp_info.get("cameraName"))
    rtsp_url = "".join(rtsp_info.get("rtsp_url"))
    frame_interval = 30  # 읽어온 프레임 중 큐에 넣을 주기 -> 30fps 일 경우, 30으로 설정 시, 1초에 1프레임 input
    cap = cv2.VideoCapture(rtsp_url)

    # # YOUTUBE TEST
    # video = pafy.new(rtsp_url)
    # best = video.getbest(preftype='mp4')  # 'webm','3gp'
    # cap = cv2.VideoCapture(best.url)

    try:
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps)
            print(f"Camera info > Cam Info: {rtsp_name} / Cam FPS: {fps} / Delay: {delay}ms")

            while True:
                ret, frame = cap.read()
                frame_cnt = cap.get(cv2.CAP_PROP_POS_FRAMES)

                if ret and frame_cnt % frame_interval == 0:
                    # print(f"APPEND! stream_id {rtsp_id} / frame_cnt {frame_cnt}")
                    frame_queue.put((stream_index, rtsp_id, rtsp_name, frame))

                if not ret:
                    break

        else:
            print("Can't open Cameras or videos!")
            cap.release()

    except Exception as e:
        logging.warning(f"Can't open Cameras or videos, Exception occurred in producer method: {e}")


def consumer(frame_queue, event_queue, send_events, vis):
    """
    get frame from queue and inference

    Args:
        frame_queue: queue with frame
        event_queue: queue for event info
        send_events: flag of send event with event queue
        vis: flag of visualization to RTSP camera frame

    Returns: input event information into event queue
    """
    num_channel = 20
    max_count = 10
    count_threshold = 5
    counter = 0
    event_counts = np.zeros((num_channel, max_count))
    rtsp_channel_div = list(range(num_channel))
    event_limit = 60
    event_limit_cnt = 0

    while True:
        stream_index, rtsp_id, rtsp_name, frame = frame_queue.get()
        print(f"now running No.2 ...  / the RTSP ID currently being processed is {rtsp_id}", end='\r')

        try:
            with torch.no_grad():
                try:
                    outputs = predictor(frame)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logging.warning("Exception occurred: Out of memory")
                        torch.cuda.empty_cache()
                    else:
                        raise e

            v = Visualizer(frame[:, :, ::-1], data_meta, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            original_frame = frame[:, :, ::-1]
            result_frame = v.get_image()[:, :, ::-1]
            labels = outputs["instances"].pred_classes.cpu().numpy()

            if send_events:
                if 2 in labels:
                    event_counts[stream_index, counter % max_count] = 1
                    event_limit_cnt += 1
                    if event_limit_cnt > event_limit:
                        event_counts = np.zeros((num_channel, max_count))
                        counter = 0
                        event_limit_cnt = 0

                counter += 1
                if counter > max_count:
                    counter = 0

                event_send = event_counts.sum(1) > count_threshold
                event_flag = [[rtsp_channel_div[cam_number], send] for cam_number, send in enumerate(event_send)]
                event_time = str(datetime.datetime.now())[:-7]

                for index, [channel, send] in enumerate(event_flag):
                    if send:
                        event_queue.put([channel, rtsp_id, rtsp_name, event_time, result_frame, original_frame])
                        event_counts = np.zeros((num_channel, max_count))
                        counter = 0

                # print(f"rtsp_id : {rtsp_id}, rtsp_name : {rtsp_name}, labels : {labels}")
                # print(f"event_counts : {event_counts} \n")

            if vis:
                resize_frame = cv2.resize(result_frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                cv2.imshow(f'RTSP ID: {rtsp_id}', resize_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        except Exception as e:
            logging.warning(f"Exception occurred in consumer method: {e}")

        time.sleep(0.1)


def send_msg(multi_event_queue, save_path_original_frame, save_path_event_frame, save_frame):
    """
    Args:
        multi_event_queue: queue to store event information
        save_path_original_frame: path to store the original RTSP camera frame image
        save_path_event_frame: path to store the event inference result frame image
        save_frame: flag of save in save path

    Returns: save frame images and call event API in Web application
    """
    while True:
        if multi_event_queue.qsize() > 0:
            try:
                while not multi_event_queue.empty():
                    row_number, rtsp_id, rtsp_name, event_time, event_frame, original_frame = multi_event_queue.get()

                    if save_frame:
                        now_time = event_time.replace(" ", "_").replace(":", "")
                        original_img = Image.fromarray(original_frame)
                        event_img = Image.fromarray(event_frame)

                        save_path_org_date = os.path.join(save_path_original_frame,
                                                          f"{rtsp_name}_original_image", now_time.split("_")[0])
                        save_path_event_date = os.path.join(save_path_event_frame,
                                                            f"{rtsp_name}_event_image", now_time.split("_")[0])

                        if not os.path.exists(save_path_org_date):
                            os.makedirs(save_path_org_date)
                        if not os.path.exists(save_path_event_date):
                            os.makedirs(save_path_event_date)

                        original_img.save(f"{save_path_org_date}/{now_time}_original.jpg")
                        event_img.save(f"{save_path_event_date}/{now_time}_event.jpg")

                        # print(f"Done Save Original image. {save_path_org_date}/{now_time}_original.jpg")
                        # print(f"Done Save Event image. {save_path_event_date}/{now_time}_event.jpg \n")

                    url = "http://localhost:8080/api/transfer/alert"
                    event_msg = "Lying action detected"
                    headers = {
                        "content-type": "application/json",
                        "charset": "utf-8",
                    }
                    data = {
                        "cameraId": f"{rtsp_id}",
                        "cameraName": f"{rtsp_name}",
                        "eventTime": event_time,
                        "eventMsg": event_msg,
                    }
                    res = requests.post(url=url, data=json.dumps(data), headers=headers)
                    resobj = res.content.decode()
                    js = json.loads(resobj)

                    if js["status"]:
                        print(f"{rtsp_name} Event sent at {str(datetime.datetime.now())[:-7]} and event occurred at {event_time} \n")
                    else:
                        logging.warning(f"{rtsp_name} Failed to send event occurred at {event_time}")
                        print(f"{rtsp_name} Failed to send event occurred at {event_time} \n")
                        multi_event_queue.put([rtsp_id, rtsp_name, event_time, event_frame, original_frame])

                    time.sleep(0.01)

            except Exception as e:
                logging.warning(f"Exception occurred in send_msg method: {e}")

        time.sleep(0.1)


if __name__ == '__main__':
    print("Starting HSENet No.2 ...")

    vis_flag = False
    send_events_flag = True
    save_frame_flag = True
    save_path_origin = "./saved_results/original"
    save_path_event = "./saved_results/event"

    multiprocessing.set_start_method('spawn')

    # When installing additional GPUs, change the settings so that each GPU is assigned N cameras.
    # CUDA_VISIBLE_DEVICES=0 python XXX.py, CUDA_VISIBLE_DEVICES=1 python YYY.py, ...
    rtsp_streams = [{"cameraId": "10", "cameraName": "11번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv011.stream"},
                    {"cameraId": "11", "cameraName": "12번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv012.stream"},
                    {"cameraId": "12", "cameraName": "13번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv013.stream"},
                    {"cameraId": "13", "cameraName": "14번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv014.stream"},
                    {"cameraId": "14", "cameraName": "15번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv015.stream"},
                    {"cameraId": "15", "cameraName": "16번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv016.stream"},
                    {"cameraId": "16", "cameraName": "17번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv017.stream"},
                    {"cameraId": "17", "cameraName": "18번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv018.stream"},
                    {"cameraId": "18", "cameraName": "19번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv019.stream"},
                    {"cameraId": "19", "cameraName": "20번 카메라", "rtsp_url": "rtsp://210.99.70.120:1935/live/cctv020.stream"}]

    # rtsp_streams = [{"cameraId": "10", "cameraName": "11번 카메라 CM10 디스커버리", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.137:554/trackID=3"},
    #                 {"cameraId": "11", "cameraName": "12번 카메라 D104 이코복스", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.138:554/trackID=3"},
    #                 {"cameraId": "12", "cameraName": "13번 카메라 D107 버터", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.139:554/trackID=3"},
    #                 {"cameraId": "13", "cameraName": "14번 카메라 A107 원더플레이스", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.170:554/trackID=3"},
    #                 {"cameraId": "14", "cameraName": "15번 카메라 C103 드코닝", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.171:554/trackID=3"},
    #                 {"cameraId": "15", "cameraName": "16번 카메라 B115 라미", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.172:554/trackID=3"},
    #                 {"cameraId": "16", "cameraName": "17번 카메라 F104 골스튜디오", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.173:554/trackID=3"},
    #                 {"cameraId": "17", "cameraName": "18번 카메라 F113 에이랜드", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.174:554/trackID=3"},
    #                 {"cameraId": "18", "cameraName": "19번 카메라 H111 코나 이비인후과", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.8:554/trackID=3"},
    #                 {"cameraId": "19", "cameraName": "20번 카메라 H113 도레도레", "rtsp_url": "rtsp://admin:!wtc2018@192.168.108.222:554/trackID=3"}]

    rtsp_queue = multiprocessing.Queue(maxsize=10)
    event_info_queue = multiprocessing.Queue(maxsize=10)

    producer_processes_list = []

    for i, stream_url in enumerate(rtsp_streams):
        producer_process = multiprocessing.Process(target=producer, args=(i, stream_url, rtsp_queue,), daemon=True)
        producer_process.start()
        producer_processes_list.append(producer_process)

    consumer_process = multiprocessing.Process(target=consumer, args=(rtsp_queue, event_info_queue, send_events_flag, vis_flag,), daemon=True)
    consumer_process.start()

    send_msg_process = Thread(target=send_msg, args=(event_info_queue, save_path_origin, save_path_event, save_frame_flag,), daemon=True)
    send_msg_process.start()

    for process in producer_processes_list:
        process.join()

    consumer_process.join()

    send_msg_process.join()
