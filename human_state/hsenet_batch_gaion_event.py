import datetime
import os.path
import timeit
import cv2
import time
import torch
import multiprocessing
import numpy as np
import hsenet_vms_gaion
import logging
import requests
import json

from threading import Thread
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, \
    KEYPOINT_CONNECTION_RULES
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor

"""
detectron2 모델은 multi GPU 환경에서의 분산 연산이 불가능하다.
따라서 batch 형태로 GPU 수만큼 나누어서 각 GPU에 py 파일 올리고 실행. -> 1 GPU 1 model

ex) 20대의 카메라 & 4개의 GPU 환경에서, 
    1) 5개씩 RTSP 카메라를 GPU 개수만큼 나눈다. -> 1번~5번, 6번~10번, 11번~15번, 16번~20번 
    2) 한 그룹씩 py 파일을 만든다. 
    3) 각 py 파일을 각 GPU를 통해 실행한다. -> CUDA_VISIBLE_DEVICES=0 python XXX.py
"""

""" GPU 할당 정보 확인 """
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# print('Current cuda device name:', torch.cuda.get_device_name(0))
# print('Current cuda device:', torch.cuda.current_device())
# print('Count of using GPUs:', torch.cuda.device_count())  # CUDA_VISIBLE_DEVICES 설정시, 1로 표시됨

""" dataset 및 meta data 설정 """
dataset_root = './datasets/'

meta = {
    "keypoint_names": COCO_PERSON_KEYPOINT_NAMES,
    "keypoint_flip_map": COCO_PERSON_KEYPOINT_FLIP_MAP,
    "keypoint_connection_rules": KEYPOINT_CONNECTION_RULES,
}

register_coco_instances("MPHBE2020_train", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_train2020.json",
                        dataset_root + "MPHBE2020/train2020")
register_coco_instances("MPHBE2020_test", meta, dataset_root + "MPHBE2020/annotations/instances_mphbE_test2020.json",
                        dataset_root + "MPHBE2020/test2020")
register_coco_instances("MPHBE2021_train", meta, dataset_root + "MPHBE2021/annotations/instances_mphbE_train2021.json",
                        dataset_root + "MPHBE2021/train2021")
register_coco_instances("MPHBE2021_test", meta, dataset_root + "MPHBE2021/annotations/instances_mphbE_test2021.json",
                        dataset_root + "MPHBE2021/test2021")
register_coco_instances("keypoints_coco_2017_train_mphbe", meta,
                        dataset_root + "coco/annotations/person_keypoints_train2017.json",
                        dataset_root + "coco/train2017")
register_coco_instances("keypoints_coco_2017_val_mphbe", meta,
                        dataset_root + "coco/annotations/person_keypoints_val2017.json", dataset_root + "coco/val2017")

""" 자세 및 컬러 세팅 """
data_meta = MetadataCatalog.get("MPHBE2020_test").set(
    thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])
data_meta = MetadataCatalog.get("MPHBE2020_test").set(
    thing_colors=[(0, 0, 0), (0, 0, 0), (0, 0, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

""" 모델 설정 및 가중치 로드 """
cfg = get_cfg()
cfg.merge_from_file("configs/mphbe_hsenet_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "checkpoints/hsenet_R_50_FPN_3x/model_0269999.pth"
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)
# predictor = AsyncPredictor(cfg)

""" 비디오 또는 이미지, 로그 파일 저장을 위한 경로 및 시간 """
now = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
save_video_path = "/home/ho/WTC_Seoul/wtc_seoul/human_state/results/event_videos/"
save_image_path = "/home/ho/WTC_Seoul/wtc_seoul/human_state/results/event_images/"
save_log_path = "/home/ho/WTC_Seoul/wtc_seoul/human_state/results/event_logs/"


def producer(stream_id, rtsp_url, frame_queue):
    """
     read RTSP and input frame into queue

    Args:
        stream_id: camera channel id
        rtsp_url: rtsp url info
        frame_queue: queue with frame

    Returns: None (just input frame into queue)
    """
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(rtsp_url)
    # 원하는 배치 크기를 설정
    batch_size = 10
    # 읽어온 프레임을 담아둘 리스트
    frames = []
    # 초당 N 프레임을 위한 대기 시간
    wait_time = 1.0 / 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((stream_id, frame))

        # 배치 크기만큼 프레임이 모이면 큐에 추가
        if len(frames) == batch_size:
            frame_queue.put(frames)
            # 큐에 넣고 리스트 flush
            frames = []

        # 일정 시간 대기
        time.sleep(wait_time)

    # 남은 프레임이 있다면 큐에 추가(배치 크기 이하)
    if frames:
        frame_queue.put(frames)


def consumer(frame_queue, event_queue):
    """
    get frame from queue and inference

    Args:
        frame_queue: queue with frame
        event_queue: queue for event info

    Returns: None for now (but must return the information results later via the event alarm API)
    """
    vis = False
    send_events = True
    save_videos = False

    num_channel = 2
    max_count = 10
    counter = 0
    count_threshold = 5
    event_counts = np.zeros((num_channel, max_count))
    rtsp_channel_div = list(range(num_channel))
    # 이벤트 저장 행렬 유지 제한(time.sleep() * event_limit 유지)
    event_limit = 60
    # 이벤트 저장 행렬 유지 제한 counter
    event_limit_cnt = 0

    while True:
        # 큐에서 (stream_id, frame)을 가져옴
        batch = frame_queue.get()

        # 프레임 측정
        start_t = timeit.default_timer()
        end_t = timeit.default_timer()
        fps = int(1. / (end_t - start_t))
        # print(f"Inference time {fps} fps")

        for stream_id, frame in batch:
            # 프레임에 대한 추론
            with torch.no_grad():
                try:
                    outputs = predictor(frame)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('Out of memory')
                        torch.cuda.empty_cache()
                    else:
                        raise e

            # 시각화
            v = Visualizer(frame[:, :, ::-1], data_meta, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_frame = v.get_image()[:, :, ::-1]
            labels = outputs["instances"].pred_classes.cpu().numpy()

            if send_events:
                # 특정 이벤트(Lying 등) 발생시,
                if 5 in labels:
                    # 이벤트가 발생한 stream_id(카메라 번호) 행에 하나씩 1을 채움
                    event_counts[stream_id, counter % max_count] = 1

                    # 이벤트 저장 행렬 유지 제한 카운터 1 증가
                    event_limit_cnt += 1
                    # 이벤트 저장 행렬 유지 제한 넘어가면 이벤트 저장 행렬, 이벤트 counter, 이벤트 저장 행렬 유지 제한 counter 초기화
                    if event_limit_cnt > event_limit:
                        event_counts = np.zeros((num_channel, max_count))
                        counter = 0
                        event_limit_cnt = 0
                        # print("Initialization event_queue, counter, event_limit_cnt: ", event_counts, counter, event_limit_cnt, "\n")

                counter += 1
                if counter > max_count:
                    counter = 0

                # 행 원소의 합이 count_threshold 이상이면 True 아니면 False
                event_send = event_counts.sum(1) > count_threshold

                # [[stream_id, event_send], [stream_id, event_send], ...] 형태로 만듦
                event_flag = [[rtsp_channel_div[cam_number], send] for cam_number, send in enumerate(event_send)]
                event_time = str(datetime.datetime.now())[:-7]
                event_msg = 'Lying action detected'

                # RTSP 카메라 번호와 event_flag가 있는 리스트에서
                for index, [channel, send] in enumerate(event_flag):
                    # send flag가 True면
                    if send:
                        # event_queue에 넣음
                        event_queue.put([fps, channel, event_time, result_frame, event_msg])
                        print("Done insert event_queue! \n")
                        # event_queue에 넣은 후 이벤트 저장 행렬 및 이벤트 counter 초기화
                        event_counts = np.zeros((num_channel, max_count))
                        counter = 0

            # 결과를 화면에 출력
            if vis:
                cv2.imshow(f'Stream ID: {stream_id}', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            # 결과를 비디오 파일에 기록
            if save_videos:
                # 경로 유뮤 확인 및 생성
                save_video_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_video_time_path = os.path.join(save_image_path, save_video_time)
                if not os.path.exists(save_video_time_path):
                    os.mkdir(save_video_time_path)

                # 비디오 파일 생성
                fps = 30
                video_resolution = (1280, 1080)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                save_video = cv2.VideoWriter(save_video_time_path + "/" + f"output_video_{save_video_time}.mp4", fourcc, fps, video_resolution)
                resize_frame = cv2.resize(result_frame, video_resolution)

                # 비디오 파일 저장
                save_video.write(resize_frame)

            time.sleep(0.5)


def send_msg(multi_event_queue):
    while True:
        if multi_event_queue.qsize() > 0:
            try:
                while not multi_event_queue.empty():
                    fps, cam_number, event_time, frame, event_msg = multi_event_queue.get()
                    print(f"fps {fps} / cam_number {cam_number} / event_time {event_time} / event_msg {event_msg}")
                    print(f"frame {frame} \n")

            except Exception as e:
                print(e)


if __name__ == '__main__':
    """ multiprocessing의 process 시작 방법 정의 """
    multiprocessing.set_start_method('spawn')

    """ RTSP URL """
    # rtsp_streams = vms.get_rtsp_urls()
    rtsp_streams = ["rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/0/media.smp",
                    "rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/2/media.smp"]

    """ RTSP 카메라의 ID와 프레임을 담을 큐 생성 """
    rtsp_queue = multiprocessing.Queue()
    """ 이벤트 정보를 담을 큐 생성 """
    event_info_queue = multiprocessing.Queue()

    """ RTSP 카메라 대수 만큼 프로세스 생성 """
    """ 다수의 producer process가 하나의 queue에 프레임을 넣음 """
    producer_processes_list = []
    for i, stream_url in enumerate(rtsp_streams):
        producer_process = multiprocessing.Process(target=producer, args=(i, stream_url, rtsp_queue,), daemon=True)
        producer_process.start()
        producer_processes_list.append(producer_process)

    """ rtsp_queue에서 프레임을 가져오고 추론하는 consumer process 생성 """
    """ 큐에 프레임을 넣는 producer process는 여러개지만, 큐에서 프레임을 가져오는 consumer process는 하나만 생성 """
    consumer_process = multiprocessing.Process(target=consumer, args=(rtsp_queue, event_info_queue,), daemon=True)
    consumer_process.start()

    """ event_queue에서 이벤트 정보를 가져오고 이벤트 API를 호출, 프레임 저장, 로그 저장 등을 수행하는 send_msg process 생성 """
    """ 큐에 프레임을 넣는 producer process는 여러개지만, 큐에서 이벤트 정보를 처리하는 send_msg process는 하나만 생성 """
    # send_msg_process = multiprocessing.Process(target=send_msg, args=(event_queue,), daemon=True)
    send_msg_process = Thread(target=send_msg, args=(event_info_queue,), daemon=True)
    send_msg_process.start()

    """ 모든 producer process가 종료될 때까지 대기 """
    for process in producer_processes_list:
        process.join()

    """ consumer process 종료될 때까지 대기 """
    consumer_process.join()

    """ send_msg process 종료될 때까지 대기 """
    send_msg_process.join()
