import datetime
import os.path
import cv2
import time
import torch
import multiprocessing
import numpy as np
import requests
import json
import logging

from PIL import Image
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

# # GPU 할당 정보 확인
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Device:', device)
# print('Current cuda device name:', torch.cuda.get_device_name(0))
# print('Current cuda device:', torch.cuda.current_device())
# print('Count of using GPUs:', torch.cuda.device_count())  # CUDA_VISIBLE_DEVICES 설정시, 1로 표시됨

# dataset 및 meta data 설정
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

# 자세 및 컬러 세팅
data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])
data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_colors=[(0, 0, 0), (0, 0, 0), (0, 0, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

# 모델 설정 및 가중치 로드
cfg = get_cfg()
cfg.merge_from_file("configs/mphbe_hsenet_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "checkpoints/hsenet_R_50_FPN_3x/model_0269999.pth"
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)

# 로그 세팅 (저장 위치, 로그 포맷, 저장할 로그 레벨 임계값)
logging.basicConfig(filename="./saved_results/system_log.txt",
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARNING)


def producer(stream_id, rtsp_url, frame_queue):
    """
     read RTSP and input frame into queue

    Args:
        stream_id: camera channel id
        rtsp_url: rtsp url info
        frame_queue: queue with frame

    Returns: input frame into frame queue
    """
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(rtsp_url)
    # 원하는 배치 크기를 설정
    batch_size = 10
    # 읽어온 프레임을 담아둘 리스트
    frames = []
    # 이전 프레임 재생 시간
    prev_time = 0
    # 지정된 FPS 값
    fps_setting = 10

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000/fps)
        print(f"Camera info > Cam ID: {stream_id} / FPS: {fps} / Delay: {delay}ms")

        while True:
            # 프레임 read
            ret, frame = cap.read()
            # 현재 시간과 이전 프레임 재생 루프에서 저장된 시간 값을 비교하여 경과된 시간 값을 계산
            current_time = time.time() - prev_time
            # 지정된 FPS 값과 비교하여 FPS 값 이상의 시간이 경과 되었을 때 새로운 프레임을 append
            if ret and (current_time > 1.0 / fps_setting):
                prev_time = time.time()
                frames.append((stream_id, frame))

            if not ret:
                break

            # 배치 크기만큼 프레임이 모이면 큐에 추가
            if len(frames) == batch_size:
                frame_queue.put(frames)
                # 큐에 넣고 리스트 flush
                frames = []

        # 남은 프레임이 있다면 큐에 추가(배치 크기 이하)
        if frames:
            frame_queue.put(frames)

    else:
        print("Can't open Cameras or videos!")
        cap.release()


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
    num_channel = 2  # RTSP 카메라 개수
    max_count = 10  # event 상태를 저장할 최대값(열의 개수)
    count_threshold = 5 # 이벤트 큐에 넣는 임계점(1개 열의 최대 합계)
    counter = 0  # 행렬에서 열의 포인터
    event_counts = np.zeros((num_channel, max_count))  # (num_channel x max_count) 크기의 영행렬
    rtsp_channel_div = list(range(num_channel))  # 카메라 번호 리스트
    event_limit = 60  # 이벤트 저장 행렬 유지 제한(time.sleep() * event_limit 유지)
    event_limit_cnt = 0  # 이벤트 저장 행렬 유지 제한 counter

    while True:
        # 큐에서 (stream_id, frame)을 가져옴
        batch = frame_queue.get()

        for stream_id, frame in batch:
            # 프레임에 대한 추론
            with torch.no_grad():
                try:
                    outputs = predictor(frame)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logging.warning("Exception occurred: Out of memory")
                        torch.cuda.empty_cache()
                    else:
                        raise e

            # 시각화
            v = Visualizer(frame[:, :, ::-1], data_meta, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            original_frame = frame[:, :, ::-1]
            result_frame = v.get_image()[:, :, ::-1]
            labels = outputs["instances"].pred_classes.cpu().numpy()

            if send_events:
                # 특정 이벤트(Lying) 발생시,
                if 2 in labels:
                    # event_counts 영행렬에 [stream_id(이벤트가 발생한 카메라 번호) 행, 0번째 열]부터 하나씩 1을 채움
                    event_counts[stream_id, counter % max_count] = 1

                    # 이벤트 저장 행렬 유지 제한 카운터 1 증가
                    event_limit_cnt += 1
                    # 이벤트 저장 행렬 유지 제한 넘어가면 이벤트 저장 행렬, 이벤트 counter, 이벤트 저장 행렬 유지 제한 counter 초기화
                    if event_limit_cnt > event_limit:
                        event_counts = np.zeros((num_channel, max_count))
                        counter = 0
                        event_limit_cnt = 0

                # 행렬에서 열의 포인터 +1
                counter += 1

                # 행렬에서 열의 포인터가 event 상태를 저장할 최대값(열의 개수)를 넘어가면 초기화
                if counter > max_count:
                    counter = 0

                # 행 원소의 합이 count_threshold 이상이면 True 아니면 False
                event_send = event_counts.sum(1) > count_threshold

                # [[stream_id, event_send(True/False)], [stream_id, event_send(True/False)], ...] 형태로 만듦
                event_flag = [[rtsp_channel_div[cam_number], send] for cam_number, send in enumerate(event_send)]
                event_time = str(datetime.datetime.now())[:-7]

                # TODO: 마지막에 삭제
                print(f"stream_id => {stream_id}")
                print(f"labels =>  {labels}")
                print(f"event_counts => {event_counts} \n")

                # RTSP 카메라 번호와 event_flag가 있는 리스트에서
                for index, [channel, send] in enumerate(event_flag):
                    # send flag가 True면
                    if send:
                        # event_queue에 넣음
                        event_queue.put([channel, event_time, result_frame, original_frame])
                        # event_queue에 넣은 후 이벤트 저장 행렬 및 이벤트 counter 초기화
                        event_counts = np.zeros((num_channel, max_count))
                        counter = 0

            # 결과를 화면에 출력
            if vis:
                cv2.imshow(f'Stream ID: {stream_id}', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


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
                    cam_number, event_time, event_frame, original_frame = multi_event_queue.get()

                    # 오리지널 프레임 이미지와 이벤트 추론 이미지를 저장
                    if save_frame:
                        now_time = event_time.replace(" ", "_").replace(":", "")
                        original_img = Image.fromarray(original_frame)
                        event_img = Image.fromarray(event_frame)

                        save_path_org_date = os.path.join(save_path_original_frame, f"camera_id_{cam_number}_original_image", now_time.split("_")[0])
                        save_path_event_date = os.path.join(save_path_event_frame, f"camera_id_{cam_number}_event_image", now_time.split("_")[0])

                        if not os.path.exists(save_path_org_date):
                            os.makedirs(save_path_org_date)
                        if not os.path.exists(save_path_event_date):
                            os.makedirs(save_path_event_date)

                        original_img.save(f"{save_path_org_date}/{now_time}_original.jpg")
                        event_img.save(f"{save_path_event_date}/{now_time}_event.jpg")

                        # TODO: 마지막에 삭제
                        print(f">>> Done save images! <<< \n")

                    # 이벤트 API 호출
                    # TODO: URL & port 수정 -> localhost:xxxx
                    # url = "http://localhost:8080/api/transfer/alert"
                    url = "http://192.168.2.18:8080/api/transfer/alert"
                    event_msg = "Lying action detected"
                    headers = {
                        "content-type": "application/json",
                        "charset": "utf-8",
                    }
                    data = {
                        "cameraId": f"{cam_number}",
                        "eventTime": event_time,
                        "eventMsg": event_msg,
                    }

                    res = requests.post(url=url, data=json.dumps(data), headers=headers)
                    resobj = res.content.decode()
                    js = json.loads(resobj)

                    if js["status"]:
                        print(f"Camera {cam_number} Event sent at {str(datetime.datetime.now())[:-7]} and event occurred at {event_time} \n")
                    else:
                        logging.warning(f"Camera {cam_number} Failed to send event occurred at {event_time}")
                        print(f"Camera {cam_number} Failed to send event occurred at {event_time} \n")
                        multi_event_queue.put([cam_number, event_time, event_frame, original_frame])

            except Exception as e:
                logging.warning(f"Exception occurred: {e}")


if __name__ == '__main__':
    # Parameter Setting
    vis_flag = False
    send_events_flag = True
    save_frame_flag = True
    save_path_origin = "./saved_results/original"
    save_path_event = "./saved_results/event"

    # multiprocessing의 process 시작 방법 정의
    multiprocessing.set_start_method('spawn')

    # RTSP URL
    # TODO: RTSP 카메라 정보 수정 -> WTC 환경
    rtsp_streams = ["rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/0/media.smp",
                    "rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/2/media.smp"]

    # RTSP 카메라의 ID와 프레임을 담을 큐 생성
    rtsp_queue = multiprocessing.Queue()
    # # 이벤트 정보를 담을 큐 생성
    event_info_queue = multiprocessing.Queue()

    # RTSP 카메라 대수 만큼 프로세스 생성
    # 다수의 producer process가 하나의 queue에 프레임을 넣음
    producer_processes_list = []
    for i, stream_url in enumerate(rtsp_streams):
        producer_process = multiprocessing.Process(target=producer, args=(i, stream_url, rtsp_queue,), daemon=True)
        producer_process.start()
        producer_processes_list.append(producer_process)

    # rtsp_queue에서 프레임을 가져오고 추론하는 consumer process 생성
    # 큐에 프레임을 넣는 producer process는 여러개지만, 큐에서 프레임을 가져오는 consumer process는 하나만 생성
    consumer_process = multiprocessing.Process(target=consumer, args=(rtsp_queue, event_info_queue, send_events_flag, vis_flag,), daemon=True)
    consumer_process.start()

    # event_queue에서 이벤트 정보를 가져오고 이벤트 API를 호출, 프레임 저장, 로그 저장 등을 수행하는 send_msg process 생성
    # 큐에 프레임을 넣는 producer process는 여러개지만, 큐에서 이벤트 정보를 처리하는 send_msg process는 하나만 생성
    # send_msg_process = multiprocessing.Process(target=send_msg, args=(event_queue,), daemon=True)
    send_msg_process = Thread(target=send_msg, args=(event_info_queue, save_path_origin, save_path_event, save_frame_flag,), daemon=True)
    send_msg_process.start()

    # 모든 producer process가 종료될 때까지 대기
    for process in producer_processes_list:
        process.join()

    # consumer process 종료될 때까지 대기
    consumer_process.join()

    # send_msg process 종료될 때까지 대기
    send_msg_process.join()
