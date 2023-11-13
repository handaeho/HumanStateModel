import cv2
import time
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, \
    KEYPOINT_CONNECTION_RULES
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
import torch
import threading, queue
import multiprocessing
import vms

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

data_meta = MetadataCatalog.get("MPHBE2020_test").set(
    thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])
# color setting
data_meta = MetadataCatalog.get("MPHBE2020_test").set(
    thing_colors=[(0, 0, 0), (0, 0, 0), (0, 0, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

# 모델 설정 및 가중치 로드
# cfg = get_cfg()
# cfg.merge_from_file("configs/mphbe_hsenet_R_50_FPN_3x.yaml")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# cfg.MODEL.WEIGHTS = "checkpoints/hsenet_R_50_FPN_3x/model_0269999.pth"
# cfg.MODEL.DEVICE = "cuda"
# predictor = DefaultPredictor(cfg)

cfg = get_cfg()
cfg.merge_from_file("configs/mphbe_hsenet_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "checkpoints/hsenet_R_50_FPN_3x/model_0269999.pth"
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)


# 결과를 기록할 비디오 파일 생성
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out_video = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

def producer(stream_id, stream, q):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(stream)
    batch_size = 10  # 원하는 배치 크기를 설정
    frames = []
    wait_time = 1.0 / 30  # 초당 30프레임을 위한 대기 시간

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append((stream_id, frame))

        # 배치 크기만큼 프레임이 모이면 큐에 추가
        if len(frames) == batch_size:
            q.put(frames)
            frames = []

        time.sleep(wait_time)  # 일정 시간 대기

    # 남은 프레임이 있다면 큐에 추가
    if frames:
        q.put(frames)


def consumer(q):
    while True:
        # 큐에서 스트림 ID와 프레임을 가져옴
        batch = q.get()

        for stream_id, frame in batch:
            # 프레임에 대한 예측
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

            # 결과를 화면에 출력
            # cv2.imshow(f'Stream ID: {stream_id}', result_frame)
            cv2.imshow('stream', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":

    # RTSP URL
    rtsp_streams = vms.get_rtsp_urls()

    multiprocessing.set_start_method('spawn')

    q = multiprocessing.Queue()

    processes = []
    for i, stream in enumerate(rtsp_streams):
        p = multiprocessing.Process(target=producer, args=(i, stream, q))
        p.start()
        processes.append(p)

    consumer_process = multiprocessing.Process(target=consumer, args=(q,))
    consumer_process.start()

    for p in processes:
        p.join()

    consumer_process.join()
