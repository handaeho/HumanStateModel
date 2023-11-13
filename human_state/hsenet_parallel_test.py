import cv2
import time, timeit
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, \
    KEYPOINT_CONNECTION_RULES
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from hsenet.modeling.meta_arch.build import build_model
import torch
from torch.nn import DataParallel
from queue import Queue
from threading import Thread

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
cfg = get_cfg()
cfg.merge_from_file("configs/mphbe_hsenet_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "checkpoints/hsenet_R_50_FPN_3x/model_0269999.pth"
cfg.MODEL_DEVICE = "cuda"

# GPU 개수 만큼 모델 생성
# num_gpus = torch.cuda.device_count()
num_gpus = 1

predictors = []
for gpu_id in range(num_gpus):
    # predictor = DefaultPredictor(cfg)
    predictor = build_model(cfg)

    torch.cuda.set_device(gpu_id)
    # predictor = DataParallel(predictor)
    predictor = DataParallel(predictor.half().cuda(gpu_id), device_ids=[gpu_id])
    predictors.append(predictor)


def producer(stream_id, stream, q, gpu_id):
    """
    read RTSP and input frame into queue

    Args:
        stream_id: camera channel id
        stream: rtsp url info
        q: queue

    Returns: None (just input frame into queue)

    """
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(stream)
    batch_size = 10
    frames = []
    wait_time = 1.0 / 30

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append((stream_id, frame))

        # 배치 크기만큼 프레임이 모이면 큐에 추가
        if len(frames) == batch_size:
            q.put(frames)
            frames = []  # flush

        time.sleep(wait_time)  # 일정 시간 대기

    # 남은 프레임이 있다면 큐에 추가
    if frames:
        q.put(frames)


def consumer(q, gpu_id):
    predictor = predictors[gpu_id]

    while True:
        # 큐에서 스트림 ID와 프레임을 가져옴
        batch = q.get()

        for stream_id, frame in batch:
            # 프레임에 대한 예측
            with torch.no_grad():
                try:
                    im = torch.from_numpy(frame).to(torch.device("cuda:0"))
                    height, width = im.shape[:2]
                    print(im, im.shape, im.ndim, height, width) #  torch.Size([720, 1280, 3]) 3 720 1280
                    inputs = [{"image": im, "height": height, "width": width}]
                    # im = im.unsqueeze(0)
                    outputs = predictor(inputs)
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

            cv2.imshow("Result", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # RTSP URL
    # rtsp_streams = vms.get_rtsp_urls()
    rtsp_streams = ["rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/2/media.smp"]

    frame_queues = [Queue(maxsize=20) for _ in rtsp_streams]

    for i, stream in enumerate(rtsp_streams):
        gpu_id = i % num_gpus
        Thread(target=producer, args=(i, stream, frame_queues[i], gpu_id)).start()
        Thread(target=consumer, args=(frame_queues[i], gpu_id)).start()

