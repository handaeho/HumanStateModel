import cv2
import timeit
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, \
    KEYPOINT_CONNECTION_RULES
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
import threading, queue
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
cfg = get_cfg()

# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file("configs/mphbe_hsenet_p_V_39_FPN_3x_daejeonCCTV.yaml")
# cfg.merge_from_file("configs/mphbe_hsenet_V_39_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "model_HSENet-P_final.pth"
cfg.merge_from_file("configs/mphbe_hsenet_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "checkpoints/hsenet_R_50_FPN_3x/model_0269999.pth"
predictor = DefaultPredictor(cfg)

# RTSP URL
# rtsp_streams = vms.get_rtsp_urls()

# 결과를 기록할 비디오 파일 생성
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out_video = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

q = queue.Queue()


def producer(stream_id, stream, q):
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

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 프레임과 스트림 ID를 함께 큐에 추가
        q.put((stream_id, frame))
        cv2.waitKey(int(1000 / 30))


def consumer(q):
    """
    get frame from queue and inference

    Args:
        q: queue with frame

    Returns: None for now (but must return the information results later via the event alarm API)

    """
    while True:
        # 큐에서 스트림 ID와 프레임을 가져옴
        stream_id, frame = q.get()

        # 프레임에 대한 예측
        start_t = timeit.default_timer()
        outputs = predictor(frame)
        end_t = timeit.default_timer()
        FPS = int(1. / (end_t - start_t))
        print("Inference time " + str(FPS) + " fps" + " / " + "Stream ID " + str(stream_id))

        # 시각화
        v = Visualizer(frame[:, :, ::-1], data_meta, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result_frame = v.get_image()[:, :, ::-1]

        # 결과를 비디오 파일에 기록
        # out_video.write(result_frame)

        # 결과를 화면에 출력
        cv2.imshow("Result", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # video.release()
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    # 프로듀서 스레드 생성 및 시작
    rtsp_url = ["rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/2/media.smp"]
    producer_threads = []
    # for i, stream in enumerate(rtsp_streams):
    for i, stream in enumerate(rtsp_url):
        t = threading.Thread(target=producer, args=(i, stream, q))
        t.start()
        producer_threads.append(t)

    # 컨슈머 스레드 생성 및 시작
    consumer_thread = threading.Thread(target=consumer, args=(q,))
    consumer_thread.start()

    # 모든 스레드가 종료될 때까지 대기
    for t in producer_threads:
        t.join()

    consumer_thread.join()

    # out_video.release()
    cv2.destroyAllWindows()
