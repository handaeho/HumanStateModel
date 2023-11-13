import atexit
import bisect
import queue
import torch
import warnings
import cv2
import multiprocessing as mp
import os

from detectron2.engine import DefaultPredictor
from detectron2.engine import default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import (
    COCO_PERSON_KEYPOINT_NAMES,
    COCO_PERSON_KEYPOINT_FLIP_MAP,
    KEYPOINT_CONNECTION_RULES
)

# HSENet_example
# from hsenet.modeling.roi_heads.roi_heads import HSENetROIHeads
# from hsenet.config.config import add_posture_config
from hsenet.utils.visualizer import Visualizer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

dataset_root = 'datasets/'

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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_posture_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


class QueueProcessor:
    def __init__(self, rtsp_path, vis, save_video):
        self.rtsp_path = rtsp_path
        self.vis = vis
        self.save_video = save_video
        self.data_meta= MetadataCatalog.get("MPHBE2020_test").set(
            thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])

    def reader(self, frames_queue, output_queue):
        parser = default_argument_parser()
        args = parser.parse_args()

        # Inference with a multitask model
        cfg = setup(args)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5  # set threshold for this model
        print(">>> Command Line Args:", args)

        predictor = DefaultPredictor(cfg)
        predictor = AsyncPredictor(cfg, num_gpus=8)

        with torch.no_grad():
            """
            stop gradient operation. backpropagation is off(train impossible).
            """
            cam = cv2.VideoCapture(self.rtsp_path)

        while True:
            # Read frames
            ret, frame = cam.read()
            if ret:
                try:
                    def _interpreter(value):
                        if value == 0:
                            return 'Walking'
                        elif value == 1:
                            return 'Crouch'
                        elif value == 2:
                            return 'Lying'
                        elif value == 3:
                            return 'Standing'
                        elif value == 4:
                            return 'Running'
                        else:
                            return 'Sitting'

                    # discard possible previous (unprocessed) frame / get current camera frame from queue
                    outputs = predictor(frames_queue.get_nowait())
                    ind = (outputs["instances"].pred_classes == 2).long().argsort(descending=True)
                    outputs["instances"] = outputs["instances"][ind]

                    label = outputs["instances"].pred_classes.cpu().numpy()
                    coordinates = outputs["instances"].pred_boxes

                    interpreted_label = [_interpreter(x) for x in label]

                    print(f'current label : {interpreted_label}, current coordinates : {coordinates}')

                    output_queue.put(outputs, block=False)

                except queue.Empty:
                    pass

                try:
                    frames_queue.put(cv2.resize(frame, (1280, 720)), block=False)

                except:
                    pass

            else:
                break

    def consumer(self, frames_queue, output_queue, stop_switch):
        while True:
            # get current camera frame from queue
            frame = frames_queue.get()

            # get current state outputs  from queue
            outputs = output_queue.get()

            if (outputs["instances"].pred_classes == 2).sum():
                ind = (outputs["instances"].pred_classes == 2).long().argsort(descending=True)
                outputs["instances"] = outputs["instances"][ind]

            if self.vis or self.save_video:
                v = Visualizer(frame[:, :, ::-1], self.data_meta, scale=1.2, instance_mode=ColorMode.SEGMENTATION)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            if self.vis:
                cv2.imshow('inference', out.get_image()[:, :, ::-1])
                # esc to quit
                if cv2.waitKey(1) == 27:
                    stop_switch.set()
                    break

    def state_interpreter(self, output_queue):
        # get current state outputs  from queue
        outputs = output_queue.get()

        label = outputs["instances"].pred_classes.cpu().numpy()
        # coordinates = outputs["instances"].pred_boxes.cpu().numpy()

        for index, value in enumerate(label):
            if value == 0:
                label[index] = 'Walking'
            elif value == 1:
                label[index] = 'Crouch'
            elif value == 2:
                label[index] = 'Lying'
            elif value == 3:
                label[index] = 'Standing'
            elif value == 4:
                label[index] = 'Running'
            elif value == 5:
                label[index] = 'Sitting'

        print(f'current label : {label}, current coordinates : {outputs["instances"].pred_boxes}')

    def main(self):
        # CUDA start_method setting spawn
        mp.set_start_method("spawn", force=True)

        # frame processing queue
        frames_queue = mp.Queue()
        output_queue = mp.Queue()
        stop_switch = mp.Event()

        reader = mp.Process(target=self.reader, args=(frames_queue, output_queue), daemon=True)
        consumer = mp.Process(target=self.consumer, args=(frames_queue, output_queue, stop_switch), daemon=True)
        state_interpreter = mp.Process(target=self.state_interpreter, args=(output_queue, ), daemon=True)

        reader.start()
        consumer.start()
        state_interpreter.start()

        stop_switch.wait()


if __name__ == '__main__':
    # Visualization flag
    vis = True
    save_video = False

    # RTSP CAM URL
    video_path_cam_00 = "rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/2/media.smp"
    # video_path_cam_01 = "rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/1/media.smp"

    worker = QueueProcessor(rtsp_path=video_path_cam_00, vis=vis, save_video=save_video)

    worker.main()

