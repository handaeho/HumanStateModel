import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures import Boxes, Keypoints

from predictor import VisualizationDemo
from vovnet import add_vovnet_config

# constants
WINDOW_NAME = "COCO detections"

# RTSP CAM URL
video_path = "rtsp://admin:!gaion3413@192.168.2.200:558/LiveChannel/0/media.smp"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_vovnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/faster_rcnn_V_19_slim_FPNLite_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def log_detection_results(predictions, demo):
    """
    Args:
        predictions (dict)
        demo (VisualizationDemo)
    """

    instances = predictions["instances"].to("cpu") if predictions["instances"] is not None else None
    num_instances = len(instances)
    if instances is not None:
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        classes = instances.pred_classes if instances.has("pred_classes") else None
        scores = instances.scores if instances.has("scores") else None
        class_names = demo.metadata.get("thing_classes", None)
        labels = [class_names[i] for i in classes]
        keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None
        if isinstance(boxes, Boxes):
            boxes = boxes.tensor.numpy()
        if isinstance(keypoints, Keypoints):
            keypoints = keypoints.tensor
            keypoints = np.asarray(keypoints)
            """
            keypoints (Tensor): a tensor of shape (N, K, 3), where N is the number of persons and 
            K (17) is the number of keypoints,
            and the last dimension corresponds to (x, y, probability).
            """
        for idx in range(num_instances):
            # print bbox results
            logger.info(f"{labels[idx]}\t{scores[idx]*100:.1f}% \
            box : x1:{int(boxes[idx][0])}, y1:{int(boxes[idx][1])}, x2:{int(boxes[idx][2])}, y2:{int(boxes[idx][3])}")

            # print keypoint results
            if keypoints is not None:
                for keypoints_per_intance in keypoints:
                    keypoint_names = demo.metadata.get("keypoint_names")
                    for idx, keypoint in enumerate(keypoints_per_intance):
                        x, y, score = keypoint
                        if keypoint_names:
                            keypoint_name = keypoint_names[idx]
                            logger.info(f"\t({idx+1}) {keypoint_name} : ({int(x)}, {int(y)})")



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {}".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished"
                )
            )
            log_detection_results(predictions, demo)

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        # cam = cv2.VideoCapture(0)
        cam = cv2.VideoCapture(video_path)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()