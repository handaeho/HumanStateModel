import cv2
# from multiprocessing import Queue, Process, Value
# import multiprocessing
# from multiprocessing import Process
from torch.multiprocessing import Process, Manager
# from multiprocessing import Process, Queue, Manager
from threading import Thread
# import threading
import time
from typing import Tuple
import sys
import numpy as np
# from open_cam_rtsp import open_cam_rtsp
from open_cam_rtsp import open_cam_login, open_cam_get_rtspUrl, RTSPConnectionRefusedError
from open_cam_rtsp import camera_id_return, open_cam_login
import detectron2.data.transforms as T
import torch
import requests
import json
import datetime
import logging
import os
from hsenet.utils.visualizer import Visualizer
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, KEYPOINT_CONNECTION_RULES
from PIL import Image
from detectron2.data import MetadataCatalog, DatasetCatalog
from utils import setup_logger, arg_as_list, to_matrixView, get_final_res, gather, create_reconnectDict

data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_classes=["Walking", "Crouch", "Lying", "Standing", "Running", "Sitting"])
data_meta = MetadataCatalog.get("MPHBE2020_test").set(thing_colors=[(0, 0, 0), (0, 0, 0), (0, 0, 255), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "timeout;10"

logger = logging.getLogger('CCTV')


class VideoScreenshot(object):
    def __init__(self, args, rtsp_infos, ngpus_per_node, region, multi_raw_inputs, multi_rtsp_queue, multi_event_queue, multi_batch_queue, skipped_frame_dict, func_time):
        # Take screenshot every x seconds
        self.args = args
        self.video_res = args.video_res
        self.server_number = args.server_number
        black_frame = np.zeros([self.video_res[1], self.video_res[0], 3], dtype=np.uint8)
        self.screenshot_interval = 1
        self.skipped_frame_dict = skipped_frame_dict
        self.func_time = func_time

        self.rtsp_infos = rtsp_infos
        # self.r_dict = r_dict
        self.region = region
        # self.event_q = event_q

        self.reconnect_attempts = Manager().dict()
        # self.reconnect_attempts['global'] = 1
        for x in self.rtsp_infos.keys():
            self.reconnect_attempts[x] = 1  # len(args.rtsp_channels)*[0]
        self.max_attempts = 10
        self.multi_rtsp_queue = multi_rtsp_queue

        # rtsp_channel_div = np.array_split(args.rtsp_channels, ngpus_per_node)[gpu]
        rtsp_channel_div = [camera_id_return(x, args.rtsp_region, args.server_number) for x in args.rtsp_channels]

        self.nosig_view = {channel_div: self.noSig_view(channel_div) for channel_div in rtsp_channel_div}
        self.nosig_view_t = {channel_div: torch.as_tensor(self.noSig_view(channel_div).astype("float16").transpose(2, 0, 1)).unsqueeze_(0) for channel_div in rtsp_channel_div}

        '''
        Currently not using thread_list, force_reconnect_list
        '''
        # init connection
        self.headers = None
        self.target_ip = None
        self.force_reconnect_with_login = True
        self.force_reconnect = False
        self.reconnecting = False
        self.skip_key = []
        # self.tryReconnection(0)
        self.send_msg_logger = setup_logger('SendMsg', 'results/logs/', filename='SendMsg_log.txt')

        self.thread_list = []
        self.event_sended = dict()
        self.rtsp_logger_dict = dict()

        self.status_q = dict()  # [True for _ in range(len(args.rtsp_channels))]
        self.frame_q = dict()  # [black_frame for _ in range(len(args.rtsp_channels))]

        gpu_sel = 0
        for i in self.rtsp_infos.keys():
            if i == 'send_msg':
                self.rtsp_logger_dict[i] = self.send_msg_logger
                continue
            if i == 'global':
                continue
            self.status_q[i] = True
            self.frame_q[i] = black_frame
            self.rtsp_logger_dict[i] = setup_logger('RTSPLogger_{}'.format(self.rtsp_infos[i]['camera_id']), 'results/logs', filename='RTSPLogger_{}_log.txt'.format(self.rtsp_infos[i]['camera_id']))
            self.event_sended[i] = -1
            # thread = Thread(target=self.update, args=(i, multi_rtsp_queue, ))
            thread = Process(target=self.update, args=(i, multi_raw_inputs, multi_rtsp_queue, gpu_sel % ngpus_per_node,))
            gpu_sel += 1
            thread.daemon = True
            thread.start()
            thread = Process(target=self.preprocess_inputs, args=(i, multi_raw_inputs, multi_rtsp_queue,))
            thread.daemon = True
            thread.start()
            # self.thread_list.append(thread)

        for gpu in range(ngpus_per_node):
            # thread = Thread(target=self.batch_inputs, args=(args, gpu, ngpus_per_node,  multi_rtsp_queue, multi_batch_queue, ))
            thread = Process(target=self.batch_inputs, args=(args, gpu, ngpus_per_node, multi_rtsp_queue, multi_batch_queue,))
            thread.daemon = True
            thread.start()

        self.event_info = []
        thread = Thread(target=self.send_msg, args=(multi_event_queue,))
        # thread = Process(target=self.send_msg, args=(multi_event_queue, ))
        thread.daemon = True
        thread.start()

    def send_msg(self, multi_event_queue):
        self.send_msg_logger.info('send_msg | Reconnecting login: {}, getRTSP: {}'.format(self.force_reconnect_with_login, self.force_reconnect))
        while True:
            if any([self.reconnect_attempts[x] for x in self.reconnect_attempts.keys()]):
                self.tryReconnection()
            # if len(self.event_info)>0:
            if multi_event_queue.qsize() > 0:
                try:
                    key = 'send_msg'
                    headers, target_ip, api_serial, auth_token, js = open_cam_login(self.region)  # , self.rtsp_infos[key]['userInput'])
                    print('\n')
                    self.send_msg_logger.info('send_msg | open_cam_login: {}'.format(js))
                    self.target_ip = target_ip
                    # set headers and api_serial as global value
                    self.rtsp_infos['send_msg']['headers'] = headers
                    self.rtsp_infos['send_msg']['api_serial'] = api_serial
                    self.rtsp_infos['send_msg']['auth_token'] = auth_token
                    self.rtsp_infos['send_msg']['target_ip'] = target_ip

                    # key, event_time, _ = self.event_info[0]

                    while not multi_event_queue.empty():
                        key, event_time, reconnect, im, output = multi_event_queue.get()

                        ## Save frame
                        if self.args.save_frame:
                            v = Visualizer(im, data_meta, scale=1.2)
                            out = v.draw_instance_predictions(output)

                            now = event_time.replace(' ', '_')

                            img = Image.fromarray(im)

                            save_path_org = self.args.save_path_org
                            save_path_result = self.args.save_path_result

                            save_path_org_date = os.path.join(save_path_org, 'camera_id_{}'.format(key), now.split('_')[0])
                            save_path_result_date = os.path.join(save_path_result, 'camera_id_{}'.format(key), now.split('_')[0])
                            if not os.path.exists(save_path_org_date):
                                os.makedirs(save_path_org_date)
                            if not os.path.exists(save_path_result_date):
                                os.makedirs(save_path_result_date)

                            img.save('{}/{}.bmp'.format(
                                save_path_org_date,
                                now
                            ))

                            img = Image.fromarray(out.get_image())

                            img.save('{}/{}_result.jpg'.format(
                                save_path_result_date,
                                now,
                            ))

                            self.send_msg_logger.info('send_msg | Result saved at save_path : {}/{}'.format(save_path_result_date, now))

                        # self.send_msg_logger.info("send_msg | Sending events...")
                        self.rtsp_infos['send_msg']['api_serial'] += 1
                        headers = {
                            'content-type': 'application/json',
                            'charset': 'utf-8',
                            'x-auth-token': self.rtsp_infos['send_msg']['auth_token'],
                            'x-api-serial': str(self.rtsp_infos['send_msg']['api_serial'])
                        }

                        event_msg = 'Lying action detected'
                        data = {
                            'camera_id': "{}_0".format(key),
                            'event_id': 208,
                            'event_time': event_time,
                            'event_msg': event_msg,
                            'event_status': 1
                        }

                        res = requests.post(url='http://{}/api/event/send-vca'.format(self.rtsp_infos['send_msg']['target_ip']), data=json.dumps(data), headers=headers)
                        resobj = res.content.decode()
                        js = json.loads(resobj)

                        if js['success']:
                            self.send_msg_logger.info('send_msg | {} Event sent at {} and event occured at {}'.format(self.rtsp_infos[key]['camera_id'], str(datetime.datetime.now())[:-3], event_time))
                            # del self.event_info[0]
                        else:
                            # self.send_msg_logger.info('send_msg | Response from send-vca api was unsuccessful')
                            self.send_msg_logger.info('send_msg | {} Failed to send event occured at {}'.format(self.rtsp_infos[key]['camera_id'], event_time))
                            # self.force_reconnect = 'global'
                            multi_event_queue.put([key, event_time, reconnect, im, output])
                            # self.event_info[0][2] = 1

                    # Delete token after sending events
                    self.send_msg_logger.info('send_msg | Delete token: {}'.format(self.rtsp_infos['send_msg']['auth_token']))
                    res = requests.delete('http://{}/api/logout'.format(self.rtsp_infos['send_msg']['target_ip']), headers=self.rtsp_infos['send_msg']['headers'])
                    resobj = res.content.decode()
                    js = json.loads(resobj)
                    self.rtsp_infos['send_msg'].pop('headers', None)
                    self.rtsp_infos['send_msg'].pop('api_serial', None)
                    self.rtsp_infos['send_msg'].pop('auth_token', None)
                    self.rtsp_infos['send_msg'].pop('target_ip', None)
                    self.send_msg_logger.info('send_msg | Delete response: {}'.format(js))

                except requests.exceptions.Timeout as errd:
                    self.send_msg_logger.info('{} send_msg | Timeout Error : {}'.format(self.rtsp_infos[key]['camera_id'], errd))
                except requests.exceptions.ConnectionError as errc:
                    self.send_msg_logger.info('{} send_msg | Error Connecting : {}'.format(self.rtsp_infos[key]['camera_id'], errc))
                except requests.exceptions.HTTPError as errb:
                    self.send_msg_logger.info('{} send_msg | HTTP Error : {}'.format(self.rtsp_infos[key]['camera_id'], errb))
                except requests.exceptions.RequestException as erra:
                    self.send_msg_logger.info('{} send_msg | AnyException : {}'.format(self.rtsp_infos[key]['camera_id'], erra))
                # except:
                #    self.send_msg_logger.info('{} send_msg | Failed to send requests'.format(self.rtsp_infos[key]['camera_id']))
                finally:
                    pass
                    # multi_event_queue.put([key, event_time, 'global'])
            else:
                time.sleep(1)

    def tryReconnection(self):
        try:
            reconnect_list = [x for x in self.reconnect_attempts.keys() if self.reconnect_attempts[x]]

            key = 'send_msg'
            self.rtsp_infos[key]['userInput'] = 'Global Login'
            headers, target_ip, api_serial, auth_token, js = open_cam_login(self.region)  # , self.rtsp_infos[key]['userInput'])
            self.rtsp_logger_dict[key].info("tryReconnection | open_cam_login {}".format(self.rtsp_infos[key]['userInput'], js))

            self.target_ip = target_ip
            # set headers and api_serial as global value
            self.rtsp_infos['global']['headers'] = headers
            self.rtsp_infos['global']['api_serial'] = api_serial
            self.rtsp_infos['global']['auth_token'] = auth_token
            self.rtsp_infos['global']['target_ip'] = target_ip

            sort_reconnect_list = np.argsort([self.reconnect_attempts[key] for key in reconnect_list])  # .sort()
            # reconnect_list = reconnect_list[sort_reconnect_list]

            prev_attempts = 1
            # for key in reconnect_list:
            for sort_idx in sort_reconnect_list:
                key = reconnect_list[sort_idx]
                # avoid self.rtsp_infos['global'] to get any rtspUrl
                if key == 'global' or key == 'send_msg':
                    self.reconnect_attempts[key] = 0
                    continue

                self.rtsp_logger_dict[key].info("tryReconnection | Channel {} | Start connecting in {}s".format(self.rtsp_infos[key]['userInput'], 10 * (self.reconnect_attempts[key] - prev_attempts)))

                if self.reconnect_attempts[key] > 0:
                    time.sleep(10 * (self.reconnect_attempts[key] - prev_attempts))
                    prev_attempts = self.reconnect_attempts[key]

                self.rtsp_infos['global']['api_serial'] += 1
                headers['x-api-serial'] = str(self.rtsp_infos['global']['api_serial'])

                rtspUrl, target_ip, camera_id, js = open_cam_get_rtspUrl(headers, target_ip, self.rtsp_infos[key]['userInput'], self.region, self.server_number)
                self.rtsp_logger_dict[key].info("tryReconnection | Channel {} | open_cam_get_rtspUrl {}".format(self.rtsp_infos[key]['userInput'], js))

                if rtspUrl == False:
                    error_key = key
                    self.rtsp_logger_dict[key].info("tryReconnection | Channel {} | open_cam_get_rtspUrl failed: {}".format(self.rtsp_infos[key]['userInput'], rtspUrl))
                    self.reconnect_attempts[key] += 1
                    self.reconnect_attempts[key] %= self.max_attempts  # max timeout
                    continue

                self.rtsp_infos[key]['rtspUrl'] = rtspUrl
                self.rtsp_infos[key]['target_ip'] = target_ip
                self.rtsp_infos[key]['camera_id'] = camera_id
                self.rtsp_logger_dict[key].info("tryReconnection | Channel {}| open_cam_rtsp succeeded".format(self.rtsp_infos[key]['userInput']))

                self.reconnect_attempts[key] = 0

            if not self.rtsp_infos['global']['auth_token'] == None:
                self.send_msg_logger.info('tryReconnection | Delete token: {}'.format(self.rtsp_infos['global']['auth_token']))
                res = requests.delete('http://{}/api/logout'.format(self.target_ip), headers=self.headers)
                resobj = res.content.decode()
                js = json.loads(resobj)
                self.rtsp_infos['global']['headers'] = None
                self.rtsp_infos['global']['api_serial'] = None
                self.rtsp_infos['global']['auth_token'] = None
                self.rtsp_infos['global']['target_ip'] = None
                self.send_msg_logger.info('tryReconnection | Delete response: {}'.format(js))

        except requests.exceptions.Timeout as errd:
            self.reconnect_attempts[key] += 1
            self.reconnect_attempts[key] = self.reconnect_attempts[key] if self.reconnect_attempts[key] < self.max_attempts else self.max_attempts
            self.rtsp_logger_dict[key].info("tryReconnection | Channel {} Timeout Error : {}".format(self.rtsp_infos[key]['userInput'], errd))
        except requests.exceptions.ConnectionError as errc:
            self.reconnect_attempts[key] += 1
            self.reconnect_attempts[key] = self.reconnect_attempts[key] if self.reconnect_attempts[key] < self.max_attempts else self.max_attempts
            self.rtsp_logger_dict[key].info("tryReconnection | Channel {} Error Connecting : {}".format(self.rtsp_infos[key]['userInput'], errc))
        except requests.exceptions.HTTPError as errb:
            self.reconnect_attempts[key] += 1
            self.reconnect_attempts[key] = self.reconnect_attempts[key] if self.reconnect_attempts[key] < self.max_attempts else self.max_attempts
            self.rtsp_logger_dict[key].info("Channel {} tryReconnection | Http Error : {}".format(self.rtsp_infos[key]['userInput'], errb))
        except requests.exceptions.RequestException as erra:
            self.reconnect_attempts[key] += 1
            self.reconnect_attempts[key] = self.reconnect_attempts[key] if self.reconnect_attempts[key] < self.max_attempts else self.max_attempts
            self.rtsp_logger_dict[key].info("Channel {} tryReconnection | AnyException : {}".format(self.rtsp_infos[key]['userInput'], erra))
        except RTSPConnectionRefusedError:
            self.reconnect_attempts[key] += 1
            self.reconnect_attempts[key] = self.reconnect_attempts[key] if self.reconnect_attempts[key] < self.max_attempts else self.max_attempts
            self.rtsp_logger_dict[key].info("Channel {} tryReconnection | RTSPConnectionRefused....".format(self.rtsp_infos[key]['userInput']))
        # except:
        #    self.reconnect_attempts[key]+=1
        #    self.reconnect_attempts[key] = self.reconnect_attempts[key] if self.reconnect_attempts[key]<self.max_attempts else self.max_attempts
        #    self.rtsp_logger_dict[key].info("Channel {} tryReconnection | Exception not related to requests".format(self.rtsp_infos[key]['userInput']))
        finally:
            pass

    def update(self, key, multi_raw_inputs, multi_rtsp_queue, gpu=0):
        gpu_decode = True
        while True:
            if not self.reconnect_attempts[key]:
                # self.rtsp_infos[key]['cam'] = 1:
                if gpu_decode and 'cudacodec' in dir(cv2):
                    cv2.cuda.setDevice(gpu)
                    try:
                        cam = cv2.cudacodec.createVideoReader(self.rtsp_infos[key]['rtspUrl'])
                        colour_code_BGR = cv2.cudacodec.ColorFormat_BGR
                        cam.set(colour_code_BGR)
                        ret = cam.grab()
                    except:
                        ret = False
                else:
                    cam = cv2.VideoCapture(self.rtsp_infos[key]['rtspUrl'])
                    ret = cam.grab()
                if not ret:
                    self.rtsp_logger_dict[key].info("Channel {} : frame not retrieved 'ret' is false".format(self.rtsp_infos[key]['userInput']))
                    self.reconnect_attempts[key] += 1
                    continue

                while True:
                    if multi_raw_inputs[key].full():
                        # time.sleep(0.01)
                        continue

                    t = time.time()

                    if gpu_decode and 'cudacodec' in dir(cv2):
                        status, frame = cam.nextFrame()
                        if self.video_res:
                            frame = cv2.cuda.resize(frame, self.video_res)
                        frame = frame.download()
                    else:
                        status, frame = cam.read()

                    if status:
                        multi_raw_inputs[key].put([status, frame])

                    else:
                        self.rtsp_logger_dict[key].info("Channel {} : VideoCapture.read() status is false".format(self.rtsp_infos[key]['userInput']))
                        self.reconnect_attempts[key] += 1
                        break

                    self.func_time['update'][key] = time.time() - t
                    # print(1000*(time.time() - t))
                    # time.sleep(0.03)
            time.sleep(self.reconnect_attempts[key] * 10 + 1)

    def preprocess_inputs(self, key, multi_raw_inputs, multi_rtsp_queue):
        while True:
            # for key in multi_raw_inputs.keys():
            if not multi_raw_inputs[key].empty():
                if multi_rtsp_queue[key].full():
                    # time.sleep(0.01)
                    continue

                t = time.time()

                status, frame = multi_raw_inputs[key].get()

                frame = frame[:, :, ::-1]
                im_t_ = frame.transpose(2, 0, 1)
                c, h, w = im_t_.shape
                im_t_ = im_t_.reshape(1, c, h, w)

                multi_rtsp_queue[key].put([status, frame, im_t_])

                self.func_time['preprocess_inputs'][key] = time.time() - t

    def batch_inputs(self, args, gpu, ngpus_per_node, multi_rtsp_queue, multi_batch_queue):
        rtsp_channel_div = np.array_split(args.rtsp_channels, ngpus_per_node)[gpu]
        rtsp_channel_div = [camera_id_return(x, args.rtsp_region, args.server_number) for x in rtsp_channel_div]

        while True:
            t = time.time()
            skipped_frame = 0
            ret_q = []
            inputs = []
            im_q = []
            ori_img = []

            for channel_div in rtsp_channel_div:
                ret_q_, im_q_, im_t_ = multi_rtsp_queue[channel_div].get_nowait()

                if not ret_q_:
                    im_q_ = self.nosig_view[channel_div]  # self.noSig_view(channel_div)
                    im_t_ = self.nosig_view_t[channel_div]  # self.noSig_view(channel_div)

                    self.skipped_frame_dict[channel_div] = 1
                    skipped_frame += 1
                else:
                    self.skipped_frame_dict[channel_div] = 0

                inputs.append(im_t_)
                ori_img.append(im_q_)

            inputs = np.concatenate(inputs, 0)

            with torch.no_grad():
                # inputs = torch.stack(inputs, 0)
                # inputs = torch.cat(inputs, 0)
                # inputs = torch.as_tensor(inputs)
                inputs = torch.from_numpy(inputs)

            multi_batch_queue[gpu].put([skipped_frame, inputs, ori_img])

            self.func_time['batch_inputs'][gpu] = time.time() - t

            # time.sleep(0.1)
            # time.sleep(0.1)

    def noSig_view(self, chnl):
        black_frame = np.zeros([self.video_res[1], self.video_res[0], 3], dtype=np.uint8)
        view_txt = 'NO SIGNAL'
        txtsize = cv2.getTextSize(view_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        textX = (black_frame.shape[1] - txtsize[0]) // 2
        textY = (black_frame.shape[0] + txtsize[1]) // 2
        cv2.putText(black_frame, view_txt, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        view_chnl = 'Channel ' + str(chnl)
        cv2.putText(black_frame, view_chnl, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return black_frame

    def get(self):
        return self.status_q, self.frame_q
