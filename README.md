# WTC_deepview

## Prerequisite Programs
```
1. conda 설치
	<리눅스>
		$ sudo apt-get update
		$ sudo apt-get install curl -y
		$ curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
		$ sha256sum anaconda.sh
		$ bash anaconda.sh
		$ sudo vi ~/.bashrc 
			(아래 내용 입력 후 저장. 경로 확인 필수)
			# >>> conda initialize >>>
			export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH

			# !! Contents within this block are managed by 'conda init' !!
			__conda_setup="$('/home/wtc/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
			if [ $? -eq 0 ]; then
        			eval "$__conda_setup"
			else
        			if [ -f "/home/wtc/anaconda3/etc/profile.d/conda.sh" ]; then
                			. "/home/wtc/anaconda3/etc/profile.d/conda.sh"
        			else
                			export PATH="/home/wtc/anaconda3/bin:$PATH"
        			fi
			fi
			unset __conda_setup
			# <<< conda initialize <<<
		$ source ~/.bashrc
		$ conda -V
	
	<윈도우>
		$ https://www.anaconda.com/ 접속 후 다운로드
	
2. conda 가상환경 생성(with python 3.10)
	$ conda create -n wtc python=3.10
	(가상환경 activate 했는데 아래와 같은 에러 발생 시,)
		- CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'. 
		$ source ~/anaconda3/etc/profile.d/conda.sh
		(실행 후 다시 activate 시도)

3. NVIDIA 그래픽 드라이버 설치
	1) nvidia 드라이버 삭제
		$ sudo apt-get remove --purge nvidia-*
		$ sudo apt-get autoremove
		$ sudo apt-get update
		(확인)
		$ sudo dpkg -l | grep nvidia
		(아무것도 안나오면 완료. 아니라면)
		$ sudo apt-get remove --purge [지울이름]
	2) nvidia 드라이버 설치
		$ ubuntu-drivers devices
		$ sudo apt-get install nvidia-driver-525
		$ sudo apt-get install dkms nvidia-modprobe
		$ sudo apt-get update
		$ sudo apt-get upgrade
		$ sudo reboot
	3) Failed to initialize NVML: Driver/library version mismatch 에러 발생 시,
		1) 로딩되어 있는 nvidia driver kernel을 확인
			$ lsmod | grep nvidia
		2) nvidia driver kernel들을 unload 해준다. 
			(순서대로 입력)
			$ sudo rmmod nvidia_drm
			$ sudo rmmod nvidia_modeset
			$ sudo rmmod nvidia_uvm
			$ sudo rmmod nvidia
			- rmmod: ERROR: Module nvidia_drm is in use 발생 시,
				$ systemctl isolate multi-user.target
				$ modprobe -r nvidia-drm
				$ systemctl start graphical.target
				$ sudo rmmod nvidia_drm -> 재시도
			-  rmmod: ERROR: Module nvidia_uvm is in use 또는 rmmod: ERROR: Module nvidia is in use 발생 시,
				(nvidia와 관련된 프로세스 번호를 추출하여 kill 해준다)
				$ sudo lsof /dev/nvidia* | awk 'NR > 1 {print $2}' | sudo xargs kill
				$ sudo rmmod nvidia_uvm 또는 $ sudo rmmod nvidia -> 재시도
		3)  다시 한 번 로딩되어 있는 nvidia driver kernel을 확인.
			$ lsmod | grep nvidia
			(아무 것도 뜨지 않는다면 성공)
		4) 확인
			$ nvidia-smi

4. CUDA 11.7.0 설치 -> https://vividian.net/2022/11/1111 참고
	1) https://developer.nvidia.com/cuda-toolkit-archive 이동(wget으로 받으면 안해도 됨)
	2) CUDA Toolkit 11.7 Runfile 다운로드 및 설치
		$ wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
		$ sudo sh cuda_11.7.0_515.43.04_linux.run
	3) nvidia 드라이버를 이미 설치했으므로, Driver는 제외하고 나머지 설치
	4) 환경변수 설정
		$ sudo sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.7/bin'>> /etc/profile"
		$ sudo sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64'>> /etc/profile"
		$ sudo sh -c "echo 'export CUDADIR=/usr/local/cuda-11.7'>> /etc/profile"
		$ source /etc/profile
	5) CUDA 삭제
		$ sudo rm -rf /usr/local/cuda*
		$ sudo apt-get remove --purge cuda*
		$ sudo apt-get autoremove --purge 'cuda*'
		$ sudo apt-get update
		(확인)
		$ sudo dpkg -l | grep cuda
		(아무것도 안나오면 완료. 아니라면)
		- sudo apt-get remove --purge [지울이름]

5. cuDNN v8.4.1 설치 -> https://vividian.net/2022/11/1111 참고
	1) https://developer.nvidia.com/cudnn 이동
	2) Download cuDNN v8.4.1 (May 27th, 2022), for CUDA 11.x 선택 후, Local Installer for Linux x86_64 (Tar) 다운로드
	3) 다운로드 받은 cuDNN 압축 풀기
		$ tar -xvf cudnn-linux-~~~.tar.xz
	4) /usr/local/cuda 디렉토리로 복사
		$ cd cudnn-linux-~~~
		$ sudo cp include/cudnn* /usr/local/cuda/include
		$ sudo cp lib/libcudnn* /usr/local/cuda/lib64
		$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
	5) cuda 디렉토리와 실제 설치된 cuda-11.7 디렉토리를 심볼릭 링크 설정
		(8.4.1인지 버전 잘 확인하기)
		$ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.4.1 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
		$ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.4.1 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
		$ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.4.1 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
		$ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.4.1 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
		$ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.4.1 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
		$ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.4.1 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
		$ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn.so.8.4.1 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn.so.8
	6) cuDNN 설정이 제대로 됐는지 확인
		$ sudo ldconfig
		$ ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
	7) CUDNN 8 이후 CUDNN 버전 확인
		cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
		(안된다면)
		cat /usr/local/cuda(설치된버전확인)/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

6. torch 설치
	<GPU 버전>
		# CUDA 11.7
		conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
	
	<CPU 버전>
		# CPU Only
		conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch

7. detectron2 설치
	<리눅스>
		$ python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
		(git 설치 필요시)
		$ conda install git

	<윈도우> 
		- error : Microsoft Visual C++ 14.0 or greater is required 발생시, Microsoft Visual C++ 설치 필요함
			- https://aka.ms/vs/17/release/vc_redist.x64.exe 다운로드 후, "C++을 사용한 데스크톱 개발"에 체크하고 설치
		- git 설치
			- https://git-scm.com/downloads
			$ conda install git
		$ pip install cython
		$ pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
		$ git clone https://github.com/facebookresearch/detectron2.git
		$ python -m pip install -e detectron2

			>> nvcc.exe failed with exit status 1 에러 발생 시,
				1) detectron2\layers\csrc\nms_rotated\nms_rotated_cuda.cu 파일에서

				2) 	#ifdef WITH_CUDA
					#include "../box_iou_rotated/box_iou_rotated_utils.h"
					#endif
					// TODO avoid this when pytorch supports "same directory" hipification
					#ifdef WITH_HIP
					#include "box_iou_rotated/box_iou_rotated_utils.h"
					#endif

				3) 이 부분을 주석 처리하고 밑에 #include "box_iou_rotated/box_iou_rotated_utils.h"를 쓴다.

				4) 	/*#ifdef WITH_CUDA
					#include "../box_iou_rotated/box_iou_rotated_utils.h"
					#endif
					// TODO avoid this when pytorch supports "same directory" hipification
					#ifdef WITH_HIP
					#include "box_iou_rotated/box_iou_rotated_utils.h"
					#endif*/
					#include "box_iou_rotated/box_iou_rotated_utils.h"

8. opencv(cpu 버전) 설치 -> 만약 opencv-GPU 버전이 필요하다면, make 명령어 써 놓은것 참고
	$ pip install opencv-python
```

## Prerequisite Models
```
------------------------------------------------------------------------------------------------------------------------------------------------------
[object detection]
python demo.py --config-file configs/faster_rcnn_V_19_slim_FPNLite_3x.yaml --video-input /home/gaion/WTC_Seoul/wtc_seoul/vedios/cctv_003.mp4 --output /home/gaion/WTC_Seoul/wtc_seoul/vedios_result --opts MODEL.WEIGHTS weights/faster_rcnn_V_19_eSE_slim_FPNLite_ms_3x.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
[keypoint]
python demo.py --config-file configs/keypoint_V_39_FPN_3x.yaml --input //home/gaion/WTC_Seoul/wtc_seoul/images/human_keypoint_test.jpg --output /home/gaion/WTC_Seoul/wtc_seoul/images_result --opts MODEL.WEIGHTS weights/keypoint_vovnet39.pth
python demo.py --config-file configs/keypoint_V_39_FPN_3x.yaml --video-input /home/gaion/WTC_Seoul/wtc_seoul/vedios/cctv_003.mp4 --output /home/gaion/WTC_Seoul/wtc_seoul/vedios_result --opts MODEL.WEIGHTS weights/keypoint_vovnet39.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
[instance segmentation] --> output에서 파일 이름 맨뒤에 숫자가 들어가면 안됨. 따라서 input도 숫자 없이 하기.
python demo/demo.py --config-file configs/centermask_lite_V_39_eSE_FPN_ms_4x.yaml --video-input /home/gaion/WTC_Seoul/wtc_seoul/vedios/cctv.mp4 --output /home/gaion/WTC_Seoul/wtc_seoul/vedios_result --opts MODEL.WEIGHTS weights/centermask_lite_V_39_eSE_FPN_ms_4x.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
[keypoint webcam]
python demo.py --config-file configs/keypoint_V_39_FPN_3x.yaml --webcam --opts MODEL.WEIGHTS weights/keypoint_vovnet39.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
[segmentation webcam]
python demo/demo.py --config-file configs/centermask_lite_V_39_eSE_FPN_ms_4x.yaml --webcam --opts MODEL.WEIGHTS weights/centermask_lite_V_39_eSE_FPN_ms_4x.pth 
------------------------------------------------------------------------------------------------------------------------------------------------------
[human-state video]
python infer_videos.py --config configs/mphbe_hsenet_R_50_FPN_3x.yaml --video /path/to/video MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
-> python infer_videos.py --config configs/mphbe_hsenet_R_50_FPN_3x.yaml --video /home/gaion/WTC_Seoul/wtc_seoul/vedios/cctv_001.mp4 --output_dir /home/gaion/WTC_Seoul/wtc_seoul/vedios_result MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
-> python infer_videos.py --config configs/mphbe_hsenet_R_50_FPN_3x.yaml --video /home/gaion/WTC_Seoul/wtc_seoul/vedios/cctv_002.mp4 --output_dir /home/gaion/WTC_Seoul/wtc_seoul/vedios_result MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
-> python infer_videos.py --config configs/mphbe_hsenet_R_50_FPN_3x.yaml --video /home/gaion/WTC_Seoul/wtc_seoul/vedios/cctv_003.mp4 --output_dir /home/gaion/WTC_Seoul/wtc_seoul/vedios_result MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
[human-state webcam]
python infer_webcam.py --config-file configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
[human-state pred_rtsp_gaion]
python pred_rtsp_gaion.py --rtsp_channels=[0] --num-gpus=1 --config-file configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth 
------------------------------------------------------------------------------------------------------------------------------------------------------
[human-state infer_rtsp_gpu]
python infer_rtsp_gpu.py --rtsp_channels=[0] --num-gpus=1 --config-file configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth 
python infer_rtsp_gpu.py --rtsp_channels=[0] --num-gpus=8 --config-file configs/mphbe_hsenet_R_50_FPN_3x.yaml 
python test_001.py --config-file configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
python test_002.py --rtsp_channels=[0,1,2,3,4,5,6,7] --num-gpus=8 --config-file configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
python test_002.py --rtsp_channels=[0] --config-file /home/ho/WTC_Seoul/wtc_seoul/human_state/configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
>>>>> ETRI HESNET <<<<<
[human-state hsenet_test.py]
python hsenet_test.py --config-file /home/ho/WTC_Seoul/wtc_seoul/human_state/configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
[human-state hsenet_vms.py]
python hsenet_vms.py --config-file /home/ho/WTC_Seoul/wtc_seoul/human_state/configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
[human-state hsenet_vms_multi.py]
python hsenet_vms_multi.py --config-file /home/ho/WTC_Seoul/wtc_seoul/human_state/configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
[human-state hsenet_vms_th.py] --> detectron2 모델은 muti-GPU 환경에서의 연산이 불가능하다
python hsenet_vms_th.py --config-file /home/ho/WTC_Seoul/wtc_seoul/human_state/configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
[human-state hsenet_vms_batch.py] --> 그래서 그냥 batch 형태로 GPU 수만큼 나누어서 각 GPU에 py 파일 올리고 실행(1 GPU 1 model)
python hsenet_vms_batch.py --config-file /home/ho/WTC_Seoul/wtc_seoul/human_state/configs/mphbe_hsenet_R_50_FPN_3x.yaml MODEL.WEIGHTS /home/ho/WTC_Seoul/wtc_seoul/human_state/model_0269999.pth
------------------------------------------------------------------------------------------------------------------------------------------------------
[human-state hsenet_batch_gaion.py]
python hsenet_batch_gaion.py
[human-state hsenet_batch_gaion.py] --> GPU 할당 후 실행(ex) 0번 GPU)
CUDA_VISIBLE_DEVICES=0 python hsenet_batch_gaion.py
```

## Command Collection Programs
```
[ETRI training]
python train_net_etri.py --config configs/etri_ver_hsenet_config.yaml --num-gpus 8 --resume
===========================================================================================
[Tensor Board]
tensorboard --logdir checkpoints/ --port 9999 --bind_all
===========================================================================================
[training with F1 Score, Presicion, Recall]
> just inference
python hsenet_gaion_train_net.py --eval-only --num-gpus 8 --config ./configs/gaion_ver_hsenet_config.yaml MODEL.WEIGHTS ./weights/model_final.pth

> if you want training, remove --eval-only
python hsenet_gaion_train_net.py --num-gpus 8 --config ./configs/gaion_ver_hsenet_config.yaml MODEL.WEIGHTS ./weights/model_final.pth

> 3 Class 훈련
python hsenet_gaion_train_net_3.py --num-gpus 8 --config ./configs/gaion_ver_hsenet_config_3.yaml MODEL.WEIGHTS ./checkpoints/gaion_ver_train_100000_0_02_3class/model_final.pth

> 3 Class 평가
python hsenet_gaion_train_net_3.py --eval-only --num-gpus 8 --config ./configs/gaion_ver_hsenet_config_3.yaml MODEL.WEIGHTS ./checkpoints/gaion_ver_train_100000_0_02_3class/model_final.pth

python hsenet_gaion_train_net_3.py --eval-only --num-gpus 8 --config ./configs/gaion_ver_hsenet_config_3.yaml MODEL.WEIGHTS ./weights/model_100000.pth

> 6 Class 훈련
python hsenet_gaion_train_net_6.py --num-gpus 8 --config ./configs/gaion_ver_hsenet_config_6.yaml MODEL.WEIGHTS ./weights/model_final.pth

> 6 Class 평가
python hsenet_gaion_train_net_6.py --eval-only --num-gpus 8 --config ./configs/gaion_ver_hsenet_config_6.yaml MODEL.WEIGHTS ./weights/model_final.pth
===========================================================================================
[FPS 측정] -> gpu-home/tiep/detectron2
python vovnet_train_net.py --eval-only --config=configs/faster_rcnn_V_19_slim_FPNLite_3x_gaion.yaml MODEL.WEIGHTS ./checkpoints/FRCN-V2-19-slim-FPNLite-3x-gaion/model_final.pth
===========================================================================================
[mAP50 측정] -> IHPE test datasets + model_final.pth
python hsenet_gaion_train_net_6.py --eval-only --num-gpus 8 --config ./configs/gaion_ver_hsenet_config_6.yaml MODEL.WEIGHTS ./weights/model_final.pth
===========================================================================================
[F1-Score 측정] =====>>>>> 모델 이름 수정 필요(뭘로 쓸건지?)
python hsenet_gaion_train_net_3.py --eval-only --num-gpus 8 --config ./configs/gaion_ver_hsenet_config_3.yaml MODEL.WEIGHTS ./checkpoints/gaion_ver_train_100000_0_02_3class/model_final.pth
===========================================================================================
```

## Command OpenCV-GPU Installation
```
1. 설치된 OpenCV 확인
	$ pkg-config --modversion opencv 
	(버전 안나오면 설치 X, 나오면)
	$ sudo apt-get purge  libopencv* python-opencv
	$ sudo apt-get autoremove

2. 기존 설치된 패키지 업그레이드
	$ sudo apt-get update
	$ sudo apt-get upgrade

3. OpenCV 컴파일 전 필요한 패키지 설치
	$ sudo apt-get install build-essential cmake
	$ sudo apt-get install pkg-config
	$ sudo apt-get install libjpeg-dev libtiff5-dev libpng-dev
	$ sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
	$ sudo apt-get install libv4l-dev v4l-utils
	$ sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev 
	$ sudo apt-get install libgtk-3-dev
	$ sudo apt-get install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev
	$ sudo apt-get install libatlas-base-dev gfortran libeigen3-dev
	$ sudo apt-get install python3-dev python3-numpy
4. 설치
	$ cd opencv_etri
	$ mkdir bulid
	$ cd bulid
	$ /usr/bin/cmake -B build -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/gaion/anaconda3/envs/wtc -D INSTALL_PYTHON_EXAMPLES=OFF \ 
	-D INSTALL_C_EXAMPLES=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D BUILD_PACKAGE=OFF -D BUILD_EXAMPLES=OFF -D WITH_TBB=ON \
        -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -D WITH_CUDA=ON -D WITH_CUBLAS=ON \
        -D WITH_CUFFT=ON -D WITH_NVCUVID=ON -D WITH_IPP=OFF -D WITH_V4L=ON -D WITH_1394=OFF -D WITH_GTK=ON -D WITH_QT=OFF \
        -D WITH_OPENGL=OFF -D WITH_EIGEN=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D BUILD_JAVA=OFF \
        -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D OPENCV_SKIP_PYTHON_LOADER=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules -D WITH_CUDNN=ON -D BUILD_opencv_cudev=ON \
        -D CUDA_ARCH_BIN=5.0:5.2:6.0:6.1:7.0:7.5:8.0:8.6 -D CUDA_ARCH_PTX=5.0:5.2:6.0:6.1:7.0:7.5:8.0:8.6 -D WITH_NVCUVID=ON \
        -D BUILD_opencv_cudacodec=ON -D BUILD_opencv_xphoto=OFF -D BUILD_opencv_xfeatures2d=OFF -D BUILD_opencv_ximgproc=OFF \
        -D BUILD_opencv_xobjdetect=OFF -D BUILD_opencv_python_bindings_generator=ON -D PYTHON3_LIBRARY=/home/gaion/anaconda3/envs/wtc/lib/python3.10 \
        -D PYTHON3_INCLUDE_DIR=/home/gaion/anaconda3/envs/wtc/include/python3.10 \
        -D PYTHON3_EXECUTABLE=/home/gaion/anaconda3/envs/wtc/bin/python \
        -D PYTHON3_PACKAGES_PATH=/home/gaion/anaconda3/envs/wtc/lib/python3.10/site-packages \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/gaion/anaconda3/envs/wtc/lib/python3.10/site-packages/numpy/core/include \
        -D OPENCV_PYTHON3_VERSION=3.10 -D CMAKE_INSTALL_PREFIX=/home/gaion/anaconda3/envs/wtc opencv

	$ make all -j 40 -C build

	$ make install -j 40 -C build


[ERROR 처리]
>> CUDA_ERROR_FILE_NOT_FOUND [Code = 301] in function 'CuvidVideoSource' -> make 할 때 ffmpeg YES/NO 확인
=> sudo apt install pkg-config     
sudo apt install ffmpeg libavformat-dev libavcodec-dev libswscale-dev 

>> The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
=> sudo apt-get install libgtk2.0-dev pkg-config
```
