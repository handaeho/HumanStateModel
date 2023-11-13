import json
import requests

"""  
RTSP 카메라의 RTSP URL을 얻기 위한 function 
    1. 카메라의 인증 토큰과 시리얼 넘버를 획득 
    2. 이 인증 토큰과 시리얼 넘버에 해당하는 디바이스 리스트를 획득 
    3. 이 디바이스 리스트에 포함된 카메라의 RTSP URL을 획득
"""

# TODO: 무역센터의 환경(아이디스)에 맞게 재구성해야 함

# VMS IP / PORT
target_ip = '129.254.75.110'
target_port = '8080'


def open_token():
    """
    VMS에 요청을 보내 카메라의 인증 토큰과 시리얼 넘버를 획득

    Returns: auth_token, api_serial
    """
    headers = {'Accept': 'application/json', 'x-account-id': 'sdk', 'x-account-pass': 'Innodep1@',
               'x-account-group': 'group1', 'x-license': 'licVCA|licPasswordSolution', 'charset': 'utf-8'}

    res = requests.get('http://{}:{}/api/login?force-login=true'.format(target_ip, target_port), headers=headers)

    resobj = res.content.decode()

    js = json.loads(resobj)

    auth_token = js['results']['auth_token']
    api_serial = js['results']['api_serial']

    return auth_token, api_serial


def device_list(auth_token, api_serial):
    """
    VMS에 요청을 보내 획득한 인증 토큰과 시리얼 넘버에 해당하는 디바이스 리스트를 획득

    Args:
        auth_token: 인증 토큰
        api_serial: 시리얼 넘버

    Returns: 디바이스 리스트
    """
    headers = {
        'Accept': 'application/json',
        'x-auth-token': auth_token,
        'x-api-serial': str(api_serial)
    }

    res = requests.get('http://{}:{}/sdk/device/list'.format(target_ip, target_port), headers=headers)

    resobj = res.content.decode()

    js = json.loads(resobj)

    dlist = js['results']

    dslist = []
    for item in dlist:
        dslist.append(item['dev_serial'])

    return dslist[12:]


def open_cam_rtsp(auth_token, api_serial, idx):
    """
    디바이스 리스트에 포함된 카메라의 RTSP URL을 획득

    Args:
        auth_token: 인증 토큰
        api_serial: 시리얼 넘버
        idx: 카메라 디바이스 번호

    Returns: RTSP URL
    """
    headers = {
        'Accept': 'application/json',
        'x-auth-token': auth_token,
        'x-api-serial': str(api_serial)
    }

    res = requests.get('http://{}:{}/api/video/rtsp-url/'.format(target_ip, target_port) + str(idx) + '/0/0',
                       headers=headers)

    resobj = res.content.decode()

    js = json.loads(resobj)

    rtspUrl = js['results']['url']

    return rtspUrl


def get_rtsp_urls():
    """
    요청된 각 카메라의 수만큼 RTSP URL을 가져오고 리스트에 담아 반환

    Returns: RTSP URL List
    """
    auth_token, api_serial = open_token()

    api_serial = api_serial + 1

    dslist = device_list(auth_token, api_serial)
    print(dslist)

    rtsp_urls = []
    for item in dslist:
        api_serial = api_serial + 1
        rtsp_url = open_cam_rtsp(auth_token, api_serial, item)
        rtsp_urls.append(rtsp_url)

    return rtsp_urls

