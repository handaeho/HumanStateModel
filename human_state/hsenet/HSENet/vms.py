import json
import requests

target_ip = '129.254.75.110'
target_port = '8080'


def open_token():
    headers = {'Accept': 'application/json', 'x-account-id': 'sdk', 'x-account-pass': 'Innodep1@',
               'x-account-group': 'group1', 'x-license': 'licVCA|licPasswordSolution', 'charset': 'utf-8'}
    res = requests.get('http://{}:{}/api/login?force-login=true'.format(target_ip, target_port), headers=headers)

    resobj = res.content.decode()
    js = json.loads(resobj)
    auth_token = js['results']['auth_token']
    api_serial = js['results']['api_serial']

    return auth_token, api_serial


def device_list(auth_token, api_serial):
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
    # return dslist[:20]


def open_cam_rtsp(auth_token, api_serial, idx):
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
    # print(rtspUrl)
    return rtspUrl


def get_rtsp_urls():
    auth_token, api_serial = open_token()
    api_serial = api_serial + 1
    dslist = device_list(auth_token, api_serial)
    print(dslist)

    rtsp_urls = []
    for item in dslist:
        api_serial = api_serial + 1
        rtsp_url = open_cam_rtsp(auth_token, api_serial, item)
        rtsp_urls.append(rtsp_url)

    # print(rtsp_urls)
    return rtsp_urls


if __name__ == "__main__":
    get_rtsp_urls()
