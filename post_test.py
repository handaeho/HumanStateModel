import datetime
import json
import requests


cam_number = 0
event_time = "2023-09-18 09:48:00"
event_msg = 'Lying action detected'

# 이벤트 API 호출
url = "http://192.168.2.18:8080/api/transfer/alert"
headers = {
    "content-type": "application/json",
    "charset": "utf-8",
}
data = {
    "camera_id": f"{cam_number}",
    "event_time": event_time,
    "event_msg": event_msg,
    "event_status": 1
}

res = requests.post(url=url, data=json.dumps(data), headers=headers)
resobj = res.content.decode()
js = json.loads(resobj)

if js['success']:
    print(f'send_msg | Camera {cam_number} Event sent at {str(datetime.datetime.now())[:-7]} and event occured at {event_time}')
else:
    print(f'send_msg | {cam_number} Failed to send event occured at {event_time}')
