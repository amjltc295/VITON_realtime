import requests
import pickle
import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)

url = 'http://localhost:5000'
url = 'http://140.112.29.182:8000/pose'
url = 'http://140.112.29.182:8000/seg'
while 1:
    t = time.time()
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    ret, img = cap.read()
    cv2.imwrite('tmp.jpg', img)
    # img_json = json.dumps(img.tolist())
    files = {'files': open('tmp.jpg', 'rb')}
    r = requests.post(url, files=files)
    # r = requests.post(url, data=img_json)
    # img = json.loads(r.content)
    # img = np.array(img)
    pose = pickle.loads(r.content)
    import pdb
    pdb.set_trace()
    print(pose)
    print(1 / (time.time() - t))
