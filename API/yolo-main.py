import warnings, socket
warnings.simplefilter('ignore')

from flask import Flask, request, Response
from config import DevConfig

app = Flask(__name__)
app.config.from_object(DevConfig)

from yolo import yolo
options = {"pbLoad": "yolo-top.pb", "metaLoad": "yolo-top.meta", "json": True}
tfnet = yolo(options).createNet()

def pred(img):
	return tfnet.return_predict(img)

def parsing(js, keys):
	detected = False
	for r_ in js:
		if r_["label"] == keys:
			detected = True
			x1 = r_["topleft"]["x"]; y1 = r_["topleft"]["y"]
			x2 = r_["bottomright"]["x"]; y2 = r_["bottomright"]["y"]
			h = y2 - y1; w = x2 - x1

	if not detected:
		return "Haven't detected."
	else:
		return [x1, y1, x2, y2, h, w]

import numpy as np, cv2, jsonpickle
print("Model test ...")
tsimg = np.ones([10,10,3], dtype="uint8")
pred(tsimg)


@app.route('/', methods=['GET'])
def index():
	return 'Hello World!'


@app.route('/yolo/top', methods=['POST'])
def top():
	r = request
	nparr = np.fromstring(r.data, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	result = pred(img)
	ps = parsing(result, keys="top")

	resp_pickled = jsonpickle.encode(ps)
	return Response(response=resp_pickled, status=200, mimetype='application/json')


@app.route('/yolo/bottom', methods=['POST'])
def bottom():
	r = request
	nparr = np.fromstring(r.data, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	result = pred(img)
	ps = parsing(result, keys="bottom")

	resp_pickled = jsonpickle.encode(ps)
	return Response(response=resp_pickled, status=200, mimetype='application/json')


if __name__ == '__main__':
	app.run(host=socket.gethostbyname(socket.gethostname()), port=508)
