import socket
import time
import struct
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import threading
import queue

HOST_CAM = '192.168.137.220'
PORT_CAM = 80
PORT_MOT = 81

client_cam = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_mot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_cam.connect((HOST_CAM, PORT_CAM))
client_mot.connect((HOST_CAM, PORT_MOT))

t_now = time.time()
t_prev = time.time()
cnt_frame = 0

model = load_model('model.h5')

names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

HOW_MANY_MESSAGES = 10
mq = [
	queue.Queue(HOW_MANY_MESSAGES),
	queue.Queue(HOW_MANY_MESSAGES),
	queue.Queue(HOW_MANY_MESSAGES),
	queue.Queue(HOW_MANY_MESSAGES)
]

flag_exit = False
def cnn_main(args) :

	print(args)

	while True:

		frame = mq[args].get()

		image = frame
		image = image/255

		image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
		# print(image_tensor.shape)

		# Add dimension to match with input mode
		image_tensor = tf.expand_dims(image_tensor, 0)
		# print(image_tensor.shape)

		y_predict = model.predict(image_tensor)
		y_predict = np.argmax(y_predict,axis=1)
		print(names[y_predict[0]], y_predict[0])

		# send y_predict
		cmd = y_predict[0].item()
		cmd = struct.pack('B', cmd)
		client_mot.sendall(cmd)

		if flag_exit: break

cnnThread_0 = threading.Thread(target=cnn_main, args=(0,))
cnnThread_0.start()
cnnThread_1 = threading.Thread(target=cnn_main, args=(1,))
cnnThread_1.start()
cnnThread_2 = threading.Thread(target=cnn_main, args=(2,))
cnnThread_2.start()
cnnThread_3 = threading.Thread(target=cnn_main, args=(3,))
cnnThread_3.start()

fn = 0

while True:

	# 영상 보내
	cmd = 12
	cmd = struct.pack('B', cmd)
	client_cam.sendall(cmd)

	# 영상 받기
	data_len_bytes = client_cam.recv(4)
	data_len = struct.unpack('I', data_len_bytes)

	data = client_cam.recv(data_len[0], socket.MSG_WAITALL)

	# 영상 출력
	np_data = np.frombuffer(data, dtype='uint8')
	frame = cv2.imdecode(np_data,1)
	frame = cv2.rotate(frame,cv2.ROTATE_180)
	frame2 = cv2.resize(frame, (320, 240))
	cv2.imshow('frame', frame2)

	mq[fn%4].put(frame)
	fn += 1

	key = cv2.waitKey(1)
	if key == 27:
		break

	cnt_frame += 1
	t_now = time.time()
	if t_now - t_prev > 1.0 :
		t_prev = t_now
		print("frame count : %f" %cnt_frame)
		cnt_frame = 0

flag_exit = True
cnnThread_0.join()
cnnThread_1.join()
cnnThread_2.join()
cnnThread_3.join()

client_cam.close()
client_mot.close()