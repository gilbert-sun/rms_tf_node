#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String,Int8
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf

#monitor = ''

import diagnostic_msgs.msg
from enum import Enum
from rostopic import ROSTopicHz
import random

from node_function_monitor.msg import BoundingBox

#from node_monitor import topic_diagnostics,node_result,DataInfo,node_monitor

# cp from https://github.com/tensorflow/models/tutorials/image/imagenet/classify_image.py
from node_function_monitor.node_monitor import node_monitor

from classify_image import tf_model_download

import classify_image

class BoundingBox():
	def __init__(self, clas="", prob=0):
		self.Class= clas
		self.probablity =prob

def data_callback(data):
	global msg_data
	global content
	#if data from the topic exists, store display message
	if data:
		msg_data = DataInfo.msg_correct.value
	else:
		msg_data = DataInfo.msg_wrong.value
	content = data.data

def topic_check_callback(msg, arg):
	global camera_is_work
	monitor = arg

	if msg.data == 1:
		#send result/status
		camera_is_work = True
		print("---------111 camera_is_work--------")

	else:
		#send status and do some handling
		camera_is_work = False
		print("---------000 camera_is_work--------")

def topic_check_callback2(msg, arg):
	monitor = arg

	if msg.data == 1:
		# node status
		data2 = {}
		data2['Status'] = 'Normal'
		monitor.node_info(data2, 'Status', 'Node is fine.', 0)
		print("---------456--------",data2.Status)
	else:
		# node status
		data2 = {}
		data2['Status'] = 'Topic is not active'
		monitor.node_info(data2, 'Status', 'Topic is not active.', 2)
		print("---------789--------", data2.Status)


def status_result(nodename, pub, data):
	# prepare the diagnostic array
	diagnosticArray = diagnostic_msgs.msg.DiagnosticArray()
	diagnosticArray.header.stamp = rospy.get_rostime()

	# prepare diagnostics message
	# determine level, node name, object recognized and probability
	statusMsg = diagnostic_msgs.msg.DiagnosticStatus()
	statusMsg.name = nodename + " -> --Status--"
	statusMsg.message = 'Detected ' + data.Class
	statusMsg.level = 0
	statusMsg.values.append(
		diagnostic_msgs.msg.KeyValue("Node name", nodename)
	)
	statusMsg.values.append(
		diagnostic_msgs.msg.KeyValue("Status:", str(data.Class))
	)
	statusMsg.values.append(
		diagnostic_msgs.msg.KeyValue("Val:", str(data.probability))
	)

	# append message to diagnostic array
	diagnosticArray.status.append(statusMsg)

	# publish array
	pub.publish(diagnosticArray)


class RosTensorFlow():
#	global monitor
	def __init__(self):
		self._session = tf.Session()

		classify_image.create_graph()

		self.node_name = 'rms_tf_node'

		rospy.init_node(self.node_name, anonymous=True)

		self.monitor = node_monitor(self.node_name)

		self.bridge = CvBridge()

		self._sub = rospy.Subscriber('/logitech_c922/image_raw', Image, self.callback, queue_size=1)

		self._pub = rospy.Publisher('result', String, queue_size=1)

		self._pub1 = rospy.Publisher('Imgnet_Tf', Image, queue_size=1)

		self.score_threshold = rospy.get_param('~score_threshold', 0.1)

		self.use_top_k = rospy.get_param('~use_top_k', 5)




	def callback(self, image_msg):

		frame= self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

		image_data = cv2.imencode('.jpg', frame)[1].tostring()

		image_data2 = cv2.imencode('.jpg', frame)[1]

		image_data3 = cv2.imdecode(image_data2, cv2.IMREAD_COLOR)

#		cv2.imshow("imgNet_tf", image_data3)

#		cv2.waitKey(1)

		#rospy.sleep(0.1)
		# Creates graph from saved GraphDef.
		softmax_tensor = self._session.graph.get_tensor_by_name('softmax:0')

		predictions = self._session.run(
			softmax_tensor, {'DecodeJpeg/contents:0': image_data})

		predictions = np.squeeze(predictions)

		# Creates node ID --> English string lookup.
		node_lookup = classify_image.NodeLookup()

		top_k = predictions.argsort()[-self.use_top_k:][::-1]


		for node_id in top_k:
			human_string = node_lookup.id_to_string(node_id)
			score = predictions[node_id]
			if score > self.score_threshold:
				rospy.loginfo('%s (score = %.5f)' % (human_string, score))
				self._pub.publish(human_string)

				#topic_diag_main("ros_tf_node_imgNet", "", human_string, (float(score)) * 100)
				data={}
				data['class'] = human_string
				data['probablity'] = (float(score)) * 100
				# send a result to be viewed on the console; in this case, data
				# node_result(node_name, diagnostics_pub, diagnostics_data)
				self.monitor.node_info(data, '--Result--', 'Detection result received.', 0)
				data2 = BoundingBox()
				data2.Class = "Status"  # object[r]
				data2.probability = 1
				pub2 = rospy.Publisher('diagnostics1', diagnostic_msgs.msg.DiagnosticArray, queue_size=10)
				status_result(self.node_name, pub2, data2)
				#_pub2 = rospy.Publisher('diagnostics1', diagnostic_msgs.msg.DiagnosticArray, queue_size=10)
				# self.monitor1.node_info(data2, '--Status--', 'Object Recgnization Stauts', 1)

	def main(self):
		rospy.spin()

if __name__ == '__main__':
	tf_model_download
	tensor = RosTensorFlow()
	tensor.main()
