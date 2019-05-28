import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String,Int8
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf

import diagnostic_msgs.msg
from enum import Enum
from rostopic import ROSTopicHz
import random
#from node_function_monitor.msg import BoundingBox


#from node_monitor import topic_diagnostics,node_result,DataInfo,node_monitor
from node_function_monitor.node_monitor import node_monitor
# cp from https://github.com/tensorflow/models/tutorials/image/imagenet/classify_image.py

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



class RosTensorFlow():
	global monitor
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

		self.human_string = ''

		self.score = 0



	def callback(self, image_msg):

		frame= self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

		image_data = cv2.imencode('.jpg', frame)[1].tostring()

		image_data2 = cv2.imencode('.jpg', frame)[1]

		image_data3 = cv2.imdecode(image_data2, cv2.IMREAD_COLOR)

#		cv2.imshow("imgNet_tf", image_data3)

#		cv2.waitKey(1)

		cv2.putText(image_data3, self.human_string +": "+ str(int(self.score*10000))+"%", (10, 30), 0, 1, (0, 8, 255), 2)

		msg = self.bridge.cv2_to_imgmsg(image_data3, "bgr8")

		self._pub1.publish(msg)

		rospy.sleep(0.9)

		softmax_tensor = self._session.graph.get_tensor_by_name('softmax:0')

		predictions = self._session.run(
			softmax_tensor, {'DecodeJpeg/contents:0': image_data})

		predictions = np.squeeze(predictions)

		# Creates node ID --> English string lookup.
		node_lookup = classify_image.NodeLookup()

		top_k = predictions.argsort()[-self.use_top_k:][::-1]


		for node_id in top_k:
			self.human_string = node_lookup.id_to_string(node_id)
			self.score = predictions[node_id]

			if self.score > self.score_threshold:

				rospy.logdebug('%s (score = %.5f)' % (self.human_string, self.score))

				self._pub.publish(self.human_string)

				#topic_diag_main("ros_tf_node_imgNet", "", human_string, (float(score)) * 100)
				data={}
				data['class'] = self.human_string
				data['probablity'] = (float(self.score)) * 100
				self.monitor.node_info(data, 'Result', 'Detection result received.', 0)

				data3={}
				data3['Status'] = 1
				self.monitor.node_info(data3, 'Status', 'Detection status received.', 0)


	def main(self):
		rospy.spin()

if __name__ == '__main__':
	tf_model_download
	tensor = RosTensorFlow()
	tensor.main()
