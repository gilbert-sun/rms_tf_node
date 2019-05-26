import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf


import diagnostic_msgs.msg
from enum import Enum
from rostopic import ROSTopicHz
import random
#from node_function_monitor.msg import BoundingBox

from node_monitor import topic_diagnostics,node_result,DataInfo

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

def topic_diag_main(ndname,topname  ,clname, probablity):
	# initialize node
	nodename = ndname if ndname != "" else "listener"

	# rospy.init_node(nodename, anonymous=True)

	# state the topic you want to subscribe to for monitoring
	topicname = topname if topname != "" else "/chatter"

	# the topic's message type
	# (if all your topic's message type are the same, you may declare only one topictype)
	topictype = String

	# describe a minimum publish rate per topic. If its the same for all topics, one minimum rate is enough
	min_rate = 2000

	# create a ROSTopicHz instance per topic you wish to get the publish rate from.
	rt = ROSTopicHz(-1, None)

	# subscribe to all the topics and run data_callback to check if the data messages are empty or not
	rospy.Subscriber(topicname, topictype, data_callback)

	# subscribe once again to get the rate of publishing messages by calling rt.callback_hz everytime a message is published
	rospy.Subscriber(topicname, topictype, rt.callback_hz)

	# we will be continually publishing to diagnostics topic
	# (first one is for monitoring topics that we are subscribed to)
	# (second one is for sending results)
	pub = rospy.Publisher('diagnostics', diagnostic_msgs.msg.DiagnosticArray, queue_size=10)
	pub2 = rospy.Publisher('diagnostics1', diagnostic_msgs.msg.DiagnosticArray, queue_size=10)

	# rate at which the diagnostic messages will be published
	rate = rospy.Rate(1)

	# example : object recognition will recognize one of these objects and store it in data
	data = BoundingBox()
	# data = 0#BoundingBox()
	# object = ['dog', 'cat', 'truck', 'person', 'tree', 'cup', 'table', 'chair', 'beer', 'chicken']
	# r = random.randint(0, 9)
	data.Class = clname #object[r]
	data.probability = probablity #random.uniform(0, 100)
	print ("---------123--------",data.Class ,data.probability)

	topic_diagnostics(topicname, nodename, rt, pub, topictype, min_rate)
	node_result(nodename, pub2, data)


class RosTensorFlow():
    def __init__(self):
        self._session = tf.Session()

        classify_image.create_graph()

        self.bridge = CvBridge()

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)

        self._pub = rospy.Publisher('result', String, queue_size=1)

        self._pub1 = rospy.Publisher('Imgnet_Tf', Image, queue_size=1)

        self.score_threshold = rospy.get_param('~score_threshold', 0.1)

        self.use_top_k = rospy.get_param('~use_top_k', 5)

    def callback(self, image_msg):

        frame= self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        image_data = cv2.imencode('.jpg', frame)[1].tostring()

        image_data2 = cv2.imencode('.jpg', frame)[1]

        image_data3 = cv2.imdecode(image_data2, cv2.IMREAD_COLOR)

        cv2.imshow("imgNet_tf", image_data3)

        cv2.waitKey(1)

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

                topic_diag_main("ros_tf_node_imgNet", "", human_string, (float(score)) * 100)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    tf_model_download
    rospy.init_node('rms_tf_node')
    tensor = RosTensorFlow()
    tensor.main()