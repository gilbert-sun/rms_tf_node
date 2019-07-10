#!/usr/bin/env python


import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Int8
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
import json
import diagnostic_msgs.msg
from enum import Enum
from rostopic import ROSTopicHz
import random
import diagnostic_msgs.msg

#from node_function_monitor.node_monitor import node_monitor


from rms_telemetry.telemetry import telemetry
from rms_telemetry.constants import MsgLevel, MsgType

# global variables
TELEMETRY_CODE_NODE_STARTED = '100'
TELEMETRY_CODE_NODE_INITIALIZED = '101'
TELEMETRY_CODE_NODE_SHUTDOWN = '102'
TELEMETRY_CODE_NODE_STATUS = '103'
TELEMETRY_CODE_PROCESS_RESULT = '104'

TELEMETRY_CODE_ERROR_CLIENT_EXCEPTION = '400'
TELEMETRY_CODE_ERROR_SERVER_EXCEPTION = '500'


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def makeCNN(x, keep_prob):
    # --- define CNN model
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv


class RosTensorFlow():
    def __init__(self):
        self.node_name = 'rms_tf_node'

        rospy.init_node(self.node_name, anonymous=True)

        self.monitor =  telemetry(self.node_name) #node_monitor(self.node_name)

        self.monitor.node_info(MsgType.event.value, TELEMETRY_CODE_NODE_STARTED, MsgLevel.INFO.value)

        self._cv_bridge = CvBridge()

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
        self.keep_prob = tf.placeholder("float")
        self.y_conv = makeCNN(self.x, self.keep_prob)

        self._saver = tf.train.Saver()
        self._session = tf.InteractiveSession()

        init_op = tf.global_variables_initializer()
        self._session.run(init_op)

        self._saver.restore(self._session,'/rms_root/catkin_ws/src/rms_tf_node/src/model/model.ckpt')

        self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', Int16, queue_size=1)

        self._pub1 = rospy.Publisher('Imgnet_Tf', Image, queue_size=1)

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        ret, cv_image_binary = cv2.threshold(cv_image_gray, 128, 255, cv2.THRESH_BINARY_INV)
        cv_image_28 = cv2.resize(cv_image_binary, (28, 28))
        np_image = np.reshape(cv_image_28, (1, 28, 28, 1))
        predict_num = self._session.run(self.y_conv, feed_dict={self.x: np_image, self.keep_prob: 1.0})
        answer = np.argmax(predict_num, 1)
        rospy.loginfo('%d' % answer)
        self._pub.publish(answer)

        cv2.putText(cv_image, str(answer), (10, 30), 0, 1,
                    (0, 8, 255), 5)
        msg = self._cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")

        self._pub1.publish(msg)

        # topic_diag_main("ros_tf_node_imgNet", "", human_string, (float(score)) * 100)
        data = {}
        data['class'] = 1

        #self.monitor.node_info(data, 'Result', 'Detection result received.', 0)
        self.monitor.node_info(MsgType.status.value, TELEMETRY_CODE_NODE_STATUS, MsgLevel.INFO.value,  data )

        data3 = {}
        data3['Status'] = 1
        #self.monitor.node_info(data3, 'Status', 'Detection status received.', 0)

        #cv2.imshow("imgNet_tf", cv_image)

        #cv2.waitKey(1)

        rospy.sleep(10)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    # rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()


