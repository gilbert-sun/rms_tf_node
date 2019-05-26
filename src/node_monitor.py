#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import diagnostic_msgs.msg
from enum import Enum
from rostopic import ROSTopicHz
import random


# from node_function_monitor.msg import BoundingBox

msg_data = ''
content = ''


# Enum for declaring a set of constants
class DataInfo(Enum):
    # Describe a small message content
    msg_wrong = "Not receiving message from topic"
    msg_correct = "Receiving messages"
    rate_normal = "Publish rate is normal"
    rate_low = "Publish rate is too low"
    rate_none = "Publish rate is at 0"


# monitor and display diagnostics messages on the console
def topic_diagnostics(topicname, nodename, rt, pub, topictype, min_rate):
    # declare variables
    global msg_data
    global content
    rate = ()

    # check if topics are being published to
    if topicname not in dict(rospy.get_published_topics()).keys():
        msg_data = DataInfo.msg_wrong.value

    # get the publish rate of the topic
    rate = rt.get_hz()

    # check the publish rate
    if rate != None:
        if rate[0] < min_rate:
            # rospy.loginfo("Warn: Rate is too low")
            rate_data = DataInfo.rate_low.value
        elif rate[0] > min_rate:
            # rospy.loginfo("OK: Rate is normal")
            rate_data = DataInfo.rate_normal.value
    else:
        # rospy.loginfo("Error: Message rate is 0")
        rate_data = DataInfo.rate_none.value

    # prepare the diagnostic array
    diagnosticArray = diagnostic_msgs.msg.DiagnosticArray()
    diagnosticArray.header.stamp = rospy.get_rostime()

    # prepare diagnostics message
    # and determine level, node state, node topics, and issues
    statusMsg = diagnostic_msgs.msg.DiagnosticStatus()
    statusMsg.name = nodename + ' -> ' + topicname
    if msg_data == DataInfo.msg_correct.value:
        if rate_data == DataInfo.rate_normal.value:
            statusMsg.message = DataInfo.msg_correct.value
            statusMsg.level = 0
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Node Name", nodename)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Topic Name", topicname)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Minimum Rate", str(min_rate))
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Current Rate", str(rate[0]))
            )
        elif rate_data == DataInfo.rate_low.value:
            statusMsg.message = DataInfo.rate_low.value
            statusMsg.level = 1
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Node Name", nodename)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Topic Name", topicname)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Minimum Rate", str(min_rate))
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Current Rate", str(rate[0]))
            )
        elif rate_data == DataInfo.rate_none.value:
            statusMsg.message = DataInfo.rate_none.value
            statusMsg.level = 2
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Node Name", nodename)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Topic Name", topicname)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Minimum Rate", str(min_rate))
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Current Rate", "0")
            )
    elif msg_data == DataInfo.msg_wrong.value:
        # if messages are not received, it means publish rate is 0
        if rate_data == DataInfo.rate_none.value:
            statusMsg.message = DataInfo.rate_none.value
            statusMsg.level = 2
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Node Name", nodename)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Topic Name", topicname)
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Minimum Rate", str(min_rate))
            )
            statusMsg.values.append(
                diagnostic_msgs.msg.KeyValue("Current Rate", "0")
            )

    # append message to diagnostic array
    diagnosticArray.status.append(statusMsg)

    # publish array
    pub.publish(diagnosticArray)


# function to send result to be viewed on the console
def node_result(nodename, pub, data):
    # prepare the diagnostic array
    diagnosticArray = diagnostic_msgs.msg.DiagnosticArray()
    diagnosticArray.header.stamp = rospy.get_rostime()

    # prepare diagnostics message
    # determine level, node name, object recognized and probability
    statusMsg = diagnostic_msgs.msg.DiagnosticStatus()
    statusMsg.name = nodename + " -> result"
    statusMsg.message = 'Detected ' + data.Class
    statusMsg.level = 0
    statusMsg.values.append(
        diagnostic_msgs.msg.KeyValue("Node name", nodename)
    )
    statusMsg.values.append(
        diagnostic_msgs.msg.KeyValue("Object Recognized", str(data.Class))
    )
    statusMsg.values.append(
        diagnostic_msgs.msg.KeyValue("Object Probability", str(data.probability))
    )

    # append message to diagnostic array
    diagnosticArray.status.append(statusMsg)

    # publish array
    pub.publish(diagnosticArray)

