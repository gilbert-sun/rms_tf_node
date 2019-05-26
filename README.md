
# Tensorflow Objtect recognization by image_net at ROS Nvidia Tx2
This is the tensorflow slim model for Object detection

Ret1):  Class 代號: coffee mup,kb, ms

Ret2):  Class 百分比: class confident percentage

rms_tf_node recognization ....
==============================

- Install TensorFlow (see [tensor flow install guide](https://www.tensorflow.org/install/install_linux))
- Install ROS (see http://wiki.ros.org)
- InstallROSTX2 (see https://github.com/jetsonhacks/installROSTX2)
    - updateRepositories.sh
    - installROS.sh
    - setupCatkinWorkspace.sh
- Install cv-bridge

```bash
$ sudo apt-get install ros-kinetic-cv-bridge ros-kinetic-opencv3
```

- (Optional1) Install camera driver (for example, cv_camera , cv_camera_node)

```bash
$ sudo apt-get install ros-kinetic-cv-camera
```

- (Optional2) Install UVC Camera for ROS
```bash
$ sudo apt-get install ros-kinetic-libuvc-camera
$ sudo apt-get install ros-kinetic-image-pipeline
$ rosdep update
```

- (Optional3) Create udev rule
    - /etc/udev/rules.d/99-uvc.rules
```bash
$ SUBSYSTEM=="usb", ATTR{idVendor}=="046d", MODE="0666"
```

- (Optional4) Test Camera
```bash
$ rosrun libuvc_camera camera_node
$ rosrun image_view image_view image:=image_raw
```

TensorFlow install note (without GPU)
-------------------------------------------
Please read official guide. This is a only note for me.

```bash
$ sudo apt-get install python-pip python-dev python-virtualenv
$ virtualenv --system-site-packages ~/tensorflow
$ source ~/tensorflow/bin/activate
$ pip install --upgrade tensorflow
```


TensorFlow install note (with GPU)
-------------------------------------------
Please read official guide. This is a only note for me.

```bash
$ sudo apt-get install python-pip python-dev python-virtualenv
$ virtualenv --system-site-packages ~/tensorflow
$ source ~/tensorflow/bin/activate
$ pip install --upgrade tensorflow-1.4.1-cp27-cp27mu-linux_aarch64.whl
$ https://github.com/peterlee0127/tensorflow-nvJetson
$ https://github.com/jetsonhacks/installTensorFlowTX2
```


img_recognization.py
--------------------
* subscribe: /CVsub  (sensor_msgs/Image)
* publish:   /CVpub  (sensor_msgs/Image)
* publish1:  /result (std_msgs/String)



# Environmental requirements
Opencv 3.2 or laster

Tensorflow 1.4.1


# Start up
How to try

```
$ cd /root/catkin_ws/
$ catkin_make 
$ source /root/catkin_ws/devel/setup.bash
$ roscore
$ roslaunch usb_cam usb_cam-test.launch
$ rosrun rms_tf_node listen.py image:=/usb_cam/image_raw
```

# Log
```
cat result.log
```

# Function Input & Output

See img_recognization.py



```
#Program execution
python img_recognization.py image:=image_raw

#function output
(Class name,%百分比 ) == (Keyboard, 0.8253038)
```


# Demo

![Total Ros Structure]( ./src/pic/1.png)
