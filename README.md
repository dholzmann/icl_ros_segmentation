## example of running the code on shared memory inputs, and uses corresponding camera calibration files.

```rosrun icl_ros_segmentation  icl_ros_segmentation_node -d /vol/famula/stable/etc/pa10/vision/camera/kinect-depth-qvga.xml -c /vol/famula/stable/etc/pa10/vision/camera/dc-qvga.xml -so /vol/famula/stable/etc/pa10/vision/segmenter.xml -file sm depth color -s 320x240 -du mm -pc robotselffilter.xml```
