#!/usr/bin/env python3
import sys
import os
import rospy
import yaml
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Point
from duckietown.dtros import DTROS, NodeType
from duckiebot_msgs.msg import DetectedObstacle, DetectedObject

from image_processing.ground_projection_geometry import GroundProjectionGeometry, Point as GPPoint
from image_processing.rectification import Rectify
from image_processing.projection_utils import load_extrinsics, pixel_msg_to_ground_msg

# Find the path to YOLOv5 and weights based on the current file's location

# object_detection/src
object_detection_src_path = os.path.dirname(os.path.abspath(__file__)) 

# object_detection
object_detection_path = os.path.dirname(object_detection_src_path)

# object_detection/yolov5
yolov5_path = os.path.join(object_detection_path, 'yolov5')

# object_detection/config/model_weights/yolov5n.pt
weight_path = os.path.join(object_detection_path, 'config', 'model_weights', 'yolov5s_duckie.pt')

# Add YOLOv5 path to sys.path so that torch.hub can find it
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # DEBUG: Check if GPU is available
        rospy.loginfo(f"YOLOv5 v6.1 yükleniyor... GPU Durumu: {torch.cuda.is_available()}")
        
        try:
            # Load the YOLOv5 model locally with custom weights
            self.model = torch.hub.load(
                yolov5_path, 
                'custom', 
                path=weight_path, 
                source='local',
                force_reload=False,
                trust_repo=True
            )
        except Exception as e:
            # Log the error and shutdown the node
            node_name = rospy.get_name()
            rospy.logerr(f"CRITICAL: Error loading YOLOv5 model, check paths:\n Error: {e.with_traceback()}")
            rospy.signal_shutdown(f"{node_name} failed. Model could not be loaded. Shutting down the node.")

        # GPU or CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # DEBUG: Log the device being used
        rospy.loginfo(f"Device used: {self.device}")
        
        # Inference Parameters (need tuning)
        self.model.conf = 0.5  # Confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        
        self.bridge = CvBridge()
        
        # Ground projection setup
        self.ground_projector = None
        self.rectifier = None
        self.homography = load_extrinsics()
        self.camera_info_received = False
        
        # Subscriber for camera info
        self.sub_camera_info = rospy.Subscriber("~camera_info", CameraInfo, self.cb_camera_info, queue_size=1)
        
        # Subscriber for compressed image
        self.sub_image = rospy.Subscriber(
            "~image/compressed", 
            CompressedImage, 
            self.cb_image, 
            queue_size=1,
            buff_size=2**24 
        )
        
        # Publisher for debug image
        self.pub_debug = rospy.Publisher(
            "~debug/image/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        # Publisher for detected objects
        self.pub_obstacle = rospy.Publisher(
            "~detected_obstacle", 
            DetectedObstacle, 
            queue_size=1
        )

    def cb_camera_info(self, msg):
        if not self.camera_info_received:
            self.rectifier = Rectify(msg)
            self.ground_projector = GroundProjectionGeometry(
                im_width=msg.width, im_height=msg.height, homography=np.array(self.homography).reshape((3, 3))
            )
        self.camera_info_received = True

    def cb_image(self, msg):
        try:
            # Change the format of the compressed image data to a numpy array and decode it using OpenCV
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # DEBUG: Check if the image is empty
            if image is None:
                rospy.logwarn("Empty image received!")
                return

            # YoLOv5 expects RGB format, OpenCV uses BGR by default
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Inference with YOLOv5
            results = self.model(img_rgb)
            
            # Get detections
            detections = results.pandas().xyxy[0]
            
            # DEBUG: Log detections
            if not detections.empty:
                for _, det in detections.iterrows():
                    rospy.loginfo(f"Detection: name={det['name']}, conf={det['confidence']}, xmin={det['xmin']}, xmax={det['xmax']}, ymin={det['ymin']}, ymax={det['ymax']}")
            
            # Detect all obstacles in frame
            detected_objects = []
            
            if self.camera_info_received:
                for _, det in detections.iterrows():
                    # Filter: Only consider 'duckie' class detections
                    if det['name'] == 'duckie':
                        
                        # Compute the center bottom point of the bounding box
                        x_center = (det['xmin'] + det['xmax']) / 2
                        y_bottom = det['ymax'] 

                        # Normalize
                        x_norm = x_center / image.shape[1]
                        y_norm = y_bottom / image.shape[0]
                        
                        point_msg = Point(x_norm, y_norm, 0)
                        ground_point = pixel_msg_to_ground_msg(point_msg, self.ground_projector, self.rectifier)
                        
                        # Check if the detected duckie is within a reasonable area in front of the robot
                        if ground_point.x > 0 and ground_point.x < 0.5 and abs(ground_point.y) < 0.15:
                            
                            distance = ground_point.x
                            
                            # Create a DetectedObject for this duckie
                            detected_object = DetectedObject()
                            detected_object.object_type = "duckie"
                            detected_object.distance = distance
                            detected_object.position = ground_point
                            detected_object.confidence = float(det['confidence'])
                            detected_objects.append(detected_object)
                            
                            rospy.loginfo(f"Duckie detected: distance={distance:.3f}m, confidence={det['confidence']:.2f}")
                        else:
                            # DEBUG: Log out-of-bounds detections
                            rospy.logdebug(f"Duckie ignored (out of bounds): x={ground_point.x:.2f}, y={ground_point.y:.2f}")
            
            # Sort by distance (closest first)
            detected_objects.sort(key=lambda obj: obj.distance)
            
            # DEBUG: Log detection results
            rospy.loginfo(f"Obstacles detected: {len(detected_objects)}")
            
            # Publish obstacle message
            obstacle_msg = DetectedObstacle()
            obstacle_msg.detected = len(detected_objects) > 0
            obstacle_msg.objects = detected_objects
            self.pub_obstacle.publish(obstacle_msg)
            
            # Annotate detections on the image for debugging
            results.render() 
            annotated_img_rgb = results.imgs[0]
            
            # Convert back to BGR for OpenCV before publishing
            annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
            
            # Publish the annotated image
            out_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img_bgr)
            self.pub_debug.publish(out_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in object detection loop: {e}")


if __name__ == '__main__':
    node = ObjectDetectionNode(node_name='object_detection_node')
    rospy.spin()

