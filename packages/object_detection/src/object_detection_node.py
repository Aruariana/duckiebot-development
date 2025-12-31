#!/usr/bin/env python3
import sys
import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Point
from duckietown.dtros import DTROS, NodeType
from duckiebot_msgs.msg import DuckieObstacle

# Artık torch ve yolov5 modülleri import edilebilir
import torch

# Ground projection imports
from image_processing.ground_projection_geometry import GroundProjectionGeometry, Point as GPPoint
from image_processing.rectification import Rectify
import yaml

# ---------------------------------------------------------
# 1. PATH AYARLAMASI (Dinamik Yol Bulma)
# ---------------------------------------------------------
# Bu dosya 'src' klasörünün içinde.
# Yapı: object_detection/src/bu_dosya.py
script_dir = os.path.dirname(os.path.abspath(__file__)) 

# Bir üst klasöre (paket kök dizinine 'object_detection') çıkıyoruz
package_root = os.path.dirname(script_dir)

# 1. YOLOv5 kütüphanesinin yolu: object_detection/yolov5
yolov5_path = os.path.join(package_root, 'yolov5')

# 2. Ağırlık dosyasının yolu: object_detection/config/model_weights/yolov5n.pt
weight_path = os.path.join(package_root, 'config', 'model_weights', 'yolov5s_duckie.pt')

# Python'ın YOLOv5 modüllerini bulabilmesi için path'e ekliyoruz
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        rospy.loginfo(f"Paket Kök Dizini: {package_root}")
        rospy.loginfo(f"YOLO Path: {yolov5_path}")
        rospy.loginfo(f"Weights Path: {weight_path}")

        # ---------------------------------------------------------
        # 2. MODEL YÜKLEME
        # ---------------------------------------------------------
        rospy.loginfo(f"YOLOv5 v6.1 yükleniyor... GPU Durumu: {torch.cuda.is_available()}")
        
        try:
            # source='local' -> Github'a gitme, gösterdiğim 'yolov5_path' klasörünü kullan.
            # path -> Ağırlık dosyasının tam yolu
            self.model = torch.hub.load(
                yolov5_path, 
                'custom', 
                path=weight_path, 
                source='local', 
                force_reload=True
            )
        except Exception as e:
            rospy.logerr(f"CRITICAL: Model yüklenirken hata oluştu! Pathleri kontrol edin.\nHata: {e}")
            # Hata durumunda node'u kapatmak güvenlidir
            sys.exit(1)

        # GPU (CUDA) Ayarı
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        rospy.loginfo(f"Kullanılan cihaz: {self.device}")
        
        # Inference Parametreleri
        self.model.conf = 0.5  # Güven eşiği (Minimum %50 emin olmalı)
        self.model.iou = 0.45  # NMS IoU eşiği
        
        self.bridge = CvBridge()
        
        # Ground projection setup
        self.ground_projector = None
        self.rectifier = None
        self.homography = self.load_extrinsics()
        self.camera_info_received = False
        
        # Subscriber for camera info
        self.sub_camera_info = rospy.Subscriber("~camera_info", CameraInfo, self.cb_camera_info, queue_size=1)
        
        # Subscriber (Robot'tan gelen görüntü)
        self.sub_image = rospy.Subscriber(
            "~image/compressed", 
            CompressedImage, 
            self.cb_image, 
            queue_size=1,
            buff_size=2**24 
        )
        
        # Publisher (Sonuç görüntüsü)
        self.pub_debug = rospy.Publisher(
            "~debug/image/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        # Publisher for duckie obstacle
        self.pub_obstacle = rospy.Publisher(
            "/duckie_obstacle", 
            DuckieObstacle, 
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
            # 1. ROS Mesajını OpenCV formatına çevir
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                rospy.logwarn("Boş görüntü alındı!")
                return

            # 2. Renk dönüşümü (ROS/OpenCV BGR kullanır, YOLO RGB ister)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 3. Inference (Tahmin)
            # size=640 varsayılan değerdir, hız için 320 veya 416 yapılabilir
            results = self.model(img_rgb)
            
            # Get detections
            detections = results.pandas().xyxy[0]  # pandas dataframe
            
            # Log detections for debugging
            if not detections.empty:
                for _, det in detections.iterrows():
                    rospy.loginfo(f"Detection: name={det['name']}, conf={det['confidence']}, xmin={det['xmin']}, xmax={det['xmax']}, ymin={det['ymin']}, ymax={det['ymax']}")
            
            # Check for duckie obstacles
            obstacle_detected = False
            min_distance = float('inf')
            obstacle_position = Point()
            
            if self.camera_info_received:
                for _, det in detections.iterrows():
                    # Filter: Only trust detections with high confidence
                    if det['name'] == 'duckie' and det['confidence'] > 0.5:
                        
                        x_center = (det['xmin'] + det['xmax']) / 2
                        y_bottom = det['ymax'] 

                        # Normalize
                        x_norm = x_center / image.shape[1]
                        y_norm = y_bottom / image.shape[0]
                        
                        point_msg = Point(x_norm, y_norm, 0)
                        ground_point = self.pixel_msg_to_ground_msg(point_msg)
                        
                        # --- LOGIC FIX START ---
                        # In ROS/Duckietown: X is Forward, Y is Left/Right
                        
                        # 1. Is it in front of me? (x > 0)
                        # 2. Is it close enough to matter? (x < 1.2 meters)
                        # 3. Is it in my lane? (abs(y) < 0.20 meters approx lane half-width)
                        if ground_point.x > 0 and ground_point.x < 0.5 and abs(ground_point.y) < 0.15:
                            
                            # Distance is the X axis (forward)
                            distance = ground_point.x 
                            
                            if distance < min_distance:
                                min_distance = distance
                                obstacle_position = ground_point
                                obstacle_detected = True
                        else:
                            rospy.logdebug(f"Duckie ignored (out of bounds): x={ground_point.x:.2f}, y={ground_point.y:.2f}")
                        # --- LOGIC FIX END ---
            
            # Log detection status
            rospy.loginfo(f"Obstacle detected: {obstacle_detected}, distance: {min_distance if obstacle_detected else 0}")
            
            # Publish obstacle message
            obstacle_msg = DuckieObstacle()
            obstacle_msg.detected = obstacle_detected
            if obstacle_detected:
                obstacle_msg.distance = min_distance
                obstacle_msg.position = obstacle_position
            else:
                obstacle_msg.distance = 0.0
                obstacle_msg.position = Point(0,0,0)
            self.pub_obstacle.publish(obstacle_msg)
            
            # 4. Çizim (Rendering)
            # Orijinal imajın üzerine kutuları çizer
            results.render() 
            
            # Sonuç, results.imgs[0] içinde RGB formatında durur
            annotated_img_rgb = results.imgs[0]
            
            # 5. Tekrar BGR'a çevir (Geri yayınlamak için)
            annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
            
            # 6. ROS Mesajına çevir ve yayınla
            out_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_img_bgr)
            self.pub_debug.publish(out_msg)
            
        except Exception as e:
            rospy.logerr(f"Tespit döngüsünde hata: {e}")

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        """
        cali_file_folder = "/data/config/calibrations/camera_extrinsic/"
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        if not os.path.isfile(cali_file):
            self.log(f"Can't find calibration file: {cali_file}. Using default calibration instead.", "warn")
            cali_file = os.path.join(cali_file_folder, "default.yaml")

        if not os.path.isfile(cali_file):
            msg = "Found no calibration file ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file, "r") as stream:
                calib_data = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError:
            msg = f"Error in parsing calibration file {cali_file} ... aborting"
            self.logerr(msg)
            rospy.signal_shutdown(msg)

        return calib_data["homography"]

    def pixel_msg_to_ground_msg(self, point_msg):
        """
        Projects a normalized point to ground coordinates.
        """
        # 1. Normalized coordinates to absolute pixel (e.g. 0.5 -> 320)
        norm_pt = GPPoint.from_message(point_msg)
        pixel = self.ground_projector.vector2pixel(norm_pt)
        
        # 2. Rectify (undistort) the pixel
        # This returns absolute pixels (e.g. 320 -> 321 due to distortion)
        rect = self.rectifier.rectify_point(pixel)
        rect_pt = GPPoint.from_message(rect)
        
        # 3. Project on ground
        # ERROR WAS HERE: Do NOT normalize (divide by width/height) again.
        # The Homography matrix expects absolute pixel coordinates.
        ground_pt = self.ground_projector.pixel2ground(rect_pt)
        
        # 4. Point to message
        ground_pt_msg = Point()
        ground_pt_msg.x = ground_pt.x
        ground_pt_msg.y = ground_pt.y
        ground_pt_msg.z = ground_pt.z

        return ground_pt_msg

if __name__ == '__main__':
    # Node başlatma
    node = ObjectDetectionNode(node_name='object_detection_node')
    rospy.spin()

