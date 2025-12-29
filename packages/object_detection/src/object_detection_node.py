#!/usr/bin/env python3
import sys
import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from duckietown.dtros import DTROS, NodeType

# Artık torch ve yolov5 modülleri import edilebilir
import torch

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

if __name__ == '__main__':
    # Node başlatma
    node = ObjectDetectionNode(node_name='object_detection_node')
    rospy.spin()