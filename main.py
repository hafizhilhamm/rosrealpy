import rospy
import cv2
import pyrealsense2 as rs
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

class ObjectDetectionNode:
    def __init__(self):
        rospy.init_node("object_detection_node")

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=50)
        self.distance_pub = rospy.Publisher("/object_distances", String, queue_size=50)

        self.template = cv2.imread("/home/hafizh/Pictures/pic9.png")
        self.template = cv2.resize(self.template, (640,480))

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.max_distance = 10000

    def run(self):
        self.pipeline.start(self.config)

        try:
            while not rospy.is_shutdown():
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_image[depth_image > self.max_distance] = 0

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )

                hsving = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)

                graying = cv2.cvtColor(hsving,cv2.COLOR_BGR2GRAY)

                lower_white = np.array([96 , 29 , 175], dtype=np.uint8)
                upper_white = np.array([118 , 50 , 245], dtype=np.uint8)

                mask = cv2.inRange(hsving , lower_white, upper_white)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                center_x = 0
                center_y = 0
                mid = 0
                center_goal_1 = 0
                center_goal_2 = 0
                in_1 = 0
                distance = 0
                distancekiri = 0
                distancekanan = 0

                for i, contour in enumerate(contours):
                  area = cv2.contourArea(contour)
                  cv2.drawContours(mask, contours, i, (0, 0, 255), 2)
                  moments = cv2.moments(contour)
                  #print("Area of object {}: {:.2f} pixels".format(i+1, area))
                  if (moments['m00'] != 0 and moments['m01'] != 0 and 4000 > area > 500 ):
                   center_x = int(moments['m10'] / moments['m00'])
                   center_y = int(moments['m01'] / moments['m00'])
                   #print(center_x,center_y)
                   distance = depth_frame.get_distance(center_x, center_y)
                   cv2.circle(mask, (center_x,center_y), 8, (0,255,0), cv2.FILLED)
                  if(in_1 == 0 and center_x !=0):
                   center_goal_1 = center_x
                   in_1 = 1
                  elif(in_1 == 1 and center_x !=0 ):
                   center_goal_2 = center_x
                   mid = (center_goal_1+center_goal_2)/2
                  if(center_x > mid):
                    distancekanan = distance
                    rospy.loginfo("Gawang kanan : {:.3f} Meters".format(distancekanan)) 
                  if(center_x < mid):
                    distancekiri = distance
                    rospy.loginfo("Gawang kiri : {:.3f} Meters".format(distancekiri)) 

                
                self.distance_pub.publish(format(distance))

                cv2.imshow("Result", color_image)
                cv2.imshow("Mask", mask)

                self.image_pub.publish(self.bridge.cv2_to_imgmsg(color_image, "bgr8"))

                key = cv2.waitKey(1)
                if key == 27:
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    node = ObjectDetectionNode()
    node.run()

