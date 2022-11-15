import rospy
import math
import cv2

from image_processing.image_parser import image_data_calc
from radar_processing.pcl_parser import pcl_data_calc
from detector import Yolov5Detector
from dataclasses import dataclass

@dataclass
class fusion_object:
    index: int = None
    distance: int = None
    

class fusion(image_data_calc, pcl_data_calc, Yolov5Detector):
    def __init__(self):
        self.fusion_index_list = []
        self.fusion_distance_list = []
        self.best_candidate = []
        self.distance_weight = 0.5
        self.azimuth_weight = 0.5
        self.euclidean_threshold = 0.1
        self.camera_weight_min = 0.2
        self.camera_weight_max = 0.4
        self.radar_weight = 0.7
        self.my_speed = 0
        
        self.visualize()
        
        
    def euclidean_distance(self, camera_index, radar_index):
        
        camera_r = self.camera_object_list[camera_index].distance
        camera_theta = self.camera_object_list[camera_index].azimuth
        
        radar_r = self.radar_object_list[radar_index].distance
        radar_theta = self.radar_object_list[radar_index].azimuth
        
        camera_coord = (camera_r * math.sin(camera_theta), camera_r * math.cos(camera_theta))
        radar_coord = (radar_r * math.sin(radar_theta), radar_r * math.cos(radar_theta))
        
        result = math.sqrt((camera_coord[0] - radar_coord[0])**2 + (camera_coord[1] - radar_coord[1])**2)
        
        return result
    
        
    def matching(self):
        camera_object_list = self.camera_object_list
        radar_object_list = self.radar_object_list
        
        first_best_candidate = []
        second_best_candidate = []
        
        # 1st for loop
        for camera_object in camera_object_list:
    
            cost_val_list = []
            
            for radar_object in radar_object_list:
                cost_val = self.distance_weight * abs(camera_object.distance - radar_object.distance)\
                            + self.azimuth_weight * abs(camera_object.azimuth - radar_object.azimuth)
                
                cost_val_list.append(cost_val)
            
            min_radar_idx = cost_val_list.index(min(cost_val_list))
            
            first_best_candidate.append((camera_object.index, min_radar_idx))
            
        # 2nd for loop
        for radar_object in radar_object_list:
            
            cost_val_list = []
            
            for camera_object in camera_object_list:
                cost_val = self.distance_weight * abs(camera_object.distance - radar_object.distance)\
                            + self.azimuth_weight * abs(camera_object.azimuth - radar_object.azimuth)
            
                cost_val_list.append(cost_val)
            
            min_camera_idx = cost_val_list.index(min(cost_val_list))
            
            second_best_candidate.append((min_camera_idx, radar_object.index))
        
        
        first_set = set(first_best_candidate)
        second_set = set(second_best_candidate)
        
        self.best_candidate = list(first_set.intersection(second_set))
        
        # Removing uncertain candidates (using euclidean distance) : Should be revised!!!!!
        for candidate in self.best_candidate:
            if self.euclidean_distance(candidate[0], candidate[1]) > self.euclidean_threshold:
                self.best_candidate.remove(candidate)
        
        # Call fusion function
        self.fusion()
        
        
    def fusion(self):
        
        for candidate in self.best_candidate:
            conf = self.camera_object_list[candidate[0]].confidence
            camera_weight = self.camera_weight_min + ((conf - self.conf_threshold) / (1 - self.conf_threshold)) * (self.camera_weight_max - self.camera_weight_min)
            radar_weight = 1 - camera_weight
            
            camera_idx = self.camera_object_list[candidate[0]].index
            radar_idx = self.radar_object_list[candidate[1]].index
            fusion_distance = camera_weight * self.camera_object_list[candidate[0]].distance + radar_weight * self.radar_object_list[candidate[1]].distance
            
            self.fusion_index_list.append((camera_idx, radar_idx))
            self.fusion_distance_list.append(fusion_distance)
    
    
    def Risk_calculate(self, distance, velocity):
        
        crash_time = 1000*distance / (3600*((1-math.sin(85*math.pi/180))*self.my_speed + velocity))
        
        lane_change_time = self.lane_width / (self.my_speed * math.cos(85*math.pi/180))
        
        # Ok to change lane
        if crash_time - lane_change_time > 3:
            return 0
        
        # Warning
        elif crash_time - lane_change_time <= 3 and crash_time - lane_change_time > 2:
            return 1
        
        # Dangerous
        else:
            return 2
    
    
    def visualize(self):
        
        self.matching()
        
        ret, frames = self.image.read()
        if not ret:
            print("None")
        
        closest_camera_idx = self.fusion_index_list[self.fusion_distance_list.index(min(self.fusion_distance_list))][0]
        closest_radar_idx = self.fusion_index_list[self.fusion_distance_list.index(min(self.fusion_distance_list))][1]
        
        closest_distance = min(self.fusion_distance_list)
        closest_velocity = self.radar_object_list[closest_radar_idx].velocity
        
        xmin, ymin, xmax, ymax = self.bounding_box_list[closest_camera_idx]
        
        if self.Risk_calculate(closest_distance, closest_velocity) == 0:
            cv2.rectangle(frames, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5, 1)
        
        if self.Risk_calculate(closest_distance, closest_velocity) == 1:
            cv2.rectangle(frames, (xmin, ymin), (xmax, ymax), (0,130,255), 5, 1)
            cv2.rectangle(frames, (0, 0), (self.image.shape[1], self.image.shape[0]), (0,130,255), 50, 1)
            
        elif self.Risk_calculate(closest_distance, closest_velocity) == 2:
            cv2.rectangle(frames, (xmin, ymin), (xmax, ymax), (0,0,255), 5, 1)
            cv2.rectangle(frames, (0, 0), (self.image.shape[1], self.image.shape[0]), (0,0,255), 50, 1)
        
        
        cv2.namedWindow("Display", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cv2.imshow("Display", frames)
        

# Main function
if __name__ == '__main__':
    fusion()
            
