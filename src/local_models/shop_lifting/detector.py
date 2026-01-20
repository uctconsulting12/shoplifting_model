"""
detector.py - Object Detection and Tracking
============================================
Contains YOLO detectors, DeepSORT tracker, container and item tracking
(No changes needed - original code works with enhanced system)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

#Production
from src.local_models.shop_lifting.config import (
    ContainerDetection, ItemDetection, ContainerType, ItemLocation,
    SystemConfig, logger
)

'''
#Local
from config import (
    ContainerDetection, ItemDetection, ContainerType, ItemLocation,
    SystemConfig, logger
)
'''

# =============================================================================
# CONTAINER TRACKER
# =============================================================================

class EnhancedContainerTracker:
    """Enhanced container detection with temporal tracking"""

    def __init__(self):
        self.container_tracker: Dict[int, ContainerDetection] = {}
        self.next_container_id = 0

    @staticmethod
    def _iou(b1, b2) -> float:
        """Calculate IoU between two bboxes"""
        x1, y1, x2, y2 = b1
        x3, y3, x4, y4 = b2
        ix1, iy1 = max(x1, x3), max(y1, y3)
        ix2, iy2 = min(x2, x4), min(y2, y4)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a1 = (x2 - x1) * (y2 - y1)
        a2 = (x4 - x3) * (y4 - y3)
        return inter / (a1 + a2 - inter + 1e-6)

    def detect_containers(self, detections: List, frame_num: int) -> List[ContainerDetection]:
        """Detect containers with temporal identity matching"""
        updated_containers: List[ContainerDetection] = []

        for det in detections:
            container_type = None

            if det.class_name in SystemConfig.CONTAINER_CLASSES:
                container_type = SystemConfig.CONTAINER_CLASSES[det.class_name]
            elif any(cart_term in det.class_name.lower()
                    for cart_term in SystemConfig.CART_INDICATORS):
                container_type = ContainerType.SHOPPING_CART

            if not container_type:
                continue

            bbox = det.bbox
            conf = det.confidence

            # Try match with existing container by IoU
            best_id = None
            best_iou = 0.0
            for cid, c in self.container_tracker.items():
                iou = self._iou(c.bbox, bbox)
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou
                    best_id = cid

            if best_id is not None:
                # Update existing container
                c = self.container_tracker[best_id]
                c.bbox = bbox
                c.last_seen_frame = frame_num
                c.confidence = conf
                updated_containers.append(c)
            else:
                # Create new container
                c = ContainerDetection(
                    container_id=self.next_container_id,
                    container_type=container_type,
                    first_seen_frame=frame_num,
                    last_seen_frame=frame_num,
                    bbox=bbox,
                    confidence=conf
                )
                self.container_tracker[self.next_container_id] = c
                updated_containers.append(c)
                self.next_container_id += 1

        return updated_containers

    def associate_containers_to_persons(self, containers: List[ContainerDetection],
                                       tracked_persons: Dict[int, List[float]],
                                       frame_num: int) -> Dict[int, List[ContainerDetection]]:
        """Associate containers with persons based on proximity"""
        associations = defaultdict(list)

        for container in containers:
            min_dist = float('inf')
            closest_person_id = None

            container_center_x = (container.bbox[0] + container.bbox[2]) / 2
            container_center_y = (container.bbox[1] + container.bbox[3]) / 2

            for person_id, person_bbox in tracked_persons.items():
                person_center_x = (person_bbox[0] + person_bbox[2]) / 2
                person_center_y = (person_bbox[1] + person_bbox[3]) / 2

                dist = np.sqrt((container_center_x - person_center_x)**2 +
                              (container_center_y - person_center_y)**2)

                is_within_bbox = (container.bbox[0] >= person_bbox[0] and
                                 container.bbox[2] <= person_bbox[2] and
                                 container.bbox[1] >= person_bbox[1] and
                                 container.bbox[3] <= person_bbox[3])

                max_distance = 200 if container.container_type == ContainerType.SHOPPING_CART else 150

                if dist < min_dist and (dist < max_distance or is_within_bbox):
                    min_dist = dist
                    closest_person_id = person_id

            if closest_person_id is not None:
                container.person_id = closest_person_id
                associations[closest_person_id].append(container)

        return associations

    def detect_container_opening(self, person_bbox: List[float], container_bbox: List[float],
                                gesture_confidence: float, container: ContainerDetection,
                                frame_num: int) -> bool:
        """Detect container opening with temporal consistency"""
        container_center_x = (container_bbox[0] + container_bbox[2]) / 2
        container_center_y = (container_bbox[1] + container_bbox[3]) / 2

        person_center_x = (person_bbox[0] + person_bbox[2]) / 2
        person_center_y = (person_bbox[1] + person_bbox[3]) / 2

        distance = np.sqrt((container_center_x - person_center_x)**2 +
                          (container_center_y - person_center_y)**2)

        person_height = person_bbox[3] - person_bbox[1]
        threshold = person_height * 0.4

        is_opening = distance < threshold and gesture_confidence > 0.5
        container.temporal_buffer.append(is_opening)

        if len(container.temporal_buffer) >= 5:
            recent_openings = list(container.temporal_buffer)[-5:]
            if sum(recent_openings) >= 3:
                return True

        return False


# =============================================================================
# ITEM TRACKER
# =============================================================================

class ItemTracker:
    """Track store items (products) and their interactions"""

    def __init__(self):
        self.tracked_items: Dict[int, ItemDetection] = {}
        self.next_item_id = 0

    def detect_items(self, detections: List, frame_num: int,
                     frame_height: Optional[int] = None) -> List[ItemDetection]:
        """Extract item detections from YOLO results"""
        items = []

        for det in detections:
            if det.class_name in SystemConfig.PRODUCT_CLASSES:
                item = ItemDetection(
                    item_id=self.next_item_id,
                    item_class=det.class_name,
                    first_seen_frame=frame_num,
                    last_seen_frame=frame_num,
                    current_location=ItemLocation.UNKNOWN,
                    bbox=det.bbox,
                    confidence=det.confidence
                )

                # Determine location (shelf vs hand)
                item.current_location = self._determine_item_location(det.bbox, frame_height)

                self.next_item_id += 1
                items.append(item)

        return items

    def _determine_item_location(self, bbox: List[float],
                                 frame_height: Optional[int]) -> ItemLocation:
        """Determine if item is on shelf or in hand (resolution-independent)"""
        item_center_y = (bbox[1] + bbox[3]) / 2

        if frame_height is None:
            return ItemLocation.UNKNOWN

        # Top 35% of frame = shelf zone (configurable)
        shelf_cutoff = SystemConfig.SHELF_CUTOFF_RATIO * frame_height

        if item_center_y < shelf_cutoff:
            return ItemLocation.SHELF
        else:
            return ItemLocation.UNKNOWN

    def associate_items_to_persons(self, items: List[ItemDetection],
                                   tracked_persons: Dict[int, List[float]]) -> Dict[int, List[ItemDetection]]:
        """Associate items with persons (items in hand)"""
        associations = defaultdict(list)

        for item in items:
            item_center_x = (item.bbox[0] + item.bbox[2]) / 2
            item_center_y = (item.bbox[1] + item.bbox[3]) / 2

            for person_id, person_bbox in tracked_persons.items():
                # Define hand region (lower 70% of person bbox)
                hand_region_y_min = person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.3
                hand_region_y_max = person_bbox[3]

                if (person_bbox[0] <= item_center_x <= person_bbox[2] and
                    hand_region_y_min <= item_center_y <= hand_region_y_max):

                    item.tracked_person_id = person_id
                    item.current_location = ItemLocation.HAND
                    associations[person_id].append(item)
                    break

        return associations


# =============================================================================
# PERSON DETECTOR AND TRACKER
# =============================================================================

class PersonDetectorTracker:
    """Handles person detection, pose estimation, and tracking"""

    def __init__(self, pose_model_path: str, object_model_path: str, device: str):
        self.device = device

        logger.info("Loading YOLO pose model...")
        self.pose_estimator = YOLO(pose_model_path)
        self.pose_estimator.to(device)

        logger.info("Loading YOLO object detector...")
        self.object_detector = YOLO(object_model_path)
        self.object_detector.to(device)

        logger.info("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3, nn_budget=100)

    def detect_persons_with_poses(self, frame: np.ndarray,
                                  config: Dict) -> Tuple[List, Dict[int, np.ndarray]]:
        """Detect persons with poses"""
        person_detections = []
        person_poses = {}

        try:
            results = self.pose_estimator(
                frame,
                conf=config['detection_confidence'],
                iou=config['nms_iou'],
                verbose=False
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()

                keypoints_data = None
                if results.keypoints is not None and len(results.keypoints.data) > 0:
                    keypoints_data = results.keypoints.data.cpu().numpy()

                for idx, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    class_name = results.names[int(cls)]

                    if class_name == 'person':
                        detection = type('Detection', (), {
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': box.tolist(),
                            'keypoints': keypoints_data[idx] if keypoints_data is not None else None
                        })()
                        person_detections.append(detection)

                        if keypoints_data is not None and idx < len(keypoints_data):
                            person_poses[idx] = keypoints_data[idx]

        except Exception as e:
            logger.error(f"Pose detection error: {e}")

        return person_detections, person_poses

    def detect_objects(self, frame: np.ndarray, config: Dict) -> List:
        """Detect objects (containers, items)"""
        detections = []

        try:
            results = self.object_detector(
                frame,
                conf=config['container_confidence'],
                iou=config['nms_iou'],
                verbose=False
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confidences, classes):
                    class_name = results.names[int(cls)]

                    detection = type('Detection', (), {
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': box.tolist()
                    })()
                    detections.append(detection)

        except Exception as e:
            logger.error(f"Object detection error: {e}")

        return detections

    def track_persons(self, frame: np.ndarray, detections: List) -> Dict[int, List[float]]:
        """Track persons using DeepSORT"""
        tracked_persons = {}

        try:
            person_detections = [
                ([det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]],
                 det.confidence, 'person')
                for det in detections if det.class_name == 'person'
            ]

            if person_detections:
                tracks = self.tracker.update_tracks(person_detections, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    tracked_persons[track.track_id] = track.to_ltrb().tolist()

        except Exception as e:
            logger.error(f"Tracking error: {e}")

        return tracked_persons

    def match_poses_to_tracks(self, tracked_persons: Dict[int, List[float]],
                             person_poses: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Match poses to tracked IDs using IoU"""
        matched_poses = {}

        for person_id, person_bbox in tracked_persons.items():
            best_match_idx = -1
            best_iou = 0.0

            for pose_idx, pose_keypoints in person_poses.items():
                visible_points = pose_keypoints[pose_keypoints[:, 2] > 0.3]
                if len(visible_points) == 0:
                    continue

                x_coords = visible_points[:, 0]
                y_coords = visible_points[:, 1]

                pose_bbox = [float(x_coords.min()), float(y_coords.min()),
                            float(x_coords.max()), float(y_coords.max())]

                iou = self._calculate_iou(person_bbox, pose_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = pose_idx

            if best_match_idx >= 0:
                matched_poses[person_id] = person_poses[best_match_idx]

        return matched_poses

    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0

        intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0