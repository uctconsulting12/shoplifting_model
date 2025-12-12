"""
config_enhanced.py - Enhanced Configuration (v4.1 - Production Ready)
======================================================================
Complete configuration with body concealment detection and behavioral analysis
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# KEYPOINT INDEX MAP
# =============================================================================

KEYPOINT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AlertLevel(Enum):
    """Alert levels for suspicious behavior"""
    NEUTRAL = 0
    ATTENTION = 1
    ATTENTION_PLUS = 2
    ALERT = 3


class ActionType(Enum):
    """Enhanced action types with body concealment detection"""
    IDLE = "idle"
    BROWSING = "browsing"
    PICKING_ITEM = "picking_item"
    EXAMINING_ITEM = "examining_item"
    OPENING_CONTAINER = "opening_container"
    CONCEALING_ITEM = "concealing_item"
    REPLACING_ITEM = "replacing_item"
    ITEM_DISAPPEARED = "item_disappeared"
    LOOKING_AROUND = "looking_around"
    SUSPICIOUS_LOITERING = "suspicious_loitering"
    
    # Original v3.1
    POCKET_TOUCH = "pocket_touch"
    WAISTBAND_CONCEAL = "waistband_conceal"
    GRAB_MOTION_DETECTED = "grab_motion_detected"
    
    # NEW v4.1: Specific body concealment locations
    FRONT_POCKET_CONCEAL = "front_pocket_conceal"
    BACK_POCKET_CONCEAL = "back_pocket_conceal"
    FRONT_WAISTBAND_CONCEAL = "front_waistband_conceal"
    BACK_WAISTBAND_CONCEAL = "back_waistband_conceal"
    INSIDE_CLOTHING_CONCEAL = "inside_clothing_conceal"
    BAG_INSERTION = "bag_insertion"
    
    # NEW v4.1: Behavioral patterns
    ERRATIC_MOVEMENT = "erratic_movement"
    PROLONGED_LOITERING = "prolonged_loitering"
    NERVOUS_BEHAVIOR = "nervous_behavior"
    SUSPICIOUS_BODY_LANGUAGE = "suspicious_body_language"
    EXIT_BEHAVIOR = "exit_behavior"


class ItemLocation(Enum):
    """Where an item is located"""
    SHELF = "shelf"
    HAND = "hand"
    CONTAINER = "container"
    UNKNOWN = "unknown"
    DISAPPEARED = "disappeared"


class PersonOrientation(Enum):
    """Person orientation relative to camera"""
    FRONT = "front"
    FRONT_LEFT = "front_left"
    LEFT = "left"
    BACK_LEFT = "back_left"
    BACK = "back"
    BACK_RIGHT = "back_right"
    RIGHT = "right"
    FRONT_RIGHT = "front_right"
    UNKNOWN = "unknown"


class ContainerType(Enum):
    """Types of containers"""
    BACKPACK = "backpack"
    HANDBAG = "handbag"
    PURSE = "purse"
    SUITCASE = "suitcase"
    SHOPPING_BAG = "shopping_bag"
    SHOPPING_CART = "shopping_cart"
    UNKNOWN = "unknown"


class StoreProfile(Enum):
    """Store security profiles for adaptive thresholds"""
    HIGH_SECURITY = "high_security"      # Jewelry, electronics, pharmacy
    GENERAL_RETAIL = "general_retail"     # Clothing, groceries
    DISCOUNT_STORE = "discount_store"     # High traffic, lower value
    CONVENIENCE = "convenience"           # Quick transactions


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContainerDetection:
    """Represents a detected container"""
    container_id: int
    container_type: ContainerType
    first_seen_frame: int
    last_seen_frame: int
    bbox: List[float]
    confidence: float
    is_open: bool = False
    opening_frames: List[int] = field(default_factory=list)
    person_id: Optional[int] = None
    temporal_buffer: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class ItemDetection:
    """Represents a detected store item"""
    item_id: int
    item_class: str
    first_seen_frame: int
    last_seen_frame: int
    current_location: ItemLocation
    bbox: List[float]
    confidence: float
    tracked_person_id: Optional[int] = None
    picked_frame: Optional[int] = None
    disappeared_frame: Optional[int] = None
    shelf_region: Optional[List[float]] = None


@dataclass
class ItemInteraction:
    """Represents an interaction with a store item"""
    item_id: int
    item_class: str
    picked_frame: int
    last_seen_frame: int
    location: ItemLocation
    bbox: List[float]
    confidence: float
    examined_duration: int = 0
    disappeared_frame: Optional[int] = None
    concealed_in_container: bool = False
    container_id: Optional[int] = None


@dataclass
class GestureBuffer:
    """Buffer for temporal gesture smoothing"""
    gesture_type: str
    frames: deque = field(default_factory=lambda: deque(maxlen=10))
    confidences: deque = field(default_factory=lambda: deque(maxlen=10))

    def add(self, frame_num: int, detected: bool, confidence: float = 1.0):
        """Add gesture detection result"""
        self.frames.append(frame_num)
        self.confidences.append(confidence if detected else 0.0)

    def is_sustained(self, min_frames: int = 5, min_confidence: float = 0.6) -> bool:
        """Check if gesture is sustained over time"""
        if len(self.frames) < min_frames:
            return False
        recent_conf = list(self.confidences)[-min_frames:]
        return np.mean(recent_conf) >= min_confidence


@dataclass
class PersonState:
    """Enhanced PersonState with concealment tracking"""
    person_id: int
    first_seen_frame: int
    
    # Core tracking
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=150))
    action_sequence: List[tuple] = field(default_factory=list)
    item_interactions: Dict[int, ItemInteraction] = field(default_factory=dict)
    containers: Dict[int, ContainerDetection] = field(default_factory=dict)
    container_openings: List[tuple] = field(default_factory=list)
    location_history: deque = field(default_factory=lambda: deque(maxlen=150))
    orientation_history: deque = field(default_factory=lambda: deque(maxlen=30))
    current_orientation: PersonOrientation = PersonOrientation.UNKNOWN
    attention_score: float = 0.0
    attention_score_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_alert_level: AlertLevel = AlertLevel.NEUTRAL
    last_alert_frame: int = 0
    suspicious_count: int = 0
    time_in_store: int = 0
    dwell_start_frame: Optional[int] = None
    concealment_events: List[Dict] = field(default_factory=list)
    has_shopping_cart: bool = False
    is_staff: bool = False
    gesture_buffers: Dict[str, GestureBuffer] = field(default_factory=dict)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # NEW v4.1: Enhanced concealment tracking
    concealment_locations: List[Dict] = field(default_factory=list)
    behavioral_suspicion_score: float = 0.0
    behavioral_flags: Dict[str, int] = field(default_factory=lambda: {
        'loitering': 0,
        'nervous': 0,
        'erratic_movement': 0,
        'concealment_gestures': 0,
        'exit_behavior': 0
    })
    
    # NEW v4.1: Keypoint history for motion analysis
    previous_keypoints: Optional[np.ndarray] = None

    def update_frame(self, frame_num: int, bbox: List[float],
                    pose_keypoints: Optional[np.ndarray] = None,
                    orientation: PersonOrientation = PersonOrientation.UNKNOWN):
        """Update person state with new frame data"""
        self.frame_buffer.append({
            'frame_num': frame_num,
            'bbox': bbox,
            'pose': pose_keypoints,
            'orientation': orientation,
            'timestamp': None
        })

        self.bbox_history.append(bbox)

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.location_history.append((center_x, center_y, frame_num))

        self.current_orientation = orientation
        self.orientation_history.append((orientation, frame_num))

        self.time_in_store = frame_num - self.first_seen_frame
        
        # Store previous keypoints for motion analysis
        if pose_keypoints is not None:
            self.previous_keypoints = pose_keypoints.copy()

    def add_action(self, frame_num: int, action: ActionType):
        """Record an action"""
        self.action_sequence.append((frame_num, action))
        cutoff_frame = frame_num - 150
        self.action_sequence = [(f, a) for f, a in self.action_sequence if f > cutoff_frame]

    def add_item_interaction(self, item_id: int, item_class: str, frame_num: int,
                            location: ItemLocation, bbox: List[float], confidence: float):
        """Record interaction with an item"""
        if item_id not in self.item_interactions:
            self.item_interactions[item_id] = ItemInteraction(
                item_id=item_id,
                item_class=item_class,
                picked_frame=frame_num,
                last_seen_frame=frame_num,
                location=location,
                bbox=bbox,
                confidence=confidence
            )
        else:
            interaction = self.item_interactions[item_id]
            interaction.last_seen_frame = frame_num
            interaction.location = location
            interaction.bbox = bbox
            interaction.confidence = confidence

            if location == ItemLocation.HAND:
                interaction.examined_duration = frame_num - interaction.picked_frame

    def get_recent_actions(self, current_frame: int, window: int = 150) -> List[ActionType]:
        """Get actions from recent frames"""
        cutoff = current_frame - window
        return [action for frame, action in self.action_sequence if frame > cutoff]

    def get_person_height(self) -> float:
        """Get average person height from recent bboxes"""
        if not self.bbox_history:
            return 180.0
        heights = [bbox[3] - bbox[1] for bbox in self.bbox_history]
        return np.mean(heights)

    def get_person_width(self) -> float:
        """Get average person width"""
        if not self.bbox_history:
            return 100.0
        widths = [bbox[2] - bbox[0] for bbox in self.bbox_history]
        return np.mean(widths)

    def add_gesture(self, gesture_type: str, frame_num: int, detected: bool, confidence: float = 1.0):
        """Add gesture to temporal buffer"""
        if gesture_type not in self.gesture_buffers:
            self.gesture_buffers[gesture_type] = GestureBuffer(gesture_type=gesture_type)
        self.gesture_buffers[gesture_type].add(frame_num, detected, confidence)

    def is_gesture_sustained(self, gesture_type: str, min_frames: int = 5) -> bool:
        """Check if gesture is sustained"""
        if gesture_type not in self.gesture_buffers:
            return False
        return self.gesture_buffers[gesture_type].is_sustained(min_frames)

    def has_container(self) -> bool:
        """Check if person has any container (excluding shopping cart)"""
        return any(c.container_type != ContainerType.SHOPPING_CART
                  for c in self.containers.values())

    def recent_container_opening(self, current_frame: int, window: int = 30) -> bool:
        """Check if container was opened recently"""
        if not self.container_openings:
            return False
        return any(frame > current_frame - window for frame, _ in self.container_openings)

    def calculate_movement_speed(self) -> float:
        """Calculate average movement speed"""
        if len(self.location_history) < 2:
            return 0.0

        speeds = []
        for i in range(1, len(self.location_history)):
            x1, y1, f1 = self.location_history[i-1]
            x2, y2, f2 = self.location_history[i]

            if f2 - f1 > 0:
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                speed = distance / (f2 - f1)
                speeds.append(speed)

        return np.mean(speeds) if speeds else 0.0

    def is_loitering(self, current_frame: int, threshold: int = 90) -> bool:
        """Check if person is loitering"""
        if len(self.location_history) < threshold:
            return False

        recent_locations = list(self.location_history)[-threshold:]
        x_coords = [loc[0] for loc in recent_locations]
        y_coords = [loc[1] for loc in recent_locations]

        x_var = np.var(x_coords)
        y_var = np.var(y_coords)

        return (x_var < 150) and (y_var < 150)

    def detect_concealment_pattern(self, current_frame: int) -> bool:
        """Detect concealment pattern: item picked → container opened → item disappeared"""
        picked_items = [item for item in self.item_interactions.values()
                       if item.location in [ItemLocation.HAND, ItemLocation.DISAPPEARED]]

        if not picked_items:
            return False

        if not self.recent_container_opening(current_frame, window=60):
            return False

        for item in picked_items:
            if item.location == ItemLocation.DISAPPEARED:
                for open_frame, container_id in self.container_openings:
                    if open_frame < item.disappeared_frame and \
                       (item.disappeared_frame - open_frame) < 60:
                        item.concealed_in_container = True
                        item.container_id = container_id
                        return True

        return False

    def add_concealment_event(self, frame_num: int, location_type: str,
                             confidence: float, has_product: bool = False):
        """Record a concealment event with location details"""
        event = {
            'frame': frame_num,
            'location': location_type,
            'confidence': confidence,
            'has_product': has_product,
            'timestamp': frame_num
        }
        self.concealment_events.append(event)
        self.concealment_locations.append(event)
        
        # Keep only recent events (last 100)
        if len(self.concealment_events) > 100:
            self.concealment_events = self.concealment_events[-100:]
        if len(self.concealment_locations) > 100:
            self.concealment_locations = self.concealment_locations[-100:]

    def get_concealment_pattern(self, current_frame: int, window: int = 60) -> Dict:
        """
        Analyze concealment pattern
        Returns: dict with location frequencies and confidence
        """
        recent_events = [
            e for e in self.concealment_events 
            if e['frame'] > current_frame - window
        ]
        
        if not recent_events:
            return {'total': 0, 'locations': {}, 'avg_confidence': 0.0, 'has_product': False}
        
        locations = {}
        for event in recent_events:
            loc = event['location']
            locations[loc] = locations.get(loc, 0) + 1
        
        avg_conf = np.mean([e['confidence'] for e in recent_events])
        
        return {
            'total': len(recent_events),
            'locations': locations,
            'avg_confidence': avg_conf,
            'has_product': any(e['has_product'] for e in recent_events)
        }

    def increment_behavioral_flag(self, flag_type: str):
        """Increment behavioral suspicion flag"""
        if flag_type in self.behavioral_flags:
            self.behavioral_flags[flag_type] += 1


# =============================================================================
# ENHANCED SYSTEM CONFIGURATION (v4.1)
# =============================================================================

class SystemConfig:
    """Enhanced system configuration with adaptive thresholds"""

    # Model paths
    POSE_MODEL_PATH = "yolov8n-pose.pt"
    OBJECT_MODEL_PATH = "yolov8n.pt"

    # Detection thresholds
    DETECTION_CONFIDENCE = 0.4
    POSE_CONFIDENCE = 0.3
    CONTAINER_CONFIDENCE = 0.3
    NMS_IOU = 0.45

    # Alert thresholds
    ALERT_THRESHOLD = 0.75
    ALERT_COOLDOWN_FRAMES = 90

    # Gesture parameters
    MIN_GESTURE_FRAMES = 5
    GESTURE_CONFIDENCE_THRESHOLD = 0.6

    # Disappearance detection
    ITEM_DISAPPEAR_WINDOW = 30

    # Shelf detection
    SHELF_CUTOFF_RATIO = 0.35

    # Performance
    SKIP_FRAMES = 1

    # =========================================================================
    # v4.1: BODY CONCEALMENT DETECTION
    # =========================================================================
    
    # Pocket concealment thresholds
    FRONT_POCKET_THRESHOLD_RATIO = 0.12
    BACK_POCKET_THRESHOLD_RATIO = 0.14
    
    # Waistband concealment thresholds
    WAISTBAND_THRESHOLD_RATIO = 0.15
    WAISTBAND_HEIGHT_TOLERANCE_RATIO = 0.1
    
    # Inside clothing concealment
    INSIDE_CLOTHING_THRESHOLD_RATIO = 0.25
    BENT_ARM_THRESHOLD_RATIO = 0.2
    
    # Bag insertion
    BAG_INSERTION_THRESHOLD_RATIO = 0.3
    
    # Temporal consistency requirements
    CONCEALMENT_SUSTAIN_FRAMES = 3
    CONCEALMENT_WINDOW_FRAMES = 5
    CONCEALMENT_CONFIDENCE_THRESHOLD = 0.6
    
    # Motion requirements
    MIN_HAND_VELOCITY = 3.0
    MAX_STATIC_FRAMES = 2
    
    # =========================================================================
    # v4.1: BEHAVIORAL PATTERN ANALYSIS
    # =========================================================================
    
    # Loitering detection
    LOITERING_VARIANCE_THRESHOLD = 100
    LOITERING_DURATION_THRESHOLD = 3.0
    
    # Nervous behavior
    NERVOUS_LOOKING_COUNT_THRESHOLD = 5
    NERVOUS_BEHAVIOR_WINDOW = 90
    
    # Erratic movement
    SHARP_TURN_THRESHOLD = np.pi / 2
    SHARP_TURN_COUNT_THRESHOLD = 5
    APPROACH_RETREAT_THRESHOLD = 2
    
    # Behavioral suspicion weights
    BEHAVIOR_LOITERING_WEIGHT = 0.25
    BEHAVIOR_NERVOUS_WEIGHT = 0.25
    BEHAVIOR_MOVEMENT_WEIGHT = 0.20
    BEHAVIOR_CONCEALMENT_WEIGHT = 0.30
    
    # No-product-detection mode threshold
    NO_PRODUCT_SUSPICION_THRESHOLD = 0.70
    
    # =========================================================================
    # v4.1: SEQUENCE VALIDATION
    # =========================================================================
    
    SEQUENCE_VALIDATION_ENABLED = True
    SEQUENCE_WINDOW_FRAMES = 120
    BEHAVIORAL_SEQUENCE_THRESHOLD = 0.7
    MIN_CONCEALMENT_GESTURES = 2
    
    # =========================================================================
    # v4.1: EXIT BEHAVIOR DETECTION
    # =========================================================================
    
    EXIT_DETECTION_ENABLED = True
    EXIT_ZONE_RATIO = 0.15
    EXIT_VELOCITY_THRESHOLD = 2.0
    
    # =========================================================================
    # ORIGINAL v3.1: SEQUENTIAL GRAB MOTION DETECTION
    # =========================================================================
    
    GRAB_MOTION_WINDOW = 25
    FAST_MOTION_THRESHOLD = 15
    SLOW_MOTION_THRESHOLD = 8
    PAUSE_MOTION_THRESHOLD = 5
    MIN_EXTEND_FRAMES = 3
    MIN_PAUSE_FRAMES = 2
    MIN_RETRACT_FRAMES = 3
    MIN_PAUSE_DISTANCE_FROM_BODY = 50
    GRAB_CONFIDENCE_THRESHOLD = 0.65
    GRAB_COOLDOWN_FRAMES = 60
    REPEAT_OFFENDER_THRESHOLD = 3
    HIGH_FALSE_POSITIVE_THRESHOLD = 3

    # Container and product classes
    CONTAINER_CLASSES = {
        'backpack': ContainerType.BACKPACK,
        'handbag': ContainerType.HANDBAG,
        'suitcase': ContainerType.SUITCASE,
    }

    CART_INDICATORS = ['cart', 'shopping_cart', 'basket']

    PRODUCT_CLASSES = {
        'bottle', 'cup', 'wine glass', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush', 'cell phone', 'laptop', 'mouse', 'remote', 'keyboard'
    }
    
    # =========================================================================
    # v4.1: ADAPTIVE CONFIGURATION BY STORE PROFILE
    # =========================================================================
    
    CURRENT_PROFILE: StoreProfile = StoreProfile.GENERAL_RETAIL
    
    @classmethod
    def configure_for_store(cls, profile: StoreProfile):
        """Adapt thresholds based on store type"""
        cls.CURRENT_PROFILE = profile
        
        if profile == StoreProfile.HIGH_SECURITY:
            cls.NERVOUS_LOOKING_COUNT_THRESHOLD = 3
            cls.NO_PRODUCT_SUSPICION_THRESHOLD = 0.60
            cls.CONCEALMENT_SUSTAIN_FRAMES = 2
            cls.MIN_CONCEALMENT_GESTURES = 1
            logger.info("Configured for HIGH SECURITY store profile")
            
        elif profile == StoreProfile.GENERAL_RETAIL:
            cls.NERVOUS_LOOKING_COUNT_THRESHOLD = 5
            cls.NO_PRODUCT_SUSPICION_THRESHOLD = 0.70
            cls.CONCEALMENT_SUSTAIN_FRAMES = 3
            cls.MIN_CONCEALMENT_GESTURES = 2
            logger.info("Configured for GENERAL RETAIL store profile")
            
        elif profile == StoreProfile.DISCOUNT_STORE:
            cls.NERVOUS_LOOKING_COUNT_THRESHOLD = 8
            cls.NO_PRODUCT_SUSPICION_THRESHOLD = 0.75
            cls.CONCEALMENT_SUSTAIN_FRAMES = 4
            cls.MIN_CONCEALMENT_GESTURES = 3
            logger.info("Configured for DISCOUNT STORE profile")
            
        elif profile == StoreProfile.CONVENIENCE:
            cls.NERVOUS_LOOKING_COUNT_THRESHOLD = 4
            cls.NO_PRODUCT_SUSPICION_THRESHOLD = 0.68
            cls.CONCEALMENT_SUSTAIN_FRAMES = 3
            cls.MIN_CONCEALMENT_GESTURES = 2
            logger.info("Configured for CONVENIENCE store profile")

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            'detection_confidence': cls.DETECTION_CONFIDENCE,
            'pose_confidence': cls.POSE_CONFIDENCE,
            'alert_threshold': cls.ALERT_THRESHOLD,
            'front_pocket_threshold': cls.FRONT_POCKET_THRESHOLD_RATIO,
            'waistband_threshold': cls.WAISTBAND_THRESHOLD_RATIO,
            'inside_clothing_threshold': cls.INSIDE_CLOTHING_THRESHOLD_RATIO,
            'concealment_sustain_frames': cls.CONCEALMENT_SUSTAIN_FRAMES,
            'min_hand_velocity': cls.MIN_HAND_VELOCITY,
            'no_product_suspicion_threshold': cls.NO_PRODUCT_SUSPICION_THRESHOLD,
            'nervous_looking_threshold': cls.NERVOUS_LOOKING_COUNT_THRESHOLD,
            'loitering_duration': cls.LOITERING_DURATION_THRESHOLD,
            'sequence_validation_enabled': cls.SEQUENCE_VALIDATION_ENABLED,
            'min_concealment_gestures': cls.MIN_CONCEALMENT_GESTURES,
            'exit_detection_enabled': cls.EXIT_DETECTION_ENABLED,
            'store_profile': cls.CURRENT_PROFILE.value,
            'grab_confidence_threshold': cls.GRAB_CONFIDENCE_THRESHOLD,
            'grab_cooldown_frames': cls.GRAB_COOLDOWN_FRAMES,
            'min_gesture_frames': cls.MIN_GESTURE_FRAMES,
            'nms_iou': cls.NMS_IOU,
            'container_confidence': cls.CONTAINER_CONFIDENCE,
        }


# =============================================================================
# METRICS TRACKER
# =============================================================================

class MetricsTracker:
    """Track performance metrics with enhanced concealment tracking"""

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
        self.alert_history = []
        self.grab_motion_count = 0
        self.grab_rejections = 0
        self.repeat_offenders = 0
        
        # NEW v4.1: Concealment tracking
        self.concealment_detections = 0
        self.concealment_by_location = {}
        self.behavioral_only_alerts = 0
        self.sequence_validated_alerts = 0

    def add_alert(self, person_id: int, frame_num: int, score: float,
                 is_true_theft: Optional[bool] = None,
                 concealment_locations: Optional[Dict] = None,
                 has_product: bool = True,
                 sequence_validated: bool = False):
        """Record an alert with enhanced metadata"""
        self.alert_history.append({
            'person_id': person_id,
            'frame_num': frame_num,
            'score': score,
            'is_true_theft': is_true_theft,
            'concealment_locations': concealment_locations or {},
            'has_product': has_product,
            'sequence_validated': sequence_validated
        })

        if is_true_theft is not None:
            if is_true_theft:
                self.true_positives += 1
            else:
                self.false_positives += 1
        
        # Track concealment locations
        if concealment_locations:
            for loc in concealment_locations.keys():
                self.concealment_by_location[loc] = \
                    self.concealment_by_location.get(loc, 0) + 1
        
        # Track behavioral-only alerts
        if not has_product:
            self.behavioral_only_alerts += 1
        
        # Track sequence-validated alerts
        if sequence_validated:
            self.sequence_validated_alerts += 1

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if self.true_positives + self.false_positives == 0:
            precision = 0.0
        else:
            precision = self.true_positives / (self.true_positives + self.false_positives)

        if self.true_positives + self.false_negatives == 0:
            recall = 0.0
        else:
            recall = self.true_positives / (self.true_positives + self.false_negatives)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_alerts': len(self.alert_history),
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'grab_motions_detected': self.grab_motion_count,
            'grab_rejections': self.grab_rejections,
            'repeat_offenders': self.repeat_offenders,
            'grab_detection_rate': self.grab_motion_count / max(1, self.grab_motion_count + self.grab_rejections),
            'concealment_detections': self.concealment_detections,
            'concealment_by_location': self.concealment_by_location,
            'behavioral_only_alerts': self.behavioral_only_alerts,
            'sequence_validated_alerts': self.sequence_validated_alerts,
            'behavioral_alert_ratio': self.behavioral_only_alerts / max(1, len(self.alert_history))
        }
