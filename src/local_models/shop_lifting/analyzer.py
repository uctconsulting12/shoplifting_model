"""
analyzer_enhanced.py - Complete v4.1 Enhanced Analyzers
========================================================
All detection systems: Body concealment, behavioral analysis, sequence validation
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import defaultdict, Counter, deque
from scipy.ndimage import gaussian_filter1d

from src.local_models.shop_lifting.config import (
    PersonOrientation, ActionType, PersonState, AlertLevel,
    ItemLocation, SystemConfig, logger, KEYPOINT
)


# =============================================================================
# BODY REGION CONCEALMENT DETECTOR (v4.1)
# =============================================================================

class BodyRegionConcealmentDetector:
    """
    Enhanced concealment detector with temporal validation and motion context
    """

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        
        self.concealment_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        
        self.concealment_buffers: Dict[int, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=SystemConfig.CONCEALMENT_WINDOW_FRAMES))
        )
        
        self.static_hand_tracker: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {'left': 0, 'right': 0}
        )

    def _calculate_hand_velocity(self, current_wrist: np.ndarray,
                                 previous_wrist: Optional[np.ndarray]) -> float:
        """Calculate hand velocity"""
        if previous_wrist is None or previous_wrist[2] < self.confidence_threshold:
            return 0.0
        
        velocity = np.linalg.norm(current_wrist[:2] - previous_wrist[:2])
        return velocity

    def _is_hand_moving(self, velocity: float, person_id: int, side: str) -> bool:
        """Check if hand is moving (not static in pocket/position)"""
        if velocity < SystemConfig.MIN_HAND_VELOCITY:
            self.static_hand_tracker[person_id][side] += 1
            
            if self.static_hand_tracker[person_id][side] > SystemConfig.MAX_STATIC_FRAMES:
                return False
        else:
            self.static_hand_tracker[person_id][side] = 0
        
        return velocity >= SystemConfig.MIN_HAND_VELOCITY

    def detect_pocket_concealment(self, keypoints: np.ndarray, 
                                   person_height: float,
                                   orientation: PersonOrientation,
                                   previous_keypoints: Optional[np.ndarray] = None,
                                   has_item_context: bool = False) -> Tuple[bool, str, float]:
        """Enhanced pocket concealment detection with motion analysis"""
        if keypoints is None or len(keypoints) != 17:
            return False, "none", 0.0

        try:
            left_wrist = keypoints[KEYPOINT["left_wrist"]]
            right_wrist = keypoints[KEYPOINT["right_wrist"]]
            left_hip = keypoints[KEYPOINT["left_hip"]]
            right_hip = keypoints[KEYPOINT["right_hip"]]

            if not all(kp[2] > self.confidence_threshold 
                      for kp in [left_wrist, right_wrist, left_hip, right_hip]):
                return False, "none", 0.0

            front_pocket_threshold = person_height * SystemConfig.FRONT_POCKET_THRESHOLD_RATIO
            
            left_front_dist = np.sqrt(
                (left_wrist[0] - left_hip[0])**2 + 
                (left_wrist[1] - left_hip[1])**2
            )
            right_front_dist = np.sqrt(
                (right_wrist[0] - right_hip[0])**2 + 
                (right_wrist[1] - right_hip[1])**2
            )

            # Check left hand
            if left_front_dist < front_pocket_threshold:
                velocity = 0.0
                if previous_keypoints is not None:
                    prev_left_wrist = previous_keypoints[KEYPOINT["left_wrist"]]
                    velocity = self._calculate_hand_velocity(left_wrist, prev_left_wrist)
                
                if velocity < SystemConfig.MIN_HAND_VELOCITY:
                    return False, "none", 0.0
                
                confidence = 1.0 - (left_front_dist / front_pocket_threshold)
                
                if has_item_context:
                    confidence *= 1.3
                else:
                    confidence *= 0.6
                
                confidence = min(1.0, confidence)
                return True, "front_left_pocket", confidence
            
            # Check right hand
            if right_front_dist < front_pocket_threshold:
                velocity = 0.0
                if previous_keypoints is not None:
                    prev_right_wrist = previous_keypoints[KEYPOINT["right_wrist"]]
                    velocity = self._calculate_hand_velocity(right_wrist, prev_right_wrist)
                
                if velocity < SystemConfig.MIN_HAND_VELOCITY:
                    return False, "none", 0.0
                
                confidence = 1.0 - (right_front_dist / front_pocket_threshold)
                
                if has_item_context:
                    confidence *= 1.3
                else:
                    confidence *= 0.6
                
                confidence = min(1.0, confidence)
                return True, "front_right_pocket", confidence

            # Back pocket detection
            if orientation in [PersonOrientation.BACK, PersonOrientation.BACK_LEFT, 
                              PersonOrientation.BACK_RIGHT]:
                back_threshold = person_height * SystemConfig.BACK_POCKET_THRESHOLD_RATIO
                
                if left_front_dist < back_threshold:
                    confidence = 0.7
                    if has_item_context:
                        confidence *= 1.2
                    else:
                        confidence *= 0.5
                    return True, "back_left_pocket", min(1.0, confidence)
                    
                if right_front_dist < back_threshold:
                    confidence = 0.7
                    if has_item_context:
                        confidence *= 1.2
                    else:
                        confidence *= 0.5
                    return True, "back_right_pocket", min(1.0, confidence)

            return False, "none", 0.0

        except Exception as e:
            logger.debug(f"Pocket concealment error: {e}")
            return False, "none", 0.0

    def detect_waistband_concealment(self, keypoints: np.ndarray,
                                     person_height: float,
                                     orientation: PersonOrientation,
                                     previous_keypoints: Optional[np.ndarray] = None,
                                     has_item_context: bool = False) -> Tuple[bool, str, float]:
        """Enhanced waistband concealment with motion analysis"""
        if keypoints is None or len(keypoints) != 17:
            return False, "none", 0.0

        try:
            left_wrist = keypoints[KEYPOINT["left_wrist"]]
            right_wrist = keypoints[KEYPOINT["right_wrist"]]
            left_hip = keypoints[KEYPOINT["left_hip"]]
            right_hip = keypoints[KEYPOINT["right_hip"]]

            if not all(kp[2] > self.confidence_threshold 
                      for kp in [left_wrist, right_wrist, left_hip, right_hip]):
                return False, "none", 0.0

            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2

            waistband_threshold = person_height * SystemConfig.WAISTBAND_THRESHOLD_RATIO
            
            for wrist, side in [(left_wrist, "left"), (right_wrist, "right")]:
                if wrist[2] < self.confidence_threshold:
                    continue

                dist_to_waist = np.sqrt(
                    (wrist[0] - hip_center_x)**2 + 
                    (wrist[1] - hip_center_y)**2
                )

                y_diff = abs(wrist[1] - hip_center_y)
                height_tolerance = person_height * SystemConfig.WAISTBAND_HEIGHT_TOLERANCE_RATIO
                
                if dist_to_waist < waistband_threshold and y_diff < height_tolerance:
                    velocity = 0.0
                    if previous_keypoints is not None:
                        wrist_key = f"{side}_wrist"
                        prev_wrist = previous_keypoints[KEYPOINT[wrist_key]]
                        velocity = self._calculate_hand_velocity(wrist, prev_wrist)
                    
                    if velocity < SystemConfig.MIN_HAND_VELOCITY:
                        continue
                    
                    confidence = 1.0 - (dist_to_waist / waistband_threshold)
                    
                    if has_item_context:
                        confidence *= 1.4
                    else:
                        confidence *= 0.5
                    
                    confidence = min(1.0, confidence)
                    
                    if orientation in [PersonOrientation.BACK, PersonOrientation.BACK_LEFT,
                                      PersonOrientation.BACK_RIGHT]:
                        return True, f"back_waistband_{side}", confidence
                    else:
                        return True, f"front_waistband_{side}", confidence

            return False, "none", 0.0

        except Exception as e:
            logger.debug(f"Waistband concealment error: {e}")
            return False, "none", 0.0

    def detect_inside_clothing_concealment(self, keypoints: np.ndarray,
                                           person_height: float,
                                           previous_keypoints: Optional[np.ndarray] = None,
                                           has_item_context: bool = False) -> Tuple[bool, str, float]:
        """Enhanced inside clothing detection"""
        if keypoints is None or len(keypoints) != 17:
            return False, "none", 0.0

        try:
            left_wrist = keypoints[KEYPOINT["left_wrist"]]
            right_wrist = keypoints[KEYPOINT["right_wrist"]]
            left_elbow = keypoints[KEYPOINT["left_elbow"]]
            right_elbow = keypoints[KEYPOINT["right_elbow"]]
            left_shoulder = keypoints[KEYPOINT["left_shoulder"]]
            right_shoulder = keypoints[KEYPOINT["right_shoulder"]]

            torso_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            torso_center_y = (left_shoulder[1] + right_shoulder[1]) / 2

            inside_threshold = person_height * SystemConfig.INSIDE_CLOTHING_THRESHOLD_RATIO

            for wrist, elbow, side in [(left_wrist, left_elbow, "left"),
                                       (right_wrist, right_elbow, "right")]:
                if wrist[2] < self.confidence_threshold or elbow[2] < self.confidence_threshold:
                    continue

                dist_to_torso = np.sqrt(
                    (wrist[0] - torso_center_x)**2 + 
                    (wrist[1] - torso_center_y)**2
                )

                arm_length = np.linalg.norm(wrist[:2] - elbow[:2])
                bent_threshold = person_height * SystemConfig.BENT_ARM_THRESHOLD_RATIO
                
                if dist_to_torso < inside_threshold and arm_length < bent_threshold:
                    if previous_keypoints is not None:
                        wrist_key = f"{side}_wrist"
                        prev_wrist = previous_keypoints[KEYPOINT[wrist_key]]
                        
                        if prev_wrist[2] > self.confidence_threshold:
                            prev_dist = np.sqrt(
                                (prev_wrist[0] - torso_center_x)**2 + 
                                (prev_wrist[1] - torso_center_y)**2
                            )
                            
                            if prev_dist < inside_threshold:
                                continue
                    
                    confidence = 1.0 - (dist_to_torso / inside_threshold)
                    
                    if has_item_context:
                        confidence *= 1.5
                    else:
                        confidence *= 0.4
                    
                    confidence = min(1.0, confidence)
                    return True, f"inside_clothing_{side}", confidence

            return False, "none", 0.0

        except Exception as e:
            logger.debug(f"Inside clothing concealment error: {e}")
            return False, "none", 0.0

    def detect_bag_insertion(self, keypoints: np.ndarray,
                            person_height: float,
                            has_bag: bool,
                            previous_keypoints: Optional[np.ndarray] = None,
                            has_item_context: bool = False) -> Tuple[bool, float]:
        """Enhanced bag insertion detection"""
        if keypoints is None or len(keypoints) != 17 or not has_bag:
            return False, 0.0

        try:
            left_wrist = keypoints[KEYPOINT["left_wrist"]]
            right_wrist = keypoints[KEYPOINT["right_wrist"]]
            left_shoulder = keypoints[KEYPOINT["left_shoulder"]]
            right_shoulder = keypoints[KEYPOINT["right_shoulder"]]

            shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
            bag_insertion_threshold = person_height * SystemConfig.BAG_INSERTION_THRESHOLD_RATIO

            for wrist, side in [(left_wrist, "left"), (right_wrist, "right")]:
                if wrist[2] < self.confidence_threshold:
                    continue

                y_diff = abs(wrist[1] - shoulder_avg_y)
                
                if y_diff < bag_insertion_threshold:
                    velocity = 0.0
                    if previous_keypoints is not None:
                        wrist_key = f"{side}_wrist"
                        prev_wrist = previous_keypoints[KEYPOINT[wrist_key]]
                        velocity = self._calculate_hand_velocity(wrist, prev_wrist)
                    
                    if velocity < SystemConfig.MIN_HAND_VELOCITY * 0.5:
                        continue
                    
                    confidence = 1.0 - (y_diff / bag_insertion_threshold)
                    
                    if has_item_context:
                        confidence *= 1.4
                    else:
                        confidence *= 0.7
                    
                    return True, min(1.0, confidence)

            return False, 0.0

        except Exception as e:
            logger.debug(f"Bag insertion error: {e}")
            return False, 0.0

    def _validate_sustained_detection(self, person_id: int, location_type: str,
                                      frame_num: int, detected: bool,
                                      confidence: float) -> Tuple[bool, float]:
        """Validate detection is sustained over multiple frames"""
        buffer = self.concealment_buffers[person_id][location_type]
        buffer.append((frame_num, detected, confidence))
        
        recent = list(buffer)[-SystemConfig.CONCEALMENT_WINDOW_FRAMES:]
        detection_count = sum(1 for _, d, _ in recent if d)
        
        if detection_count >= SystemConfig.CONCEALMENT_SUSTAIN_FRAMES:
            avg_conf = np.mean([c for _, d, c in recent if d])
            return True, avg_conf
        
        return False, 0.0

    def analyze_concealment_sequence(self, person_id: int,
                                    frame_num: int,
                                    keypoints: np.ndarray,
                                    person_height: float,
                                    orientation: PersonOrientation,
                                    has_bag: bool,
                                    previous_keypoints: Optional[np.ndarray] = None,
                                    has_item_context: bool = False,
                                    grab_motion_active: bool = False) -> List[Tuple[ActionType, str, float]]:
        """Comprehensive concealment analysis"""
        validated_detections = []

        # 1. Pocket concealment
        pocket_detected, pocket_type, pocket_conf = self.detect_pocket_concealment(
            keypoints, person_height, orientation, previous_keypoints, has_item_context
        )
        
        if pocket_detected:
            sustained, avg_conf = self._validate_sustained_detection(
                person_id, pocket_type, frame_num, True, pocket_conf
            )
            
            if sustained:
                if grab_motion_active:
                    avg_conf = min(1.0, avg_conf * 1.5)
                    logger.info(
                        f"Person {person_id}: GRAB MOTION + POCKET TOUCH "
                        f"(high theft probability!)"
                    )
                
                if avg_conf >= SystemConfig.CONCEALMENT_CONFIDENCE_THRESHOLD:
                    validated_detections.append((ActionType.POCKET_TOUCH, pocket_type, avg_conf))

        # 2. Waistband concealment
        waist_detected, waist_type, waist_conf = self.detect_waistband_concealment(
            keypoints, person_height, orientation, previous_keypoints, has_item_context
        )
        
        if waist_detected:
            sustained, avg_conf = self._validate_sustained_detection(
                person_id, waist_type, frame_num, True, waist_conf
            )
            
            if sustained:
                if grab_motion_active:
                    avg_conf = min(1.0, avg_conf * 1.6)
                
                if avg_conf >= SystemConfig.CONCEALMENT_CONFIDENCE_THRESHOLD:
                    validated_detections.append((ActionType.WAISTBAND_CONCEAL, waist_type, avg_conf))

        # 3. Inside clothing concealment
        clothing_detected, clothing_type, clothing_conf = self.detect_inside_clothing_concealment(
            keypoints, person_height, previous_keypoints, has_item_context
        )
        
        if clothing_detected:
            sustained, avg_conf = self._validate_sustained_detection(
                person_id, clothing_type, frame_num, True, clothing_conf
            )
            
            if sustained:
                if grab_motion_active:
                    avg_conf = min(1.0, avg_conf * 1.7)
                
                if avg_conf >= SystemConfig.CONCEALMENT_CONFIDENCE_THRESHOLD:
                    validated_detections.append((ActionType.INSIDE_CLOTHING_CONCEAL, clothing_type, avg_conf))

        # 4. Bag insertion
        bag_detected, bag_conf = self.detect_bag_insertion(
            keypoints, person_height, has_bag, previous_keypoints, has_item_context
        )
        
        if bag_detected:
            sustained, avg_conf = self._validate_sustained_detection(
                person_id, "bag_insertion", frame_num, True, bag_conf
            )
            
            if sustained:
                if grab_motion_active:
                    avg_conf = min(1.0, avg_conf * 1.5)
                
                if avg_conf >= SystemConfig.CONCEALMENT_CONFIDENCE_THRESHOLD:
                    validated_detections.append((ActionType.CONCEALING_ITEM, "bag_insertion", avg_conf))

        if validated_detections:
            self.concealment_history[person_id].append({
                'frame': frame_num,
                'detections': validated_detections,
                'has_item_context': has_item_context,
                'grab_motion_active': grab_motion_active,
                'timestamp': frame_num
            })

        return validated_detections


# =============================================================================
# BEHAVIORAL PATTERN ANALYZER (v4.1)
# =============================================================================

class BehaviorPatternAnalyzer:
    """Enhanced behavioral pattern analysis"""

    def __init__(self):
        self.behavior_scores: Dict[int, Dict] = defaultdict(lambda: {
            'loitering_score': 0.0,
            'nervous_behavior_score': 0.0,
            'suspicious_movement_score': 0.0,
            'concealment_gesture_score': 0.0,
            'exit_behavior_score': 0.0,
            'total_suspicion_score': 0.0
        })

    def analyze_loitering_pattern(self, person_state: PersonState,
                                  current_frame: int) -> float:
        """Enhanced loitering detection"""
        if len(person_state.location_history) < 30:
            return 0.0

        recent_locations = list(person_state.location_history)[-90:]
        x_coords = [loc[0] for loc in recent_locations]
        y_coords = [loc[1] for loc in recent_locations]

        x_var = np.var(x_coords)
        y_var = np.var(y_coords)

        if x_var < SystemConfig.LOITERING_VARIANCE_THRESHOLD and \
           y_var < SystemConfig.LOITERING_VARIANCE_THRESHOLD:
            time_in_area = len(recent_locations) / 30.0
            
            if time_in_area > SystemConfig.LOITERING_DURATION_THRESHOLD:
                score = min(1.0, time_in_area / 10.0)
                person_state.increment_behavioral_flag('loitering')
                return score

        return 0.0

    def analyze_nervous_behavior(self, person_state: PersonState,
                                 current_frame: int) -> float:
        """Enhanced nervous behavior detection"""
        recent_actions = person_state.get_recent_actions(
            current_frame, 
            window=SystemConfig.NERVOUS_BEHAVIOR_WINDOW
        )
        
        looking_count = recent_actions.count(ActionType.LOOKING_AROUND)
        
        if looking_count >= SystemConfig.NERVOUS_LOOKING_COUNT_THRESHOLD:
            score = min(1.0, looking_count / 10.0)
            person_state.increment_behavioral_flag('nervous')
            return score
        elif looking_count >= (SystemConfig.NERVOUS_LOOKING_COUNT_THRESHOLD // 2):
            return 0.4
        
        return 0.0

    def analyze_suspicious_movement(self, person_state: PersonState,
                                   current_frame: int) -> float:
        """Improved movement analysis based on directional changes"""
        if len(person_state.location_history) < 30:
            return 0.0

        recent_locs = list(person_state.location_history)[-30:]
        
        sharp_turns = 0
        approach_retreat_cycles = 0
        
        for i in range(2, len(recent_locs)):
            prev = recent_locs[i-2]
            curr = recent_locs[i-1]
            next_pos = recent_locs[i]
            
            dir1 = np.arctan2(curr[1] - prev[1], curr[0] - prev[0])
            dir2 = np.arctan2(next_pos[1] - curr[1], next_pos[0] - curr[0])
            
            angle_change = abs(dir2 - dir1)
            if angle_change > np.pi:
                angle_change = 2 * np.pi - angle_change
            
            if angle_change > SystemConfig.SHARP_TURN_THRESHOLD:
                sharp_turns += 1
        
        if len(recent_locs) >= 20:
            segment_size = 5
            avg_distances = []
            
            for i in range(0, len(recent_locs) - segment_size, segment_size):
                segment = recent_locs[i:i+segment_size]
                start = segment[0]
                avg_dist = np.mean([
                    np.sqrt((loc[0] - start[0])**2 + (loc[1] - start[1])**2)
                    for loc in segment
                ])
                avg_distances.append(avg_dist)
            
            for i in range(1, len(avg_distances) - 1):
                if (avg_distances[i] > avg_distances[i-1] and 
                    avg_distances[i] > avg_distances[i+1]):
                    approach_retreat_cycles += 1

        suspicion = 0.0
        if sharp_turns > SystemConfig.SHARP_TURN_COUNT_THRESHOLD:
            suspicion += 0.3
            person_state.increment_behavioral_flag('erratic_movement')
        
        if approach_retreat_cycles > SystemConfig.APPROACH_RETREAT_THRESHOLD:
            suspicion += 0.5
            person_state.increment_behavioral_flag('erratic_movement')

        return min(1.0, suspicion)

    def analyze_concealment_gestures(self, person_state: PersonState,
                                    current_frame: int) -> float:
        """Analyze concealment-like gestures"""
        recent_actions = person_state.get_recent_actions(current_frame, window=60)
        
        concealment_indicators = [
            ActionType.POCKET_TOUCH,
            ActionType.WAISTBAND_CONCEAL,
            ActionType.INSIDE_CLOTHING_CONCEAL,
            ActionType.CONCEALING_ITEM,
            ActionType.FRONT_POCKET_CONCEAL,
            ActionType.BACK_POCKET_CONCEAL,
            ActionType.FRONT_WAISTBAND_CONCEAL,
            ActionType.BACK_WAISTBAND_CONCEAL,
        ]
        
        concealment_count = sum(
            1 for action in recent_actions 
            if action in concealment_indicators
        )
        
        if concealment_count >= SystemConfig.MIN_CONCEALMENT_GESTURES:
            score = min(1.0, concealment_count / 4.0)
            person_state.increment_behavioral_flag('concealment_gestures')
            return score
        elif concealment_count == 1:
            return 0.3

        return 0.0

    def analyze_exit_behavior(self, person_state: PersonState,
                              current_frame: int,
                              frame_height: int,
                              frame_width: int) -> float:
        """Detect suspicious exit behavior"""
        if not SystemConfig.EXIT_DETECTION_ENABLED:
            return 0.0
        
        if len(person_state.location_history) < 20:
            return 0.0
        
        recent_locs = list(person_state.location_history)[-20:]
        
        x_coords = [loc[0] for loc in recent_locs]
        y_coords = [loc[1] for loc in recent_locs]
        
        if len(x_coords) >= 10:
            x_trend = np.polyfit(range(len(x_coords)), x_coords, 1)[0]
            
            exit_zone_width = frame_width * SystemConfig.EXIT_ZONE_RATIO
            exit_zone_height = frame_height * SystemConfig.EXIT_ZONE_RATIO
            
            near_edge = (
                x_coords[-1] < exit_zone_width or 
                x_coords[-1] > frame_width - exit_zone_width or
                y_coords[-1] < exit_zone_height or 
                y_coords[-1] > frame_height - exit_zone_height
            )
            
            moving_to_exit = abs(x_trend) > SystemConfig.EXIT_VELOCITY_THRESHOLD
            
            recent_actions = person_state.get_recent_actions(current_frame, window=90)
            has_concealment = any(
                a in recent_actions for a in [
                    ActionType.POCKET_TOUCH,
                    ActionType.WAISTBAND_CONCEAL,
                    ActionType.INSIDE_CLOTHING_CONCEAL,
                    ActionType.CONCEALING_ITEM
                ]
            )
            
            if near_edge and moving_to_exit and has_concealment:
                person_state.increment_behavioral_flag('exit_behavior')
                return 0.8
            elif near_edge and has_concealment:
                return 0.5
        
        return 0.0

    def calculate_behavior_suspicion_score(self, person_state: PersonState,
                                          current_frame: int,
                                          frame_height: int = 720,
                                          frame_width: int = 1280) -> float:
        """Calculate overall behavioral suspicion score"""
        person_id = person_state.person_id
        
        loitering = self.analyze_loitering_pattern(person_state, current_frame)
        nervous = self.analyze_nervous_behavior(person_state, current_frame)
        movement = self.analyze_suspicious_movement(person_state, current_frame)
        concealment = self.analyze_concealment_gestures(person_state, current_frame)
        exit_behavior = self.analyze_exit_behavior(
            person_state, current_frame, frame_height, frame_width
        )

        behavior_score = (
            loitering * SystemConfig.BEHAVIOR_LOITERING_WEIGHT +
            nervous * SystemConfig.BEHAVIOR_NERVOUS_WEIGHT +
            movement * SystemConfig.BEHAVIOR_MOVEMENT_WEIGHT +
            concealment * SystemConfig.BEHAVIOR_CONCEALMENT_WEIGHT +
            exit_behavior * 0.15
        )

        self.behavior_scores[person_id].update({
            'loitering_score': loitering,
            'nervous_behavior_score': nervous,
            'suspicious_movement_score': movement,
            'concealment_gesture_score': concealment,
            'exit_behavior_score': exit_behavior,
            'total_suspicion_score': behavior_score
        })

        return behavior_score


# =============================================================================
# SEQUENCE VALIDATOR (v4.1)
# =============================================================================

class TheftSequenceValidator:
    """Validates complete theft sequences"""
    
    def validate_theft_sequence_with_product(self, person_state: PersonState,
                                            current_frame: int) -> Tuple[bool, float, str]:
        """Validate complete theft sequence WITH product evidence"""
        if not SystemConfig.SEQUENCE_VALIDATION_ENABLED:
            return False, 0.0, "validation_disabled"
        
        recent_actions = person_state.get_recent_actions(
            current_frame, 
            window=SystemConfig.SEQUENCE_WINDOW_FRAMES
        )
        
        has_grab = ActionType.GRAB_MOTION_DETECTED in recent_actions
        
        has_conceal_gesture = any(
            a in recent_actions for a in [
                ActionType.POCKET_TOUCH,
                ActionType.WAISTBAND_CONCEAL,
                ActionType.INSIDE_CLOTHING_CONCEAL,
                ActionType.CONCEALING_ITEM
            ]
        )
        
        has_disappearance = ActionType.ITEM_DISAPPEARED in recent_actions
        
        if has_grab and has_conceal_gesture and has_disappearance:
            return True, 0.95, "complete_theft_sequence"
        
        if has_grab and has_conceal_gesture:
            return True, 0.75, "partial_theft_sequence"
        
        if person_state.detect_concealment_pattern(current_frame):
            return True, 0.85, "concealment_pattern_detected"
        
        return False, 0.0, "incomplete_sequence"
    
    def validate_behavioral_sequence(self, person_state: PersonState,
                                    current_frame: int) -> Tuple[bool, float, str]:
        """Validate suspicious behavior sequence WITHOUT product"""
        if not SystemConfig.SEQUENCE_VALIDATION_ENABLED:
            return False, 0.0, "validation_disabled"
        
        recent_actions = person_state.get_recent_actions(
            current_frame, 
            window=SystemConfig.SEQUENCE_WINDOW_FRAMES
        )
        
        concealment_count = sum(
            1 for a in recent_actions 
            if a in [
                ActionType.POCKET_TOUCH,
                ActionType.WAISTBAND_CONCEAL,
                ActionType.INSIDE_CLOTHING_CONCEAL,
                ActionType.FRONT_POCKET_CONCEAL,
                ActionType.BACK_POCKET_CONCEAL
            ]
        )
        
        has_nervous = ActionType.NERVOUS_BEHAVIOR in recent_actions
        has_loitering = person_state.is_loitering(current_frame)
        has_exit = ActionType.EXIT_BEHAVIOR in recent_actions
        
        if concealment_count >= 3 and has_nervous and (has_loitering or has_exit):
            return True, SystemConfig.BEHAVIORAL_SEQUENCE_THRESHOLD, "strong_behavioral_pattern"
        
        if concealment_count >= 2 and (has_nervous or has_loitering):
            return True, 0.60, "moderate_behavioral_pattern"
        
        return False, 0.0, "insufficient_behavioral_evidence"


# =============================================================================
# SEQUENTIAL GRAB MOTION ANALYZER (v3.1 - Original)
# =============================================================================

class SequentialGrabMotionAnalyzer:
    """Sequential grab motion detection with pattern validation"""

    def __init__(self):
        self.hand_trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=SystemConfig.GRAB_MOTION_WINDOW)
        )

        self.grab_states: Dict[int, Dict] = defaultdict(lambda: {
            'current_phase': 'idle',
            'phase_start_frame': 0,
            'extend_count': 0,
            'pause_count': 0,
            'retract_count': 0,
            'total_grab_detections': 0,
            'false_positive_count': 0,
            'last_detection_frame': 0,
            'pause_location': None,
            'extend_direction': None,
        })

    def update_trajectory(self, person_id: int, frame_num: int,
                         left_wrist: np.ndarray, right_wrist: np.ndarray):
        """Record wrist positions"""
        self.hand_trajectories[person_id].append({
            'frame': frame_num,
            'left': left_wrist[:2] if left_wrist[2] > 0.3 else None,
            'right': right_wrist[:2] if right_wrist[2] > 0.3 else None,
            'timestamp': frame_num
        })

    def detect_sequential_grab(self, person_id: int, current_frame: int,
                               has_item_nearby: bool = False) -> Tuple[bool, float, Dict]:
        """Sequential pattern detection with strict validation"""
        trajectory = list(self.hand_trajectories.get(person_id, []))
        state = self.grab_states[person_id]

        if len(trajectory) < 20:
            return False, 0.0, {'reason': 'insufficient_trajectory'}

        try:
            movements = []
            positions = []

            for i in range(1, len(trajectory)):
                prev = trajectory[i-1]
                curr = trajectory[i]

                prev_pos = prev.get('left') if prev.get('left') is not None else prev.get('right')
                curr_pos = curr.get('left') if curr.get('left') is not None else curr.get('right')

                if prev_pos is not None and curr_pos is not None:
                    movement = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                    direction = np.array(curr_pos) - np.array(prev_pos)

                    movements.append({
                        'speed': movement,
                        'direction': direction,
                        'position': curr_pos,
                        'frame': curr['frame']
                    })
                    positions.append(curr_pos)

            if len(movements) < 15:
                return False, 0.0, {'reason': 'insufficient_movements'}

            phases = self._identify_phases_sequential(movements)
            has_valid_sequence = self._validate_grab_sequence(phases, state)

            if not has_valid_sequence:
                return False, 0.0, {'reason': 'invalid_sequence', 'phases': phases}

            pause_location = self._get_pause_location(phases, positions)
            if pause_location is not None:
                body_center = np.array(positions[0])
                pause_distance = np.linalg.norm(np.array(pause_location) - body_center)

                if pause_distance < 50:
                    state['false_positive_count'] += 1
                    return False, 0.0, {'reason': 'pause_too_close_to_body'}

            if not has_item_nearby:
                confidence_penalty = 0.3
            else:
                confidence_penalty = 0.0

            frames_since_last = current_frame - state['last_detection_frame']
            if frames_since_last < SystemConfig.GRAB_COOLDOWN_FRAMES:
                return False, 0.0, {'reason': 'cooldown_active'}

            base_confidence = self._calculate_pattern_confidence(phases)

            if state['total_grab_detections'] > 2:
                base_confidence *= 1.2

            if state['false_positive_count'] > 3:
                base_confidence *= 0.7

            final_confidence = base_confidence - confidence_penalty
            final_confidence = max(0.0, min(1.0, final_confidence))

            if final_confidence >= SystemConfig.GRAB_CONFIDENCE_THRESHOLD:
                state['total_grab_detections'] += 1
                state['last_detection_frame'] = current_frame

                logger.warning(
                    f"⚡ Person {person_id}: SEQUENTIAL GRAB DETECTED! "
                    f"(confidence: {final_confidence:.2f}, "
                    f"total_grabs: {state['total_grab_detections']}, "
                    f"fps: {state['false_positive_count']})"
                )

                debug_info = {
                    'phases': phases,
                    'confidence': final_confidence,
                    'total_grabs': state['total_grab_detections'],
                    'false_positives': state['false_positive_count'],
                    'has_item': has_item_nearby
                }

                return True, final_confidence, debug_info

            return False, final_confidence, {'reason': 'below_threshold', 'confidence': final_confidence}

        except Exception as e:
            logger.debug(f"Sequential grab detection error: {e}")
            return False, 0.0, {'reason': 'error', 'error': str(e)}

    def _identify_phases_sequential(self, movements: List[Dict]) -> Dict:
        """Identify movement phases"""
        fast_threshold = SystemConfig.FAST_MOTION_THRESHOLD
        pause_threshold = SystemConfig.PAUSE_MOTION_THRESHOLD

        phases = {
            'extend': [],
            'pause': [],
            'retract': [],
            'has_sequence': False
        }

        current_phase = None
        phase_start = 0

        for i, mov in enumerate(movements):
            speed = mov['speed']

            if speed > fast_threshold:
                if current_phase != 'fast':
                    if current_phase == 'pause' and phase_start > 0:
                        phases['retract'].append({
                            'start': i,
                            'frames': [],
                            'avg_speed': speed
                        })
                    elif current_phase is None or current_phase == 'slow':
                        phases['extend'].append({
                            'start': i,
                            'frames': [],
                            'avg_speed': speed
                        })

                    current_phase = 'fast'
                    phase_start = i

                if len(phases['extend']) > 0 and len(phases['pause']) == 0:
                    phases['extend'][-1]['frames'].append(i)
                elif len(phases['retract']) > 0:
                    phases['retract'][-1]['frames'].append(i)

            elif speed < pause_threshold:
                if current_phase != 'pause':
                    if len(phases['extend']) > 0 and len(phases['pause']) == 0:
                        phases['pause'].append({
                            'start': i,
                            'frames': [],
                            'position': mov['position']
                        })

                    current_phase = 'pause'
                    phase_start = i

                if len(phases['pause']) > 0:
                    phases['pause'][-1]['frames'].append(i)

            else:
                current_phase = 'slow'

        return phases

    def _validate_grab_sequence(self, phases: Dict, state: Dict) -> bool:
        """Validate complete grab sequence"""
        min_extend_frames = SystemConfig.MIN_EXTEND_FRAMES
        min_pause_frames = SystemConfig.MIN_PAUSE_FRAMES
        min_retract_frames = SystemConfig.MIN_RETRACT_FRAMES

        if not phases['extend'] or len(phases['extend'][0]['frames']) < min_extend_frames:
            return False

        if not phases['pause'] or len(phases['pause'][0]['frames']) < min_pause_frames:
            return False

        if not phases['retract'] or len(phases['retract'][0]['frames']) < min_retract_frames:
            return False

        extend_start = phases['extend'][0]['start']
        pause_start = phases['pause'][0]['start']
        retract_start = phases['retract'][0]['start']

        if not (extend_start < pause_start < retract_start):
            return False

        state['extend_count'] += 1
        state['pause_count'] += 1
        state['retract_count'] += 1

        return True

    def _get_pause_location(self, phases: Dict, positions: List) -> Optional[np.ndarray]:
        """Get the location where hand paused"""
        if not phases['pause']:
            return None

        pause_phase = phases['pause'][0]
        if 'position' in pause_phase:
            return np.array(pause_phase['position'])

        return None

    def _calculate_pattern_confidence(self, phases: Dict) -> float:
        """Calculate confidence based on pattern quality"""
        confidence = 0.4

        if phases['extend']:
            extend_frames = len(phases['extend'][0]['frames'])
            confidence += min(0.2, extend_frames * 0.04)

        if phases['pause']:
            pause_frames = len(phases['pause'][0]['frames'])
            confidence += min(0.2, pause_frames * 0.06)

        if phases['retract']:
            retract_frames = len(phases['retract'][0]['frames'])
            confidence += min(0.2, retract_frames * 0.04)

        return min(1.0, confidence)


# =============================================================================
# ENHANCED KEYPOINT ANALYZER (v4.1)
# =============================================================================

class EnhancedKeypointAnalyzerV4:
    """Master keypoint analyzer integrating all detection systems"""

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.concealment_detector = BodyRegionConcealmentDetector(confidence_threshold)
        self.behavior_analyzer = BehaviorPatternAnalyzer()
        self.sequence_validator = TheftSequenceValidator()

    def detect_orientation(self, keypoints: np.ndarray,
                          history: Optional[List] = None) -> PersonOrientation:
        """Detect orientation with temporal smoothing"""
        if keypoints is None or len(keypoints) != 17:
            return PersonOrientation.UNKNOWN

        current_orientation = self._detect_orientation_single_frame(keypoints)

        if history and len(history) >= 3:
            recent_orientations = [h[0] for h in history[-3:]]
            recent_orientations.append(current_orientation)

            orientation_counts = Counter(recent_orientations)
            smoothed_orientation = orientation_counts.most_common(1)[0][0]
            return smoothed_orientation

        return current_orientation

    def _detect_orientation_single_frame(self, keypoints: np.ndarray) -> PersonOrientation:
        """Single frame orientation detection"""
        try:
            nose = keypoints[KEYPOINT["nose"]]
            left_eye = keypoints[KEYPOINT["left_eye"]]
            right_eye = keypoints[KEYPOINT["right_eye"]]
            left_ear = keypoints[KEYPOINT["left_ear"]]
            right_ear = keypoints[KEYPOINT["right_ear"]]
            left_shoulder = keypoints[KEYPOINT["left_shoulder"]]
            right_shoulder = keypoints[KEYPOINT["right_shoulder"]]

            def is_visible(kp):
                return kp[2] > self.confidence_threshold

            nose_vis = is_visible(nose)
            left_eye_vis = is_visible(left_eye)
            right_eye_vis = is_visible(right_eye)
            left_ear_vis = is_visible(left_ear)
            right_ear_vis = is_visible(right_ear)
            left_shoulder_vis = is_visible(left_shoulder)
            right_shoulder_vis = is_visible(right_shoulder)

            if left_shoulder_vis and right_shoulder_vis:
                shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                nose_offset = nose[0] - shoulder_center_x if nose_vis else 0
            else:
                shoulder_width = 0
                nose_offset = 0

            if (left_eye_vis and right_eye_vis and left_shoulder_vis and right_shoulder_vis
                and nose_vis and abs(nose_offset) < shoulder_width * 0.3):
                return PersonOrientation.FRONT

            if (not nose_vis and not left_eye_vis and not right_eye_vis
                and (left_ear_vis or right_ear_vis) and left_shoulder_vis and right_shoulder_vis):
                return PersonOrientation.BACK

            if right_shoulder_vis and not left_shoulder_vis:
                return PersonOrientation.FRONT_LEFT if (right_eye_vis or nose_vis) else PersonOrientation.BACK_LEFT

            if left_shoulder_vis and not right_shoulder_vis:
                return PersonOrientation.FRONT_RIGHT if (left_eye_vis or nose_vis) else PersonOrientation.BACK_RIGHT

            if left_shoulder_vis and right_shoulder_vis and shoulder_width < 50:
                if left_ear_vis or left_eye_vis:
                    return PersonOrientation.LEFT
                elif right_ear_vis or right_eye_vis:
                    return PersonOrientation.RIGHT

            return PersonOrientation.UNKNOWN

        except Exception as e:
            logger.debug(f"Orientation error: {e}")
            return PersonOrientation.UNKNOWN

    def is_reaching_gesture(self, keypoints: np.ndarray, orientation: PersonOrientation,
                           person_height: float) -> Tuple[bool, float]:
        """Detect reaching gesture"""
        if keypoints is None or len(keypoints) != 17:
            return False, 0.0

        try:
            left_shoulder = keypoints[KEYPOINT["left_shoulder"]]
            right_shoulder = keypoints[KEYPOINT["right_shoulder"]]
            left_elbow = keypoints[KEYPOINT["left_elbow"]]
            right_elbow = keypoints[KEYPOINT["right_elbow"]]
            left_wrist = keypoints[KEYPOINT["left_wrist"]]
            right_wrist = keypoints[KEYPOINT["right_wrist"]]

            left_arm_vis = all(kp[2] > self.confidence_threshold
                              for kp in [left_shoulder, left_elbow, left_wrist])
            right_arm_vis = all(kp[2] > self.confidence_threshold
                               for kp in [right_shoulder, right_elbow, right_wrist])

            if not (left_arm_vis or right_arm_vis):
                return False, 0.0

            extension_threshold = person_height * 0.35
            max_confidence = 0.0

            if left_arm_vis:
                left_extension = np.linalg.norm(left_wrist[:2] - left_shoulder[:2])
                if left_extension > extension_threshold:
                    confidence = min(1.0, left_extension / (extension_threshold * 1.5))
                    max_confidence = max(max_confidence, confidence)

            if right_arm_vis:
                right_extension = np.linalg.norm(right_wrist[:2] - right_shoulder[:2])
                if right_extension > extension_threshold:
                    confidence = min(1.0, right_extension / (extension_threshold * 1.5))
                    max_confidence = max(max_confidence, confidence)

            return max_confidence > 0.0, max_confidence

        except Exception as e:
            logger.debug(f"Reaching gesture error: {e}")
            return False, 0.0

    def is_container_manipulation(self, keypoints: np.ndarray, orientation: PersonOrientation,
                                   person_height: float) -> Tuple[bool, float]:
        """Detect container manipulation"""
        if keypoints is None or len(keypoints) != 17:
            return False, 0.0

        try:
            left_wrist = keypoints[KEYPOINT["left_wrist"]]
            right_wrist = keypoints[KEYPOINT["right_wrist"]]
            left_shoulder = keypoints[KEYPOINT["left_shoulder"]]
            right_shoulder = keypoints[KEYPOINT["right_shoulder"]]

            if not all(kp[2] > self.confidence_threshold
                      for kp in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
                return False, 0.0

            torso_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            torso_center_y = (left_shoulder[1] + right_shoulder[1]) / 2

            left_dist = np.sqrt((left_wrist[0] - torso_center_x)**2 +
                               (left_wrist[1] - torso_center_y)**2)
            right_dist = np.sqrt((right_wrist[0] - torso_center_x)**2 +
                                (right_wrist[1] - torso_center_y)**2)

            threshold = person_height * 0.25

            if left_dist < threshold and right_dist < threshold:
                confidence = 1.0 - ((left_dist + right_dist) / (2 * threshold))
                return True, confidence

            return False, 0.0

        except Exception as e:
            logger.debug(f"Container manipulation error: {e}")
            return False, 0.0

    def is_looking_around(self, keypoints: np.ndarray,
                         previous_keypoints: Optional[np.ndarray]) -> Tuple[bool, float]:
        """Detect looking around (head movement)"""
        if keypoints is None or previous_keypoints is None:
            return False, 0.0

        try:
            nose_current = keypoints[KEYPOINT["nose"]]
            nose_previous = previous_keypoints[KEYPOINT["nose"]]

            if (nose_current[2] < self.confidence_threshold or
                nose_previous[2] < self.confidence_threshold):
                return False, 0.0

            head_movement = abs(nose_current[0] - nose_previous[0])

            if head_movement > 15:
                confidence = min(1.0, head_movement / 45.0)
                return True, confidence

            return False, 0.0

        except Exception as e:
            logger.debug(f"Looking around error: {e}")
            return False, 0.0

    def detect_all_concealment_behaviors(self, person_id: int,
                                        frame_num: int,
                                        keypoints: np.ndarray,
                                        person_height: float,
                                        orientation: PersonOrientation,
                                        has_bag: bool,
                                        previous_keypoints: Optional[np.ndarray] = None,
                                        has_item_context: bool = False,
                                        grab_motion_active: bool = False) -> List[Tuple[ActionType, str, float]]:
        """Comprehensive concealment detection"""
        return self.concealment_detector.analyze_concealment_sequence(
            person_id, frame_num, keypoints, person_height, orientation, has_bag,
            previous_keypoints, has_item_context, grab_motion_active
        )

    def calculate_behavioral_risk(self, person_state: PersonState,
                                  current_frame: int,
                                  frame_height: int = 720,
                                  frame_width: int = 1280) -> float:
        """Calculate behavioral risk score"""
        return self.behavior_analyzer.calculate_behavior_suspicion_score(
            person_state, current_frame, frame_height, frame_width
        )


# =============================================================================
# FALSE POSITIVE REDUCTION & BEHAVIOR ANALYZER
# =============================================================================

class EnhancedFPReduction:
    """Enhanced false positive reduction"""

    def __init__(self):
        self.alert_history: Dict[int, List[Dict]] = defaultdict(list)

    def validate_alert(self, person_state: PersonState, alert_data: Dict,
                      current_frame: int) -> Tuple[bool, float, List[str]]:
        """Multi-stage validation"""
        rejection_reasons = []
        confidence_score = alert_data.get('attention_score', 0.0)

        if person_state.has_shopping_cart:
            rejection_reasons.append("Person has shopping cart (legitimate shopper)")
            return False, 0.0, rejection_reasons

        if not self._check_alert_cooldown(person_state, current_frame):
            rejection_reasons.append("Alert cooldown active")
            return False, 0.0, rejection_reasons

        if person_state.time_in_store < 60:
            rejection_reasons.append("Insufficient observation time")
            confidence_score *= 0.5

        if person_state.detect_concealment_pattern(current_frame):
            confidence_score *= 1.8
            logger.info(f"Person {person_state.person_id}: Concealment pattern confirmed!")
        else:
            if ActionType.CONCEALING_ITEM in person_state.get_recent_actions(current_frame, window=60):
                if not person_state.has_container():
                    confidence_score *= 0.4
                    rejection_reasons.append("Concealment detected but no container visible")

        recent_actions = person_state.get_recent_actions(current_frame, window=90)
        grab_count = recent_actions.count(ActionType.GRAB_MOTION_DETECTED)
        
        concealment_pattern = person_state.get_concealment_pattern(current_frame, window=90)
        concealment_count = concealment_pattern['total']

        if grab_count > 0 or concealment_count > 0:
            if grab_count >= 1 and concealment_count >= 2:
                confidence_score *= 1.7
                logger.warning(
                    f"Person {person_state.person_id}: GRAB + CONCEALMENT "
                    f"(grabs: {grab_count}, concealments: {concealment_count})"
                )
            elif grab_count >= 3:
                confidence_score *= 1.5
                logger.warning(f"Person {person_state.person_id}: MULTIPLE GRABS ({grab_count})")
            elif concealment_count >= 3:
                if concealment_pattern['has_product']:
                    confidence_score *= 1.4
                else:
                    confidence_score *= 1.2
                    logger.info(
                        f"Person {person_state.person_id}: Multiple concealments "
                        f"without product ({concealment_count})"
                    )
            elif grab_count == 1 and concealment_count == 0:
                has_other_signals = any(a in recent_actions for a in [
                    ActionType.OPENING_CONTAINER,
                    ActionType.ITEM_DISAPPEARED
                ])
                if not has_other_signals:
                    confidence_score *= 0.6
                    rejection_reasons.append("Single grab without supporting evidence")

        if not self._check_behavioral_consistency(person_state, current_frame):
            rejection_reasons.append("Inconsistent behavior pattern")
            confidence_score *= 0.7
        else:
            confidence_score *= 1.2

        smoothed_score = self._apply_temporal_smoothing(person_state, confidence_score)

        if self._validate_theft_sequence(person_state, current_frame):
            smoothed_score *= 1.3

        if person_state.is_loitering(current_frame):
            smoothed_score *= 1.1

        threshold = SystemConfig.ALERT_THRESHOLD
        should_alert = smoothed_score >= threshold

        if not should_alert:
            rejection_reasons.append(f"Score below threshold ({smoothed_score:.2f} < {threshold})")

        return should_alert, smoothed_score, rejection_reasons

    def _check_alert_cooldown(self, person_state: PersonState, current_frame: int,
                             cooldown_frames: int = None) -> bool:
        """Prevent alert spam"""
        if cooldown_frames is None:
            cooldown_frames = SystemConfig.ALERT_COOLDOWN_FRAMES

        if person_state.last_alert_frame == 0:
            return True
        return (current_frame - person_state.last_alert_frame) >= cooldown_frames

    def _check_behavioral_consistency(self, person_state: PersonState,
                                     current_frame: int) -> bool:
        """Check if behavior is consistent with theft"""
        recent_actions = person_state.get_recent_actions(current_frame, window=100)

        suspicious_actions = [
            ActionType.CONCEALING_ITEM,
            ActionType.LOOKING_AROUND,
            ActionType.ITEM_DISAPPEARED,
            ActionType.OPENING_CONTAINER,
            ActionType.POCKET_TOUCH,
            ActionType.WAISTBAND_CONCEAL,
            ActionType.GRAB_MOTION_DETECTED,
            ActionType.INSIDE_CLOTHING_CONCEAL,
            ActionType.FRONT_POCKET_CONCEAL,
            ActionType.BACK_POCKET_CONCEAL,
        ]

        suspicious_count = sum(1 for action in recent_actions if action in suspicious_actions)

        normal_actions = [
            ActionType.BROWSING,
            ActionType.EXAMINING_ITEM,
            ActionType.REPLACING_ITEM
        ]

        normal_count = sum(1 for action in recent_actions if action in normal_actions)

        return suspicious_count >= normal_count

    def _apply_temporal_smoothing(self, person_state: PersonState,
                                 current_score: float) -> float:
        """Apply temporal smoothing to reduce jitter"""
        person_state.attention_score_history.append(current_score)

        if len(person_state.attention_score_history) < 5:
            return current_score

        history_array = np.array(list(person_state.attention_score_history))
        smoothed_array = gaussian_filter1d(history_array, sigma=2)

        return float(smoothed_array[-1])

    def _validate_theft_sequence(self, person_state: PersonState,
                                current_frame: int) -> bool:
        """Enhanced theft sequence validation"""
        recent_actions = person_state.get_recent_actions(current_frame, window=120)

        has_pick = ActionType.PICKING_ITEM in recent_actions
        has_grab = ActionType.GRAB_MOTION_DETECTED in recent_actions
        has_container_open = ActionType.OPENING_CONTAINER in recent_actions
        
        has_conceal = (
            ActionType.CONCEALING_ITEM in recent_actions or
            ActionType.ITEM_DISAPPEARED in recent_actions or
            ActionType.POCKET_TOUCH in recent_actions or
            ActionType.WAISTBAND_CONCEAL in recent_actions or
            ActionType.INSIDE_CLOTHING_CONCEAL in recent_actions
        )

        if (has_pick or has_grab) and has_conceal:
            return True
        
        if has_pick and has_container_open and has_conceal:
            return True

        return False

    def deduplicate_alerts(self, alerts: List[Dict], time_window: int = 5) -> List[Dict]:
        """Remove duplicate alerts within time window"""
        deduplicated = []
        seen_persons = set()

        for alert in alerts:
            person_id = alert['person_id']
            if person_id not in seen_persons:
                deduplicated.append(alert)
                seen_persons.add(person_id)

        return deduplicated


class BehaviorAnalyzer:
    """Enhanced behavior analyzer with concealment and sequence validation"""

    def __init__(self):
        self.fp_reducer = EnhancedFPReduction()
        self.grab_analyzer = SequentialGrabMotionAnalyzer()

    def calculate_attention_score(self, state: PersonState, current_frame: int) -> float:
        """Calculate attention score with enhanced grab motion + concealment scoring"""
        score = 0.0
        recent_actions = state.get_recent_actions(current_frame, window=150)

        if state.detect_concealment_pattern(current_frame):
            score += 0.6
            logger.info(f"Person {state.person_id}: Concealment pattern confirmed!")

        if ActionType.PICKING_ITEM in recent_actions:
            if any(item.location == ItemLocation.HAND
                  for item in state.item_interactions.values()):
                score += 0.2
            else:
                score += 0.1

        if ActionType.OPENING_CONTAINER in recent_actions:
            score += 0.2
            if ActionType.PICKING_ITEM in recent_actions:
                score += 0.15

        if ActionType.ITEM_DISAPPEARED in recent_actions:
            score += 0.3

        if ActionType.CONCEALING_ITEM in recent_actions:
            score += 0.2

        # Enhanced concealment location analysis
        concealment_pattern = state.get_concealment_pattern(current_frame, window=60)
        
        if concealment_pattern['total'] > 0:
            base_concealment_score = 0.3
            
            if len(concealment_pattern['locations']) >= 2:
                base_concealment_score *= 1.5
                logger.warning(
                    f"Person {state.person_id}: Multiple concealment locations "
                    f"({list(concealment_pattern['locations'].keys())})"
                )
            
            base_concealment_score *= concealment_pattern['avg_confidence']
            
            if not concealment_pattern['has_product']:
                base_concealment_score *= 1.3
                logger.warning(
                    f"Person {state.person_id}: Concealment gestures WITHOUT visible product! "
                    f"Locations: {concealment_pattern['locations']}"
                )
            
            score += base_concealment_score
        
        # Specific body concealment actions
        if ActionType.INSIDE_CLOTHING_CONCEAL in recent_actions:
            score += 0.4
        if ActionType.WAISTBAND_CONCEAL in recent_actions:
            score += 0.35
        if ActionType.BACK_POCKET_CONCEAL in recent_actions:
            score += 0.3
        if ActionType.FRONT_POCKET_CONCEAL in recent_actions:
            score += 0.25

        if ActionType.POCKET_TOUCH in recent_actions:
            score += 0.25

        looking_count = recent_actions.count(ActionType.LOOKING_AROUND)
        if looking_count >= 5:
            score += 0.25
        elif looking_count >= 3:
            score += 0.15

        if state.is_loitering(current_frame):
            score += 0.1

        # Enhanced Grab Motion Scoring with Concealment
        grab_count = recent_actions.count(ActionType.GRAB_MOTION_DETECTED)
        
        if grab_count > 0:
            concealment_actions = [
                ActionType.POCKET_TOUCH,
                ActionType.WAISTBAND_CONCEAL,
                ActionType.INSIDE_CLOTHING_CONCEAL,
                ActionType.FRONT_POCKET_CONCEAL,
                ActionType.BACK_POCKET_CONCEAL,
            ]
            
            concealment_count = sum(1 for a in recent_actions if a in concealment_actions)
            
            base_grab_score = 0.35
            
            if grab_count >= 1 and concealment_count >= 2:
                grab_score = base_grab_score * 2.0
                logger.critical(
                    f"Person {state.person_id}: GRAB + CONCEALMENT DETECTED! "
                    f"(grabs: {grab_count}, concealments: {concealment_count})"
                )
            elif grab_count >= 3:
                grab_score = base_grab_score * 1.5
                logger.warning(f"Person {state.person_id}: MULTIPLE GRABS ({grab_count})")
            elif grab_count >= 2:
                grab_score = base_grab_score * 1.2
            else:
                has_item_evidence = (
                    ActionType.ITEM_DISAPPEARED in recent_actions or
                    ActionType.PICKING_ITEM in recent_actions or
                    len(state.item_interactions) > 0
                )
                
                if has_item_evidence or concealment_count > 0:
                    grab_score = base_grab_score
                else:
                    grab_score = base_grab_score * 0.6
                    logger.debug(
                        f"Person {state.person_id}: Single grab without evidence"
                    )
            
            score += grab_score

        # Behavioral Suspicion (No Product)
        if state.behavioral_suspicion_score > 0.5:
            behavioral_bonus = state.behavioral_suspicion_score * 0.35
            score += behavioral_bonus
            
            if state.behavioral_flags['loitering'] > 3:
                logger.debug(f"Person {state.person_id}: Prolonged loitering detected")
            if state.behavioral_flags['nervous'] > 3:
                logger.debug(f"Person {state.person_id}: Nervous behavior detected")
            if state.behavioral_flags['erratic_movement'] > 2:
                logger.debug(f"Person {state.person_id}: Erratic movement detected")
            if state.behavioral_flags['exit_behavior'] > 0:
                logger.info(f"Person {state.person_id}: Exit behavior detected")

        score = max(0.0, min(1.0, score))
        return score

    def determine_alert_level(self, score: float) -> AlertLevel:
        """Determine alert level from score"""
        threshold = SystemConfig.ALERT_THRESHOLD

        if score >= threshold + 0.1:
            return AlertLevel.ALERT
        elif score >= threshold - 0.05:
            return AlertLevel.ATTENTION_PLUS
        elif score >= threshold - 0.2:
            return AlertLevel.ATTENTION
        else:
            return AlertLevel.NEUTRAL

    def generate_alert_message(self, alert_level: AlertLevel, reasons: List[str]) -> str:
        """Generate human-readable alert message"""
        if alert_level == AlertLevel.ALERT:
            return f"🚨 THEFT ALERT: {', '.join(reasons[:3]).upper()}"
        elif alert_level == AlertLevel.ATTENTION_PLUS:
            return f"⚠️ HIGH SUSPICION: {', '.join(reasons[:2])}"
        elif alert_level == AlertLevel.ATTENTION:
            return f"⚡ SUSPICIOUS: {', '.join(reasons[:2])}"
        else:
            return "Normal"
