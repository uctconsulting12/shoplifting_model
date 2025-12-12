"""
inference.py - Production Inference Module for Theft Detection System v4.1 (COMPLETE)
====================================================================================
Provides complete inference pipeline for API deployment with exact I/O format
"""

import sys
import base64
import json
import io
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import torch
from collections import deque

from  src.local_models.shop_lifting.config import (
    PersonState, PersonOrientation, ActionType, AlertLevel, ItemLocation,
    ContainerType, SystemConfig, MetricsTracker, StoreProfile, logger
)
from src.local_models.shop_lifting.analyzer import EnhancedKeypointAnalyzerV4, BehaviorAnalyzer
from src.local_models.shop_lifting.detector import PersonDetectorTracker, EnhancedContainerTracker, ItemTracker


# =============================================================================
# MODEL INITIALIZATION (model_fn)
# =============================================================================

class TheftDetectionModel:
    """Production-ready theft detection model with stateful tracking"""
    
    def __init__(self,
                 pose_model_path: str = None,
                 object_model_path: str = None,
                 device: str = None,
                 skip_frames: int = None,
                 store_profile: StoreProfile = StoreProfile.GENERAL_RETAIL):
        
        pose_model_path = pose_model_path or SystemConfig.POSE_MODEL_PATH
        object_model_path = object_model_path or SystemConfig.OBJECT_MODEL_PATH
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        skip_frames = skip_frames or SystemConfig.SKIP_FRAMES

        
        # Configure system for store type
        SystemConfig.configure_for_store(store_profile)

        self.device = device
        self.skip_frames = skip_frames
        self.process_counter = 0
        
        # Per-camera state management
        self.camera_states: Dict[int, Dict] = {}
        
        # Initialize core components
        self.detector_tracker = PersonDetectorTracker(pose_model_path, object_model_path, device)
        self.keypoint_analyzer = EnhancedKeypointAnalyzerV4()
        self.container_tracker_template = EnhancedContainerTracker
        self.item_tracker_template = ItemTracker
        self.behavior_analyzer_template = BehaviorAnalyzer
        
        self.config = SystemConfig.get_config_dict()
        
        # Performance stats
        self.performance_stats = {
            'total_frames_processed': 0,
            'avg_inference_time': 0.0,
            'total_alerts': 0
        }
        

    def get_or_create_camera_state(self, cam_id: int) -> Dict:
        """Get or initialize state for a specific camera"""
        if cam_id not in self.camera_states:
            self.camera_states[cam_id] = {
                'person_states': {},
                'frame_count': 0,
                'frame_height': None,
                'frame_width': None,
                'container_tracker': EnhancedContainerTracker(),
                'item_tracker': ItemTracker(),
                'behavior_analyzer': BehaviorAnalyzer(),
                'metrics_tracker': MetricsTracker(),
                'last_global_alert_frame': 0,
                'person_alert': {}
            }
        
        return self.camera_states[cam_id]
    
    def reset_camera_state(self, cam_id: int):
        """Reset state for a specific camera"""
        if cam_id in self.camera_states:
            del self.camera_states[cam_id]


def model_fn(model_dir: Optional[str] = None) -> TheftDetectionModel:
    """Load and initialize the model"""
    try:
        
        # Check dependencies
        required_modules = ['cv2', 'torch', 'ultralytics', 'scipy', 'numpy']
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            raise ImportError(f"Missing required modules: {', '.join(missing)}")
        
        # Initialize model
        model = TheftDetectionModel(
            pose_model_path=f"{model_dir}/yolov8n-pose.pt" if model_dir else None,
            object_model_path=f"{model_dir}/yolov8n.pt" if model_dir else None,
            store_profile=StoreProfile.GENERAL_RETAIL
        )
        
        return model
        
    except Exception as e:
        raise


# =============================================================================
# INPUT PROCESSING (input_fn)
# =============================================================================

def input_fn(request_body: str, content_type: str = 'application/json') -> Dict[str, Any]:
    """
    Parse and validate input request
    Expected format: {"cam_id": 123, "org_id": 2, "user_id": 2, "encoding": "base64_jpeg_data"}
    """
    try:
        if content_type != 'application/json':
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Parse JSON
        input_data = json.loads(request_body)
        
        # Validate required fields
        required_fields = ['cam_id', 'org_id', 'user_id', 'encoding']
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types
        if not isinstance(input_data['cam_id'], int):
            raise ValueError("cam_id must be an integer")
        if not isinstance(input_data['org_id'], int):
            raise ValueError("org_id must be an integer")
        if not isinstance(input_data['user_id'], int):
            raise ValueError("user_id must be an integer")
        
        # Decode frame from base64
        try:
            encoded_frame = input_data['encoding']
            
            # Remove data URI prefix if present
            if ',' in encoded_frame:
                encoded_frame = encoded_frame.split(',')[1]
            
            # Decode base64 to bytes
            frame_bytes = base64.b64decode(encoded_frame)
            
            # Convert to numpy array
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            
            # Decode image
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode frame image")
            
            input_data['frame'] = frame
            
        except Exception as e:
            raise ValueError(f"Failed to decode frame: {e}")
        
        return input_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise


# =============================================================================
# PREDICTION (predict_fn)
# =============================================================================

def predict_fn(input_data: Dict[str, Any], model: TheftDetectionModel) -> Dict[str, Any]:
    """Run inference on input frame"""
    inference_start = time.time()
    
    try:
        # Extract input fields
        cam_id = input_data['cam_id']
        org_id = input_data['org_id']
        user_id = input_data['user_id']
        frame = input_data['frame']
        
        # Get camera state
        cam_state = model.get_or_create_camera_state(cam_id)
        cam_state['frame_count'] += 1
        frame_count = cam_state['frame_count']
        
        # Initialize frame dimensions
        if cam_state['frame_height'] is None:
            cam_state['frame_height'], cam_state['frame_width'] = frame.shape[:2]
        
        # Generate frame_id: {CAMID}{USERID}{ORGID}{DDMMYYYY}{HHMMSS}
        now = datetime.now()
        frame_id = f"{cam_id}{user_id}{org_id}{now.strftime('%d%m%Y%H%M%S')}"
        
        results = {
            'cam_id': cam_id,
            'org_id': org_id,
            'user_id': user_id,
            'frame_id': frame_id,
            'timestamp': now.isoformat() + 'Z',
            'frame_num': frame_count,
            'persons': [],
            'alerts': [],
            'annotated_frame': None,
            'status': 1
        }
        
        # Process frame
        model.process_counter += 1
        
        # STEP 1: Detect persons with poses
        person_detections, person_poses = model.detector_tracker.detect_persons_with_poses(
            frame, model.config
        )
        
        # STEP 2: Detect objects
        object_detections = model.detector_tracker.detect_objects(frame, model.config)
        
        # STEP 3: Track persons
        tracked_persons = model.detector_tracker.track_persons(frame, person_detections)
        results['persons'] = list(tracked_persons.keys())
        
        # STEP 4: Detect and associate containers
        containers = cam_state['container_tracker'].detect_containers(
            object_detections, frame_count
        )
        container_associations = cam_state['container_tracker'].associate_containers_to_persons(
            containers, tracked_persons, frame_count
        )
        
        # STEP 5: Detect and associate items
        items = cam_state['item_tracker'].detect_items(
            object_detections, frame_count, cam_state['frame_height']
        )
        item_associations = cam_state['item_tracker'].associate_items_to_persons(
            items, tracked_persons
        )
        
        # STEP 6: Match poses to tracks
        matched_poses = model.detector_tracker.match_poses_to_tracks(tracked_persons, person_poses)
        
        # STEP 7: Detect orientations
        orientations = _detect_orientations(model, matched_poses, cam_state)
        
        # STEP 8: Recognize actions
        actions = _recognize_actions(
            model, cam_state, frame, tracked_persons, matched_poses,
            orientations, container_associations, item_associations, frame_count
        )
        
        # STEP 9: Update person states
        for person_id, person_bbox in tracked_persons.items():
            _update_person_state(
                cam_state, person_id, person_bbox,
                matched_poses.get(person_id),
                orientations.get(person_id, PersonOrientation.UNKNOWN),
                actions.get(person_id, []),
                container_associations.get(person_id, []),
                item_associations.get(person_id, []),
                frame_count
            )
        
        # STEP 10: Analyze behaviors and generate alerts
        raw_alerts = _analyze_behaviors(model, cam_state, frame_count)
        
        # STEP 11: Validate and format alerts
        validated_alerts = _validate_and_format_alerts(
            model, cam_state, raw_alerts, frame_count, cam_id, now
        )
        
        results['alerts'] = validated_alerts
        
        # STEP 12: Annotate frame
        annotated_frame = _visualize_frame(
            frame, tracked_persons, orientations, validated_alerts,
            container_associations, item_associations, cam_state
        )
        
        # Encode annotated frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        results['annotated_frame'] = frame_base64
        
        # Update performance stats
        inference_time = time.time() - inference_start
        model.performance_stats['total_frames_processed'] += 1
        model.performance_stats['avg_inference_time'] = (
            0.9 * model.performance_stats['avg_inference_time'] + 0.1 * inference_time
        )
        
        if len(validated_alerts) > 0:
            model.performance_stats['total_alerts'] += len(validated_alerts)
        
        
        return results
        
    except Exception as e:
        
        return {
            'cam_id': input_data.get('cam_id', -1),
            'org_id': input_data.get('org_id', -1),
            'user_id': input_data.get('user_id', -1),
            'frame_id': '',
            'timestamp': datetime.now().isoformat() + 'Z',
            'persons': [],
            'alerts': [],
            'annotated_frame': None,
            'status': 0,
            'error': str(e)
        }


# =============================================================================
# OUTPUT FORMATTING (output_fn)
# =============================================================================

def output_fn(prediction: Dict[str, Any], accept: str = 'application/json') -> str:
    """Format prediction output for API response"""
    try:
        if accept != 'application/json':
            raise ValueError(f"Unsupported accept type: {accept}")
        
        # Format output exactly as specified
        output = {
            'cam_id': prediction['cam_id'],
            'org_id': prediction['org_id'],
            'user_id': prediction['user_id'],
            'frame_id': prediction['frame_id'],
            'timestamp': prediction['timestamp'],
            'persons': prediction['persons'],
            'alerts': prediction['alerts'],
            'message': prediction.get('message', ''),
            'annotated_frame': prediction.get('annotated_frame', ''),
            'status': prediction['status']
        }
        
        # Add error field if present
        if 'error' in prediction:
            output['error'] = prediction['error']
        
        return json.dumps(output, indent=2)
        
    except Exception as e:
        
        error_response = {
            'status': 0,
            'error': str(e)
        }
        return json.dumps(error_response)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _detect_orientations(model: TheftDetectionModel,
                         matched_poses: Dict[int, np.ndarray],
                         cam_state: Dict) -> Dict[int, PersonOrientation]:
    """Detect person orientations"""
    orientations = {}
    person_states = cam_state['person_states']
    
    for person_id, keypoints in matched_poses.items():
        history = None
        if person_id in person_states:
            history = list(person_states[person_id].orientation_history)
        
        orientation = model.keypoint_analyzer.detect_orientation(keypoints, history)
        orientations[person_id] = orientation
    
    return orientations


def _recognize_actions(model: TheftDetectionModel,
                       cam_state: Dict,
                       frame: np.ndarray,
                       tracked_persons: Dict,
                       poses: Dict,
                       orientations: Dict,
                       container_associations: Dict,
                       item_associations: Dict,
                       frame_count: int) -> Dict[int, List[ActionType]]:
    """Recognize actions for all tracked persons"""
    actions = {}
    person_states = cam_state['person_states']
    behavior_analyzer = cam_state['behavior_analyzer']
    
    for person_id, person_bbox in tracked_persons.items():
        person_actions = []
        
        try:
            state = person_states.get(person_id)
            pose = poses.get(person_id)
            orientation = orientations.get(person_id, PersonOrientation.UNKNOWN)
            person_containers = container_associations.get(person_id, [])
            person_items = item_associations.get(person_id, [])
            
            if pose is not None and state is not None:
                person_height = state.get_person_height()
                has_bag = any(
                    c.container_type != ContainerType.SHOPPING_CART
                    for c in state.containers.values()
                )
                
                # Gesture detection
                reaching_detected, reaching_conf = model.keypoint_analyzer.is_reaching_gesture(
                    pose, orientation, person_height
                )
                state.add_gesture('reaching', frame_count, reaching_detected, reaching_conf)
                
                if state.is_gesture_sustained('reaching', min_frames=model.config['min_gesture_frames']):
                    if len(person_items) > 0:
                        person_actions.append(ActionType.PICKING_ITEM)
                
                # Container manipulation
                container_gesture_detected, container_conf = model.keypoint_analyzer.is_container_manipulation(
                    pose, orientation, person_height
                )
                state.add_gesture('container_manip', frame_count, container_gesture_detected, container_conf)
                
                if state.is_gesture_sustained('container_manip', min_frames=model.config['min_gesture_frames']):
                    if len(person_containers) > 0:
                        person_actions.append(ActionType.OPENING_CONTAINER)
                        
                        for container in person_containers:
                            if cam_state['container_tracker'].detect_container_opening(
                                person_bbox, container.bbox, container_conf,
                                container, frame_count
                            ):
                                container.is_open = True
                                container.opening_frames.append(frame_count)
                
                # Looking around
                previous_pose = state.previous_keypoints
                looking_detected, looking_conf = model.keypoint_analyzer.is_looking_around(
                    pose, previous_pose
                )
                state.add_gesture('looking', frame_count, looking_detected, looking_conf)
                
                if state.is_gesture_sustained('looking', min_frames=5):
                    person_actions.append(ActionType.LOOKING_AROUND)
                
                # Grab motion detection
                left_wrist = pose[9]
                right_wrist = pose[10]
                behavior_analyzer.grab_analyzer.update_trajectory(
                    person_id, frame_count, left_wrist, right_wrist
                )
                
                has_item_nearby = (
                    len(person_items) > 0 or
                    len(state.item_interactions) > 0 or
                    ActionType.PICKING_ITEM in state.get_recent_actions(frame_count, window=30)
                )
                
                grab_detected, grab_conf, grab_debug = \
                    behavior_analyzer.grab_analyzer.detect_sequential_grab(
                        person_id, frame_count, has_item_nearby
                    )
                
                grab_motion_active = False
                if grab_detected:
                    person_actions.append(ActionType.GRAB_MOTION_DETECTED)
                    grab_motion_active = True
                
                # Concealment detection
                has_item_context = (
                    len(person_items) > 0 or
                    len(state.item_interactions) > 0 or
                    ActionType.PICKING_ITEM in state.get_recent_actions(frame_count, window=60) or
                    grab_motion_active
                )
                
                concealment_detections = model.keypoint_analyzer.detect_all_concealment_behaviors(
                    person_id=person_id,
                    frame_num=frame_count,
                    keypoints=pose,
                    person_height=person_height,
                    orientation=orientation,
                    has_bag=has_bag,
                    previous_keypoints=state.previous_keypoints,
                    has_item_context=has_item_context,
                    grab_motion_active=grab_motion_active
                )
                
                for action_type, location, confidence in concealment_detections:
                    person_actions.append(action_type)
                    
                    has_product = (len(person_items) > 0 or len(state.item_interactions) > 0)
                    state.add_concealment_event(
                        frame_num=frame_count,
                        location_type=location,
                        confidence=confidence,
                        has_product=has_product
                    )
                
                # Behavioral risk analysis
                behavioral_risk = model.keypoint_analyzer.calculate_behavioral_risk(
                    state, frame_count, cam_state['frame_height'], cam_state['frame_width']
                )
                state.behavioral_suspicion_score = behavioral_risk
                
                if behavioral_risk > SystemConfig.NO_PRODUCT_SUSPICION_THRESHOLD:
                    person_actions.append(ActionType.SUSPICIOUS_BODY_LANGUAGE)
            
            actions[person_id] = person_actions
            
        except Exception as e:
            pass
    
    return actions


def _update_person_state(cam_state: Dict,
                         person_id: int,
                         bbox: List[float],
                         pose: Any,
                         orientation: PersonOrientation,
                         actions: List[ActionType],
                         containers: List,
                         items: List,
                         frame_count: int):
    """Update person state"""
    person_states = cam_state['person_states']
    
    if person_id not in person_states:
        person_states[person_id] = PersonState(
            person_id=person_id,
            first_seen_frame=frame_count
        )
    
    state = person_states[person_id]
    state.update_frame(frame_count, bbox, pose, orientation)
    
    # Update containers
    for container in containers:
        state.containers[container.container_id] = container
        
        if container.container_type == ContainerType.SHOPPING_CART:
            state.has_shopping_cart = True
        
        if container.is_open and frame_count in container.opening_frames:
            state.container_openings.append((frame_count, container.container_id))
    
    # Update items
    for item in items:
        if item.item_id not in state.item_interactions:
            state.add_item_interaction(
                item.item_id, item.item_class, frame_count,
                item.current_location, item.bbox, item.confidence
            )
        else:
            interaction = state.item_interactions[item.item_id]
            interaction.last_seen_frame = frame_count
            interaction.location = item.current_location
            interaction.bbox = item.bbox
    
    # Check for disappeared items
    for interaction in state.item_interactions.values():
        if interaction.location != ItemLocation.DISAPPEARED:
            if interaction.last_seen_frame < frame_count - SystemConfig.ITEM_DISAPPEAR_WINDOW:
                interaction.location = ItemLocation.DISAPPEARED
                interaction.disappeared_frame = frame_count
                actions.append(ActionType.ITEM_DISAPPEARED)
    
    # Record actions
    for action in actions:
        state.add_action(frame_count, action)


def _analyze_behaviors(model: TheftDetectionModel,
                       cam_state: Dict,
                       frame_count: int) -> List[Dict]:
    """Analyze behaviors and generate alerts"""
    alerts = []
    person_states = cam_state['person_states']
    behavior_analyzer = cam_state['behavior_analyzer']
    person_alert = cam_state['person_alert']
    
    for person_id, state in person_states.items():
        try:
            score = behavior_analyzer.calculate_attention_score(state, frame_count)
            state.attention_score = score
            
            alert_level = behavior_analyzer.determine_alert_level(score)
            
            if alert_level.value >= AlertLevel.ATTENTION.value:
                # Sequence validation
                has_product_evidence = len(state.item_interactions) > 0
                
                if has_product_evidence:
                    sequence_valid, seq_conf, seq_type = \
                        model.keypoint_analyzer.sequence_validator.validate_theft_sequence_with_product(
                            state, frame_count
                        )
                else:
                    sequence_valid, seq_conf, seq_type = \
                        model.keypoint_analyzer.sequence_validator.validate_behavioral_sequence(
                            state, frame_count
                        )
                
                if sequence_valid:
                    score = min(1.0, score * 1.2)
                    state.attention_score = score
                
                if alert_level != state.last_alert_level or alert_level == AlertLevel.ALERT:
                    alert = _generate_alert(
                        person_id, state, alert_level, score,
                        frame_count, sequence_valid, behavior_analyzer,
                        person_alert
                    )
                    alerts.append(alert)
                    state.last_alert_level = alert_level
        
        except Exception as e:
            pass
    
    return alerts


def _generate_alert(person_id: int,
                    state: PersonState,
                    alert_level: AlertLevel,
                    score: float,
                    frame_count: int,
                    sequence_validated: bool,
                    behavior_analyzer: BehaviorAnalyzer,
                    person_alert: Dict) -> Dict:
    """Generate alert dictionary"""
    recent_actions = state.get_recent_actions(frame_count, window=60)

    reasons = []

    # Initialize: False means "no alert sent yet"
    if person_id not in person_alert:
        person_alert[person_id] = False  # ← Changed to False

    concealment_pattern = state.get_concealment_pattern(frame_count, window=60)
    if concealment_pattern['total'] > 0:
        locations = concealment_pattern['locations']
        location_str = ", ".join(
            f"{loc}({count}x)"
            for loc, count in sorted(locations.items(), key=lambda x: x[1], reverse=True)[:3]
        )

        if not concealment_pattern['has_product']:
            reasons.append(f"CONCEALMENT GESTURES: {location_str} [NO PRODUCT VISIBLE]")
        else:
            reasons.append(f"CONCEALMENT: {location_str}")

    if ActionType.GRAB_MOTION_DETECTED in recent_actions:
        grab_count = recent_actions.count(ActionType.GRAB_MOTION_DETECTED)
        if grab_count > 2:
            reasons.append(f"MULTIPLE FAST GRABS ({grab_count}x)")
        else:
            reasons.append("FAST GRAB MOTION")

    if state.behavioral_suspicion_score > SystemConfig.NO_PRODUCT_SUSPICION_THRESHOLD:
        behavior_details = []
        if state.behavioral_flags['loitering'] > 2:
            behavior_details.append("prolonged loitering")
        if state.behavioral_flags['nervous'] > 3:
            behavior_details.append("nervous behavior")
        if state.behavioral_flags['erratic_movement'] > 2:
            behavior_details.append("erratic movement")
        if state.behavioral_flags['exit_behavior'] > 0:
            behavior_details.append("exit behavior")

        if behavior_details:
            reasons.append(f"SUSPICIOUS BEHAVIOR: {', '.join(behavior_details)}")

    if state.detect_concealment_pattern(frame_count):
        reasons.append("ITEM CONCEALED IN CONTAINER")
    if ActionType.OPENING_CONTAINER in recent_actions:
        reasons.append("container opened")
    if ActionType.ITEM_DISAPPEARED in recent_actions:
        reasons.append("item disappeared")
    if state.is_loitering(frame_count):
        reasons.append("loitering")

    base_message = behavior_analyzer.generate_alert_message(alert_level, reasons)

    # Send alert only if: it's a real alert AND no alert sent yet
    if base_message != "✅ Normal" and not person_alert[person_id]:  # ← Changed logic
        message = base_message
        person_alert[person_id] = True  # ← Mark as "alert sent"
    else:
        message = "already alert sent!"


    return {
        'person_id': person_id,
        'alert_level': alert_level.name,
        'attention_score': round(score, 3),
        'validated_confidence': round(score * 0.95, 3),
        'frame_num': frame_count,
        'reasons': reasons,
        'message': message,
        'concealment_locations': concealment_pattern['locations'],
        'concealment_count': concealment_pattern['total'],
        'behavioral_risk': round(state.behavioral_suspicion_score, 3),
        'has_product_evidence': concealment_pattern['has_product'],
        'sequence_validated': sequence_validated
    }


def _validate_and_format_alerts(model: TheftDetectionModel,
                                cam_state: Dict,
                                raw_alerts: List[Dict],
                                frame_count: int,
                                cam_id: int,
                                timestamp: datetime) -> List[Dict]:
    """
    Validate alerts and format them according to output specification
    Format: alert_id = {CAM_ID}{DDMMYYYY}{HHMMSS}_{PERSON_ID}
    """
    validated_alerts = []
    person_states = cam_state['person_states']
    behavior_analyzer = cam_state['behavior_analyzer']
    
    for raw_alert in raw_alerts:
        try:
            person_id = raw_alert['person_id']
            
            if person_id not in person_states:
                continue
            
            state = person_states[person_id]
            
            # Multi-stage validation through FP reducer
            should_alert, final_score, rejection_reasons = \
                behavior_analyzer.fp_reducer.validate_alert(
                    state, raw_alert, frame_count
                )
            
            if not should_alert:
                continue
            
            # Generate alert_id: {CAM_ID}{DDMMYYYY}{HHMMSS}_{PERSON_ID}
            alert_id = f"{cam_id}{timestamp.strftime('%d%m%Y%H%M%S')}_{person_id}"
            
            # Format alert according to specification
            formatted_alert = {
                'alert_id': alert_id,
                'person_id': person_id,
                'alert_level': raw_alert['alert_level'],
                'attention_score': round(final_score, 2),
                'validated_confidence': round(final_score * 0.92, 2),
                'reasons': raw_alert['reasons'][:5],
                'message': raw_alert.get('message', '')
            }
            
            validated_alerts.append(formatted_alert)
            
            # Update state
            state.last_alert_frame = frame_count
            state.suspicious_count += 1
            
            # Track in metrics
            cam_state['metrics_tracker'].add_alert(
                person_id=person_id,
                frame_num=frame_count,
                score=final_score,
                concealment_locations=raw_alert.get('concealment_locations', {}),
                has_product=raw_alert.get('has_product_evidence', False),
                sequence_validated=raw_alert.get('sequence_validated', False)
            )
            
            '''logger.warning(
                f"🚨 ALERT GENERATED: Person {person_id} | "
                f"Score: {final_score:.2f} | "
                f"Reasons: {', '.join(raw_alert['reasons'][:3])}"
            )'''
        
        except Exception as e:
            pass
    
    # Deduplicate alerts
    if len(validated_alerts) > 1:
        validated_alerts = behavior_analyzer.fp_reducer.deduplicate_alerts(validated_alerts)
    
    # Update global alert frame
    if validated_alerts:
        cam_state['last_global_alert_frame'] = frame_count
    
    return validated_alerts


def _visualize_frame(frame: np.ndarray,
                     tracked_persons: Dict[int, List[float]],
                     orientations: Dict[int, PersonOrientation],
                     alerts: List[Dict],
                     container_associations: Dict,
                     item_associations: Dict,
                     cam_state: Dict) -> np.ndarray:
    """
    Visualize detections and alerts on frame
    """
    annotated = frame.copy()
    person_states = cam_state['person_states']
    
    # Create alert lookup
    alert_map = {alert['person_id']: alert for alert in alerts}
    
    # Draw persons
    for person_id, bbox in tracked_persons.items():
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Determine color based on alert level
            if person_id in alert_map:
                alert = alert_map[person_id]
                alert_level = alert['alert_level']
                
                if alert_level == 'ALERT':
                    color = (0, 0, 255)  # Red
                    thickness = 3
                elif alert_level == 'ATTENTION_PLUS':
                    color = (0, 140, 255)  # Orange
                    thickness = 3
                elif alert_level == 'ATTENTION':
                    color = (0, 255, 255)  # Yellow
                    thickness = 2
                else:
                    color = (0, 255, 0)  # Green
                    thickness = 2
            else:
                color = (0, 255, 0)  # Green for normal
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label_parts = [f"ID:{person_id}"]
            
            # Add orientation
            if person_id in orientations:
                orientation = orientations[person_id]
                if orientation != PersonOrientation.UNKNOWN:
                    label_parts.append(orientation.value.upper())
            
            # Add alert info
            if person_id in alert_map:
                alert = alert_map[person_id]
                score = alert['attention_score']
                label_parts.append(f"{alert['alert_level']}")
                label_parts.append(f"{score:.2f}")
            elif person_id in person_states:
                state = person_states[person_id]
                if state.attention_score > 0.3:
                    label_parts.append(f"{state.attention_score:.2f}")
            
            # Draw label
            label = " | ".join(label_parts)
            
            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
            
            # Draw alert reasons
            if person_id in alert_map:
                alert = alert_map[person_id]
                reasons = alert['reasons'][:2]  # Top 2 reasons
                
                y_offset = y2 + 20
                for i, reason in enumerate(reasons):
                    reason_text = reason[:50]  # Truncate long reasons
                    cv2.putText(
                        annotated,
                        reason_text,
                        (x1, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1
                    )
        
        except Exception as e:
            pass
    
    # Draw containers
    for person_id, containers in container_associations.items():
        for container in containers:
            try:
                x1, y1, x2, y2 = map(int, container.bbox)
                
                # Color based on container type
                if container.container_type == ContainerType.SHOPPING_CART:
                    color = (255, 200, 0)  # Cyan
                else:
                    color = (255, 100, 200)  # Pink
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = container.container_type.value
                if container.is_open:
                    label += " [OPEN]"
                
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
            except Exception as e:
                pass
    
    # Draw items
    for person_id, items in item_associations.items():
        for item in items:
            try:
                x1, y1, x2, y2 = map(int, item.bbox)
                
                # Green for items in hand
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                
                # Label
                cv2.putText(
                    annotated,
                    item.item_class,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1
                )
            except Exception as e:
                pass
    
    # Draw system info
    info_y = 25
    info_x = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Frame count
    cv2.putText(
        annotated,
        f"Frame: {cam_state['frame_count']}",
        (info_x, info_y),
        font,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Tracked persons count
    cv2.putText(
        annotated,
        f"Persons: {len(tracked_persons)}",
        (info_x, info_y + 25),
        font,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Active alerts count
    if alerts:
        cv2.putText(
            annotated,
            f"Alerts: {len(alerts)}",
            (info_x, info_y + 50),
            font,
            0.6,
            (0, 0, 255),
            2
        )
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        timestamp,
        (annotated.shape[1] - 200, 25),
        font,
        0.5,
        (255, 255, 255),
        1
    )
    
    return annotated


# =============================================================================
# MAIN INFERENCE ENTRY POINT
# =============================================================================

def run_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for inference

    Args:
        input_data: Dictionary with structure:
            {
                "cam_id": int,
                "org_id": int,
                "user_id": int,
                "encoding": str  # base64 encoded JPEG
            }

    Returns:
        Dictionary with structure:
            {
                "cam_id": int,
                "org_id": int,
                "user_id": int,
                "frame_id": str,  # {CAMID}{USERID}{ORGID}{DDMMYYYY}{HHMMSS}
                "timestamp": str,
                "persons": List[int],
                "alerts": List[Dict],
                "message": str,
                "annotated_frame": str,
                "status": int
            }
    """
    global _inference_model

    try:
        # Initialize model if not already done
        if '_inference_model' not in globals():
            _inference_model = model_fn()

        # Process input
        processed_input = input_fn(json.dumps(input_data), 'application/json')

        # Run prediction
        prediction = predict_fn(processed_input, _inference_model)

        # Generate overall message
        if prediction['alerts']:
            alert_count = len(prediction['alerts'])
            max_alert = max(prediction['alerts'], key=lambda x: x.get('attention_score', 0))

            prediction['message'] = max_alert.get('message', '')
            
            if not prediction['message']:
                reasons = max_alert.get('reasons', [])
                if reasons and len(reasons) > 0:
                    prediction['message'] = reasons[0]
                else:
                    prediction['message'] = f"{max_alert.get('alert_level', 'ALERT')} detected"

            if alert_count > 1 and prediction['message'] != "already alert sent!":
                prediction['message'] += f" (+{alert_count - 1} more)"
        else:
            prediction['message'] = "✅ No suspicious activity detected"

        # Format output
        output_str = output_fn(prediction, 'application/json')

        return json.loads(output_str)

    except Exception as e:

        return {
            'cam_id': input_data.get('cam_id', -1),
            'org_id': input_data.get('org_id', -1),
            'user_id': input_data.get('user_id', -1),
            'frame_id': '',
            'timestamp': datetime.now().isoformat() + 'Z',
            'persons': [],
            'alerts': [],
            'message': f"❌ Error: {str(e)}",
            'annotated_frame': '',
            'status': 0,
            'error': str(e)
        }


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the inference system
    """
    
    # Example 1: Load an image and run inference
    def test_with_image(image_path: str):
        """Test with an actual image file"""
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare input
            input_data = {
                "cam_id": 123,
                "org_id": 2,
                "user_id": 2,
                "encoding": base64_frame
            }
            
            # Run inference
            result = run_inference(input_data)
            
            # Print results
            print("\n" + "="*80)
            print("INFERENCE RESULT")
            print("="*80)
            print(json.dumps(result, indent=2))
            print("="*80 + "\n")
            
            # Decode and save annotated frame
            if result.get('annotated_frame'):
                annotated_bytes = base64.b64decode(result['annotated_frame'])
                annotated_array = np.frombuffer(annotated_bytes, dtype=np.uint8)
                annotated_frame = cv2.imdecode(annotated_array, cv2.IMREAD_COLOR)
                
                output_path = "annotated_output.jpg"
                cv2.imwrite(output_path, annotated_frame)
            
            return result
            
        except Exception as e:
            return None
    
    # Example 2: Test with synthetic data
    def test_with_synthetic():
        """Test with synthetic frame data"""
        try:
            # Create a blank test image
            test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Add some text
            cv2.putText(
                test_frame,
                "TEST FRAME",
                (500, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3
            )
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', test_frame)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare input
            input_data = {
                "cam_id": 456,
                "org_id": 3,
                "user_id": 5,
                "encoding": base64_frame
            }
            
            # Run inference
            result = run_inference(input_data)
            
            print("\n" + "="*80)
            print("SYNTHETIC TEST RESULT")
            print("="*80)
            print(json.dumps(result, indent=2))
            print("="*80 + "\n")
            
            return result
            
        except Exception as e:
            return None
    
    # Run tests
    print("\n" + "="*80)
    print("THEFT DETECTION INFERENCE SYSTEM v4.1")
    print("="*80 + "\n")
    
    # Test with synthetic data
    test_with_synthetic()
    
    # Uncomment to test with real image
    # test_with_image("path/to/your/test_image.jpg")
    
    
    # Print model statistics
    if '_inference_model' in globals():
        stats = _inference_model.performance_stats
        print("\n" + "="*80)
        print("PERFORMANCE STATISTICS")
        print("="*80)
        print(f"Total Frames Processed: {stats['total_frames_processed']}")
        print(f"Average Inference Time: {stats['avg_inference_time']:.3f}s")
        print(f"Total Alerts Generated: {stats['total_alerts']}")
        print("="*80 + "\n")