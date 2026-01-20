"""
test_inference.py - Simple Test Script for Theft Detection
==========================================================
Just run this file - it will process your video!
"""

import cv2
import base64
import json
import time
from datetime import datetime
import os

# Import the inference module
from inference import run_inference


# =============================================================================
# CONFIGURATION - CHANGE THESE SETTINGS
# =============================================================================

# Your video path
#VIDEO_PATH = r"E:\UTC project\utc\cctv\CCTV_Project\Darshil\3nov2025\Transformer Testing Theft\Pose Detection MEthod\23Nov2025\Test1.mp4"
#VIDEO_PATH = r"E:\UTC project\utc\cctv\CCTV_Project\Darshil\3nov2025\Transformer Testing Theft\Kaggel Model\V5Th.mp4"
VIDEO_PATH = r"E:\UTC project\utc\cctv\CCTV_Project\Darshil\3nov2025\Transformer Testing Theft\Pose Detection MEthod\23Nov2025\thftDect.mp4"

# Camera settings
CAM_ID = 123
ORG_ID = 2
USER_ID = 2

# Processing settings
PROCESS_EVERY_N_FRAMES = 1  # 1 = process all frames, 2 = every other frame, etc.
MAX_FRAMES = None           # None = process all, or set a number like 100
DISPLAY_VIDEO = True        # Show video window while processing
SAVE_ALERT_FRAMES = True    # Save frames that have alerts


# =============================================================================
# SIMPLE VIDEO PROCESSOR
# =============================================================================

def process_video():
    """Process video and display results"""

    print("\n" + "="*80)
    print("THEFT DETECTION - VIDEO TEST")
    print("="*80)
    print(f"Video: {VIDEO_PATH}")
    print(f"Camera ID: {CAM_ID} | Org ID: {ORG_ID} | User ID: {USER_ID}")
    print("="*80 + "\n")

    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video file not found!")
        print(f"Path: {VIDEO_PATH}")
        return

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open video file!")
        return

    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds\n")

    # Create output folder
    output_folder = "alert_frames"
    if SAVE_ALERT_FRAMES:
        os.makedirs(output_folder, exist_ok=True)
        print(f"💾 Alert frames will be saved to: {output_folder}/\n")

    # Statistics
    frame_count = 0
    processed_count = 0
    alert_count = 0
    total_inference_time = 0
    alert_frames = []

    print("🚀 Starting processing...\n")
    print("-"*80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if configured
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue

            # Check max frames limit
            if MAX_FRAMES and processed_count >= MAX_FRAMES:
                print(f"\n⏹️  Reached maximum frame limit ({MAX_FRAMES})")
                break

            processed_count += 1

            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_frame = base64.b64encode(buffer).decode('utf-8')

            # Prepare input
            input_data = {
                "cam_id": CAM_ID,
                "org_id": ORG_ID,
                "user_id": USER_ID,
                "encoding": base64_frame
            }

            # Run inference
            start_time = time.time()
            result = run_inference(input_data)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Check results
            has_alerts = len(result.get('alerts', [])) > 0
            num_persons = len(result.get('persons', []))

            # Print progress
            status = "🚨 ALERT" if has_alerts else "✅ OK"
            print(f"Frame {frame_count:04d} | {status} | "
                  f"Persons: {num_persons} | "
                  f"Alerts: {len(result.get('alerts', []))} | "
                  f"Time: {inference_time:.2f}s")

            # Print alert details
            if has_alerts:
                alert_count += 1
                alert_frames.append(frame_count)

                for alert in result['alerts']:
                    print(f"  └─ Person {alert['person_id']}: {alert['alert_level']} "
                          f"(Score: {alert['attention_score']:.2f})")
                    print(f"     Reasons: {', '.join(alert['reasons'][:2])}")

                # Save alert frame
                if SAVE_ALERT_FRAMES and result.get('annotated_frame'):
                    frame_bytes = base64.b64decode(result['annotated_frame'])
                    import numpy as np
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    annotated = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    filename = f"{output_folder}/alert_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"     💾 Saved: {filename}")

            # Display frame
            if DISPLAY_VIDEO and result.get('annotated_frame'):
                frame_bytes = base64.b64decode(result['annotated_frame'])
                import numpy as np
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                annotated = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                # Add progress info
                progress_text = f"Frame: {frame_count}/{total_frames} | FPS: {1/inference_time:.1f}"
                cv2.putText(annotated, progress_text, (10, annotated.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow('Theft Detection Test', annotated)

                # Press 'q' to quit, 'p' to pause
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopped by user (pressed 'q')")
                    break
                elif key == ord('p'):
                    print("\n⏸️  Paused (press any key to continue)")
                    cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user (Ctrl+C)")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Print final statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"✅ Frames Processed: {processed_count}/{total_frames}")
    print(f"⏱️  Total Time: {total_inference_time:.2f}s")
    print(f"⚡ Average Time: {total_inference_time/processed_count:.2f}s per frame")
    print(f"📊 Average FPS: {processed_count/total_inference_time:.2f}")
    print(f"\n🚨 Total Alerts: {alert_count}")

    if alert_frames:
        print(f"📍 Alert Frames: {alert_frames[:10]}")
        if len(alert_frames) > 10:
            print(f"   ... and {len(alert_frames) - 10} more")

    if SAVE_ALERT_FRAMES and alert_count > 0:
        print(f"\n💾 {alert_count} alert frames saved to: {output_folder}/")

    print("="*80 + "\n")

    # Save results to JSON
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_data = {
        "video_path": VIDEO_PATH,
        "video_info": {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames
        },
        "processing": {
            "frames_processed": processed_count,
            "total_time": total_inference_time,
            "avg_time_per_frame": total_inference_time/processed_count,
            "avg_fps": processed_count/total_inference_time
        },
        "results": {
            "total_alerts": alert_count,
            "alert_frames": alert_frames
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"📄 Results saved to: {results_file}\n")


# =============================================================================
# RUN THE TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🔍"*40)
    print("THEFT DETECTION INFERENCE SYSTEM - VIDEO TEST")
    print("🔍"*40)

    # Run the test
    process_video()

    print("\n✅ Test Complete!\n")
    print("Controls:")
    print("  - Press 'Q' to quit during playback")
    print("  - Press 'P' to pause/resume")
    print("  - Press Ctrl+C to stop anytime\n")