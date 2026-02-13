import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
from collections import deque

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# 1. Camera & Processing
CAM_W, CAM_H = 640, 480
FPS_TARGET = 60                   
HAND_MODEL_COMPLEXITY = 1         

# 2. Movement (Absolute Mapping)
FRAME_MARGIN = 75                 
MIN_ALPHA = 0.05                  
MAX_ALPHA = 1.0                   
ALPHA_RAMP_DIST = 150             

# 3. Gestures & Click Judgment
# Robustness Tweaks
OPEN_THRESHOLD = 1.6              

# Pinch Thresholds (Tuned for Precision)
# We use a hysteresis gap to prevent "flickering" clicks.
# Lower = Harder to Click (More Precise). Higher = Easiest.
PINCH_START_RATIO = 0.20          # Must pinch TIGHT to click
PINCH_RELEASE_RATIO = 0.35        # Must open WIDE to release (Reliable Drag)

CLICK_COOLDOWN = 0.4              # Slightly increased to prevent accidental double-fires

# 4. Scroll
SCROLL_DEAD_ZONE = 10.0           
SCROLL_SPEED_MULT = 14.0          
SCROLL_SMOOTHING = 0.15           

# ──────────────────────────────────────────────────────────────────────────────

def to_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def get_dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def main():
    # ─── SETUP ────────────────────────────────────────────────────────────────
    pyautogui.FAILSAFE = False      
    pyautogui.PAUSE = 0             
    screen_w, screen_h = pyautogui.size()

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # Mediapipe
    mp_hands = mp.solutions.hands
    
    hands = mp_hands.Hands(
        model_complexity=HAND_MODEL_COMPLEXITY,
        max_num_hands=1,
        min_detection_confidence=0.8, # High confidence for robust tracking
        min_tracking_confidence=0.8
    )

    # State
    prev_cx, prev_cy = screen_w // 2, screen_h // 2
    curr_cx, curr_cy = prev_cx, prev_cy
    
    # Click States
    left_click_active = False
    
    right_click_last_time = 0
    double_click_last_time = 0
    
    scroll_val = 0.0
    
    # Buffers (Smoothing)
    # Movement Buffer
    raw_x_buf = deque(maxlen=6) # Smoother cursor
    raw_y_buf = deque(maxlen=6)
    
    # Click Judgment Buffers (Crucial for precision)
    # Averages pinch distance over frames to eliminate jitter
    pinch_dist_buf = deque(maxlen=4) 
    rc_dist_buf = deque(maxlen=4)
    dc_dist_buf = deque(maxlen=4)

    # FPS Calc
    prev_frame_time = 0

    print("[Virtual Mouse] PRECISION CLICK MODE. Running... Press 'q' to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip & Color
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            h, w, _ = frame.shape
            
            # Status vars
            mode = "IDLE"
            active_color = (100, 100, 100)

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                lm = hand_lms.landmark
                
                # ─── 1. CORE METRICS ──────────────────────────────────────────
                
                # Key Points
                wrist = to_px(lm[0], w, h)
                thumb_tip = to_px(lm[4], w, h)
                index_mcp = to_px(lm[5], w, h)
                index_tip = to_px(lm[8], w, h)
                middle_mcp = to_px(lm[9], w, h)
                middle_tip = to_px(lm[12], w, h)
                ring_tip = to_px(lm[16], w, h)
                pinky_mcp = to_px(lm[17], w, h)

                # Palm Center (Visuals)
                palm_cx = (wrist[0] + index_mcp[0] + pinky_mcp[0]) // 3
                palm_cy = (wrist[1] + index_mcp[1] + pinky_mcp[1]) // 3
                
                # Scale Reference (Wrist to Middle MCP) - Stable invariant
                palm_size = get_dist(wrist, middle_mcp)
                if palm_size < 1: palm_size = 1

                # ─── 2. STATE DETECTION ──────────────────────────────────────
                
                # Openness Check
                tips = [4, 8, 12, 16, 20]
                total_tip_dist = sum(get_dist(to_px(lm[i], w, h), wrist) for i in tips)
                avg_dist = total_tip_dist / 5.0
                open_ratio = avg_dist / palm_size
                
                is_open = open_ratio > OPEN_THRESHOLD
                
                # ─── 3. EXECUTION ─────────────────────────────────────────────

                if is_open:
                    # ─── SCROLL MODE ──────────────────────────────────────────
                    mode = "SCROLL"
                    active_color = (0, 255, 255) # Yellow
                    
                    if left_click_active:
                        pyautogui.mouseUp()
                        left_click_active = False

                    # Clean Move buffers
                    raw_x_buf.clear()
                    raw_y_buf.clear()
                    pinch_dist_buf.clear() # Clear click buffers on mode switch
                    
                    # Tilt Angle
                    dx = middle_mcp[0] - wrist[0]
                    dy = wrist[1] - middle_mcp[1]
                    angle = math.degrees(math.atan2(dy, abs(dx)))
                    
                    target_scroll = 0
                    if abs(angle) > SCROLL_DEAD_ZONE:
                        exceed = abs(angle) - SCROLL_DEAD_ZONE
                        intensity = min(exceed / 20.0, 1.0) 
                        sign = 1 if angle > 0 else -1
                        target_scroll = sign * (intensity ** 1.6) * SCROLL_SPEED_MULT
                    
                    scroll_val = (scroll_val * (1.0 - SCROLL_SMOOTHING)) + (target_scroll * SCROLL_SMOOTHING)
                    
                    if abs(scroll_val) >= 1.0:
                        steps = int(scroll_val)
                        pyautogui.scroll(steps * 25) 
                    
                    cv2.putText(frame, f"SCROLL: {int(scroll_val)}", (palm_cx + 20, palm_cy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, active_color, 2)
                    cv2.line(frame, wrist, middle_mcp, active_color, 3)

                else:
                    # ─── CURSOR MODE ──────────────────────────────────────────
                    mode = "CURSOR"
                    active_color = (0, 255, 0) # Green

                    # 1. Smoothed Movement
                    raw_x_buf.append(palm_cx)
                    raw_y_buf.append(palm_cy)
                    x_stable = sum(raw_x_buf) / len(raw_x_buf)
                    y_stable = sum(raw_y_buf) / len(raw_y_buf)

                    # 2. Map & Clip
                    x_norm = (x_stable - FRAME_MARGIN) / (CAM_W - 2 * FRAME_MARGIN)
                    y_norm = (y_stable - FRAME_MARGIN) / (CAM_H - 2 * FRAME_MARGIN)
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    
                    target_x = x_norm * screen_w
                    target_y = y_norm * screen_h
                    
                    # 3. Adaptive Gamma Smoothing
                    dx = target_x - curr_cx
                    dy = target_y - curr_cy
                    dist_err = math.hypot(dx, dy)
                    
                    scale = min(dist_err / ALPHA_RAMP_DIST, 1.0)
                    alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * (scale ** 1.5)
                    
                    curr_cx = curr_cx + alpha * dx
                    curr_cy = curr_cy + alpha * dy
                    
                    pyautogui.moveTo(curr_cx, curr_cy)
                    
                    # ─── PRECISE CLICK JUDGMENT ──────────────────────────────
                    
                    # A. Left Click / Drag (Index + Thumb)
                    # We utilize a buffer to smooth out the pinch distance
                    raw_pinch = get_dist(thumb_tip, index_tip)
                    pinch_dist_buf.append(raw_pinch)
                    avg_pinch = sum(pinch_dist_buf) / len(pinch_dist_buf)
                    
                    pinch_ratio = avg_pinch / palm_size
                    
                    # Visual Feedback for Pinch Strength
                    # Draw a line that changes color based on proximity
                    line_col = (0, 255, 0)
                    if pinch_ratio < PINCH_RELEASE_RATIO: line_col = (0, 165, 255) # Orange (Warning)
                    if pinch_ratio < PINCH_START_RATIO: line_col = (0, 0, 255)     # Red (Click)
                    
                    if pinch_ratio < 0.8: # Only draw if somewhat close
                        cv2.line(frame, thumb_tip, index_tip, line_col, 2)

                    # Logic
                    if pinch_ratio < PINCH_START_RATIO:
                        if not left_click_active:
                             pyautogui.mouseDown()
                             left_click_active = True
                        
                        # Visual: Solid Circle
                        cv2.circle(frame, ( (thumb_tip[0]+index_tip[0])//2, (thumb_tip[1]+index_tip[1])//2 ), 
                                   8, (0, 0, 255), cv2.FILLED)
                        cv2.putText(frame, "HOLD", (palm_cx, palm_cy-40),
                                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        
                    elif pinch_ratio > PINCH_RELEASE_RATIO:
                        if left_click_active:
                            pyautogui.mouseUp()
                            left_click_active = False

                    # B. Right Click (Thumb + Middle)
                    raw_rc = get_dist(thumb_tip, middle_tip)
                    rc_dist_buf.append(raw_rc)
                    avg_rc = sum(rc_dist_buf) / len(rc_dist_buf)
                    rc_ratio = avg_rc / palm_size
                    
                    if rc_ratio < PINCH_START_RATIO:
                        if (time.time() - right_click_last_time) > CLICK_COOLDOWN:
                            pyautogui.rightClick()
                            right_click_last_time = time.time()
                            cv2.putText(frame, "RIGHT CLICK", (palm_cx, palm_cy-70),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                    
                    # C. Double Click (Thumb + Ring)
                    raw_dc = get_dist(thumb_tip, ring_tip)
                    dc_dist_buf.append(raw_dc)
                    avg_dc = sum(dc_dist_buf) / len(dc_dist_buf)
                    dc_ratio = avg_dc / palm_size
                    
                    if dc_ratio < PINCH_START_RATIO:
                         if (time.time() - double_click_last_time) > CLICK_COOLDOWN:
                            pyautogui.doubleClick()
                            double_click_last_time = time.time()
                            cv2.putText(frame, "DOUBLE CLICK", (palm_cx, palm_cy-70),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

                    # Cursor Visual
                    cv2.circle(frame, (palm_cx, palm_cy), 5, active_color, cv2.FILLED)
                    
            # Info
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
            prev_frame_time = new_frame_time
            
            cv2.putText(frame, f"FPS: {int(fps)} | Mode: {mode}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, active_color, 2)

            cv2.imshow("Virtual Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if left_click_active:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()
        try:
            hands.close()
        except:
            pass
        print("Program Safe Exit.")

if __name__ == "__main__":
    main()
