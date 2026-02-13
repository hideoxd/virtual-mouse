# AI Virtual Mouse 🖱️✋

This project implements a highly accurate, gesture-controlled virtual mouse using **Python**, **OpenCV**, and **MediaPipe**. It allows you to control your system cursor, click, drag, and scroll using simple hand movements captured by your webcam.

## ✨ Features

*   **High Precision Tracking:** Utilizes adaptive smoothing algorithms (dynamic alpha) for jitter-free hovering and responsive movement.
*   **Robust Gesture Recognition:** Distinguishes clearly between cursor movement, clicking, and scrolling modes.
*   **Full Mouse Functionality:**
    *   **Cursor Movement:** Mapped to your hand's position.
    *   **Left Click / Drag:** Pinch gesture with hysteresis for reliable dragging.
    *   **Right Click:** Thumb + Middle finger pinch.
    *   **Double Click:** Thumb + Ring finger pinch.
    *   **Scrolling:** Rate-controlled scrolling by tilting an open hand.
*   **Visual Feedback:** On-screen indicators for clicks, modes, and active gestures.

## 🛠️ Requirements

*   Python 3.7+
*   Webcam

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ai-virtual-mouse.git
    cd ai-virtual-mouse
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a requirements file, install manually:*
    ```bash
    pip install opencv-python mediapipe pyautogui numpy
    ```

## 🚀 Usage

Run the main application:

```bash
python virtual_mouse.py
```

### 🎮 Controls / Gestures

| Action | Mode | Gesture | Description |
| :--- | :--- | :--- | :--- |
| **Move Cursor** | **Cursor Mode** | ☝️ **Pointing / Relaxed** | Move your hand around the screen. |
| **Left Click** | **Cursor Mode** | 🤏 **Index + Thumb Pinch** | Quick pinch and release. |
| **Drag & Drop** | **Cursor Mode** | ✊ **Index + Thumb (Hold)** | Pinch and hold. Move hand to drag. Release pinch to drop. |
| **Right Click** | **Cursor Mode** | 🖕 **Middle + Thumb Pinch** | Pinch your middle finger and thumb together. |
| **Double Click** | **Cursor Mode** | 🖖 **Ring + Thumb Pinch** | Pinch your ring finger and thumb together. |
| **Scroll Mode** | **Scroll Mode** | ✋ **Open Hand** | Spread all fingers to enter Scroll Mode. **Tilt** your hand Up/Down to scroll. |

*   **Exit:** Press `q` to quit the application.

## ⚙️ Configuration

You can fine-tune the behavior by modifying the uppercase constants at the top of `virtual_mouse.py`:

*   `CAM_W`, `CAM_H`: Camera capture resolution.
*   `FPS_TARGET`: Desired framerate.
*   `PINCH_START_RATIO`: Sensitivity of the click pinch (Lower = harder pinch required).
*   `MIN_ALPHA`: Smoothing factor (Lower = smoother cursor, Higher = more responsive).

## 📝 Troubleshooting

*   **Jittery Cursor:** Adjust `MIN_ALPHA` in the code to a lower value (e.g., `0.02`).
*   **Clicks not registering:** Ensure your hand is well-lit and facing the camera. You may need to adjust `PINCH_START_RATIO` if your hand size differs significantly from the default calibration.
