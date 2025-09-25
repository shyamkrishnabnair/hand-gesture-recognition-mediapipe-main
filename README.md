# 🎵 Music Generation Through Hand Gesture with Custom Notation

> **Turn your hands into instruments, and your gestures into melodies.**

---

## About the Project

Imagine a world where your hand gestures don’t just express, but *compose* music. This project uses advanced computer vision and a Fully Connected Neural Network (FCNN) trained to recognize simple hand gestures like palm, stop, start, and more. It converts these gestures into live musical notes with a custom notation system — all controlled by YOU.

We are currently working on the “train your own gesture” feature to empower users with personalized gesture recognition.

**Technologies:**  
- MediaPipe & OpenCV for hand landmark detection  
- FCNN for gesture classification  
- MIDI integration for dynamic sound generation  
- CustomTkinter (v6) for smooth, interactive GUI  
- Multithreaded audio playback for seamless, overlapping notes  

---

## Features

- 🎶 Real-time hand gesture recognition (palm, stop, start, and others)  
- 🎹 Custom MIDI note playback with dynamic velocity  
- 🎛️ Volume & mute control via pinch-drag gestures  
- 📜 Live notation and logging display  
- 🖥️ Lightweight and efficient for smooth performance  
- 🚧 “Train your own gesture” phase — coming soon!  

---

## Installation

1. Clone this repository  
2. Make sure you have **Python 3.8+** installed  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main app with: <br>
```python customTkinter.v6.py```

---
## Devs

[Vineet](https://github.com/vineet-k09)

[Shyamkrishna](https://github.com/shyamkrishnabnair) 

---
## License
MIT License © 2025 Vineet & Shyamkrishna

Turn your gestures into symphonies. Because music isn’t just heard — it’s _felt_.

```bash
pyinstaller --noconfirm --onefile --windowed \
--add-data "utils;utils" \
--add-data "model;model" \
customTKinter.v7.py
```