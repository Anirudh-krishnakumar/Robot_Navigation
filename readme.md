# Context-Aware Robot Navigation System

## Overview

The Context-Aware Robot Navigation System is an intelligent navigation framework that combines computer vision, scene understanding, and large language models (LLMs) to generate safe and context-aware navigation decisions in indoor environments.

Unlike traditional obstacle avoidance systems, this project interprets the surrounding scene, identifies relevant objects, estimates their distances, understands contextual relationships, and generates navigation actions based on a user-defined target.

---

## Features

- Real-time object detection using YOLO
- Scene understanding using BLIP
- Distance estimation for detected objects
- Context-aware navigation using Llama (Hugging Face Inference API)
- Target-based navigation through natural language instructions
- Rule-based safety checks for obstacle avoidance
- Dynamic navigation decisions in changing environments

---

## System Architecture

```
Camera Input
      │
      ▼
YOLO Object Detection
      │
      ▼
Distance Estimation
      │
      ▼
BLIP Scene Understanding
      │
      ▼
Prompt Generation
      │
      ▼
Llama (LLM)
      │
      ▼
Navigation Decision
      │
      ▼
Safety Validation
      │
      ▼
Robot Navigation Command
```

---

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- BLIP Vision-Language Model
- Hugging Face Inference API
- Transformers
- NumPy

---

## Project Structure

```
Robot_Navigation/
│
├── SLM&VLM.py
├── requirements.txt
└── README.md
```

---

## Workflow

1. Capture an image from the camera.
2. Detect objects in the environment using YOLO.
3. Estimate the relative distance of each detected object.
4. Generate a scene description using BLIP.
5. Combine the scene description, detected objects, and user target into a structured prompt.
6. Send the prompt to the Llama model.
7. Generate a navigation decision.
8. Apply safety rules before producing the final navigation command.

---

## Example User Instructions

```
Navigate to the chair.

Move towards the bottle.

Find the backpack.

Locate the laptop.

Navigate to the exit.
```

---

## Sample Output

```
Detected Objects
----------------
Person      : 1.3 m
Chair       : 2.4 m
Bottle      : 1.8 m

Scene Description
-----------------
A person is standing near a chair with a bottle placed on the table.

Navigation Decision
-------------------
Turn Right

Safety Status
-------------
Safe to Proceed
```

---

## Applications

- Indoor autonomous robots
- Warehouse automation
- Smart home assistants
- Hospital service robots
- Human-robot interaction research
- Intelligent navigation systems

---

## Future Enhancements

- ROS2 integration
- SLAM-based mapping
- Voice-controlled navigation
- Multi-object tracking
- Edge deployment on NVIDIA Jetson
- Autonomous path planning
- Reinforcement learning for adaptive navigation

---

## Author

**Anirudh Krishnakumar**

GitHub: https://github.com/Anirudh-krishnakumar

---

## License

This project is intended for educational and research purposes.