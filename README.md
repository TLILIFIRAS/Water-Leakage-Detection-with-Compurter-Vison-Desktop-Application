# Water Leak Detection with YOLO11 and PyQt5

This desktop application allows users to detect water leaks in real-time or from uploaded videos using the YOLO11 deep learning model. The application is built with PyQt5 for the user interface and integrates with OpenCV for video processing. The application enables water leak detection either by using a webcam feed or by uploading a video file.

## Features

- **Real-time Water Leak Detection**: Use your webcam to detect water leaks in real-time.
- **Video File Upload**: Upload pre-recorded videos to detect water leaks.
- - **Roboflow Object Detection Model**: Model trained using Roboflow for detecting water leaks.
- **YOLO11 Model Integration**: Leverages the YOLO11 model to detect water leaks.
- **Custom UI**: A clean and simple interface using PyQt5.
- **Bounding Box Visualization**: Shows bounding boxes around detected leaks in the video stream.

## Demo

![App Screenshot](./screenshot.png)

[Watch the demo on YouTube](https://youtu.be/zFJNLmpjs7I)


## Requirements

To run this project, you need the following:
- Pytorch
- Python 3
- PyQt5
- OpenCV
- Ultralytics
- NumPy

## Usage

- Upload a video: Click on the 'Upload Video' button to select a video file from your system. The app will process the video and detect water leaks.
- Start Webcam: Click on the 'Start Webcam' button to start detecting water leaks in real-time using your webcam

## Future Enhancements
- Add more models for detecting different kinds of leaks.
- Implement advanced settings for configuring detection thresholds.
- Add localization support for different languages.
## Author
Eng. Firas Tlili
[Linkedin](https://www.linkedin.com/in/firastlili/)
