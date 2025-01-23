import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout
from PyQt5.QtCore import QTimer , Qt 
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load YOLO11 model
model = YOLO("./Model_Training/best.pt").to(device)

def get_detections(img):
    results = model(img)  # Run detection
    detections = []

    for result in results:
        for box in result.boxes:
            detection = {
                'class': result.names[int(box.cls[0])],  # Class name
                'confidence': box.conf[0].item(),  # Confidence score
                'coordinates': box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
            }
            detections.append(detection)
    
    return detections

class WaterLeakDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def initUI(self):
        # Set up the window
        self.setWindowTitle("Water Leak Detection")
        self.setGeometry(150, 150, 800, 600)

        # Create layout
        layout = QVBoxLayout()

        # Top Title Text (add it above everything)
        title_label = QLabel("Water Leaking Detection with Computer Vision")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; text-align: center;")
        title_label.setAlignment(Qt.AlignCenter)
        

        # Video label (to show the camera feed or video)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        # Upload Button to choose video
        upload_button = QPushButton('Upload Video', self)
        upload_button.clicked.connect(self.upload_video)

        # Start Webcam Button
        webcam_button = QPushButton('Start Webcam', self)
        webcam_button.clicked.connect(self.start_camera)

        # Arrange buttons in a horizontal layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(upload_button)
        button_layout.addWidget(webcam_button)

        # Bottom Text for "By Eng. Firas Tlili"
        author_label = QLabel("By Eng. Firas Tlili")
        author_label.setStyleSheet("font-size: 16px; font-style: italic; text-align: center;")
        author_label.setAlignment(Qt.AlignCenter)

        # Add widgets to layout
        layout.addWidget(title_label)  # Top title text
        layout.addWidget(self.video_label)  # Video/camera display area
        layout.addLayout(button_layout)  # Buttons layout
        layout.addWidget(author_label)  # Bottom author text

        # Set the layout for the window
        self.setLayout(layout)
    
    def upload_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video Files (*.mp4 *.avi)')
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.start(30)  # Start the video playback
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)  # Open default camera
        self.timer.start(30)  # Start the camera feed
    
    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
    
    def update_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Perform detection on the frame
                frame = self.detect_leak(frame)
                
                # Convert frame to RGB format for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(convert_to_qt_format)
                self.video_label.setPixmap(pixmap)
    
    def detect_leak(self, frame):
        # Get detections
        detections = get_detections(frame)
        
        # Draw bounding boxes on the frame
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['coordinates']
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Display class and confidence score
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return frame

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WaterLeakDetectorApp()
    window.show()
    sys.exit(app.exec_())
