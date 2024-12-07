from deepface import DeepFace
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

class VideoEmotionAnalyzer:
    def __init__(self):
        self.emotions_history = {
            'angry': [], 'disgust': [], 'fear': [], 
            'happy': [], 'sad': [], 'surprise': [], 'neutral': []
        }
        
    def analyze_frame(self, frame):
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            return result[0]['emotion']
        except Exception as e:
            return None

    def update_history(self, emotions):
        for emotion in self.emotions_history:
            self.emotions_history[emotion].append(emotions.get(emotion, 0))
            # Keep only last 30 frames of history
            if len(self.emotions_history[emotion]) > 30:
                self.emotions_history[emotion].pop(0)

    def draw_emotions_on_frame(self, frame, emotions):
        if emotions is None:
            return frame
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (10, 10), (200, 150), (0, 0, 0), -1)
        
        y = 30
        for emotion, score in emotions.items():
            text = f"{emotion}: {score:.2f}%"
            cv2.putText(frame, text, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 255, 255), 1)
            y += 20
            
        return frame

    def analyze_video(self, source=0):  # 0 for webcam, or provide video file path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        print("Starting video analysis... Press 'q' to quit")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Only analyze every 3rd frame to improve performance
            if frame_count % 3 == 0:
                emotions = self.analyze_frame(frame)
                if emotions:
                    self.update_history(emotions)
                    frame = self.draw_emotions_on_frame(frame, emotions)

            # Show FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Emotion Analysis', frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Save final emotion history plot
        self.save_emotion_plot()

    def save_emotion_plot(self):
        plt.figure(figsize=(12, 6))
        for emotion in self.emotions_history:
            plt.plot(self.emotions_history[emotion], label=emotion)
        
        plt.title('Emotion Analysis Over Time')
        plt.xlabel('Frames')
        plt.ylabel('Confidence Score (%)')
        plt.legend()
        plt.grid(True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'emotion_analysis_{timestamp}.png')
        print(f"\nEmotion analysis plot saved as emotion_analysis_{timestamp}.png")

def main():
    analyzer = VideoEmotionAnalyzer()
    
    # For webcam, use:
    analyzer.analyze_video(0)
    
    # For video file, use:
    # analyzer.analyze_video("path_to_your_video.mp4")

if __name__ == "__main__":
    main()