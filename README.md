# Real-time Emotion Analysis with DeepFace

A Python application that performs real-time emotion analysis using the DeepFace framework. This tool can analyze emotions from webcam feed or video files, displaying results in real-time and generating emotion trend graphs.

## Features

- Real-time emotion detection
- Support for webcam and video file input
- Live emotion confidence scoring
- FPS monitoring
- Historical emotion tracking
- Automated graph generation
- Multi-emotion detection:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

## Prerequisites

- Python 3.7+
- Required packages:
```bash
pip install deepface opencv-python numpy matplotlib
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
python deepface.py
```
This will start emotion analysis using your default webcam.

### Using Video Files
Modify the `main()` function to analyze a video file:
```python
analyzer = VideoEmotionAnalyzer()
analyzer.analyze_video("path_to_your_video.mp4")
```

## Controls

- Press 'q' to quit the analysis
- Close the window to end the session

## Output

### Real-time Display
- Emotion confidence scores
- FPS counter
- Live video feed with overlay

### Generated Files
- Emotion analysis graph (`emotion_analysis_YYYYMMDD_HHMMSS.png`)
  - X-axis: Frames
  - Y-axis: Confidence scores
  - Individual lines for each emotion

## Performance Optimization

- Analyzes every 3rd frame to maintain performance
- Uses OpenCV backend for faster detection
- Maintains rolling 30-frame history
- Silent mode for reduced console output

## Class: VideoEmotionAnalyzer

### Methods

#### `analyze_frame(frame)`
- Processes single video frame
- Returns emotion confidence scores
- Handles detection failures gracefully

#### `update_history(emotions)`
- Maintains emotion history
- Rolling 30-frame window
- Tracks all seven emotions

#### `draw_emotions_on_frame(frame, emotions)`
- Overlays emotion scores on video
- Includes background for readability
- Updates every analyzed frame

#### `analyze_video(source)`
- Handles video input stream
- Displays real-time analysis
- Generates performance metrics

#### `save_emotion_plot()`
- Creates visualization of emotion trends
- Saves timestamped plot file
- Includes all tracked emotions

## Error Handling

- Video source validation
- Frame processing failures
- Detection errors
- Resource cleanup

## Best Practices

1. Ensure good lighting for accurate detection
2. Position face clearly in frame
3. Monitor system resources for long sessions
4. Keep face at moderate distance from camera
5. Check saved plots for emotion patterns

## Known Limitations

- Performance depends on hardware
- May struggle with multiple faces
- Lighting affects accuracy
- Processing speed varies by system

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- DeepFace framework
- OpenCV project
- Matplotlib library
