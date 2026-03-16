"""
===============================================================================
===============================================================================

This file is the main backend server of my Human Action Recognition project.
It connects the frontend interface (HTML pages) with the AI model that
performs action recognition.

The system allows users to:

1) Upload a video and analyze human actions
2) Run real-time webcam action recognition
3) Perform browser-based frame prediction
4) Generate a timeline of detected actions

The backend is implemented using Flask and integrates OpenCV, NumPy,
and my CNN-LSTM model.

-------------------------------------------------------------------------------
1. LIBRARIES USED
-------------------------------------------------------------------------------

Flask Components:

Flask
    Creates the web application server.

render_template
    Renders HTML pages from the templates folder.

request
    Used to receive uploaded files and data from the frontend.

Response
    Used to stream video frames to the browser.

jsonify
    Returns prediction results in JSON format for APIs.


Computer Vision and Processing:

OpenCV (cv2)
    Used for video capture, frame processing, and drawing labels.

NumPy
    Used for converting image data into arrays for model input.


Utilities:

os
    Used for file management and directory creation.

collections.deque
    Used as a frame buffer to store recent frames for sequence prediction.

Counter
    Used to stabilize predictions by taking the most common action.

-------------------------------------------------------------------------------
2. MODEL FUNCTIONS
-------------------------------------------------------------------------------

Two main model functions are imported from:

scripts.predict_action

run_video()
    Used to process a complete uploaded video.

run_frames()
    Used to process a sequence of frames (for webcam or browser prediction).

These functions use the trained CNN-LSTM model to predict actions.

-------------------------------------------------------------------------------
3. APPLICATION INITIALIZATION
-------------------------------------------------------------------------------

app = Flask(__name__)

This creates the Flask web application.

UPLOAD_FOLDER = "static/uploads"

This directory stores uploaded video files.

The following command ensures the folder exists:

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

-------------------------------------------------------------------------------
4. GLOBAL VARIABLES
-------------------------------------------------------------------------------

action_timeline
    Stores the timeline of actions detected in uploaded videos.

browser_buffer
    Stores the latest 16 frames from browser input for prediction.

browser_history
    Stores recent predictions to smooth the final output.

Frame buffering is important because the LSTM model needs a sequence
of frames to understand motion over time.

-------------------------------------------------------------------------------
5. HOME PAGE ROUTE
-------------------------------------------------------------------------------

Route:

    "/"

This route serves the main page of the application.

Users can upload a video using the form on index.html.

Workflow:

1) User uploads a video
2) The file is saved to the upload folder
3) The filename is passed to the HTML page
4) The page displays the video and starts prediction

-------------------------------------------------------------------------------
6. WEBCAM PAGE
-------------------------------------------------------------------------------

Route:

    "/webcam"

This loads the webcam interface page (webcam.html).

The page connects to the backend video streaming endpoint.

-------------------------------------------------------------------------------
7. WEBCAM STREAMING SYSTEM
-------------------------------------------------------------------------------

Function:

generate_frames()

This function captures frames from the user's webcam.

Process:

1) Open webcam using OpenCV
2) Capture frames continuously
3) Resize frames to 224x224 (model input size)
4) Convert frames to RGB format
5) Store frames in a sequence buffer

Once 16 frames are collected:

The model predicts the action using:

run_frames()

Prediction smoothing:

Recent predictions are stored in a history queue,
and the most frequent action is displayed.

-------------------------------------------------------------------------------
8. DRAWING PREDICTIONS ON VIDEO
-------------------------------------------------------------------------------

OpenCV is used to draw text on frames:

Action label
Confidence score
Explanation

Functions used:

cv2.putText()

This overlays prediction information directly on the video.

-------------------------------------------------------------------------------
9. STREAMING VIDEO TO BROWSER
-------------------------------------------------------------------------------

Frames are encoded as JPEG and streamed using:

multipart/x-mixed-replace

This creates a continuous video stream in the browser.

The route used:

    /video_feed

-------------------------------------------------------------------------------
10. VIDEO STREAMING FOR UPLOADED VIDEOS
-------------------------------------------------------------------------------

Route:

    /video_stream/<filename>

This processes uploaded videos frame-by-frame and streams
AI predictions while the video plays.

Function:

generate_video_frames()

Steps:

1) Read frames from the uploaded video
2) Store frames in sequence buffer
3) Run model prediction
4) Draw results on video frames
5) Stream frames to the browser

-------------------------------------------------------------------------------
11. ACTION TIMELINE GENERATION
-------------------------------------------------------------------------------

While processing the video, timestamps of detected actions are recorded.

Example:

(0.5s, Punch)
(2.1s, Kick)
(3.8s, Jump)

After processing, actions are compressed into time segments:

Example timeline:

0.0s – 2.0s : Punch
2.0s – 3.5s : Kick
3.5s – 5.0s : Jump

This timeline is stored in:

action_timeline

-------------------------------------------------------------------------------
12. FRAME PREDICTION API
-------------------------------------------------------------------------------

Route:

    /predict_frame

Used for browser-based prediction.

The frontend sends frames captured from the video player.

Steps:

1) Receive frame as image blob
2) Decode frame using OpenCV
3) Resize and convert color
4) Add frame to sequence buffer
5) Run model prediction
6) Return JSON response

Example JSON response:

{
    "action": "Punch",
    "confidence": 92.4,
    "explanation": "Rapid forward arm movement detected."
}

-------------------------------------------------------------------------------
13. TIMELINE API
-------------------------------------------------------------------------------

Route:

    /timeline

Returns the detected action timeline as JSON.

Example:

{
    "timeline":[
        [0.0,2.0,"Punch"],
        [2.0,4.0,"Kick"]
    ]
}

This can be used for visualization or analysis.

-------------------------------------------------------------------------------
14. RUNNING THE APPLICATION
-------------------------------------------------------------------------------

The Flask server starts with:

app.run(debug=True, host="0.0.0.0", port=5000)

This allows:

• local testing
• browser access
• debugging during development

-------------------------------------------------------------------------------
15. PROJECT ARCHITECTURE
-------------------------------------------------------------------------------

Frontend
(index.html / webcam.html)
        ↓
Flask Backend (app.py)
        ↓
Frame Processing (OpenCV)
        ↓
CNN Feature Extraction
        ↓
LSTM Temporal Analysis
        ↓
Action Prediction + Explanation
        ↓
Displayed in Browser

-------------------------------------------------------------------------------
16. PURPOSE OF THIS SYSTEM
-------------------------------------------------------------------------------

The goal of this project is to create an explainable human action
recognition system that:

• Detects actions from video streams
• Uses deep learning (CNN-LSTM)
• Provides confidence scores
• Generates explanations for predictions
• Works with uploaded videos and live webcam

This makes the system useful for research, surveillance, sports analysis,
and intelligent video understanding.

-------------------------------------------------------------------------------
===============================================================================
"""
from flask import Flask, render_template, request, Response, jsonify
import os
import cv2
import numpy as np
from collections import Counter, deque
from scripts.predict_action import run_video, run_frames

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

action_timeline = []

#  Increased buffer for better motion understanding
browser_buffer = deque(maxlen=16)
browser_history = []


# HOME PAGE 
@app.route("/", methods=["GET", "POST"])
def index():

    video_file = None

    if request.method == "POST":

        video = request.files.get("video")

        if video and video.filename != "":
            filepath = os.path.join(UPLOAD_FOLDER, video.filename)
            video.save(filepath)
            video_file = video.filename

    return render_template("index.html", video_file=video_file)


#  WEBCAM PAGE

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


#  WEBCAM STREAM 

def generate_frames():

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    frame_buffer = deque(maxlen=16)
    frame_count = 0

    action = "Detecting..."
    confidence = 0
    explanation = ""

    history = deque(maxlen=10)

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame_count += 1

        small = cv2.resize(frame, (224,224))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        frame_buffer.append(rgb)

        #  Predict every few frames 
        if len(frame_buffer) == 16 and frame_count % 12 == 0:

            try:

                action, confidence, explanation = run_frames(list(frame_buffer))

                history.append(action)

                action = Counter(history).most_common(1)[0][0]

            except:
                action = "Processing..."
                confidence = 0
                explanation = ""

        #  DRAW LABELS

        cv2.putText(frame, f"Action: {action}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)

        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        cv2.putText(frame, explanation[:60], (20,135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    camera.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


#  REAL-TIME VIDEO STREAM 

@app.route("/video_stream/<filename>")
def video_stream(filename):

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    return Response(
        generate_video_frames(filepath),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_video_frames(video_path):

    global action_timeline

    cap = cv2.VideoCapture(video_path)

    frame_buffer = deque(maxlen=16)
    frame_count = 0

    action = "Detecting..."
    confidence = 0
    explanation = ""

    history = deque(maxlen=10)
    raw_timeline = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        small = cv2.resize(frame, (224,224))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        frame_buffer.append(rgb)

        if len(frame_buffer) == 16 and frame_count % 12 == 0:

            try:

                action, confidence, explanation = run_frames(list(frame_buffer))

                history.append(action)

                action = Counter(history).most_common(1)[0][0]

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                raw_timeline.append((round(timestamp,1), action))

            except:
                action = "Processing..."
                confidence = 0
                explanation = ""

        #  DRAW LABELS 

        cv2.putText(frame, f"Action: {action}", (20,70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)

        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20,120),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.putText(frame, explanation[:60], (20,170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()

    compressed = []

    if raw_timeline:

        start_time = raw_timeline[0][0]
        current_action = raw_timeline[0][1]

        for t, act in raw_timeline[1:]:

            if act != current_action:
                compressed.append((start_time, t, current_action))
                start_time = t
                current_action = act

        compressed.append((start_time, raw_timeline[-1][0], current_action))

    action_timeline = compressed


#  FRAME PREDICTION API 

@app.route("/predict_frame", methods=["POST"])
def predict_frame():

    file = request.files["frame"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    small = cv2.resize(frame,(224,224))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    browser_buffer.append(rgb)

    if len(browser_buffer) < 16:
        return jsonify({
            "action":"Detecting...",
            "confidence":0,
            "explanation":"Collecting frames..."
        })

    action, confidence, explanation = run_frames(list(browser_buffer))

    browser_history.append(action)

    if len(browser_history) > 10:
        browser_history.pop(0)

    action = Counter(browser_history).most_common(1)[0][0]

    return jsonify({
        "action": action,
        "confidence": round(confidence,2),
        "explanation": explanation
    })


#  TIMELINE API 

@app.route("/timeline")
def timeline():
    return jsonify({"timeline": action_timeline})


# RUN APP
# ============================================================
# Flask Application Entry Point
# ============================================================
# This section starts the Flask server.
# It is required when deploying to cloud platforms like Render.

import os

if __name__ == "__main__":
    
    # ------------------------------------------------------------
    # Render (and most cloud platforms) provide a PORT environment
    # variable. We must run the server on that port so the platform
    # can route traffic to our application.
    #
    # If the PORT variable does not exist (for example when running
    # locally on our laptop), we default to port 5000.
    # ------------------------------------------------------------
    port = int(os.environ.get("PORT", 5000))
    
    # ------------------------------------------------------------
    # host="0.0.0.0"
    # This allows the server to accept requests from outside
    # the local machine (required for cloud deployment).
    #
    # Without this, the app would only listen to localhost
    # and Render would not detect the open port.
    # ------------------------------------------------------------
    app.run(host="0.0.0.0", port=port)