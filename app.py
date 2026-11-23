"""Flask web entry point for the FSL sign-to-text app."""

from collections import Counter, deque
import os
from threading import Lock
import uuid

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "local-development-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

_recognizer = None
_model_lock = Lock()
_states = {}


def get_recognizer():
    """Load the existing recognition models only when the first frame arrives."""
    global _recognizer
    if _recognizer is None:
        with _model_lock:
            if _recognizer is None:
                from body_sequence_recog import BodySequenceRecog
                from hand_pose_recog import HandPoseRecog

                _recognizer = {
                    "hand": HandPoseRecog(),
                    "body": BodySequenceRecog(),
                }
    return _recognizer


def get_state():
    client_id = session.get("client_id")
    if not client_id:
        client_id = uuid.uuid4().hex
        session["client_id"] = client_id
    if client_id not in _states:
        _states[client_id] = {
            "hand_predictions": deque(maxlen=25),
            "body_frames": deque(maxlen=20),
            "body_predictions": deque(maxlen=6),
            "output": [],
        }
    return _states[client_id]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/manifest.webmanifest")
def manifest():
    return app.send_static_file("manifest.webmanifest")


@app.post("/api/recognize")
def recognize():
    uploaded = request.files.get("frame")
    if uploaded is None:
        return jsonify(error="No camera frame was uploaded."), 400

    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv.imdecode(raw, cv.IMREAD_COLOR)
    if image is None:
        return jsonify(error="The uploaded frame could not be read."), 400

    # Keep processing bounded on phones and when the service is hosted remotely.
    height, width = image.shape[:2]
    if width > 640:
        scale = 640 / width
        image = cv.resize(image, (640, int(height * scale)))
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    models = get_recognizer()
    state = get_state()

    with _model_lock:
        hands = models["hand"].mp_hands.process(rgb)
        pose = models["body"].mp_pose.process(rgb)

        hand_frames = {"Left": models["hand"].zeros, "Right": models["hand"].zeros}
        current_sign = None
        if hands.multi_hand_landmarks:
            for landmarks, handedness in zip(hands.multi_hand_landmarks, hands.multi_handedness):
                side = handedness.classification[0].label
                points = models["hand"].calc_landmarks(image, landmarks)
                # A perfectly still/degenerated landmark set cannot be normalized.
                if max(abs(value) for point in points for value in point) == 0:
                    continue
                normalized = models["hand"].preprocess_landmarks(points, side)
                hand_frames[side] = normalized
                prediction_id, confidence = models["hand"].recog_model.predict_with_confidence(normalized)
                if confidence < models["hand"].recog_model.MIN_CONFIDENCE:
                    state["hand_predictions"].clear()
                    continue
                prediction = models["hand"].labels[prediction_id]
                current_sign = prediction
                if prediction not in ("Error", "Finger heart"):
                    state["hand_predictions"].append(prediction)
                else:
                    state["hand_predictions"].clear()
        else:
            state["hand_predictions"].clear()

        if pose.pose_landmarks:
            points = models["body"].calc_landmarks(image, pose.pose_landmarks)
            pose_frame = models["body"].preprocess_landmarks(points)
        else:
            pose_frame = models["body"].zeros
        hand_pose_ready = len(state["hand_predictions"]) == state["hand_predictions"].maxlen
        if hand_pose_ready:
            state["body_frames"].clear()
            state["body_predictions"].clear()
        else:
            state["body_frames"].append(pose_frame + hand_frames["Left"] + hand_frames["Right"])

            if len(state["body_frames"]) == 20:
                sign_id, confidence = models["body"].recog_model(list(state["body_frames"]))
                if confidence >= 80 and models["body"].labels[sign_id] != "error":
                    state["body_predictions"].append(models["body"].labels[sign_id])
                if len(state["body_predictions"]) == 6 and not state["hand_predictions"]:
                    word = Counter(state["body_predictions"]).most_common(1)[0][0]
                    if not state["output"] or state["output"][-1] != word:
                        state["output"].append(word)
                    state["body_predictions"].clear()
                    state["body_frames"].clear()

        if len(state["hand_predictions"]) == 25:
            word = Counter(state["hand_predictions"]).most_common(1)[0][0]
            if not state["output"] or state["output"][-1] != word:
                state["output"].append(word)
            state["hand_predictions"].clear()

    return jsonify(sign=current_sign, text=" ".join(state["output"]))


@app.post("/api/clear")
def clear_output():
    state = get_state()
    state["output"].clear()
    state["hand_predictions"].clear()
    state["body_predictions"].clear()
    state["body_frames"].clear()
    return jsonify(text="")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
