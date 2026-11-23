# Filipino Sign Language Translator

A computer-vision prototype for recognizing selected Filipino Sign Language
(FSL) hand poses and body gestures. The project includes a Windows desktop
application and a mobile-friendly Flask web interface for sign-to-text.

> **Project status:** This is an experimental project, not a certified
> translation service. Recognition is limited to the classes represented in
> the included models and data; results may be inaccurate and should not be
> relied on for important communication.

## Features

- **Desktop app:** camera-based sign recognition, speech/sign interface, and
  tools for collecting and training examples.
- **Web app:** browser camera preview and live sign-to-text transcript.
- **Hand-pose model:** the included labels are the letters A through I.
- **Facial-expression model:** the included labels are Angry, Questioning,
  Happy, Neutral, Sad, and Surprise. Facial-expression display is part of the
  desktop app, not the current web interface.
- **Body-sequence model:** model code and weights are included, but the
  `data/body_sequence` folder has no training examples in this checkout.
  Collect and train body-sequence data before expecting body gestures to be
  recognized.

## Requirements

- Windows is the primary target for the desktop application.
- Python 3.11 is used by the included development environment. The pinned
  dependencies in `reqs.txt` are from the project's original setup and may
  require adjustment for other Python versions or operating systems.
- A webcam is needed for live recognition.

## Setup

Open PowerShell in the project directory and create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r reqs.txt
```

If PowerShell prevents virtual-environment activation, either allow activation
for your user according to your organization's policy or invoke the environment
interpreter directly as `.\.venv\Scripts\python.exe`.

## Run the desktop app

```powershell
python main.py
```

Allow camera access if prompted. The desktop application opens its sign-to-text
camera view; use its options to access the available collection and training
controls.

## Run the web app

Start the Flask development server:

```powershell
python app.py
```

Open <http://127.0.0.1:5000/> on the same computer and allow camera access.
Flask loads the recognition models when the first camera frame is submitted.
Stop the server with `Ctrl+C`.

The browser captures camera frames and sends them to this server for
recognition while the camera is active. The web interface currently provides
sign-to-text only; desktop speech-to-sign, training controls, and the facial
expression overlay have not been ported.

### Access from another device

The browser needs a secure context for camera access. `http://127.0.0.1` is
treated as secure on the same computer, but a phone generally needs an HTTPS
address. For a temporary demo, use a trusted HTTPS tunnel. Anyone with the
tunnel URL may be able to access the app, so share it carefully and stop the
tunnel when finished.

The included Flask development server is **not intended for public production
hosting**. The app does not add authentication or access control. For a real
deployment, use a production WSGI server behind HTTPS, configure a private
`FLASK_SECRET_KEY`, and add suitable access controls. Never commit real
secrets.

## Project layout

| Path | Description |
| --- | --- |
| `main.py` | Desktop app entry point |
| `sign_textv2.py`, `gui_sign_to_text.py` | Desktop sign-to-text interface |
| `app.py` | Flask web server and recognition API |
| `templates/`, `static/` | Web page, styles, browser code, and app manifest |
| `hand_pose_recog.py`, `body_sequence_recog.py`, `face_recog.py` | Landmark processing and recognition wrappers |
| `models/recog_models.py` | TensorFlow Lite and Keras model definitions |
| `models/weights/` | Trained model weights |
| `data/hand/`, `data/face/` | Included training data and class labels |
| `data/body_sequence/` | Location for body-sequence training examples |
| `train_func.py` | Desktop model-training functions |
| `reqs.txt` | Pinned Python dependencies |

Large animation assets are excluded from this repository. Local environments,
temporary training data, and Python cache files are also excluded by
`.gitignore`.
