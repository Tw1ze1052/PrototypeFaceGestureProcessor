import time
from math import dist
from pathlib import Path
from threading import Lock
from urllib.request import urlretrieve

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# Configuración general
# =========================================================

st.set_page_config(
    page_title="Detector de Gestos Faciales",
    page_icon="🙂",
    layout="wide"
)

st.title("Detector de cambios faciales con MediaPipe Face Landmarker")

st.write(
    "Este programa usa la cámara web para reconocer gestos faciales básicos: "
    "ojo izquierdo cerrado, ojo derecho cerrado, ambos ojos cerrados, boca abierta "
    "y rostro inexpresivo."
)

st.info(
    "Presiona **START** y acepta el permiso de cámara del navegador. "
    "La cámara se mostrará como espejo en el recuadro izquierdo."
)


# =========================================================
# Descarga automática del modelo Face Landmarker
# =========================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


def ensure_model_exists():
    if MODEL_PATH.exists():
        return

    try:
        with st.spinner("Descargando modelo Face Landmarker..."):
            urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        st.error(
            "No se pudo descargar el modelo de MediaPipe. "
            "Revisa tu conexión a internet o descarga manualmente el archivo "
            "`face_landmarker.task` y colócalo en la misma carpeta que este script."
        )
        st.exception(error)
        st.stop()


ensure_model_exists()


# =========================================================
# Datos iniciales
# =========================================================

def default_data():
    return {
        "face_detected": False,
        "gestures": [
            {
                "icon": "🙂",
                "name": "Rostro inexpresivo"
            }
        ],
        "left_ear": 0.0,
        "right_ear": 0.0,
        "mouth_ratio": 0.0,
    }


# =========================================================
# Utilidades de cálculo
# =========================================================

def landmark_point(landmarks, index):
    point = landmarks[index]
    return point.x, point.y


def eye_aspect_ratio(landmarks, indexes):
    """
    Calcula qué tan abierto está un ojo.
    Mientras menor sea el valor, más cerrado está el ojo.
    """
    p1 = landmark_point(landmarks, indexes[0])
    p2 = landmark_point(landmarks, indexes[1])
    p3 = landmark_point(landmarks, indexes[2])
    p4 = landmark_point(landmarks, indexes[3])
    p5 = landmark_point(landmarks, indexes[4])
    p6 = landmark_point(landmarks, indexes[5])

    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_open_ratio(landmarks):
    """
    Calcula la apertura de la boca.
    Mientras mayor sea el valor, más abierta está la boca.
    """
    upper_lip = landmark_point(landmarks, 13)
    lower_lip = landmark_point(landmarks, 14)
    left_mouth = landmark_point(landmarks, 78)
    right_mouth = landmark_point(landmarks, 308)

    vertical = dist(upper_lip, lower_lip)
    horizontal = dist(left_mouth, right_mouth)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def classify_gestures(
    left_ear,
    right_ear,
    mouth_ratio,
    eye_threshold,
    mouth_threshold
):
    left_eye_closed = left_ear < eye_threshold
    right_eye_closed = right_ear < eye_threshold
    mouth_open = mouth_ratio > mouth_threshold

    gestures = []

    if left_eye_closed and right_eye_closed:
        gestures.append({
            "icon": "😑",
            "name": "Ambos ojos cerrados"
        })
    elif left_eye_closed:
        gestures.append({
            "icon": "👈😉",
            "name": "Ojo izquierdo cerrado"
        })
    elif right_eye_closed:
        gestures.append({
            "icon": "😉👉",
            "name": "Ojo derecho cerrado"
        })

    if mouth_open:
        gestures.append({
            "icon": "😮",
            "name": "Boca abierta"
        })

    if not gestures:
        gestures.append({
            "icon": "🙂",
            "name": "Rostro inexpresivo"
        })

    return gestures


def draw_landmarks_overlay(image, landmarks):
    """
    Dibuja puntos faciales básicos sobre el frame.
    Esto reemplaza el antiguo mp.solutions.drawing_utils.
    """
    height, width = image.shape[:2]

    # Dibuja puntos de toda la cara
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Contorno aproximado de ojos y boca
    left_eye = [362, 385, 387, 263, 373, 380, 362]
    right_eye = [33, 160, 158, 133, 153, 144, 33]
    mouth = [
        61, 185, 40, 39, 37, 0, 267, 269, 270,
        409, 291, 375, 321, 405, 314, 17, 84,
        181, 91, 146, 61
    ]

    for region in [left_eye, right_eye, mouth]:
        points = []

        for index in region:
            landmark = landmarks[index]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed=False, color=(255, 255, 255), thickness=2)


def render_gesture_panel(data):
    lines = []

    lines.append("## Gestos")
    lines.append("")

    for gesture in data["gestures"]:
        lines.append(f"### {gesture['icon']} {gesture['name']}")

    lines.append("---")

    if data["face_detected"]:
        lines.append("✅ **Rostro detectado correctamente**")
    else:
        lines.append("**Esperando detección del rostro...**")

    lines.append("")
    lines.append(f"**Ojo izquierdo:** `{data['left_ear']:.3f}`")
    lines.append(f"**Ojo derecho:** `{data['right_ear']:.3f}`")
    lines.append(f"**Apertura de boca:** `{data['mouth_ratio']:.3f}`")

    return "\n\n".join(lines)


# =========================================================
# Procesador de video usando MediaPipe Tasks
# =========================================================

class FaceGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = Lock()
        self.data = default_data()

        self.eye_threshold = 0.21
        self.mouth_threshold = 0.18
        self.show_landmarks = True
        self.swap_eye_labels = False

        self.frame_timestamp_ms = 0

        # Índices de Face Landmarker / Face Mesh
        # Interpretados desde la perspectiva anatómica del usuario.
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        base_options = python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")

        # MediaPipe trabaja con RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        # En modo VIDEO, MediaPipe necesita timestamps crecientes
        self.frame_timestamp_ms += 33

        result = self.landmarker.detect_for_video(
            mp_image,
            self.frame_timestamp_ms
        )

        current_gestures = [
            {
                "icon": "🙂",
                "name": "Rostro inexpresivo"
            }
        ]

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            raw_left_ear = eye_aspect_ratio(landmarks, self.LEFT_EYE)
            raw_right_ear = eye_aspect_ratio(landmarks, self.RIGHT_EYE)
            mouth_ratio = mouth_open_ratio(landmarks)

            if self.swap_eye_labels:
                left_ear = raw_right_ear
                right_ear = raw_left_ear
            else:
                left_ear = raw_left_ear
                right_ear = raw_right_ear

            current_gestures = classify_gestures(
                left_ear=left_ear,
                right_ear=right_ear,
                mouth_ratio=mouth_ratio,
                eye_threshold=self.eye_threshold,
                mouth_threshold=self.mouth_threshold
            )

            with self.lock:
                self.data = {
                    "face_detected": True,
                    "gestures": current_gestures,
                    "left_ear": left_ear,
                    "right_ear": right_ear,
                    "mouth_ratio": mouth_ratio,
                }

            if self.show_landmarks:
                draw_landmarks_overlay(image_bgr, landmarks)

        else:
            with self.lock:
                self.data = default_data()

        # Se voltea al final para que funcione como espejo
        image_bgr = cv2.flip(image_bgr, 1)

        # Texto sobre la cámara
        y = 40
        for gesture in current_gestures:
            cv2.putText(
                image_bgr,
                gesture["name"],
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y += 40

        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.header("Ajustes de detección")

    eye_threshold = st.slider(
        "Sensibilidad para ojos cerrados",
        min_value=0.10,
        max_value=0.35,
        value=0.21,
        step=0.01
    )

    mouth_threshold = st.slider(
        "Sensibilidad para boca abierta",
        min_value=0.05,
        max_value=0.40,
        value=0.18,
        step=0.01
    )

    show_landmarks = st.checkbox(
        "Mostrar puntos faciales",
        value=True
    )

    swap_eye_labels = st.checkbox(
        "Invertir etiquetas izquierda/derecha",
        value=False
    )

    st.caption(
        "Si el ojo izquierdo y derecho aparecen invertidos, activa "
        "'Invertir etiquetas izquierda/derecha'."
    )


# =========================================================
# Layout principal
# =========================================================

camera_col, gestures_col = st.columns([4, 1])

with camera_col:
    st.subheader("Cámara / Espejo")

    ctx = webrtc_streamer(
        key="facemesh-new-mediapipe-tasks",
        video_processor_factory=FaceGestureProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
    )

with gestures_col:
    gesture_placeholder = st.empty()

    with gesture_placeholder.container(border=True):
        st.markdown(render_gesture_panel(default_data()))


# =========================================================
# Actualización del panel de gestos
# =========================================================

while ctx.state.playing:
    if ctx.video_processor:
        ctx.video_processor.eye_threshold = eye_threshold
        ctx.video_processor.mouth_threshold = mouth_threshold
        ctx.video_processor.show_landmarks = show_landmarks
        ctx.video_processor.swap_eye_labels = swap_eye_labels

        with ctx.video_processor.lock:
            current_data = ctx.video_processor.data.copy()

        gesture_placeholder.empty()

        with gesture_placeholder.container(border=True):
            st.markdown(render_gesture_panel(current_data))

    time.sleep(0.15)