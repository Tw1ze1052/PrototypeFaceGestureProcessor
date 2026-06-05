# Detector de Gestos Faciales con MediaPipe y Streamlit

Este proyecto es una aplicación web desarrollada con Streamlit que utiliza la cámara web del usuario para detectar gestos faciales en tiempo real mediante MediaPipe Face Landmarker.

La aplicación funciona como un pequeño sistema de reconocimiento facial básico. Muestra la transmisión de la cámara como espejo y, a un costado, indica qué gesto facial está siendo reconocido.

## Gestos reconocidos

El programa puede identificar los siguientes gestos:

- Ojo izquierdo cerrado
- Ojo derecho cerrado
- Ambos ojos cerrados
- Boca abierta
- Rostro inexpresivo

Cada gesto se muestra en un panel llamado **Gestos**, acompañado de un ícono representativo.

## Tecnologías utilizadas

- Python
- Streamlit
- streamlit-webrtc
- MediaPipe
- OpenCV
- NumPy
- PyAV

## Requisitos

Se recomienda usar Python 3.13, ya que este proyecto está adaptado para las versiones recientes de MediaPipe.

Dependencias principales:

streamlit
streamlit-webrtc
mediapipe>=0.10.30
opencv-python-headless
numpy
av

## Instalación

Primero, clona o descarga este proyecto.

Después, abre una terminal en la carpeta del proyecto e instala las dependencias:

pip install -r requirements.txt

Si estás en Windows y quieres asegurarte de usar Python 3.13, puedes instalar las dependencias así:

python -m pip install -r requirements.txt

## Ejecución del proyecto

Para iniciar la aplicación, ejecuta:

streamlit run mediapipe_cam.py

En Windows, si quieres usar directamente tu instalación de Python 3.13, puedes ejecutar:

python -m streamlit run mediapipe_cam.py

Al abrirse la aplicación en el navegador, se mostrará un botón START. Al presionarlo, el navegador solicitará permiso para usar la cámara web.

## Funcionamiento general

La aplicación captura video en tiempo real desde la cámara del usuario. Cada frame es procesado con MediaPipe Face Landmarker para obtener puntos de referencia del rostro.

Con esos puntos se calculan medidas aproximadas de:

Apertura del ojo izquierdo
Apertura del ojo derecho
Apertura de la boca

A partir de esos valores, el programa clasifica el gesto facial detectado y lo muestra en el panel lateral.

## Ajustes disponibles

Desde la barra lateral se pueden modificar algunos parámetros:

Sensibilidad para detectar ojos cerrados
Sensibilidad para detectar boca abierta
Mostrar u ocultar puntos faciales
Invertir etiquetas izquierda/derecha en caso de que la cámara muestre los lados invertidos

Estos ajustes ayudan a mejorar la detección dependiendo de la iluminación, la cámara o la posición del rostro.

## Privacidad

La aplicación no guarda fotos ni videos del usuario. La cámara se utiliza únicamente para procesar los gestos faciales en tiempo real dentro de la sesión activa.
