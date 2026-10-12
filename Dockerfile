FROM python:3.11-slim

# opencv (both opencv-python-headless, which requirements.txt asks for, and
# plain opencv-python, which easyocr pulls in transitively) still dlopens a
# couple of shared libs at import time; fonts-dejavu-core gives render.py a
# real font to draw translated text with.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Plain `pip install torch` (pulled in via manga-ocr in requirements.txt)
# picks a build that supports CUDA if the machine building/running this
# image has it, and runs on CPU otherwise — no build-time choice needed.
# That also means a larger image than a CPU-only build. If you specifically
# want the smaller CPU-only wheel (e.g. you know this is going on a
# GPU-less box and want the disk back), install it explicitly first:
#   RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
