# VRD-IU Track B - YOLO11 Object Detection
# 最新のYOLO11を使用（YOLOv8より高性能）
FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /workspace

# Install YOLO11 (ultralytics>=8.3.0) and dependencies
RUN pip install --no-cache-dir \
    ultralytics>=8.3.0 \
    pandas \
    opencv-python-headless \
    albumentations \
    tqdm \
    scikit-learn \
    Pillow

# Set environment
ENV YOLO_VERBOSE=False

CMD ["bash"]
