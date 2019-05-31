FROM tensorflow/tensorflow:1.9.0-gpu-py3
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    python3-tk \
    make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
EXPOSE 8888 8888
CMD /bin/bash
