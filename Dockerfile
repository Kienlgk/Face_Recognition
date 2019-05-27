FROM tensorflow/tensorflow:1.9.0-gpu-py3
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y python3-tk
RUN pip install -r requirements.txt
RUN export PYTHONPATH=lib/src
EXPOSE 5000
CMD /bin/bash