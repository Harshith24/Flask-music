FROM python:3.8-slim-buster
WORKDIR /app
# RUN useradd -m developer
RUN apt-get update -y && apt-get install -y build-essential cmake \
libsm6 libxext6 libxrender-dev \
python3 python3-pip python3-dev

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
# USER root
# RUN usermod -a -G video developer
# USER developer
COPY . /app
# DEVICES /dev/video0:/dev/video0
# EXPOSE 5000
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]