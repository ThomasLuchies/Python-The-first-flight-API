FROM python:3.8.0-buster

#Directory for the api
WORKDIR .

#install dependancies
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install qt5-default -y

RUN pip install cmake
RUN touch CMakeLists.txt

RUN cmake -DWITH_QT=OFF -DWITH_GTK=OFF

COPY /API .

CMD ["python", "detect.py"]
