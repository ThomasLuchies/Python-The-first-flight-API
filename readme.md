# Introduction

The code in this repository makes it possible for the students to use AI with the Tello drone. The student communicates with an API. Both the API and the AI are in this repository.

# Setup 



## install docker

you can install docker from the tutorial on this website: https://docs.docker.com/engine/install/debian/



## setup shared volume

For the AI and API to communicate they have to use a volume. In this volume there will be two files that are automatically generated by the AI and then use d by the API.

To create it use the following command:

```bash
docker volume create shared-resources
```



## Setup API

To setup the API you need to first build it. You can do that with the following command:

```bash
cd API
docker build -t api .
```

After the docker image has been build it you need to run it. You can do that with the following command:

```bash
docker run -ti --rm -v shared-volume:/shared-volume -p 5000:5000 api
```



## Setup AI

To setup the AI you need to first build it. You can do that with the following command:

```bash
cd AI
docker build -t ai .
```

After the docker image has been build it you need to run it. You can do that with the following command:

```bash
docker run -ti --rm -v shared-volume:/shared-volume --network=host -v /run/dbus/:/run/dbus/ -p 11111:11111 ai
```

## API documentation

To view the API documentation download postman from the following link: https://www.postman.com/. Once downloaded click on FIle->import and select TelloDroneAPI.postman_collection.json in the API folder.ss