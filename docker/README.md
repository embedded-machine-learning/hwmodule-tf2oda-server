# How to Setup a Docker Container for Tensorflow 2.7 (for EML Tools TF2ODA Inference)

In the following guide, we setup and run a docker container. As an example, we setup inference with the EML Tools (https://github.com/embedded-machine-learning/eml-tools) for TF2ODA (https://github.com/embedded-machine-learning/hwmodule-tf2oda-server). This container can be used to automatically setup the inference with TF2ODA.

This guide can also be found here: https://github.com/embedded-machine-learning/hwmodule-tf2oda-server   

## Commands
Basic commands: https://www.educative.io/edpresso/what-is-the-workdir-command-in-docker

```
docker ps: Show running containers
docker container ls: Show running containers
docker image ls: Show all images
docker rmi -f [IMAGEID]: Remove image with force by image id
docker rm -f [CONTAINER ID]: Remove container with force
docker commit [CONTAINER ID] [NAME]:[TAG]: Commit container as new image
```

Running scripts in the Dockerfile over the image: https://stackoverflow.com/questions/34549859/run-a-script-in-dockerfile

RUN and ENTRYPOINT are two different ways to execute a script. RUN means it creates an intermediate container, runs the script and freeze the new state of that container in a new intermediate image. The script won't be run after that: your final image is supposed to reflect the result of that script. ENTRYPOINT means your image (which has not executed the script yet) will create a container, and runs that script. In both cases, the script needs to be added, and a RUN chmod +x *.sh is a good idea to pass the rights


## How to build a docker image
CUDA and TF2 Base images: https://docs.valohai.com/howto/docker/docker-build-image/

Setup container 1: https://itproguru.com/expert/2016/10/docker-create-container-change-container-save-as-new-image-and-connect-to-container/

Setup container 2: https://docs.docker.com/get-started/02_our_app/

Setting up an ML Docker and running it with scripts: https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f


1. Create setup script
Filename ```Dockerfile```. The Dockerfile name is the default build script name and should not be changed:
```
FROM tensorflow/tensorflow:2.7.0-gpu

RUN apt-get update
RUN apt-get install curl
RUN apt-get install python3-pip
RUN pip3 install virtualenv
RUN apt -y install git

RUN mkdir -p /eml-tools
COPY ./bootstrap.sh /eml-tools
WORKDIR /eml-tools
RUN chmod +x *.sh
RUN ./bootstrap.sh

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility 
```

RUN apt -y install git: Use -y to automatically always select Yes for all options. Else, the program aborts.


2. Build the Docker image from Dockerfile
```
docker build -t eml_tf2oda_inference:tf2_2.7.0-gpu .
```
NAME: eml_tf2oda_inference

TAG: tf2_2.7.0-gpu

3. Update Image in Interactive Mode
If not everything could be loaded from the image or you forgot something, the container has to be updated. This is done by using ```docker commit```. 
https://phoenixnap.com/kb/how-to-commit-changes-to-docker-image 

First start the container and do your changes within the container, e.g. create new folder or install new libs.

Open another PS Window and look for the running container ID
```
docker ps
```
From the state in the container, create an image by using 
```
docker commit 092ac5e9d46b eml_tf2oda_inference:tf2_2.7.0-gpu
```
Where docker commit [CONTAINER ID] [NAME]:[TAG]

After everything has been installed, the image can be run.

## Running a Container

1. Prepare folders to share in the container
We need to share the dataset and the project directory with the exported network file.

Dataset: C:\Projekte\21_SoC_EML\datasets\dataset-oxford-pets-val-debug. The dataset has the EML format and will be directly linked to the container at start

Project: eml_projects\test-project. The project has an exported network and a customized script tf2oda_inf_eval_saved_model_tf2oda_ssdmobilenetv2_300x300_pets_D100.sh

The best way of editing files in a docker container is to edit the files in a shared folder.

Guide for Powershell: https://adamtheautomator.com/docker-volume-create/

Guide for Linux: https://blog.softwaremill.com/editing-files-in-a-docker-container-f36d76b9613c

The following parts are added to docker run
```
-v C:\Projekte\Docker_training\tf27\eml_projects\test-project:/eml-tools/eml_projects/test-project `
-v C:\Projekte\21_SoC_EML\datasets\dataset-oxford-pets-val-debug:/eml-tools/dataset `
```
-v [SOURCE FOLDER]:[DESTINATION_FOLDER]

How to mount a local volume under Windows (a little bit more complicated than Linux): 
https://stackoverflow.com/questions/62045513/docker-run-rm-v-getting-error-response-from-daemon-status-code-not-ok-but . “when you run Docker in windows you need to specifically give Docker access to this location. To give Docker access to your computer’s drives, right click on the Docker icon in your taskbar, then click “Settings…” and look for the "File Sharing" section. Add the location”

2. Run container in interactive mode and execute the inference script

Execute ``` start_inference_real_data_ir.ps1```
```
docker run `
--rm `
-it `
-v C:\Projekte\Docker_training\tf27\eml_projects\test-project:/eml-tools/eml_projects/test-project `
-v C:\Projekte\21_SoC_EML\datasets\dataset-oxford-pets-val-debug:/eml-tools/dataset `
-p 80:80 `
eml_tf2oda_inference:tf2_2.7.0-gpu_2
```
Go to the project folder 
```
cd eml_projects/test-project/
```

And run

```
./tf2oda_inf_eval_saved_model_tf2oda_ssdmobilenetv2_300x300_pets_D100.sh 
```

The result can then be found in ```eml_projects\test-project\results```.

3. Run container script from Powershell
In a script, ``` start_inference_real_data.ps1```, the system runs linux commands from a windows Powershell.

Docker Execution Script for Powershell
```
docker run `
-v C:\Projekte\Docker_training\tf27\eml_projects\test-project:/eml-tools/eml_projects/test-project `
-v C:\Projekte\21_SoC_EML\datasets\dataset-oxford-pets-val-debug:/eml-tools/dataset `
-p 80:80 `
eml_tf2oda_inference:tf2_2.7.0-gpu_2 `
/bin/bash `
-c "cd eml_projects && cd test-project && ./tf2oda_inf_eval_saved_model_tf2oda_ssdmobilenetv2_300x300_pets_D100.sh" 
```

Where 
"cd eml_projects && 
cd test-project && 
./tf2oda_inf_eval_saved_model_tf2oda_ssdmobilenetv2_300x300_pets_D100.sh”

Is the command chain to run in the container.

The result can then be found in ```eml_projects\test-project\results```.


## Issues
Solutions for syntax problems: https://jhooq.com/invalid-reference-format-error/#3-pwd-and-pwd-path
