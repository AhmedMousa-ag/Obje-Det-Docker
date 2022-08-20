FROM tensorflow/tensorflow:devel-gpu

RUN pip install tensorflow==2.8

WORKDIR /app/

VOLUME "H:\Programming\ML\object_detection\dockerized-obj-det":/app/

# You must copy your train/test data to your docker image
# COPY Data .
COPY . /app/

RUN /app/bash-script/install-packages.sh


ENTRYPOINT ["python"]

