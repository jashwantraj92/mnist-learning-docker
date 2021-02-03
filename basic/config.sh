#!/bin/bash

docker_image_name="mnist-training-cpu"
docker_image_version="latest"
docker_repository="jashwant"

export docker_image_id="${docker_repository}/${docker_image_name}:${docker_image_version}"
