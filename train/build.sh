set -e

IMAGE_NAME="ebaa95/shufflenetv1_train:v1.2"

docker build -f Dockerfile -t $IMAGE_NAME .