set -e

IMAGE_NAME="ebaa95/shufflenetv1_train:v1.4"

docker build -f Dockerfile -t $IMAGE_NAME .