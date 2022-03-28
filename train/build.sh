set -e

IMAGE_NAME="ebaa95/shufflenetv1_train:v1.0"

docker build -f Dockerfile -t $IMAGE_NAME .