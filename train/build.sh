set -e

IMAGE_NAME="ebaa95/shufflenetv1_train"

docker build -f Dockerfile -t $IMAGE_NAME .