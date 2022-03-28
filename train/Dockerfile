FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

# Install linux packages
RUN apt update && apt install -y zip htop libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# Install tool for session handling
RUN apt-get install -y tmux

WORKDIR ../
RUN git clone https://github.com/Ebiz95/ShuffleNetV1.git