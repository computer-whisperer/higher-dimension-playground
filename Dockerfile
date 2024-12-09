FROM nvidia/cuda:12.6.3-base-ubuntu24.04

WORKDIR /usr/src/tesseract
COPY . .


RUN apt update && apt install -y \
    build-essential \
    libxext6 \
    libegl1 \
    python3 \
    cmake \
    git \
    rustup \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    libnvidia-gl-565
RUN rustup default nightly-2024-04-24
RUN cargo build

COPY nvidia_icd.json /etc/vulkan/icd.d

CMD ["/bin/sh -c cargo run -- --headless"]
