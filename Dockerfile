FROM golang:1.21

# Install Python3, pip, git, bash, and dependencies needed for matplotlib
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    bash \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install matplotlib via pip
RUN pip3 install matplotlib

WORKDIR /app

# Clone the repo (replace with your repo URL)
ARG REPO_URL=https://github.com/JasZal/dyno.git
RUN git clone ${REPO_URL} .

# Make run.sh executable
RUN chmod +x run.sh

# Declare results folder as a volume
VOLUME ["/app/results"]

# Run the script
CMD ["./run.sh"]
