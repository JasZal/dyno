FROM golang:1.24

# Install Python3, pip, git, bash, and dependencies needed for matplotlib
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-matplotlib \
    python3-numpy \
    git \
    bash \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Clone the repo (replace with your repo URL)
ARG REPO_URL=https://github.com/JasZal/dyno.git
RUN git clone ${REPO_URL} .

# Make run.sh executable
RUN chmod +x ./claim1/run.sh

# Declare results folder as a volume
VOLUME ["/app/claim1Results"]

# Run the script either with or without the biggest dataset Nhanes
# default: run without Nhanes
CMD ["./run.sh"]

#to include Nhanes change above line to 
#CMD ["./run.sh --includeNahnes"]
