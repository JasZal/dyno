# Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning

This repository is a benchmarking of a secure noisy functional encryption scheme called DyNo and a demonstration how it can be used to train a logistic regression. 
It is implemented under go version 1.21.4. 

The artifact consists of software that was used to create Table 3, as well as Figures 2-3 in the paper  'Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning' by Scheu-Hachtel and Zalonis, 2025. 

The example uses a forked and modified implementation of the [gofe library](https://github.com/JasZal/gofe) and the [differential privacy library](https://github.com/google/differential-privacy). 
The addition to the gofe library are different schemes including the scheme "DyNo" itself. 
Our example in the folder "benchmarking" shows a comparison between the "Dyno" and another noisy FE scheme called "DiffPipe", that are used to create Table 3 in the original Paper ("Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning").
The code in folder "log_reg" demonstrates the training of a logistic regression using the proposed protocoll with DyNo. 

## Description
This artifact is the source code that was used to measure the scheme linked to Table 3-5 and  Figures 2-3 in the paper  'Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning' by Scheu-Hachtel and Zalonis, 2025. 

## Basic Requirements

### Hardware Requirements
at least 8 GB RAM (to include the largest Dataset 50GB)

### Software Requirements
- OS: Ubuntu (at least version 20.04)
- Software: go (at least version 1.21.4), bazel (at least version 7.5.0), python using matplotlib and numpy


## Set up the environment
Either use the presented Dockerfile or follow the above instruction:

## Docker: (Assuming Ubuntu 22)
- install docker
```bash
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo tee /etc/apt/sources.list.d/docker.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```
- download Dockerfile

## Without Docker:
(Assuming Ubuntu 22)
- install go on your system (https://go.dev/doc/install)
```bash
curl -O -L "https://golang.org/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz" 
tar -xf "go${GO_VERSION}.linux-${ARCH}.tar.gz" && mv -v go /usr/local
echo 'export PATH=$PATH:/usr/local/go/bin' >>$HOME/.profile
echo 'export PATH=$PATH:$HOME/go/bin' >>$HOME/.profile
```

-install python on your system, including matplotlib and numpy
```bash
sudo apt install -y python3 python3-pip
pip3 install matplotlib numpy
```

- clone the artifact (https://github.com/JasZal/dyno)
```bash
git clone https://github.com/JasZal/dyno
```

## Run Code
Docker:
navigate in the folder where the dockerfile is and run
```bash
docker build -t dyno .
docker run --rm -v $(pwd)/artifactResults:/artifactResults dyno
```

Without Docker:
navigate in the cloned git and run
```bash
./run.sh
```

This will start the benchmarking and log_reg evaluation and will store all results in the folder "artifactResults". 
Following a python script will build the figures as displayed in the paper. 
Runtimes are stored in the respective txt files. 
Note that the number of rounds over which the average is taken is currently 3 due to efficiency but can be increased in the file 'log_reg/main.go' in Line 60 (resp. 93 for Nhanes) to obtain a more robust result. 


## License
This code is licensed under [GPLv3](https://github.com/JasZal/dyno/blob/main/LICENSE.txt)


