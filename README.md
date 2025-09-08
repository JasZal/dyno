# Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning

This repository is a benchmarking of a secure noisy functional encryption scheme called DyNo and a demonstration how it can be used to train a logistic regression. 
It is implemented under go version 1.21.4. 

The artifact consists of software that was used to create Table 3, as well as Figures 2-3 in the paper  'Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning' by Scheu-Hachtel and Zalonis, 2025. 

The example uses a forked and modified implementation of the [gofe library](https://github.com/JasZal/gofe) and the [differential privacy library](https://github.com/google/differential-privacy). 
The addition to the gofe library are different schemes including the scheme "DyNo" itself. 
Our example in the folder "benchmarking" shows a comparison between the "Dyno" and another noisy FE scheme called "DiffPipe", that are used to create Table 3 in the original Paper ("Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning").
The code in folder "log_reg" demonstrates the training of a logistic regression using the proposed protocoll with DyNo. 


## TL;DR

To run the example first install the forked [gofe library](https://github.com/JasZal/gofe) and the [differential privacy library](https://github.com/google/differential-privacy). Make sure you have installed all dependencies, e.g. bazel.
You can either run the run.sh script or to produce the individual claims for benchmarking and log reg individually as described below. 
For the runtimes of Table 3, please into the 'benchmarking' folder and run benchmarking.go.
To compute the runtime for the logistic regression navigate to 'log_reg' folder and run main.go


## Description
This artifact is the source code that was used to measure the scheme linked to Table 3-5 and  Figures 2-3 in the paper  'Enhancing Noisy Functional Encryption for Privacy-Preserving Machine Learning' by Scheu-Hachtel and Zalonis, 2025. 

## Basic Requirements

### Hardware Requirements
at least 8 GB RAM

### Software Requirements
- OS: Ubuntu (at least version 20.04)
- Software: go (at least version 1.21.4), bazel (at least version 6.4.0), python using matplotlib


## Set up the environment
Follow the above instruction:

(Assuming Ubuntu 20.04)
- install go on your system (https://go.dev/doc/install)
```bash
curl -O -L "https://golang.org/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz" 
tar -xf "go${GO_VERSION}.linux-${ARCH}.tar.gz" && mv -v go /usr/local
echo 'export PATH=$PATH:/usr/local/go/bin' >>$HOME/.profile
echo 'export PATH=$PATH:$HOME/go/bin' >>$HOME/.profile
```
  
- install bazel on your system, at least version 7.5.0 (https://bazel.build/install)
```bash
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list 
sudo apt  update -y 
sudo apt install bazel=7.5.0
```

- clone and install the differntial privacy library from google (https://github.com/google/differential-privacy)
```bash 
git clone https://github.com/google/differential-privacy.git
cd differential-privacy
cd go
bazel build ...
cd ../..
```

- clone the artifact (https://github.com/JasZal/dyno)
```bash
git clone https://github.com/JasZal/dyno
```

## Run Code
now you can run the source code of the experiment by starting the run.sh script or start the claims individually as described above. This will start the benchmarking and log_reg evaluation and will store all results in the folder "results". 
Following a python script will build the figures as displayed in the paper. Runtimes are stored in the respective txt files. 






