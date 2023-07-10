#!/bin/bash
g++ -O3 -DNDEBUG -std=c++20 -Wno-deprecated -pthread -I/HOME1/users/guests/jannr/github/BAEMM -I/HOME1/users/guests/jannr/anaconda3/envs/BEM/include -L/HOME1/users/guests/jannr/anaconda3/envs/BEM/lib -lOpenCL -lopenblas GaussNewtonStep_TPM.cpp -o /HOME1/users/guests/jannr/BEM/GaussNewton_TPM