#!/bin/bash
g++ -g -O3 -DNDEBUG -std=c++20 -Wno-deprecated -I/HOME1/users/guests/jannr/github/BAEMM -I/HOME1/users/guests/jannr/anaconda3/envs/BEM/include -L/HOME1/users/guests/jannr/anaconda3/envs/BEM/lib -lOpenCL -lopenblas FarFieldOperator.cpp -o /HOME1/users/guests/jannr/BEM/FarField