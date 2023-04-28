#!/bin/bash

g++ -g -Wall -I~/github/BAEMM -I~/anaconda3/envs/BEM/include -fopenmp -L~/anaconda3/envs/BEM/lib -lgomp -OpenCL test.cpp -o test