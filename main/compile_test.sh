#!/bin/bash

g++ -g -Wall -O3 -DNDEBUG -I~/github/BAEMM -I~/anaconda3/envs/BEM/include -fopenmp -L~/anaconda3/envs/BEM/lib -lgomp -lOpenCL test.cpp -o test