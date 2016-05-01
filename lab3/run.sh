#!/bin/bash
g++ -std=c++11 -c pgm.cpp -o pgm.o
echo "pgm done"
nvcc -std=c++11 -c lab3.cu -o lab3.o
echo "lab3.o done"
nvcc -std=c++11  main.cu lab3.o pgm.o -o main.run -arch sm_30 -D_MWAITXINTRIN_H_INCLUDED 
echo "main.run done"
./main.run ./lab3_test/img_background.ppm ./lab3_test/img_target.ppm ./lab3_test/img_mask.pgm 130 600 myoutput.ppm  

