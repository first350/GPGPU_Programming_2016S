#!/bin/bash
nvcc -std=c++11 -c lab2.cu -o lab2.o
echo "lab2.o done"
nvcc -std=c++11 main.cu lab2.o -o main.run 
echo "main.run done"
./main.run
avconv -i result.y4m result.mkv
