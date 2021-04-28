# File              : Makefile
# Author            : Anton Riedel <anton.riedel AT tum.de>
# Date              : 26.04.2021
# Last Modified Date: 28.04.2021
# Last Modified By  : Philipp Haas <philipp.haas AT tum.de>

NVCC := /usr/bin/nvcc
CXX := /usr/bin/g++-8

default: all

all: HelloWorld vectorAdd MatrixMultiplication MatrixMultOneBlock MatrixMultOneThread PascalTriangle OpenGL

HelloWorld: HelloWorld.cu
	$(NVCC) -ccbin $(CXX) -o $@ $^

vectorAdd: vectorAdd.cu
	$(NVCC) -ccbin $(CXX) -o $@ $^

MatrixMultiplication: MatrixMultiplication.cu
	$(NVCC) -ccbin $(CXX) -o $@ $^

MatrixMultOneBlock: MatrixMultOneBlock.cu
	$(NVCC) -ccbin $(CXX) -o $@ $^

MatrixMultOneThread: MatrixMultOneThread.cu
	$(NVCC) -ccbin $(CXX) -o $@ $^

PascalTriangle: PascalTriangle.cu
	$(NVCC) -ccbin $(CXX) -o $@ $^

OpenGL: OpenGL.cpp
	$(CXX) -o $@ $^ -lglut -lGLU -lGL

clean:
	$(RM) HelloWorld vectorAdd MatrixMultiplication MatrixMultOneBlock MatrixMultOneThread PascalTriangle OpenGL

.PHONY: all clean
