object/libclblas.so: src/*.c
	gcc -g -fPIC -shared -o $@ $^ -I./include -lOpenCL

object/blastest:
	gcc -g -o $@ ./test/level1.c -I./include -lOpenCL -L./object -lclblas

all: object/libclblas.so object/blastest

clean:
	rm object/libclblas.so object/blastest