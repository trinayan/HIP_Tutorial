all:
	hipcc -I/opt/rocm/include  -L/opt/rocm/lib -lhipblas *.cu -o CNN

run:
	./CNN
clean:
	rm CNN
