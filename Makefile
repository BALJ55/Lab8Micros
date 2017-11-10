NVCC =/usr/local/cuda-7.5/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
OPENCV_FLAGS = `pkg-config --cflags --libs opencv`
TARGET =filter


$(TARGET): main.o kernel.o
	$(NVCC) $(OPENCV_FLAGS) $^ -o $@

main.o: main.cpp kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
