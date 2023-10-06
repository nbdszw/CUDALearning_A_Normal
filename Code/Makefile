# 指定编译器
CC = nvcc

# 指定编译选项
CFLAGS = -arch=sm_86  # 根据你的GPU架构选择适当的选项

# 获取当前文件夹内的所有cu文件
CUDA_SRC = $(wildcard *.cu)

# 生成可执行文件的列表，将每个cu文件的名称用于可执行文件名
EXECUTABLES = $(patsubst %.cu,%,$(CUDA_SRC))

all: $(EXECUTABLES)

%: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(EXECUTABLES)
