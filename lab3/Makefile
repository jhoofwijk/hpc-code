
all: clean run

run:
	sbatch power_gpu.job

all_block_sizes:
	# 32
	make run

	# 64
	cat power_gpu.cu | sed 's/32/64/' | tee power_gpu.cu
	make run

	# 128
	cat power_gpu.cu | sed 's/64/128/' | tee power_gpu.cu
	make run

	# cleanup
	cat power_gpu.cu | sed 's/128/32/' | tee power_gpu.cu

all_sizes:
	cat power_gpu.job | sed -E 's/size [0-9]+/size 50/' | tee power_gpu.job
	make all_block_sizes

	cat power_gpu.job | sed -E 's/size [0-9]+/size 500/' | tee power_gpu.job
	make all_block_sizes

	cat power_gpu.job | sed -E 's/size [0-9]+/size 2000/' | tee power_gpu.job
	make all_block_sizes


clean:
	rm -f slurm*.out
	rm -f HelloWorld
	rm -f power_gpu
