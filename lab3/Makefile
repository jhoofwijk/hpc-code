
all: clean run

run:
	sbatch power_gpu.job

all_sizes:
	# GRID 50
	cat power_gpu.job | sed -E 's/size [0-9]+/size 50/' | tee power_gpu.job
	# 32
	sbatch power_gpu.job
	# 64
	cat power_gpu.cu | sed 's/32/64/' | tee power_gpu.cu
	sbatch power_gpu.job
	# 128
	cat power_gpu.cu | sed 's/64/128/' | tee power_gpu.cu
	sbatch power_gpu.job
	# cleanup
	cat power_gpu.cu | sed 's/128/32/' | tee power_gpu.cu


	# GRID 500
	cat power_gpu.job | sed -E 's/size [0-9]+/size 500/' | tee power_gpu.job
	# 32
	sbatch power_gpu.job
	# 64
	cat power_gpu.cu | sed 's/32/64/' | tee power_gpu.cu
	sbatch power_gpu.job
	# 128
	cat power_gpu.cu | sed 's/64/128/' | tee power_gpu.cu
	sbatch power_gpu.job
	# cleanup
	cat power_gpu.cu | sed 's/128/32/' | tee power_gpu.cu


	# GRID 2000
	cat power_gpu.job | sed -E 's/size [0-9]+/size 2000/' | tee power_gpu.job
	# 32
	sbatch power_gpu.job
	# 64
	cat power_gpu.cu | sed 's/32/64/' | tee power_gpu.cu
	sbatch power_gpu.job
	# 128
	cat power_gpu.cu | sed 's/64/128/' | tee power_gpu.cu
	sbatch power_gpu.job
	# cleanup
	cat power_gpu.cu | sed 's/128/32/' | tee power_gpu.cu
	


clean:
	rm -f slurm*.out
	rm -f HelloWorld
	rm -f power_gpu
