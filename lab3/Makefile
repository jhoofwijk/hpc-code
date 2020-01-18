
global: clean
	sbatch power_gpu_global.job

all: clean run

run:
	sbatch power_gpu.job

all_sizes:
	cat power_gpu.cu | sed 's/32/32/'  | tee power_gpu32.cu
	cat power_gpu.cu | sed 's/32/64/'  | tee power_gpu64.cu
	cat power_gpu.cu | sed 's/32/16/'  | tee power_gpu16.cu
	# cat power_gpu.cu | sed 's/32/128/' | tee power_gpu128.cu


	# GRID 50
	cat power_gpu.job | sed -E 's/size [0-9]+/size 50/' | tee power_gpu.job
	# 16
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu16/' | tee power_gpu.job
	sbatch power_gpu.job
	# 32
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu32/' | tee power_gpu.job
	sbatch power_gpu.job
	# 64
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu64/' | tee power_gpu.job
	sbatch power_gpu.job


	# GRID 500
	cat power_gpu.job | sed -E 's/size [0-9]+/size 500/' | tee power_gpu.job
	# 16
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu16/' | tee power_gpu.job
	sbatch power_gpu.job
	# 32
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu32/' | tee power_gpu.job
	sbatch power_gpu.job
	# 64
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu64/' | tee power_gpu.job
	sbatch power_gpu.job


	# GRID 2000
	cat power_gpu.job | sed -E 's/size [0-9]+/size 2000/' | tee power_gpu.job
	# 16
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu16/' | tee power_gpu.job
	sbatch power_gpu.job
	# 32
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu32/' | tee power_gpu.job
	sbatch power_gpu.job
	# 64
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu64/' | tee power_gpu.job
	sbatch power_gpu.job


	# GRID 5000
	cat power_gpu.job | sed -E 's/size [0-9]+/size 5000/' | tee power_gpu.job
	# 16
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu16/' | tee power_gpu.job
	sbatch power_gpu.job
	# 32
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu32/' | tee power_gpu.job
	sbatch power_gpu.job
	# 64
	cat power_gpu.job | sed -E 's/power_gpu[0-9]*/power_gpu64/' | tee power_gpu.job
	sbatch power_gpu.job




clean:
	rm -f slurm*.out
	rm -f HelloWorld
	rm -f power_gpu
	rm -f power_gpu16 power_gpu32 power_gpu64 power_gpu128 power_gpu_global
	rm -f power_gpu16.cu power_gpu32.cu power_gpu64.cu power_gpu128.cu 

