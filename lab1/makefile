
all: a.out run

a.out: PAR_Poisson.c
	mpicc PAR_Poisson.c

run: 11

11: a.out
	mpirun -c 1 a.out 1 1

12: a.out
	mpirun -c 2 a.out 1 2

13: a.out
	mpirun -c 3 a.out 1 3

14: a.out
	mpirun -c 4 a.out 1 4

22: a.out
	mpirun -c 4 a.out 2 2

23: a.out
	mpirun -c 6 a.out 2 3

24: a.out
	mpirun -c 8 a.out 2 4

33: a.out
	mpirun -c 9 a.out 3 3


