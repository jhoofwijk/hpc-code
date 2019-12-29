CC = mpicc

.PHONY: all
all: MPI_Fempois GridDist

.PHONY: clean
clean:
	rm -f *.o 
	rm -f MPI_Fempois
	rm -f GridDist

MPI_Fempois: MPI_Fempois.o
	mpicc -o $@ $< -lm

GridDist: GridDist.o
	gcc -o $@ $< -lm

MPI_Fempois.o:
	mpicc -c MPI_Fempois.c

GridDist.o:
	gcc -c GridDist.c

.PHONY: inputs
inputs: GridDist
	./GridDist 2 2 100 100

.PHONY: run
run: MPI_Fempois inputs
	mpirun -c 4 MPI_Fempois
