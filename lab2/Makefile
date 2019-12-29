CC = mpicc

all: MPI_Fempois GridDist

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



