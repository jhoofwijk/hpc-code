GridX = 600
GridY = 600

X = 10
Y = 10
P = $$(( $(X) * $(Y) ))

.PHONY: all
all: run

.PHONY: clean
clean:
	rm -f *.o 
	rm -f MPI_Fempois
	rm -f GridDist
	rm -f input*-*.dat
	rm -f output*.dat
	rm -f mapping*.dat

MPI_Fempois: MPI_Fempois.o
	mpicc -o $@ $< -lm

GridDist: GridDist.o
	gcc -o $@ $< -lm

MPI_Fempois.o: MPI_Fempois.c
	mpicc -c $<

GridDist.o: GridDist.c
	gcc -c $<

.PHONY: inputs
inputs: GridDist
	./GridDist $(X) $(Y) $(GridX) $(GridY)

.PHONY: adaptive
adaptive: GridDist MPI_Fempois 
	rm -f output*.dat
	./GridDist $(X) $(Y) $(GridX) $(GridY) adapt
	mpirun -c $(P) MPI_Fempois

.PHONY: run
run: MPI_Fempois inputs
	rm -f output*.dat
	mpirun -c $(P) MPI_Fempois

.PHONY: concat
concat: run
	cat output*-*.dat | sort > output.dat


.PHONY: 4_2
4_2: GridDist MPI_Fempois
	rm -f output*.dat
	./GridDist 1 4 100 100
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 1 4 200 200
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 1 4 400 400
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 100 100
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 200 200
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 400 400
	mpirun -c 4 MPI_Fempois


.PHONY: 4_5a
4_5a: GridDist MPI_Fempois
	rm -f output*.dat
	./GridDist 2 2 10 10
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 20 20
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 30 30
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 25 25
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 8 8
	mpirun -c 4 MPI_Fempois

.PHONY: 4_5b
4_5b: GridDist MPI_Fempois
	rm -f output*.dat
	./GridDist 2 2 1000 1000
	prun -v -np 1 -4 -sge-script $ PRUN ETC/prun-openmpi ./MPI_Fempois

	rm -f output*.dat
	./GridDist 3 3 1000 1000
	prun -v -np 1 -9 -sge-script $ PRUN ETC/prun-openmpi ./MPI_Fempois

	rm -f output*.dat
	./GridDist 3 3 1000 1000
	prun -v -np 3 -3 -sge-script $ PRUN ETC/prun-openmpi ./MPI_Fempois

	rm -f output*.dat
	./GridDist 4 4 1000 1000
	prun -v -np 2 -8 -sge-script $ PRUN ETC/prun-openmpi ./MPI_Fempois

	rm -f output*.dat
	./GridDist 5 5 1000 1000
	prun -v -np 5 -5 -sge-script $ PRUN ETC/prun-openmpi ./MPI_Fempois

	rm -f output*.dat
	./GridDist 6 6 1000 1000
	prun -v -np 3 -12 -sge-script $ PRUN ETC/prun-openmpi ./MPI_Fempois

	rm -f output*.dat
	./GridDist 10 10 1000 1000
	prun -v -np 10 -10 -sge-script $ PRUN ETC/prun-openmpi ./MPI_Fempois

.PHONY: 4_6
4_6: GridDist MPI_Fempois
	rm -f output*.dat
	./GridDist 2 2 100 100 adapt
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 200 200 adapt
	mpirun -c 4 MPI_Fempois

	rm -f output*.dat
	./GridDist 2 2 400 400 adapt
	mpirun -c 4 MPI_Fempois

