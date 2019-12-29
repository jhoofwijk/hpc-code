GridX = 100
GridY = 100

X = 2
Y = 2
P = $$(( $(X) * $(Y) ))

.PHONY: all
all: run

.PHONY: clean
clean:
	rm -f *.o 
	rm -f MPI_Fempois
	rm -f GridDist
	rm input*-*.dat
	rm output*.dat

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