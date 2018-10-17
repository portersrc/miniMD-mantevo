# Multiple-machine Makefile

SHELL = /bin/sh

# Files

SRC =	ljs.c input.c integrate.c atom.c force_lj.c force_eam.c neighbor.c \
	thermo.c comm.c timer.c output.c setup.c
INC =	ljs.h atom.h force.h neighbor.h thermo.h timer.h comm.h integrate.h threadData.h variant.h openmp.h \
	force_lj.h force_eam.h types.h

# Definitions

ROOT =	miniMD
EXE =	$(ROOT)_$@
OBJ =	$(SRC:.c=.o)

# Help

help:
	@echo 'Type "make target" where target is one of:'
	@echo '      nvidia-pgi  (Compile with PGI for NVIDIA GPUs)'
	@echo '      host-pgi    (Compile with PGI for CPUs)'

# Targets

nvidia-pgi:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

host-pgi:
	@if [ ! -d Obj_$@ ]; then mkdir Obj_$@; fi
	@cp -p $(SRC) $(INC) Obj_$@
	@cp Makefile.$@ Obj_$@/Makefile
	@cd Obj_$@; \
	$(MAKE)  "OBJ = $(OBJ)" "INC = $(INC)" "EXE = ../$(EXE)" ../$(EXE)
#       @if [ -d Obj_$@ ]; then cd Obj_$@; rm $(SRC) $(INC) Makefile*; fi

# Clean

clean:
	rm -r Obj_*

clean_nvidia-pgi:
	rm -r Obj_nvidia-pgi

clean_host-pgi:
	rm -r Obj_host-pgi

# Test

scope=0
input=lj
halfneigh=0
path=""
test:
	bash run_tests ${scope} ${input} ${halfneigh} ${path}   
