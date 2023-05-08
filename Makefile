default:all

HYPRE_DIR = /home/dyt/Software/hypre_2.27/src

include ${HYPRE_DIR}/config/Makefile.config


CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

CDEFS = -DHYPRE_TIMING -DHYPRE_FORTRAN
CXXDEFS = -DNOFEI -DHYPRE_TIMING -DMPICH_SKIP_MPICXX

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I${HYPRE_BUILD_DIR}/include\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}\
 ${CDEFS}

CXX_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/../FEI_mv/fei-base\
 -I${HYPRE_BUILD_DIR}/include\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}\
 ${CXXDEFS}

F77_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}

MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
LAPACKLIBFLAGS = ${LAPACKLIBDIRS} ${LAPACKLIBS}
BLASLIBFLAGS = ${BLASLIBDIRS} ${BLASLIBS}
LIBFLAGS = ${LDFLAGS} ${LIBS}
LIBHYPRE = -L${HYPRE_BUILD_DIR}/lib -lHYPRE

ifeq ($(notdir $(firstword ${LINK_CC})), nvcc)
   XLINK = -Xlinker=-rpath,${HYPRE_BUILD_DIR}/lib
else
   XLINK = -Wl,-rpath,${HYPRE_BUILD_DIR}/lib
endif

LFLAGS =\
 $(LIBHYPRE)\
 ${XLINK}\
 ${DSUPERLU_LIBS}\
 ${SUPERLU_LIBS}\
 ${MPILIBFLAGS}\
 ${LAPACKLIBFLAGS}\
 ${BLASLIBFLAGS}\
 ${LIBFLAGS}

##################################################################
# Targets
##################################################################

HYPRE_DRIVERS =\
 maxwell_ams.c \
 hypre_superlu.c

HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.c=}
HYPRE_F77_EXAMPLES_DRIVER_EXECS=${HYPRE_F77_EXAMPLES_DRIVERS:.c=}
HYPRE_DRIVER_F77_EXECS=${HYPRE_DRIVERS_F77:.f=}
HYPRE_DRIVER_CXX_EXECS=${HYPRE_DRIVERS_CXX:.cxx=}

all: ${HYPRE_DRIVER_EXECS}

all77: ${HYPRE_DRIVER_F77_EXECS}

all++: ${HYPRE_DRIVER_CXX_EXECS}

install:

clean:
	rm -f maxwell_ams hypre_superlu
	rm -f *.o *.obj *.csv
	rm -rf pchdir tca.map *inslog*

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}
	rm -f ${HYPRE_F77_EXAMPLES_DRIVER_EXECS}
	rm -f ${HYPRE_DRIVER_F77_EXECS}
	rm -f ${HYPRE_DRIVER_CXX_EXECS} cxx_*
	rm -f maxwell_ams hypre_superlu

##################################################################
# Rules
##################################################################

# C

maxwell_ams: maxwell_ams.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS} 

hypre_superlu: hypre_superlu.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}