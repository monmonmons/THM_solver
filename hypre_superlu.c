#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* hypre/AMS prototypes */
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE.h"
#include "_hypre_utilities.hpp"

/* hypre/dsuperlu pretotypes*/
// #include "dsuperlu.h"
#include "superlu_ddefs.h"

/*----------------------------------------------------------------------
 * Build matrix from one file on Proc. 0. Expects matrix to be in
 * CSR format. Distributes matrix across processors giving each about
 * the same number of rows.
 *----------------------------------------------------------------------*/

HYPRE_Int
ReadParFromOneFile(char *filename,
                   HYPRE_ParCSRMatrix *A_ptr,
                   HYPRE_BigInt *row_partitioning,
                   HYPRE_BigInt *col_partitioning)
{
    HYPRE_ParCSRMatrix A;
    HYPRE_CSRMatrix A_CSR = NULL;

    HYPRE_Int myid;

    FILE *test;
    /*-----------------------------------------------------------
     * Initialize some stuff
     *-----------------------------------------------------------*/

    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

    /*-----------------------------------------------------------
     * Parse command line
     *-----------------------------------------------------------*/
    if (!(test = fopen(filename, "r")))
    {
        hypre_printf("Can't find the input file \"%s\"\n", filename);
        exit(1);
    }

    /*-----------------------------------------------------------
     * Print driver parameters
     *-----------------------------------------------------------*/

    if (myid == 0)
    {
        hypre_printf("  FromFile: %s\n", filename);

        /*-----------------------------------------------------------
         * Generate the matrix
         *-----------------------------------------------------------*/

        A_CSR = HYPRE_CSRMatrixRead(filename);

        // hypre_printf("%d , %d\n", myid, hypre_CSRMatrixNumNonzeros((hypre_CSRMatrix *)A_CSR) );
    }

    HYPRE_CSRMatrixToParCSRMatrix(hypre_MPI_COMM_WORLD, A_CSR, row_partitioning, col_partitioning, &A);

    *A_ptr = A;

    if (myid == 0)
    {
        HYPRE_CSRMatrixDestroy(A_CSR);
    }

    fclose(test);

    return (0);
}

/*----------------------------------------------------------------------
 * Build rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

HYPRE_Int
ReadRhsParFromOneFile(char *filename,
                      HYPRE_BigInt *partitioning,
                      HYPRE_ParVector *b_ptr)
{
    HYPRE_Int myid;
    HYPRE_ParVector b;
    HYPRE_Vector b_CSR = NULL;

    FILE *test;
    /*-----------------------------------------------------------
     * Initialize some stuff
     *-----------------------------------------------------------*/

    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

    /*-----------------------------------------------------------
     * Parse command line
     *-------------------------------------------------N----------*/

    if (!(test = fopen(filename, "r")))
    {
        hypre_printf("Can't find the input file \"%s\"\n", filename);
        exit(1);
    }

    /*-----------------------------------------------------------
     * Print driver parameters
     *-----------------------------------------------------------*/

    if (myid == 0)
    {
        hypre_printf("  Rhs FromFile: %s\n", filename);

        /*-----------------------------------------------------------
         * Generate the matrix
         *-----------------------------------------------------------*/

        b_CSR = HYPRE_VectorRead(filename);
    }

    HYPRE_VectorToParVector(hypre_MPI_COMM_WORLD, b_CSR, partitioning, &b);

    *b_ptr = b;

    HYPRE_VectorDestroy(b_CSR);

    return (0);
}

hypre_int main(hypre_int argc, char *argv[])
{
    HYPRE_Int num_procs, myid;
    HYPRE_Int time_index;

    HYPRE_Solver solver, precond, slusolver;

    /* iterative options */
    HYPRE_Int example, print_level;
    HYPRE_Int solver_id, refine, rhs_type;

    /* ams options */
    HYPRE_Int dim, coordinates;

    HYPRE_Real rhs_norm, final_res_norm;

    HYPRE_ParCSRMatrix A = 0, G = 0, Aalpha = 0, Abeta = 0, M = 0;
    HYPRE_ParCSRMatrix A_G = 0;
    HYPRE_ParVector x0 = 0, b = 0;
    HYPRE_ParVector x = 0, y = 0, z = 0;

    /* default execution policy and memory space */
#if defined(HYPRE_TEST_USING_HOST)
    HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
    HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_HOST;
#else
    HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_DEVICE;
    HYPRE_ExecutionPolicy default_exec_policy = HYPRE_EXEC_DEVICE;
#endif

    /* Initialize MPI */
    hypre_MPI_Init(&argc, &argv);
    hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs);
    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);

    /*-----------------------------------------------------------------
     * GPU Device binding
     * Must be done before HYPRE_Init() and should not be changed after
     *-----------------------------------------------------------------*/
    hypre_bind_device(myid, num_procs, hypre_MPI_COMM_WORLD);

    /*-----------------------------------------------------------
     * Initialize : must be the first HYPRE function to call
     *-----------------------------------------------------------*/
    HYPRE_Init();

    /* default memory location */
    HYPRE_SetMemoryLocation(memory_location);

    /* default execution policy */
    HYPRE_SetExecutionPolicy(default_exec_policy);

    example = 1;
    rhs_type = 2;
    print_level = 3;
    solver_id = 0;
    refine = 0;
    dim = 3;
    coordinates = 1;

    HYPRE_Int arg_index = 0;
    HYPRE_Int print_usage = 0;

    /* Parse command line */
    {
        while (arg_index < argc)
        {
            if (strcmp(argv[arg_index], "-solver") == 0)
            {
                arg_index++;
                solver_id = atoi(argv[arg_index++]);
            }
            else if (strcmp(argv[arg_index], "-refine") == 0)
            {
                arg_index++;
                refine = atoi(argv[arg_index++]);
            }
            else if (strcmp(argv[arg_index], "-rhs") == 0)
            {
                arg_index++;
                rhs_type = atoi(argv[arg_index++]);
            }
            else if (strcmp(argv[arg_index], "-help") == 0)
            {
                print_usage = 1;
                break;
            }
            else
            {
                arg_index++;
            }
        }
    }

    if (argc == 1)
    {
        print_usage = 1;
    }

    if (print_usage)
    {
        hypre_MPI_Finalize();
        return (0);
    }

    {
        /*   From one file  ---  hypre CSR format  */

#if 1
        char *file_locate = "/home/dyt/matrix/filter_matrix_hypre";
        char M_file[100], G_file[100];
        char rhs_file_real[100], rhs_file_imag[100], rhs_file_real_iter[100], rhs_file_imag_iter[100];
        char xcood_file[100], ycood_file[100], zcood_file[100];

        if (refine == 0)
        {
            sprintf(M_file, "%s/refine_0/hypre_CSR.M", file_locate);
            sprintf(G_file, "%s/refine_0/hypre_CSR.G", file_locate);
            sprintf(rhs_file_real, "%s/refine_0/seq_b0_real", file_locate);
            sprintf(rhs_file_imag, "%s/refine_0/seq_b0_imag", file_locate);
            sprintf(rhs_file_real_iter, "%s/refine_0/seq_b1_real", file_locate);
            sprintf(rhs_file_imag_iter, "%s/refine_0/seq_b1_imag", file_locate);
            sprintf(xcood_file, "%s/refine_0/seq_x", file_locate);
            sprintf(ycood_file, "%s/refine_0/seq_y", file_locate);
            sprintf(zcood_file, "%s/refine_0/seq_z", file_locate);
        }
        else if (refine == 1)
        {
            sprintf(M_file, "%s/refine_1/hypre_CSR.M", file_locate);
            sprintf(G_file, "%s/refine_1/hypre_CSR.G", file_locate);
            sprintf(rhs_file_real, "%s/refine_1/seq_b0_real", file_locate);
            sprintf(rhs_file_imag, "%s/refine_1/seq_b0_imag", file_locate);
            sprintf(rhs_file_real_iter, "%s/refine_1/seq_b1_real", file_locate);
            sprintf(rhs_file_imag_iter, "%s/refine_1/seq_b1_imag", file_locate);
            sprintf(xcood_file, "%s/refine_1/seq_x", file_locate);
            sprintf(ycood_file, "%s/refine_1/seq_y", file_locate);
            sprintf(zcood_file, "%s/refine_1/seq_z", file_locate);
        }

        // ReadParFromOneFile("/home/matrix/solverchanllenge2021/solverchallenge21_01/hypre_matrix01", &A, NULL, NULL);

        ReadParFromOneFile(M_file, &A, NULL, NULL);

        HYPRE_BigInt *row_partition, *col_partition;
        HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A, &row_partition);
        HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)A, &col_partition);

        // for (int i = 0; i <= num_procs; i++)
        //    hypre_printf("myid = %d -- row_partition[%d] = %d ;\n", myid, i, row_partition[i]);

        if (rhs_type == 0)
        {
            b = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A), row_partition[num_procs], row_partition);
            hypre_ParVectorInitialize(b);
            hypre_ParVectorSetRandomValues(b, 1);
        }
        else if (rhs_type == 1)
        {
            ReadRhsParFromOneFile(rhs_file_real, col_partition, &b);
        }
        else if (rhs_type == 2)
        {
            ReadRhsParFromOneFile(rhs_file_imag, col_partition, &b);
        }
        else if (rhs_type == 3)
        {
            ReadRhsParFromOneFile(rhs_file_real_iter, col_partition, &b);
        }
        else if (rhs_type == 4)
        {
            ReadRhsParFromOneFile(rhs_file_imag_iter, col_partition, &b);
        }

        x0 = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A), row_partition[num_procs], row_partition);
        hypre_ParVectorInitialize(x0);
        hypre_ParVectorSetConstantValues(x0, 0.0);

        ReadParFromOneFile(G_file, &G, NULL, NULL);
        if (row_partition)
        {
            hypre_TFree(row_partition, HYPRE_MEMORY_HOST);
        }
        if (col_partition)
        {
            hypre_TFree(row_partition, HYPRE_MEMORY_HOST);
        }

        HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)G, &row_partition);
        HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)G, &col_partition);

        /* Vectors x, y and z */
        if (coordinates)
        {
            ReadRhsParFromOneFile(xcood_file, col_partition, &x);
            ReadRhsParFromOneFile(ycood_file, col_partition, &y);
            if (dim == 3)
            {
                ReadRhsParFromOneFile(zcood_file, col_partition, &z);
            }
        }

        if (row_partition)
        {
            hypre_TFree(row_partition, HYPRE_MEMORY_HOST);
        }
        if (col_partition)
        {
            hypre_TFree(row_partition, HYPRE_MEMORY_HOST);
        }
#endif
    }
    hypre_ParCSRMatrixMigrate(A, hypre_HandleMemoryLocation(hypre_handle()));
    hypre_ParCSRMatrixMigrate(G, hypre_HandleMemoryLocation(hypre_handle()));
    hypre_ParCSRMatrixMigrate(Aalpha, hypre_HandleMemoryLocation(hypre_handle()));
    hypre_ParCSRMatrixMigrate(Abeta, hypre_HandleMemoryLocation(hypre_handle()));

    hypre_ParVectorMigrate(x0, hypre_HandleMemoryLocation(hypre_handle()));
    hypre_ParVectorMigrate(b, hypre_HandleMemoryLocation(hypre_handle()));
    hypre_ParVectorMigrate(x, hypre_HandleMemoryLocation(hypre_handle()));
    hypre_ParVectorMigrate(y, hypre_HandleMemoryLocation(hypre_handle()));
    hypre_ParVectorMigrate(z, hypre_HandleMemoryLocation(hypre_handle()));

    hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

    /* Suprelu */
    if (solver_id == 0)
    {
        /* Start timing */
        time_index = hypre_InitializeTiming("DSupreLU Setup");
        hypre_BeginTiming(time_index);

        hypre_BoomerAMGBuildCoarseOperator(G, A, G, &A_G);
        
        if (myid == 0)
        {
            hypre_printf("\n");
            hypre_printf("RAP for AG");
            hypre_printf("\n");
        }

#if 0
        hypre_SLUDistSetup(&slusolver, A_G, print_level);
        
        HYPRE_BigInt *row_partition, *col_partition;
        HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A_G, &row_partition);
        HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)A_G, &col_partition);

        b = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_G), row_partition[num_procs], row_partition);
        hypre_ParVectorInitialize(b);
        hypre_ParVectorSetRandomValues(b, 1);

        x0 = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_G), row_partition[num_procs], row_partition);
        hypre_ParVectorInitialize(x0);
        hypre_ParVectorSetConstantValues(x0, 0.0);
#endif

        hypre_SLUDistSetup(&slusolver, A, print_level);

        /* Finalize setup timing */
        hypre_EndTiming(time_index);
        hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
        hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();

        /* Start timing again */
        time_index = hypre_InitializeTiming("DSupreLU Solve");
        hypre_BeginTiming(time_index);


        hypre_SLUDistSolve(slusolver, b, x0);

        /* Finalize solve timing */
        hypre_EndTiming(time_index);
        hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
        hypre_FinalizeTiming(time_index);
        hypre_ClearTiming();
        
        rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(b, b));

        hypre_ParCSRMatrixMatvec(-1.0, A, x0, 1.0, b);
        final_res_norm = hypre_sqrt(hypre_ParVectorInnerProd(b, b));

        /* Run info - needed logging turned on */
        if (myid == 0)
        {
            hypre_printf("\n");
            hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm / rhs_norm);
            hypre_printf("\n");
        }

        /* Destroy solver */
        hypre_SLUDistDestroy(slusolver);
    }

    /* Clean-up */
    HYPRE_ParCSRMatrixDestroy(A);
    HYPRE_ParVectorDestroy(x0);
    HYPRE_ParVectorDestroy(b);
    HYPRE_ParCSRMatrixDestroy(G);

    if (M)
    {
        HYPRE_ParCSRMatrixDestroy(M);
    }

    if (x)
    {
        HYPRE_ParVectorDestroy(x);
    }
    if (y)
    {
        HYPRE_ParVectorDestroy(y);
    }
    if (z)
    {
        HYPRE_ParVectorDestroy(z);
    }

    if (Aalpha)
    {
        HYPRE_ParCSRMatrixDestroy(Aalpha);
    }
    if (Abeta)
    {
        HYPRE_ParCSRMatrixDestroy(Abeta);
    }

    /* Finalize Hypre */
    HYPRE_Finalize();

    /* Finalize MPI */
    hypre_MPI_Finalize();

    if (HYPRE_GetError() && !myid)
    {
        hypre_fprintf(stderr, "hypre_error_flag = %d\n", HYPRE_GetError());
    }
    return 0;
}