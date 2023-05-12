/*
1. Read a linear system corresponding to a parallel finite element
   discretization of Maxwell's equations.

   curl alpha curl E + beta E = 0

   default : alpha = 1.0
             beta  = 1.0

2. Call the AMS solver in HYPRE to solve that linear system.
*/

/*
Input:
    - the discrete mass matrix: A
    - the Gradient matrix: G
    - coordinates:
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* hypre/AMS prototypes */
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE.h"
#include "_hypre_utilities.hpp"

#define RAP_Matrix 0
#define Subspace_Matrix 0

/*  Hypre  IJ  format  */
void AMSDriverMatrixRead(const char *file, HYPRE_ParCSRMatrix *A)
{
   FILE *test;
   char file0[100];
   sprintf(file0, "%s.D.0", file);
   if (!(test = fopen(file0, "r")))
   {
      sprintf(file0, "%s.00000", file);
      if (!(test = fopen(file0, "r")))
      {
         hypre_MPI_Finalize();
         hypre_printf("Can't find the input file \"%s\"\n", file);
         exit(1);
      }
      else /* Read in IJ format*/
      {
         HYPRE_IJMatrix ij_A;
         void *object;
         HYPRE_IJMatrixRead(file, hypre_MPI_COMM_WORLD, HYPRE_PARCSR, &ij_A);
         HYPRE_IJMatrixGetObject(ij_A, &object);
         *A = (HYPRE_ParCSRMatrix)object;
         hypre_IJMatrixObject((hypre_IJMatrix *)ij_A) = NULL;
         HYPRE_IJMatrixDestroy(ij_A);
      }
   }
   else /* Read in ParCSR format*/
   {
      HYPRE_ParCSRMatrixRead(hypre_MPI_COMM_WORLD, file, A);
   }
   fclose(test);
}

void AMSDriverVectorRead(const char *file, HYPRE_ParVector *x)
{
   FILE *test;
   char file0[100];
   sprintf(file0, "%s.0", file);
   if (!(test = fopen(file0, "r")))
   {
      sprintf(file0, "%s.00000", file);
      if (!(test = fopen(file0, "r")))
      {
         hypre_MPI_Finalize();
         hypre_printf("Can't find the input file \"%s\"\n", file);
         exit(1);
      }
      else /* Read in IJ format*/
      {
         HYPRE_IJVector ij_x;
         void *object;
         HYPRE_IJVectorRead(file, hypre_MPI_COMM_WORLD, HYPRE_PARCSR, &ij_x);
         HYPRE_IJVectorGetObject(ij_x, &object);
         *x = (HYPRE_ParVector)object;
         hypre_IJVectorObject((hypre_IJVector *)ij_x) = NULL;
         HYPRE_IJVectorDestroy(ij_x);
      }
   }
   else /* Read in ParCSR format*/
   {
      HYPRE_ParVectorRead(hypre_MPI_COMM_WORLD, file, x);
   }
   fclose(test);
}

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

   HYPRE_Solver solver, precond, amg_precond;

   /* iterative options */
   HYPRE_Int example, spd_example, print_level, save_rhs, save_sol;
   HYPRE_Int solver_id, refine, rhs_type, rand_seed;
   HYPRE_Int maxit, p_maxit, kdim;
   HYPRE_Real ptol, tol; // tolerance for the preconditioner
   HYPRE_Int num_iterations;
   HYPRE_Real final_res_norm;

   /* amg options */
   HYPRE_Int amg_rlx_type, alpha_coarse_rlx_type, beta_coarse_rlx_type;
   HYPRE_Real theta;
   HYPRE_Int amg_coarsen_type, amg_interp_type, amg_agg_levels, amg_Pmax, amg_rlx_sweeps;
   HYPRE_Int beta_amg_max_level, beta_amg_print_level, beta_amg_max_iter;
   HYPRE_Int alpha_amg_x_max_iter, alpha_amg_y_max_iter, alpha_amg_z_max_iter;

   /* ams options */
   HYPRE_Int dim, cycle_type, ams_rlx_type, ams_rlx_sweeps;
   HYPRE_Int h1_method, singular_problem, coordinates;
   HYPRE_Real ams_rlx_weight, ams_rlx_omega;
   HYPRE_Int recompute_residual_p;
   HYPRE_Int zero_cond;

   /* amg-dd options */
   HYPRE_Int amgdd_start_level = 0;
   HYPRE_Int amgdd_padding = 1;
   HYPRE_Int amgdd_fac_num_relax = 1;
   HYPRE_Int amgdd_num_comp_cycles = 2;
   HYPRE_Int amgdd_fac_relax_type = 3;
   HYPRE_Int amgdd_fac_cycle_type = 1;
   HYPRE_Int amgdd_num_ghost_layers = 1;

   HYPRE_ParCSRMatrix A = 0, G = 0, Aalpha = 0, Abeta = 0, M = 0;
   HYPRE_ParVector x0 = 0, b = 0;
   HYPRE_ParVector Gx = 0, Gy = 0, Gz = 0;
   HYPRE_ParVector x = 0, y = 0, z = 0;

   HYPRE_ParVector interior_nodes = 0;

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

   /* Set defaults */
   example = 0;
   spd_example = 0;
   refine = 0;
   save_rhs = 0;
   save_sol = 0;
   rhs_type = 2;
   rand_seed = 1;
   print_level = 2;
   solver_id = 3;
   maxit = 100;
   p_maxit = 1;
   tol = 1e-1;
   ptol = 1e-1;
   dim = 3;
   coordinates = 1;
   h1_method = 0;
   singular_problem = 0;

   kdim = 30;

   cycle_type = 13;
   ams_rlx_type = 2; /* 1 - l1-scaled JAC; 3 - kaczmarz; 2/4: sym Hybrid GS; */
   ams_rlx_sweeps = 1;
   ams_rlx_weight = 1.0;
   ams_rlx_omega = 1.0;

   beta_amg_max_level = 25;
   beta_amg_max_iter = 1;
   beta_amg_print_level = 0;
   beta_coarse_rlx_type = 8; /* 1 - Jacobi; 8 - SSOR; 9 - direct */

   alpha_amg_x_max_iter = 1;
   alpha_amg_y_max_iter = 1;
   alpha_amg_z_max_iter = 1;

   alpha_coarse_rlx_type = 8; /* 1 - Jacobi; 8 - SSOR; 9 - direct */
   amg_coarsen_type = 10;     /* 10: HMIS-1 ; 8: PMIS ; 6: Falgout*/
   amg_agg_levels = 1;
   amg_rlx_type = 8; /* 3 - GS; 6 - sym GS */

   /* cycle_type = 1; amg_coarsen_type = 10; amg_agg_levels = 0; amg_rlx_type = 3; */ /* HMIS-0 */
   /* cycle_type = 1; amg_coarsen_type = 8; amg_agg_levels = 1; amg_rlx_type = 3;  */ /* PMIS-1 */
   /* cycle_type = 1; amg_coarsen_type = 8; amg_agg_levels = 0; amg_rlx_type = 3;  */ /* PMIS-0 */
   /* cycle_type = 7; amg_coarsen_type = 6; amg_agg_levels = 0; amg_rlx_type = 6;  */ /* Falgout-0 */

   amg_interp_type = 6;                                                  /* 6 - extended+i interpolation; 0 - standard interpolation */
   amg_Pmax = 4; /* long-range interpolation */                          /* PMIS */
   /* amg_interp_type = 0; amg_Pmax = 0; */ /* standard interpolation */ /* HMIS */
   amg_rlx_sweeps = 1;
   theta = 0.25;
   recompute_residual_p = 0;
   zero_cond = 0;

   /* Parse command line */
   {
      HYPRE_Int arg_index = 0;
      HYPRE_Int print_usage = 0;

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
         else if (strcmp(argv[arg_index], "-seed") == 0)
         {
            arg_index++;
            rand_seed = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-saverhs") == 0)
         {
            arg_index++;
            save_rhs = 1;
         }
         else if (strcmp(argv[arg_index], "-savesol") == 0)
         {
            arg_index++;
            save_sol = 1;
         }
         else if (strcmp(argv[arg_index], "-ptlv") == 0)
         {
            arg_index++;
            print_level = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-bmaxlv") == 0)
         {
            arg_index++;
            beta_amg_max_level = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-bptlv") == 0)
         {
            arg_index++;
            beta_amg_print_level = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-bgmaxit") == 0)
         {
            arg_index++;
            beta_amg_max_iter = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-bxmaxit") == 0)
         {
            arg_index++;
            alpha_amg_x_max_iter = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-bymaxit") == 0)
         {
            arg_index++;
            alpha_amg_y_max_iter = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-bzmaxit") == 0)
         {
            arg_index++;
            alpha_amg_z_max_iter = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-maxit") == 0)
         {
            arg_index++;
            maxit = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-pmaxit") == 0)
         {
            arg_index++;
            p_maxit = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-tol") == 0)
         {
            arg_index++;
            tol = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-type") == 0)
         {
            arg_index++;
            cycle_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-amsrlx") == 0)
         {
            arg_index++;
            ams_rlx_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-amgrlx") == 0)
         {
            arg_index++;
            amg_rlx_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-amgrlxn") == 0)
         {
            arg_index++;
            amg_rlx_sweeps = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-rlxn") == 0)
         {
            arg_index++;
            ams_rlx_sweeps = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-rlxw") == 0)
         {
            arg_index++;
            ams_rlx_weight = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-rlxo") == 0)
         {
            arg_index++;
            ams_rlx_omega = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-ctype") == 0)
         {
            arg_index++;
            amg_coarsen_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-bcrlx") == 0)
         {
            arg_index++;
            beta_coarse_rlx_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-acrlx") == 0)
         {
            arg_index++;
            alpha_coarse_rlx_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-agg") == 0)
         {
            arg_index++;
            amg_agg_levels = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-itype") == 0)
         {
            arg_index++;
            amg_interp_type = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-pmax") == 0)
         {
            arg_index++;
            amg_Pmax = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-dim") == 0)
         {
            arg_index++;
            dim = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-coord") == 0)
         {
            arg_index++;
            coordinates = 1;
         }
         else if (strcmp(argv[arg_index], "-h1") == 0)
         {
            arg_index++;
            h1_method = 1;
         }
         else if (strcmp(argv[arg_index], "-sing") == 0)
         {
            arg_index++;
            singular_problem = 1;
         }
         else if (strcmp(argv[arg_index], "-theta") == 0)
         {
            arg_index++;
            theta = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-ptol") == 0)
         {
            arg_index++;
            ptol = (HYPRE_Real)atof(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-rr") == 0)
         {
            arg_index++;
            recompute_residual_p = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-zc") == 0)
         {
            arg_index++;
            zero_cond = 1;
         }
         else if (strcmp(argv[arg_index], "-ex") == 0)
         {
            example = 1;
            arg_index++;
            break;
         }
         else if (strcmp(argv[arg_index], "-spdex") == 0)
         {
            spd_example = 1;
            arg_index++;
            break;
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

      if (argc == 1)
      {
         print_usage = 1;
      }

      if ((print_usage) && (myid == 0))
      {
         hypre_printf("\n");
         hypre_printf("Usage: mpirun -np <np> %s [<options>]\n", argv[0]);
         hypre_printf("\n");
         hypre_printf("  Hypre solvers options:                                       \n");
         hypre_printf("    -solver <ID>         : solver ID                           \n");
         hypre_printf("                           0  - AMG                            \n");
         hypre_printf("                           1  - AMG-PCG                        \n");
         hypre_printf("                           2  - AMS                            \n");
         hypre_printf("                           3  - AMS-PCG (default)              \n");
         hypre_printf("                           4  - DS-PCG                         \n");
         hypre_printf("                           5  - PCG                            \n");
         hypre_printf("                           11 - AMG-PGMRES                     \n");
         hypre_printf("                           13 - AMS-PGMRES                     \n");
         hypre_printf("                           14 - DS-PGMRES                      \n");
         hypre_printf("                           15 - GMRES                          \n");
         hypre_printf("                           16 - AMG-DD-PGMRES                  \n");
         hypre_printf("                           21 - AMG-FGMRES                     \n");
         hypre_printf("                           23 - AMS-FGMRES                     \n");
         hypre_printf("                           24 - DS-FGMRES                      \n");
         hypre_printf("                           25 - FGMRES                         \n");
         hypre_printf("    -maxit <num>         : maximum number of iterations (100)  \n");
         hypre_printf("    -tol <num>           : convergence tolerance (1e-6)        \n");
         hypre_printf("    -ptol <num>          : prec convergence tolerance (1e-6)   \n");
         hypre_printf("\n");
         hypre_printf("  AMS solver options:                                          \n");
         hypre_printf("    -dim <num>           : space dimension                     \n");
         hypre_printf("    -type <num>          : 3-level cycle type (0-8, 11-14)     \n");
         hypre_printf("    -theta <num>         : BoomerAMG threshold (0.25)          \n");
         hypre_printf("    -ctype <num>         : BoomerAMG coarsening type           \n");
         hypre_printf("    -agg <num>           : Levels of BoomerAMG agg. coarsening \n");
         hypre_printf("    -amgrlx <num>        : BoomerAMG relaxation type           \n");
         hypre_printf("    -itype <num>         : BoomerAMG interpolation type        \n");
         hypre_printf("    -pmax <num>          : BoomerAMG interpolation truncation  \n");
         hypre_printf("    -rlx <num>           : relaxation type                     \n");
         hypre_printf("    -rlxn <num>          : number of relaxation sweeps         \n");
         hypre_printf("    -rlxw <num>          : damping parameter (usually <=1)     \n");
         hypre_printf("    -rlxo <num>          : SOR parameter (usuallyin (0,2))     \n");
         hypre_printf("    -coord               : use coordinate vectors              \n");
         hypre_printf("    -h1                  : use block-diag Poisson solves       \n");
         hypre_printf("    -sing                : curl-curl only (singular) problem   \n");
         hypre_printf("\n");
         hypre_printf("  AME eigensolver options:                                     \n");
         hypre_printf("    -bsize<num>          : number of eigenvalues to compute    \n");
         hypre_printf("\n");
      }

      if (print_usage)
      {
         hypre_MPI_Finalize();
         return (0);
      }
   }

   if (example)
   {
      AMSDriverMatrixRead("TEST_ams/mfem.A", &A);
      AMSDriverVectorRead("TEST_ams/mfem.x0", &x0);
      AMSDriverVectorRead("TEST_ams/mfem.b", &b);
      AMSDriverMatrixRead("TEST_ams/mfem.G", &G);

      /* Vectors Gx, Gy and Gz */
      if (!coordinates)
      {
         AMSDriverVectorRead("TEST_ams/mfem.Gx", &Gx);
         AMSDriverVectorRead("TEST_ams/mfem.Gy", &Gy);
         if (dim == 3)
         {
            AMSDriverVectorRead("TEST_ams/mfem.Gz", &Gz);
         }
      }

      /* Vectors x, y and z */
      if (coordinates)
      {
         AMSDriverVectorRead("TEST_ams/mfem.x", &x);
         AMSDriverVectorRead("TEST_ams/mfem.y", &y);
         if (dim == 3)
         {
            AMSDriverVectorRead("TEST_ams/mfem.z", &z);
         }
      }

      if (zero_cond)
      {
         AMSDriverVectorRead("TEST_ams/mfem.inodes", &interior_nodes);
      }
   }

   else if (spd_example)
   {
#if 0
      char A_file[100], rhs_file[100];
      // sprintf(A_file, "/home/matrix/dyt_matrix/jxpamg_560_560");
      // sprintf(rhs_file, "/home/matrix/dyt_matrix/jxpamg_560_560_rhs");

      sprintf(A_file, "/tmp/fzb/amg_mtrix/SM/SM-CS3.matrix");
      sprintf(rhs_file, "/tmp/fzb/amg_mtrix/SM/SM-CS3.rhs");

      ReadParFromOneFile(A_file, &A, NULL, NULL);

      HYPRE_BigInt *row_partition, *col_partition;
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A, &row_partition);
      HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)A, &col_partition);

      ReadRhsParFromOneFile(rhs_file, col_partition, &b);

      x0 = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A), row_partition[num_procs], row_partition);
      hypre_ParVectorInitialize(x0);
      hypre_ParVectorSetConstantValues(x0, 0.0);
#endif
      char *file_locate = "/home/dyt/matrix/SPD_matrix_hypre";
      char M_file[100], G_file[100];
      char rhs_file[100];
      char xcood_file[100], ycood_file[100], zcood_file[100];

      if (refine == 0 || refine == 1 || refine == 2)
      {
         sprintf(M_file, "%s/refine_%d/hypre_CSR.A", file_locate, refine);
         sprintf(G_file, "%s/refine_%d/hypre_CSR.G", file_locate, refine);
         sprintf(rhs_file, "%s/refine_%d/seq_b", file_locate, refine);
         sprintf(xcood_file, "%s/refine_%d/seq_x", file_locate, refine);
         sprintf(ycood_file, "%s/refine_%d/seq_y", file_locate, refine);
         sprintf(zcood_file, "%s/refine_%d/seq_z", file_locate, refine);
      }

      ReadParFromOneFile(M_file, &A, NULL, NULL);

      HYPRE_BigInt *row_partition, *col_partition;
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)A, &row_partition);
      HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix)A, &col_partition);

      // b = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A), row_partition[num_procs], row_partition);
      // hypre_ParVectorInitialize(b);
      // hypre_ParVectorSetRandomValues(b, 1);

      ReadRhsParFromOneFile(rhs_file, col_partition, &b);

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
   }
   else
   {
      /*   From one file  ---  hypre CSR format  */

#if 1
      char *file_locate = "/home/dyt/matrix/filter_matrix_hypre";
      char M_file[100], G_file[100];
      char rhs_file_real[100], rhs_file_imag[100], rhs_file_real_iter[100], rhs_file_imag_iter[100];
      char xcood_file[100], ycood_file[100], zcood_file[100];

      if (refine == 0 || refine == 1 )
      {
         sprintf(M_file, "%s/refine_%d/hypre_CSR.M", file_locate, refine);
         sprintf(G_file, "%s/refine_%d/hypre_CSR.G", file_locate, refine);
         sprintf(rhs_file_real, "%s/refine_%d/seq_b0_real", file_locate, refine);
         sprintf(rhs_file_imag, "%s/refine_%d/seq_b0_imag", file_locate, refine);
         sprintf(rhs_file_real_iter, "%s/refine_%d/seq_b1_real", file_locate, refine);
         sprintf(rhs_file_imag_iter, "%s/refine_%d/seq_b1_imag", file_locate, refine);
         sprintf(xcood_file, "%s/refine_%d/seq_x", file_locate, refine);
         sprintf(ycood_file, "%s/refine_%d/seq_y", file_locate, refine);
         sprintf(zcood_file, "%s/refine_%d/seq_z", file_locate, refine);
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
         hypre_ParVectorSetRandomValues(b, rand_seed);
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
      else if (rhs_type == 5)
      {
         b = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A), row_partition[num_procs], row_partition);
         hypre_ParVectorInitialize(b);
         hypre_ParVectorSetConstantValues(b, 1.0);
      }
      
      // Print rhs
      if (save_rhs)
      {
         if (spd_example)
         {
            char *file_locate = "/home/dyt/matrix/SPD_matrix_hypre/";
            char hypre_rhs_file[100];
            sprintf(hypre_rhs_file, "%s/refine_%d/hypre_rhs", file_locate, refine);
            HYPRE_ParVectorPrint(b, hypre_rhs_file);
         }
         else
         {
            char *file_locate = "/home/dyt/matrix/filter_matrix_hypre/";
            char hypre_rhs_file[100];
            sprintf(hypre_rhs_file, "%s/refine_%d/hypre_rhs_%d", file_locate, refine, rhs_type);
            HYPRE_ParVectorPrint(b, hypre_rhs_file);
         }
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

      // for (int i = 0; i <= num_procs; i++)
      //    hypre_printf("%d -- %d, %d ; %d\n", myid, i, row_partition[i], col_partition[i]);

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

      /*   Parallel from files  --  hypre IJ format  */

#if 0
   // AMSDriverMatrixRead("jpsol_ams/matrix.G", &G);


   AMSDriverMatrixRead("jpsol_ams/IJ.A", &A);

   HYPRE_BigInt *row_partition, *col_partition;
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &row_partition);
   HYPRE_ParCSRMatrixGetColPartitioning((HYPRE_ParCSRMatrix) A, &col_partition);

   for (int i = 0; i <= num_procs; i++)
      hypre_printf("%d -- %d, %d ; %d\n", myid, i, row_partition[i], col_partition[i]);


   ReadRhsParFromOneFile("jpsol_ams/seq_b", row_partition, &b);

   ReadRhsParFromOneFile("jpsol_ams/seq_x0", row_partition, &x0);

   // AMSDriverVectorRead("jpsol_ams/IJ.x0", &x0); 
   // AMSDriverVectorRead("jpsol_ams/IJ.b", &b);

   /* Vectors x, y and z */
   if (coordinates)
   {
      AMSDriverVectorRead("jpsol_ams/IJ.x", &x);
      AMSDriverVectorRead("jpsol_ams/IJ.y", &y);
      if (dim == 3)
      {
         AMSDriverVectorRead("jpsol_ams/IJ.z", &z);
      }
   }

#endif
   }

   if (!myid)
   {
      hypre_printf("Problem size of matrix A: %d x %d\n",
                   hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *)A),
                   hypre_ParCSRMatrixGlobalNumCols((hypre_ParCSRMatrix *)A));
#if 1
      hypre_printf("Problem size of matrix G: %d x %d\n\n",
                   hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix *)G),
                   hypre_ParCSRMatrixGlobalNumCols((hypre_ParCSRMatrix *)G));

      hypre_printf("Size of xcoord: %d\n",
                   hypre_ParVectorGlobalSize((hypre_ParVector *)x));

      hypre_printf("Size of ycoord: %d\n",
                   hypre_ParVectorGlobalSize((hypre_ParVector *)y));

      hypre_printf("Size of zcoord: %d\n\n",
                   hypre_ParVectorGlobalSize((hypre_ParVector *)z));

      hypre_printf("Size of x0: %d\n",
                   hypre_ParVectorGlobalSize((hypre_ParVector *)x0));
      hypre_printf("Size of rhs: %d\n\n",
                   hypre_ParVectorGlobalSize((hypre_ParVector *)b));

#endif
   }
   hypre_ParCSRMatrixMigrate(A, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParCSRMatrixMigrate(G, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParCSRMatrixMigrate(Aalpha, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParCSRMatrixMigrate(Abeta, hypre_HandleMemoryLocation(hypre_handle()));

   hypre_ParVectorMigrate(x0, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(b, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(Gx, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(Gy, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(Gz, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(x, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(y, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(z, hypre_HandleMemoryLocation(hypre_handle()));
   hypre_ParVectorMigrate(interior_nodes, hypre_HandleMemoryLocation(hypre_handle()));

   hypre_MPI_Barrier(hypre_MPI_COMM_WORLD);

   /* AMG */
   if (solver_id == 0)
   {
      /* Start timing */
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_BoomerAMGSetPrintLevel(solver, print_level); /* print solve info + parameters */
      HYPRE_BoomerAMGSetCoarsenType(solver, amg_coarsen_type);   /* Falgout coarsening */
      // HYPRE_BoomerAMGSetCoarsenType(solver, 8);            /* PMIS coarsening */
      HYPRE_BoomerAMGSetRelaxType(solver, amg_rlx_type);   /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetNumSweeps(solver, amg_rlx_sweeps); /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(solver, 20);              /* maximum number of levels */
      HYPRE_BoomerAMGSetTol(solver, tol);                  /* conv. tolerance */
      HYPRE_BoomerAMGSetMaxIter(solver, maxit);            /* maximum number of iterations */
      HYPRE_BoomerAMGSetStrongThreshold(solver, theta);
      // HYPRE_BoomerAMGSetDSLUThreshold(solver, 1000);

      HYPRE_BoomerAMGSetup(solver, A, b, x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_BoomerAMGSolve(solver, A, b, x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver */
      HYPRE_BoomerAMGDestroy(solver);
   }

   /* AMS */
   if (solver_id == 2)
   {
      /* Start timing */
      time_index = hypre_InitializeTiming("AMS Setup");
      hypre_BeginTiming(time_index);

      /* Create solver */
      HYPRE_AMSCreate(&solver);

      /* Set AMS parameters */
      HYPRE_AMSSetDimension(solver, dim);
      HYPRE_AMSSetMaxIter(solver, maxit);
      HYPRE_AMSSetTol(solver, tol);
      HYPRE_AMSSetCycleType(solver, cycle_type);
      HYPRE_AMSSetPrintLevel(solver, print_level);
      HYPRE_AMSSetDiscreteGradient(solver, G);

      /* Vectors Gx, Gy and Gz */
      if (!coordinates)
      {
         HYPRE_AMSSetEdgeConstantVectors(solver, Gx, Gy, Gz);
      }

      /* Vectors x, y and z */
      if (coordinates)
      {
         HYPRE_AMSSetCoordinateVectors(solver, x, y, z);
      }

      /* Poisson matrices */
      if (h1_method)
      {
         HYPRE_AMSSetAlphaPoissonMatrix(solver, Aalpha);
         HYPRE_AMSSetBetaPoissonMatrix(solver, Abeta);
      }

      if (singular_problem)
      {
         HYPRE_AMSSetBetaPoissonMatrix(solver, NULL);
      }

      /* Smoothing and AMG options */
      HYPRE_AMSSetSmoothingOptions(solver, ams_rlx_type, ams_rlx_sweeps, ams_rlx_weight, ams_rlx_omega);
      HYPRE_AMSSetAlphaAMGOptions(solver, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                  amg_interp_type, amg_Pmax);
      HYPRE_AMSSetAlphaAMGCoarseRelaxType(solver, alpha_coarse_rlx_type);

      HYPRE_AMSSetBetaAMGOptions(solver, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                 amg_interp_type, amg_Pmax);
      HYPRE_AMSSetBetaAMGCoarseRelaxType(solver, beta_coarse_rlx_type);

      HYPRE_AMSSetBetaAMGMaxLevels(solver, beta_amg_max_level);
      HYPRE_AMSSetBetaAMGMaxIter(solver, beta_amg_max_iter);
      HYPRE_AMSSetBetaAMGPrintLevel(solver, beta_amg_print_level);

      HYPRE_AMSSetAlphaXAMGMaxIter(solver, alpha_amg_x_max_iter);
      HYPRE_AMSSetAlphaYAMGMaxIter(solver, alpha_amg_y_max_iter);
      HYPRE_AMSSetAlphaZAMGMaxIter(solver, alpha_amg_z_max_iter);

      HYPRE_AMSSetup(solver, A, b, x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

#if Subspace_Matrix
      hypre_AMSData *ams_data = (hypre_AMSData *)solver;
      char *file_locate = "/home/dyt/matrix/filter_matrix_hypre/subspace";
      char hypre_A_G_file[100];
      char hypre_A_Pix_file[100], hypre_A_Piy_file[100], hypre_A_Piz_file[100];
      sprintf(hypre_A_G_file, "%s/refine_%d/matrix_A_G", file_locate, refine);
      sprintf(hypre_A_Pix_file, "%s/refine_%d/matrix_A_Pix", file_locate, refine);
      sprintf(hypre_A_Piy_file, "%s/refine_%d/matrix_A_Piy", file_locate, refine);
      sprintf(hypre_A_Piz_file, "%s/refine_%d/matrix_A_Piz", file_locate, refine);

      hypre_ParCSRMatrixPrintIJ(ams_data->A_G, 0, 0, hypre_A_G_file);
      hypre_ParCSRMatrixPrintIJ(ams_data->A_Pix, 0, 0, hypre_A_Pix_file);
      hypre_ParCSRMatrixPrintIJ(ams_data->A_Piy, 0, 0, hypre_A_Piy_file);
      hypre_ParCSRMatrixPrintIJ(ams_data->A_Piz, 0, 0, hypre_A_Piz_file);
#endif

#if RAP_Matrix
      hypre_AMSData *ams_data = (hypre_AMSData *)solver;
      char *file_locate = "/home/matrix/RAP_matrix";
      char hypre_A_file[100], hypre_G_file[100];
      char hypre_Pix_file[100], hypre_Piy_file[100], hypre_Piz_file[100];
      sprintf(hypre_A_file, "%s/matrix_A", file_locate);
      sprintf(hypre_G_file, "%s/matrix_G", file_locate);
      sprintf(hypre_Pix_file, "%s/matrix_Pix", file_locate);
      sprintf(hypre_Piy_file, "%s/matrix_Piy", file_locate);
      sprintf(hypre_Piz_file, "%s/matrix_Piz", file_locate);

      hypre_ParCSRMatrixPrintIJ(ams_data->A, 0, 0, hypre_A_file);
      hypre_ParCSRMatrixPrintIJ(ams_data->G, 0, 0, hypre_G_file);
      hypre_ParCSRMatrixPrintIJ(ams_data->Pix, 0, 0, hypre_Pix_file);
      hypre_ParCSRMatrixPrintIJ(ams_data->Piy, 0, 0, hypre_Piy_file);
      hypre_ParCSRMatrixPrintIJ(ams_data->Piz, 0, 0, hypre_Piz_file);
#endif

      /* Start timing again */
      time_index = hypre_InitializeTiming("AMS Solve");
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_AMSSolve(solver, A, b, x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_AMSGetNumIterations(solver, &num_iterations);
      HYPRE_AMSGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver */
      HYPRE_AMSDestroy(solver);
   }

   /* PCG solvers */
   else if (solver_id == 1 || solver_id == 3 || solver_id == 4 || solver_id == 5)
   {

      /* Start timing */
      if (solver_id == 1)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-PCG Setup");
      }
      else if (solver_id == 3)
      {
         time_index = hypre_InitializeTiming("AMS-PCG Setup");
      }
      else if (solver_id == 4)
      {
         time_index = hypre_InitializeTiming("DS-PCG Setup");
      }
      else if (solver_id == 5)
      {
         time_index = hypre_InitializeTiming("CG Setup");
      }
      hypre_BeginTiming(time_index);

      /* Create CG solver */
      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, maxit);          /* max iterations */
      HYPRE_PCGSetTol(solver, tol);                /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1);              /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, print_level); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1);              /* needed to get run info later */

      /* PCG with AMG preconditioner */
      if (solver_id == 1)
      {
         /* Now set up the AMG preconditioner and specify any parameters */
         HYPRE_BoomerAMGCreate(&precond);
         HYPRE_BoomerAMGSetPrintLevel(precond, 1);             /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType(precond, amg_coarsen_type);            /* Falgout coarsening */
         HYPRE_BoomerAMGSetRelaxType(precond, amg_rlx_type);   /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps(precond, amg_rlx_sweeps); /* Sweeeps on each level */
         HYPRE_BoomerAMGSetMaxLevels(precond, 20);             /* maximum number of levels */
         HYPRE_BoomerAMGSetTol(precond, ptol);                 /* conv. tolerance (if needed) */
         HYPRE_BoomerAMGSetMaxIter(precond, p_maxit);          /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold(precond, theta);

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                             precond);
      }

      /* PCG with AMS preconditioner */
      else if (solver_id == 3)
      {
         /* Now set up the AMS preconditioner and specify any parameters */
         HYPRE_AMSCreate(&precond);
         HYPRE_AMSSetDimension(precond, dim);
         HYPRE_AMSSetMaxIter(precond, p_maxit);
         HYPRE_AMSSetTol(precond, ptol);
         HYPRE_AMSSetCycleType(precond, cycle_type);
         HYPRE_AMSSetPrintLevel(precond, 0);
         HYPRE_AMSSetDiscreteGradient(precond, G);

         if (zero_cond)
         {
            HYPRE_AMSSetInteriorNodes(precond, interior_nodes);
            HYPRE_AMSSetProjectionFrequency(precond, 5);
         }
         HYPRE_PCGSetResidualTol(solver, 0.0);
         HYPRE_PCGSetRecomputeResidualP(solver, recompute_residual_p);

         /* Vectors Gx, Gy and Gz */
         if (!coordinates)
         {
            HYPRE_AMSSetEdgeConstantVectors(precond, Gx, Gy, Gz);
         }

         /* Vectors x, y and z */
         if (coordinates)
         {
            HYPRE_AMSSetCoordinateVectors(precond, x, y, z);
         }

         /* Poisson matrices */
         if (h1_method)
         {
            HYPRE_AMSSetAlphaPoissonMatrix(precond, Aalpha);
            HYPRE_AMSSetBetaPoissonMatrix(precond, Abeta);
         }

         if (singular_problem)
         {
            HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
         }

         /* Smoothing and AMG options */
         HYPRE_AMSSetSmoothingOptions(precond, ams_rlx_type, ams_rlx_sweeps, ams_rlx_weight, ams_rlx_omega);
         HYPRE_AMSSetAlphaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                     amg_interp_type, amg_Pmax);
         HYPRE_AMSSetBetaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                    amg_interp_type, amg_Pmax);
         HYPRE_AMSSetAlphaAMGCoarseRelaxType(precond, alpha_coarse_rlx_type);
         HYPRE_AMSSetBetaAMGCoarseRelaxType(precond, beta_coarse_rlx_type);

         HYPRE_AMSSetBetaAMGMaxLevels(precond, beta_amg_max_level);
         HYPRE_AMSSetBetaAMGMaxIter(precond, beta_amg_max_iter);
         HYPRE_AMSSetBetaAMGPrintLevel(precond, beta_amg_print_level);

         HYPRE_AMSSetAlphaXAMGMaxIter(precond, alpha_amg_x_max_iter);
         HYPRE_AMSSetAlphaYAMGMaxIter(precond, alpha_amg_y_max_iter);
         HYPRE_AMSSetAlphaZAMGMaxIter(precond, alpha_amg_z_max_iter);

         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn)HYPRE_AMSSolve,
                             (HYPRE_PtrToSolverFcn)HYPRE_AMSSetup,
                             precond);
      }

      /* PCG with diagonal scaling preconditioner */
      else if (solver_id == 4)
      {
         /* Set the PCG preconditioner */
         HYPRE_PCGSetPrecond(solver,
                             (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                             NULL);
      }

      /* Setup */
      HYPRE_ParCSRPCGSetup(solver, A, b, x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      if (solver_id == 1)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-PCG Solve");
      }
      else if (solver_id == 3)
      {
         time_index = hypre_InitializeTiming("AMS-PCG Solve");
      }
      else if (solver_id == 4)
      {
         time_index = hypre_InitializeTiming("DS-PCG Solve");
      }
      else if (solver_id == 5)
      {
         time_index = hypre_InitializeTiming("CG Solve");
      }
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_ParCSRPCGSolve(solver, A, b, x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
      if (solver_id == 1)
      {
         HYPRE_BoomerAMGDestroy(precond);
      }
      else if (solver_id == 3)
      {
         HYPRE_AMSDestroy(precond);
      }
   }

   /* PGMRES solvers */
   else if (solver_id == 11 || solver_id == 13 || solver_id == 14 || solver_id == 15 || solver_id == 16)
   {

      /* Start timing */
      if (solver_id == 11)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-GMRES Setup");
      }
      else if (solver_id == 13)
      {
         time_index = hypre_InitializeTiming("AMS-GMRES Setup");
      }
      else if (solver_id == 14)
      {
         time_index = hypre_InitializeTiming("DS-GMRES Setup");
      }
      else if (solver_id == 15)
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
      }
      else if (solver_id == 16)
      {
         time_index = hypre_InitializeTiming("AMG-DD-GMRES Setup");
      }
      hypre_BeginTiming(time_index);

      /* Create GMRES solver */
      HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &solver);

      /* Set GMRES parameters (See Reference Manual for more parameters) */
      HYPRE_GMRESSetKDim(solver, kdim);
      HYPRE_GMRESSetMaxIter(solver, maxit);
      HYPRE_GMRESSetTol(solver, tol);
      HYPRE_GMRESSetPrintLevel(solver, print_level);
      HYPRE_GMRESSetLogging(solver, 1);

      /* GMRES with AMG preconditioner */
      if (solver_id == 11)
      {
         /* Now set up the AMG preconditioner and specify any parameters */
         HYPRE_BoomerAMGCreate(&precond);
         HYPRE_BoomerAMGSetPrintLevel(precond, 1);             /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType(precond, amg_coarsen_type);            /* Falgout coarsening */
         HYPRE_BoomerAMGSetRelaxType(precond, amg_rlx_type);   /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps(precond, amg_rlx_sweeps); /* Sweeeps on each level */
         HYPRE_BoomerAMGSetMaxLevels(precond, 20);             /* maximum number of levels */
         HYPRE_BoomerAMGSetTol(precond, ptol);                 /* conv. tolerance (if needed) */
         HYPRE_BoomerAMGSetMaxIter(precond, p_maxit);          /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold(precond, theta);
         // HYPRE_BoomerAMGSetDSLUThreshold(precond, 1000000);

         /* Set the GMRES preconditioner */
         HYPRE_GMRESSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                               precond);
      }
      /* GMRES with AMS preconditioner */
      else if (solver_id == 13)
      {
         /* Now set up the AMS preconditioner and specify any parameters */
         HYPRE_AMSCreate(&precond);
         HYPRE_AMSSetDimension(precond, dim);
         HYPRE_AMSSetMaxIter(precond, p_maxit);
         HYPRE_AMSSetTol(precond, ptol);
         HYPRE_AMSSetCycleType(precond, cycle_type);
         HYPRE_AMSSetPrintLevel(precond, 0);
         HYPRE_AMSSetDiscreteGradient(precond, G);

         if (zero_cond)
         {
            HYPRE_AMSSetInteriorNodes(precond, interior_nodes);
            HYPRE_AMSSetProjectionFrequency(precond, 5);
         }

         /* Vectors Gx, Gy and Gz */
         if (!coordinates)
         {
            HYPRE_AMSSetEdgeConstantVectors(precond, Gx, Gy, Gz);
         }

         /* Vectors x, y and z */
         if (coordinates)
         {
            HYPRE_AMSSetCoordinateVectors(precond, x, y, z);
         }

         /* Poisson matrices */
         if (h1_method)
         {
            HYPRE_AMSSetAlphaPoissonMatrix(precond, Aalpha);
            HYPRE_AMSSetBetaPoissonMatrix(precond, Abeta);
         }

         if (singular_problem)
         {
            HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
         }

         /* Smoothing and AMG options */
         HYPRE_AMSSetSmoothingOptions(precond, ams_rlx_type, ams_rlx_sweeps, ams_rlx_weight, ams_rlx_omega);
         HYPRE_AMSSetAlphaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                     amg_interp_type, amg_Pmax);
         HYPRE_AMSSetBetaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                    amg_interp_type, amg_Pmax);
         HYPRE_AMSSetAlphaAMGCoarseRelaxType(precond, alpha_coarse_rlx_type);
         HYPRE_AMSSetBetaAMGCoarseRelaxType(precond, beta_coarse_rlx_type);

         HYPRE_AMSSetBetaAMGMaxLevels(precond, beta_amg_max_level);
         HYPRE_AMSSetBetaAMGMaxIter(precond, beta_amg_max_iter);
         HYPRE_AMSSetBetaAMGPrintLevel(precond, beta_amg_print_level);

         HYPRE_AMSSetAlphaXAMGMaxIter(precond, alpha_amg_x_max_iter);
         HYPRE_AMSSetAlphaYAMGMaxIter(precond, alpha_amg_y_max_iter);
         HYPRE_AMSSetAlphaZAMGMaxIter(precond, alpha_amg_z_max_iter);

         /* Set the GMRES preconditioner */
         HYPRE_GMRESSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn)HYPRE_AMSSolve,
                               (HYPRE_PtrToSolverFcn)HYPRE_AMSSetup,
                               precond);
      }

      /* GMRES with diagonal scaling preconditioner */
      else if (solver_id == 14)
      {
         /* Set the GMRES preconditioner */
         HYPRE_GMRESSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                               NULL);
      }

      /* GMRES with AMG-DD */
      else if (solver_id == 16)
      {
         HYPRE_BoomerAMGDDCreate(&precond);
         HYPRE_BoomerAMGDDGetAMG(precond, &amg_precond);

         /* AMG-DD options */
         HYPRE_BoomerAMGDDSetStartLevel(precond, amgdd_start_level);
         HYPRE_BoomerAMGDDSetPadding(precond, amgdd_padding);
         HYPRE_BoomerAMGDDSetFACNumRelax(precond, amgdd_fac_num_relax);
         HYPRE_BoomerAMGDDSetFACNumCycles(precond, amgdd_num_comp_cycles);
         HYPRE_BoomerAMGDDSetFACRelaxType(precond, amgdd_fac_relax_type);
         HYPRE_BoomerAMGDDSetFACCycleType(precond, amgdd_fac_cycle_type);
         HYPRE_BoomerAMGDDSetNumGhostLayers(precond, amgdd_num_ghost_layers);

         /* AMG options */
         HYPRE_BoomerAMGSetRestriction(amg_precond, 0);                /* 0: P^T, 1: AIR, 2: AIR-2 */
         HYPRE_BoomerAMGSetInterpType(amg_precond, amg_interp_type);   /* 0: classical; 6: extended */
         HYPRE_BoomerAMGSetCoarsenType(amg_precond, amg_coarsen_type); /* 0: cljp; 8: PMIS; 10: HMIS-1 */
         HYPRE_BoomerAMGSetRelaxType(amg_precond, amg_rlx_type);
         HYPRE_BoomerAMGSetNumSweeps(amg_precond, amg_rlx_sweeps); /* Sweeeps on each level */
         HYPRE_BoomerAMGSetMaxLevels(amg_precond, 20);             /* maximum number of levels */
         HYPRE_BoomerAMGSetPrintLevel(amg_precond, 0);
         HYPRE_BoomerAMGSetMaxIter(amg_precond, p_maxit);
         HYPRE_BoomerAMGSetTol(amg_precond, 0.);
         HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);

         HYPRE_GMRESSetPrecond(solver,
                               (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGDDSolve,
                               (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGDDSetup,
                               precond);
      }

      /* Setup */
      HYPRE_ParCSRGMRESSetup(solver, A, b, x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      if (solver_id == 11)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-GMRES Solve");
      }
      else if (solver_id == 13)
      {
         time_index = hypre_InitializeTiming("AMS-GMRES Solve");
      }
      else if (solver_id == 14)
      {
         time_index = hypre_InitializeTiming("DS-GMRES Solve");
      }
      else if (solver_id == 15)
      {
         time_index = hypre_InitializeTiming("GMRES Solve");
      }
      else if (solver_id == 16)
      {
         time_index = hypre_InitializeTiming("AMG-DD-GMRES Solve");
      }
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_ParCSRGMRESSolve(solver, A, b, x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* get some info */
      HYPRE_GMRESGetNumIterations(solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      // /* Calculate the true residual norm */
      // hypre_GMRESData *gmres_data = (hypre_GMRESData *)solver;
      // hypre_GMRESFunctions *gmres_functions = (gmres_data->functions);
      // HYPRE_Real norm_b = hypre_sqrt((*(gmres_functions->InnerProd))(b, b));
      // (*(gmres_functions->Matvec))((gmres_data->matvec_data), -1.0, A, x0, 1.0, b);
      // HYPRE_Real norm_res = hypre_sqrt((*(gmres_functions->InnerProd))(b, b));
      // if (myid == 0)
      // {
      //    hypre_printf("\n");
      //    hypre_printf("Right-hand-side Norm = %e\n", norm_b);
      //    hypre_printf("Final Risidual Norm = %e\n", norm_res);
      //    hypre_printf("Final Relative Residual Norm = %e\n", norm_res / norm_b);
      //    hypre_printf("\n");
      // }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRGMRESDestroy(solver);
      if (solver_id == 11)
      {
         HYPRE_BoomerAMGDestroy(precond);
      }
      else if (solver_id == 13)
      {
         HYPRE_AMSDestroy(precond);
      }
      else if (solver_id == 16)
      {
         HYPRE_BoomerAMGDDDestroy(precond);
      }
   }

   /* FGMRES solvers */
   else if (solver_id == 21 || solver_id == 23 || solver_id == 24 || solver_id == 25)
   {
      /* Start timing */
      if (solver_id == 21)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-FlexGMRES Setup");
      }
      else if (solver_id == 23)
      {
         time_index = hypre_InitializeTiming("AMS-FlexGMRES Setup");
      }
      else if (solver_id == 24)
      {
         time_index = hypre_InitializeTiming("DS-FlexGMRES Setup");
      }
      else if (solver_id == 25)
      {
         time_index = hypre_InitializeTiming("FlexGMRES Setup");
      }
      hypre_BeginTiming(time_index);

      /* Create FlexGMRES solver */
      HYPRE_ParCSRFlexGMRESCreate(hypre_MPI_COMM_WORLD, &solver);

      /* Set FlexGMRES parameters */
      HYPRE_FlexGMRESSetKDim(solver, kdim);
      HYPRE_FlexGMRESSetMaxIter(solver, maxit);
      HYPRE_FlexGMRESSetTol(solver, tol);
      HYPRE_FlexGMRESSetPrintLevel(solver, print_level);
      HYPRE_FlexGMRESSetLogging(solver, 1);

      /* FlexGMRES with AMG preconditioner */
      if (solver_id == 21)
      {
         /* Now set up the AMG preconditioner and specify any parameters */
         HYPRE_BoomerAMGCreate(&precond);
         HYPRE_BoomerAMGSetPrintLevel(precond, 1);             /* print amg solution info */
         HYPRE_BoomerAMGSetCoarsenType(precond, amg_coarsen_type);            /* Falgout coarsening */
         HYPRE_BoomerAMGSetRelaxType(precond, amg_rlx_type);   /* Sym G.S./Jacobi hybrid */
         HYPRE_BoomerAMGSetNumSweeps(precond, amg_rlx_sweeps); /* Sweeeps on each level */
         HYPRE_BoomerAMGSetMaxLevels(precond, 20);             /* maximum number of levels */
         HYPRE_BoomerAMGSetTol(precond, ptol);                 /* conv. tolerance (if needed) */
         HYPRE_BoomerAMGSetMaxIter(precond, p_maxit);          /* do only one iteration! */
         HYPRE_BoomerAMGSetStrongThreshold(precond, theta);

         /* Set the GMRES preconditioner */
         HYPRE_FlexGMRESSetPrecond(solver,
                                   (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                                   (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                                   precond);
      }
      /* FlexGMRES with AMS preconditioner */
      else if (solver_id == 23)
      {
         /* Now set up the AMS preconditioner and specify any parameters */
         HYPRE_AMSCreate(&precond);
         HYPRE_AMSSetDimension(precond, dim);
         HYPRE_AMSSetMaxIter(precond, p_maxit);
         HYPRE_AMSSetTol(precond, ptol);
         HYPRE_AMSSetCycleType(precond, cycle_type);
         HYPRE_AMSSetPrintLevel(precond, 0);
         HYPRE_AMSSetDiscreteGradient(precond, G);

         if (zero_cond)
         {
            HYPRE_AMSSetInteriorNodes(precond, interior_nodes);
            HYPRE_AMSSetProjectionFrequency(precond, 5);
         }

         /* Vectors Gx, Gy and Gz */
         if (!coordinates)
         {
            HYPRE_AMSSetEdgeConstantVectors(precond, Gx, Gy, Gz);
         }

         /* Vectors x, y and z */
         if (coordinates)
         {
            HYPRE_AMSSetCoordinateVectors(precond, x, y, z);
         }

         /* Poisson matrices */
         if (h1_method)
         {
            HYPRE_AMSSetAlphaPoissonMatrix(precond, Aalpha);
            HYPRE_AMSSetBetaPoissonMatrix(precond, Abeta);
         }

         if (singular_problem)
         {
            HYPRE_AMSSetBetaPoissonMatrix(precond, NULL);
         }

         /* Smoothing and AMG options */
         HYPRE_AMSSetSmoothingOptions(precond, ams_rlx_type, ams_rlx_sweeps, ams_rlx_weight, ams_rlx_omega);
         HYPRE_AMSSetAlphaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                     amg_interp_type, amg_Pmax);
         HYPRE_AMSSetBetaAMGOptions(precond, amg_coarsen_type, amg_agg_levels, amg_rlx_type, theta,
                                    amg_interp_type, amg_Pmax);
         HYPRE_AMSSetAlphaAMGCoarseRelaxType(precond, alpha_coarse_rlx_type);
         HYPRE_AMSSetBetaAMGCoarseRelaxType(precond, beta_coarse_rlx_type);

         HYPRE_AMSSetBetaAMGMaxLevels(precond, beta_amg_max_level);
         HYPRE_AMSSetBetaAMGMaxIter(precond, beta_amg_max_iter);
         HYPRE_AMSSetBetaAMGPrintLevel(precond, beta_amg_print_level);

         HYPRE_AMSSetAlphaXAMGMaxIter(precond, alpha_amg_x_max_iter);
         HYPRE_AMSSetAlphaYAMGMaxIter(precond, alpha_amg_y_max_iter);
         HYPRE_AMSSetAlphaZAMGMaxIter(precond, alpha_amg_z_max_iter);

         /* Set the GMRES preconditioner */
         HYPRE_FlexGMRESSetPrecond(solver,
                                   (HYPRE_PtrToSolverFcn)HYPRE_AMSSolve,
                                   (HYPRE_PtrToSolverFcn)HYPRE_AMSSetup,
                                   precond);
      }

      /* FlexGMRES with diagonal scaling preconditioner */
      else if (solver_id == 14)
      {
         /* Set the GMRES preconditioner */
         HYPRE_FlexGMRESSetPrecond(solver,
                                   (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScale,
                                   (HYPRE_PtrToSolverFcn)HYPRE_ParCSRDiagScaleSetup,
                                   NULL);
      }

      /* Setup */
      HYPRE_FlexGMRESSetup(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x0);

      /* Finalize setup timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* Start timing again */
      if (solver_id == 21)
      {
         time_index = hypre_InitializeTiming("BoomerAMG-FlexGMRES Solve");
      }
      else if (solver_id == 23)
      {
         time_index = hypre_InitializeTiming("AMS-FlexGMRES Solve");
      }
      else if (solver_id == 24)
      {
         time_index = hypre_InitializeTiming("DS-FlexGMRES Solve");
      }
      else if (solver_id == 25)
      {
         time_index = hypre_InitializeTiming("FlexGMRES Solve");
      }
      hypre_BeginTiming(time_index);

      /* Solve */
      HYPRE_FlexGMRESSolve(solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x0);

      /* Finalize solve timing */
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      /* get some info */
      HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRFlexGMRESDestroy(solver);
      if (solver_id == 21)
      {
         HYPRE_BoomerAMGDestroy(precond);
      }
      else if (solver_id == 23)
      {
         HYPRE_AMSDestroy(precond);
      }
   }

   /* Save the solution */
   if (save_sol)
   {
      if (spd_example)
      {
         char *file_locate = "/home/dyt/matrix/SPD_matrix_hypre/";
         char hypre_sol_file[100];
         sprintf(hypre_sol_file, "%s/refine_%d/hypre_sol", file_locate, refine);
         HYPRE_ParVectorPrint(x0, hypre_sol_file);
      }
      else
      {
         char *file_locate = "/home/dyt/matrix/filter_matrix_hypre/";
         char hypre_sol_file[100];
         sprintf(hypre_sol_file, "%s/refine_%d/hypre_sol_rhs_%d", file_locate, refine, rhs_type);
         HYPRE_ParVectorPrint(x0, hypre_sol_file);
      }
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

   if (Gx)
   {
      HYPRE_ParVectorDestroy(Gx);
   }
   if (Gy)
   {
      HYPRE_ParVectorDestroy(Gy);
   }
   if (Gz)
   {
      HYPRE_ParVectorDestroy(Gz);
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

   if (zero_cond)
   {
      HYPRE_ParVectorDestroy(interior_nodes);
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
