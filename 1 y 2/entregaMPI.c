#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#if defined _OPENMP
#include <omp.h>
#endif

#define obtenerIndiceMatrizFila(F, C, N) ((F)*(N)+(C))

#define obtenerIndiceMatrizColumna(F, C, N) ((F)+(N)*(C))

#define obtenerIndiceMatrizTriaSupColumna(F, C) ((F)+((C)*((C) + 1))/2)

enum distribucion{COLUMNAS,FILAS};

void inicializarMatriz(double *A, double valor, int N, enum distribucion dist)
{
    if(dist==FILAS)
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                    A[obtenerIndiceMatrizFila(i,j,N)] = valor;
            }
        }
    else
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                    A[obtenerIndiceMatrizColumna(i,j,N)] = valor;
            }
        }
}

void inicializarMatrizInfFil(double *L, double valor, int N)
{
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < i + 1; j++)
		{
			L[obtenerIndiceMatrizFila(i, j, N)] = valor;
		}
	} 
}

void inicializarMatrizSupCol(double *U, double valor, int N)
{
	for(int i = 0; i < N; i++)
	{
		for(int j = i; j < N; j++)
		{
			U[obtenerIndiceMatrizTriaSupColumna(i, j)] = valor;
		}
	} 
}

void partialSquareRowMatXSquareColMat(double *A, double *B, double *C, int N, int workingRows)
{
	double res;
	#pragma omp parallel for private(res)
	for(int i = 0; i < workingRows; i++)
	{
		for(int j = 0; j < N; j++)
		{
			res = 0.0;
			for(int k = 0; k < N; k++)
			{
				res += A[obtenerIndiceMatrizFila(i, k, N)] * B[obtenerIndiceMatrizColumna(k, j, N)];
			}
			C[obtenerIndiceMatrizFila(i, j, N)] = res;
		}
	}  
}

void partialSquareRowMatXUpperColMat(double *A, double *U, double *C, int N, int workingRows)
{
	double res;
	#pragma omp parallel for private(res) schedule(dynamic, 64)
	for(int i = 0; i < workingRows; i++)
	{
		for(int j = 0; j < N; j++)
		{
			res = 0.0;
			for(int k = 0; k < j + 1; k++)
			{
				res += A[obtenerIndiceMatrizFila(i, k, N)] * U[obtenerIndiceMatrizTriaSupColumna(k, j)];
			}
			C[obtenerIndiceMatrizFila(i, j, N)] = res;
		}
	}  
}

void addMatrix(double *A, double *B, double *C, int length)
{
	#pragma omp parallel for
	for(int i = 0; i < length; i++)
	{
		C[i] = A[i] + B[i];
	}
}

double sumMatrix(double *A, int length)
{
	double suma = 0.0;
	#pragma omp parallel for reduction(+ : suma)
	for(int i = 0; i < length; i++)
	{
		suma += A[i];
	}
	return suma;
}

void escalarPorMatriz(double *A, double esc, double *C, int length)
{
	#pragma omp parallel for
	for(int i = 0; i < length; i++)
	{
		C[i] = esc * A[i];
	}
}


double dwalltime()
{
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size, N;

    #if defined _OPENMP
        int Threads;
        if ((argc != 3) || ((N = atoi(argv[1])) <= 0) || ((T = atoi(argv[2])) <= 0))
        {
            printf("Dimension no especificada o Threads incorrectos");
            exit(1);
        }
        omp_set_num_threads(Threads);
    #else
        if(argc != 2 || (N = atoi(argv[1])) <= 0)
        {
            printf("Dimension no especificada");
            exit(1);
        }
    #endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int length = N*N;
    int partialSize = length / size;
    int workingRows = N / size;
    double *A, *B, *C, *D, *L, *M, *U;
    double *partA, *partL, *partD, *partM;
    double *partAB, *partLC, *partDU;
    double ulAvg;
    double sumU, sumL;
    double sumaPartU, sumaPartL;

    B = (double*)malloc(sizeof(double) * length);
	C = (double*)malloc(sizeof(double) * length);
	U = (double*)malloc(sizeof(double) * (N * (N + 1)) / 2);

    partA = (double*)malloc(sizeof(double)*partialSize);
    partL = (double*)malloc(sizeof(double)*partialSize);
    partD = (double*)malloc(sizeof(double)*partialSize);
    partM = (double*)malloc(sizeof(double)*partialSize);
    partAB = (double*)malloc(sizeof(double)*partialSize);
    partLC = (double*)malloc(sizeof(double)*partialSize);
    partDU = (double*)malloc(sizeof(double)*partialSize);

    if(rank == 0)
    {
        A = (double*)malloc(sizeof(double) * length);
        D = (double*)malloc(sizeof(double) * length);
		L = (double*)malloc(sizeof(double) * length);
		M = (double*)malloc(sizeof(double) * length);
        inicializarMatriz(A, 1, N, FILAS);
        inicializarMatriz(D, 2, N, FILAS);
		inicializarMatriz(B, 1, N, COLUMNAS);
        inicializarMatriz(C, 1, N, COLUMNAS);
        inicializarMatriz(L, 0, N, FILAS);
		inicializarMatrizInfFil(L, 1, N);
		inicializarMatrizSupCol(U, 2, N);
    }
    
    double tiempoInit = dwalltime();

    MPI_Scatter(A, partialSize, MPI_DOUBLE, partA, partialSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(L, partialSize, MPI_DOUBLE, partL, partialSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(D, partialSize, MPI_DOUBLE, partD, partialSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(B, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(C, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(U, (N * (N + 1)) / 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double tiempoSlotIni = dwalltime();

    partialSquareRowMatXSquareColMat(partA, B, partAB, N, workingRows);
    partialSquareRowMatXSquareColMat(partL, C, partLC, N, workingRows);
    partialSquareRowMatXUpperColMat(partD, U, partDU, N, workingRows);

    addMatrix(partAB, partLC, partM, partialSize);
    addMatrix(partM, partDU, partM, partialSize);

    sumaPartU = sumMatrix(U + (((N * (N + 1)) / 2)/size * rank), ((N * (N + 1)) / 2)/size);
    sumaPartL = sumMatrix(partL, partialSize);

    MPI_Allreduce(&sumaPartU, &sumU, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sumaPartL, &sumL, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    ulAvg = (sumU/(N*N)) * (sumL/(N*N));
    escalarPorMatriz(partM, ulAvg, partM, partialSize);

	double tiempoSlotFin = dwalltime() - tiempoSlotIni;

	printf("Tiempo de proceso nro %d: %lf.\n",rank,tiempoSlotFin);

    MPI_Gather(partM, partialSize, MPI_DOUBLE, M, partialSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        printf("El tiempo de ejecucion es: %lf.\n", dwalltime() - tiempoInit);
		free(A);
		free(D);
		free(L);
		free(M);
    }
	else
	{
		free(partA);
		free(partL);
		free(partD);
		free(partM);
		free(partAB);
		free(partLC);
		free(partDU);
	}

	free(B);
	free(C);
	free(U);

    MPI_Finalize();
    return 0;
}
