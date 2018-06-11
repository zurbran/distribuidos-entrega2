#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#if defined _OPENMP
#include <omp.h>
#endif

#define obtenerValorMatrizFila(M, F, C, N) (M[(F)*(N)+(C)])
#define asignarValorMatrizFila(M, F, C, N, VALOR) (M[(F)*(N)+(C)] = (VALOR))

#define obtenerValorMatrizColumna(M, F, C, N) (M[(F)+(N)*(C)])
#define asignarValorMatrizColumna(M, F, C, N, VALOR) (M[(F)+(N)*(C)] = (VALOR))

#define obtenerValorMatrizTriaInfFila(M, F, C) (M[(C)+((F)*((F) + 1))/2])
#define asignarValorMatrizTriaInfFila(M, F, C, VALOR) (M[(C)+((F)*((F) + 1))/2]= (VALOR))
#define desplazamientoMatrizTriaInfFila(F, C) ((C)+((F)*((F) + 1))/2)

#define obtenerValorMatrizTriaSupColumna(M, F, C) (M[(F)+((C)*((C) + 1))/2])
#define asignarValorMatrizTriaSupColumna(M, F, C, VALOR) (M[(F)+((C)*((C) + 1))/2]= (VALOR))

enum distribucion{COLUMNAS,FILAS};

void inicializarMatriz(double *A, double valor, int N, enum distribucion dist)
{
    if(dist==FILAS)
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                    asignarValorMatrizFila(A,i,j,N,valor);
            }
        }
    else
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                    asignarValorMatrizColumna(A,i,j,N,valor);
            }
        }
}

void inicializarMatrizInfFil(double *L, double valor, int N)
{
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < i + 1; j++)
		{
			asignarValorMatrizFila(L, i, j, N, valor);
		}
	} 
}

void inicializarMatrizSupCol(double *U, double valor, int N)
{
	for(int i = 0; i < N; i++)
	{
		for(int j = i; j < N; j++)
		{
			asignarValorMatrizTriaSupColumna(U, i, j, valor);
		}
	} 
}

void cuadradaParcialPorCuadrada(double *A, double *B, double *C, int N, int workingRows)
{
	double sum;
	int i, j, k;
	#pragma omp parallel for private(sum, j, k)
	for(i = 0; i < workingRows; i++)
	{
		for(j = 0; j < N; j++)
		{
			sum = 0.0;
			for(k = 0; k < N; k++)
			{
				sum += obtenerValorMatrizFila(A, i, k, N) * obtenerValorMatrizColumna(B, k, j, N);
			}
			asignarValorMatrizFila(C, i, j, N, sum);
		}
	}  
}

void parcialTriangularInferiorPorCuadrada(double *L, double *A, double *C, int N, int workingRows)
{
	double sum;
	int i, j, k;
	#pragma omp parallel for private(sum, j, k)
	for(i = 0; i < workingRows; i++)
	{
		for(j = 0; j < N; j++)
		{
			sum = 0.0;
			for(k = 0; k < i + 1; k++)
			{
				sum += obtenerValorMatrizFila(L, i, k, N) * obtenerValorMatrizColumna(A, k, j, N);
			}
			asignarValorMatrizFila(C, i, j, N, sum);
		}
	}  
}

void parcialCuadradaPorTriangularSuperior(double *A, double *U, double *C, int N, int workingRows)
{
	double sum;
	int i, j, k;
	#pragma omp parallel for private(sum, j, k)
	for(i = 0; i < workingRows; i++)
	{
		for(j = 0; j < N; j++)
		{
			sum = 0.0;
			for(k = 0; k < j+ 1; k++)
			{
				sum += obtenerValorMatrizFila(A, i, k, N) * obtenerValorMatrizTriaSupColumna(U, k, j);
			}
			asignarValorMatrizFila(C, i, j, N, sum);
		}
	}  
}

void sumarMatrices(double *A, double *B, double *C, int length)
{
	#pragma omp parallel for
	for(int i = 0; i < length; i++)
	{
		C[i] = A[i] + B[i];
	}
}

double sumarMatriz(double *A, int length)
{
	double sum = 0.0;
	#pragma omp parallel for reduction(+ : sum)
	for(int i = 0; i < length; i++)
	{
		sum += A[i];
	}
	return sum;
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

void ejercicioUno(int N, int rank, int size)
{
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

    cuadradaParcialPorCuadrada(partA, B, partAB, N, workingRows);
    parcialTriangularInferiorPorCuadrada(partL, C, partLC, N, workingRows);
    parcialCuadradaPorTriangularSuperior(partD, U, partDU, N, workingRows);

    sumarMatrices(partAB, partLC, partM, partialSize);
    sumarMatrices(partM, partDU, partM, partialSize);

    sumaPartU = sumarMatriz(U + (((N * (N + 1)) / 2)/size * rank), ((N * (N + 1)) / 2)/size);

    sumaPartL = sumarMatriz(partL, partialSize);

    MPI_Allreduce(&sumaPartU, &sumU, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sumaPartL, &sumL, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    ulAvg = (sumU/(N*N)) * (sumL/(N*N));
    escalarPorMatriz(partM, ulAvg, partM, partialSize);

    MPI_Gather(partM, partialSize, MPI_DOUBLE, M, partialSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        printf("El tiempo de ejecucion es: %lf.\n", dwalltime() - tiempoInit);
    }
}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size, N;

    #if defined _OPENMP
        int T;
        if ((argc != 3) || ((N = atoi(argv[1])) <= 0) || ((T = atoi(argv[2])) <= 0))
        {
            printf("\nUsar: %s n t\n n: Dimension de la matriz\n T: Cantidad de threads", argv[0]);
            exit(1);
        }

        omp_set_num_threads(T);
    #else
        if(argc != 2 || (N = atoi(argv[1])) <= 0)
        {
            printf("\nUsar: %s n t\n n: Dimension de la matriz\n", argv[0]);
            exit(1);
        }
    #endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ejercicioUno(N,rank,size);

    MPI_Finalize();
    return 0;
}