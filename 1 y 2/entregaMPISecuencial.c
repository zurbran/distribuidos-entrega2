#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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
			asignarValorMatrizTriaInfFila(L, i, j, valor);
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

void cuadradaFilPorCuadradaCol(double *A, double *B, double *C, int N)
{
	double sum;
	int i, j, k;
	for(i = 0; i < N; i++)
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

void triangularInferiorPorCuadrada(double *L, double *B, double *C, int N)
{
	double sum;
	int i, j, k;
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			sum = 0.0;
			for(k = 0; k < i + 1; k++)
			{
				sum += obtenerValorMatrizTriaInfFila(L, i, k) * obtenerValorMatrizColumna(B, k, j, N);
			}
			asignarValorMatrizFila(C, i, j, N, sum);
		}
	}  
}


void cuadradaPorTriangularSuperior(double *A, double *U, double *C, int N)
{
	double sum;
	int i, j, k;
	for(i = 0; i < N; i++)
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
	for(int i = 0; i < length; i++)
	{
		C[i] = A[i] + B[i];
	}
}

double sumarMatriz(double *A, int length)
{
	double sum = 0.0;
	for(int i = 0; i < length; i++)
	{
		sum += A[i];
	}
	return sum;
}

void escalarPorMatriz(double *A, double esc, double *C, int length)
{
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
    int N;
    double *A, *B, *C, *D, *L, *M, *U;
    double *AB, *LC, *DU;
    double ulAvg;
    double sumU, sumL;
    if(argc != 2 || (N = atoi(argv[1])) <= 0)
    {
        printf("\nUsar: %s n \n n: Dimension de la matriz\n", argv[0]);
        exit(1);
    }

    int length = N*N;

    A = (double*)malloc(sizeof(double) * length);
    B = (double*)malloc(sizeof(double) * length);
	C = (double*)malloc(sizeof(double) * length);
    D = (double*)malloc(sizeof(double) * length);
    L = (double*)malloc(sizeof(double) * (N * (N + 1)) / 2);
    M = (double*)malloc(sizeof(double) * length);
    U = (double*)malloc(sizeof(double) * (N * (N + 1)) / 2);
    AB = (double*)malloc(sizeof(double) * length);
    LC = (double*)malloc(sizeof(double) * length);
	DU = (double*)malloc(sizeof(double) * length);

    inicializarMatriz(A, 1, N, FILAS);
    inicializarMatriz(D, 2, N, FILAS);
    inicializarMatriz(B, 1, N, COLUMNAS);
    inicializarMatriz(C, 1, N, COLUMNAS);
    inicializarMatrizInfFil(L, 1, N);
    inicializarMatrizSupCol(U, 2, N);
    
    double tiempoInit = dwalltime();

    cuadradaFilPorCuadradaCol(A, B, AB, N);
    triangularInferiorPorCuadrada(L, C, LC, N);
    cuadradaPorTriangularSuperior(D, U, DU, N);

    sumarMatrices(AB, LC, M, length);
    sumarMatrices(M, DU, M, length);

    sumU = sumarMatriz(U, ((N * (N + 1)) / 2));
    sumL = sumarMatriz(L, ((N * (N + 1)) / 2));
   
    ulAvg = (sumU/(N*N)) * (sumL/(N*N));
    escalarPorMatriz(M, ulAvg, M, length);

    printf("El tiempo de ejecucion es: %lf.\n", dwalltime() - tiempoInit);

    return 0;
}