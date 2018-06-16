// Acceso y asignacion por filas
#define obtenerValorMatrizFila(M, F, C, N) (M[(F)*(N)+(C)])
#define asignarValorMatrizFila(M, F, C, N, VALOR) (M[(F)*(N)+(C)] = (VALOR))
// Acceso y asignacion por columnas
#define obtenerValorMatrizColumna(M, F, C, N) (M[(F)+(N)*(C)])
#define asignarValorMatrizColumna(M, F, C, N, VALOR) (M[(F)+(N)*(C)] = (VALOR))
//Funcion para multiplicar matrices
void mulMatrices(double *A, double *B, double *C)
{
	double sum;
	int i, j, k;
	#pragma omp parallel for private(sum, j, k)
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
// Funcion para pasar de filas a columnas
void filasAColumnas(double *A, double *B)
{
	#pragma omp parallel for
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			asignarValorMatrizColumna(B, i, j, N, obtenerValorMatrizFila(A, i, j, N));
		}
	}
}