void triangularSuperiorPorCuadrada(double *U, double *B, double *C)
{
	double sum;
	int i, j, k;
	#pragma omp parallel for private(sum, j, k) schedule(dynamic, 64)
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			sum = 0.0;
			for(k = i; k < N ; k++)
			{
				sum += obtenerValorMatrizTriaSupFila(U, i, k, N) * obtenerValorMatrizColumna(B, k, j, N);
			}
			asignarValorMatrizFila(C, i, j, N, sum);
		}
	}  
}

void triangularInferiorPorCuadrada(double *L, double *A, double *C)
{
	double sum;
	int i, j, k;
	#pragma omp parallel for private(sum, j, k) schedule(dynamic, 64)
	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
		{
			sum = 0.0;
			for(k = 0; k < i + 1; k++)
			{
				sum += obtenerValorMatrizTriaInfFila(L, i, k) * obtenerValorMatrizColumna(A, k, j, N);
			}
			asignarValorMatrizFila(C, i, j, N, sum);
		}
	}  
}