typedef struct threads_args
{
	int start;
	int end;
}

void *ejercicioUno(void *args)
{
	threads_args *arg = (threads_args*) args;

	// Copia A en B pero ordenado por columnas
	filasAColumnas(A, B, arg->start, arg->end);

	pthread_barrier_wait(&barrier);
	
	// Realiza la multiplicacion
	mulMatrices(A, B, C, arg->start, arg->end);

}