void sumarMatriz(double *M, int start, int end, pthread_mutex_t *mutex)
{
    double total = 0.0;

    for (int i = start; i <= end; i++)
    {
        total += M[i];
    }

	pthread_mutex_lock(mutex);
    sumaPromedio += total;
    pthread_mutex_unlock(mutex);
}