	int64_t pares = 0;
	#pragma omp parallel for reduction(+:pares)
	for (int64_t i = 0; i < N; i++)
	{
		if(A[i] & 1 == 0)
		{
			pares += 1;
		}
	}