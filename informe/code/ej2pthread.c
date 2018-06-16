void *ejercicioDos(void *args){

	threads_args *arg = (threads_args*) args;

	// Calcular promedios
	sumarMatriz(U,arg->triaStart,arg->triaEnd,&sumMutex);

	if(pthread_barrier_wait(&threadBarrier) == PTHREAD_BARRIER_SERIAL_THREAD)
	{
		uAvg = sumaPromedio / (N*N);
		sumaPromedio = 0;
	}

	pthread_barrier_wait(&threadBarrier);

	sumarMatriz(L,arg->triaStart,arg->triaEnd,&sumMutex);

	if(pthread_barrier_wait(&threadBarrier) == PTHREAD_BARRIER_SERIAL_THREAD)
	{
		lAvg = sumaPromedio / (N*N);
		sumaPromedio = 0;
	}

	pthread_barrier_wait(&threadBarrier);

	ulAvg = uAvg * lAvg;

	sumarMatriz(B,arg->vectStart,arg->vectEnd,&sumMutex);

	if(pthread_barrier_wait(&threadBarrier) == PTHREAD_BARRIER_SERIAL_THREAD)
	{
		bAvg = sumaPromedio / (N*N);
		sumaPromedio = 0;
	}
	// Fin de calcular promedios
	pthread_barrier_wait(&threadBarrier);
	// Paso matriz C a columnas para hacer A*C
	filasAColumnas(C, tC, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	// Multiplicar AtC y guardar en TAC
	mulMatrices(A, tC, TAC, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);
	// Paso a columna nuevamente
	filasAColumnas(TAC, tTAC, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	// Multiplicar A por tTAC y guardar en TAAC
	mulMatrices(A, tTAC, TAAC, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	// Multiplicar ulAvg por TAAC y almacenar en ulTAAC
	escalarPorMatriz(TAAC, ulTAAC, ulAvg, arg->vectStart, arg->vectEnd);
	// ulTAAC ahora contiene el primer termino ordenado por filas
	pthread_barrier_wait(&threadBarrier);
	// Ordeno a E por columnas
	filasAColumnas(E, tE, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	// Multiplicar B por tE y almacenar en TBE
	mulMatrices(B, tE, TBE, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	filasAColumnas(TBE, tTBE, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	// Multiplicar L por tTBE (BE) y almacenar en TLBE
	triangularInferiorPorCuadrada(L, tTBE, TLBE, arg->start, arg->end);
	/// TLBE ahora contiene el segundo termino sin el escalar multiplicado
	pthread_barrier_wait(&threadBarrier);

	// Preparo F para ser multiplicada pasandola a columnas
	filasAColumnas(F,tF, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	// Multiplicar U por tF y almacenar en TUF
	triangularSuperiorPorCuadrada(U, tF, TUF, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

	filasAColumnas(TUF,tTUF, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);

    // Multiplicar D por tTUF y almacenar en TDUF (ordenado por filas)
	mulMatrices(D, tTUF, TDUF, arg->start, arg->end);

	pthread_barrier_wait(&threadBarrier);
	// Dado que TLBE y TDUF estan ordenadas por filas se puede sumar como un vector
	sumarMatrices(TLBE,TDUF,TLBEDUF, arg->vectStart, arg->vectEnd);

	pthread_barrier_wait(&threadBarrier);
	// Multiplico el escalar (promedio de B, bAvg) a la matriz resultante de la suma (TLBEDUF)
	escalarPorMatriz(TLBEDUF, M, bAvg, arg->vectStart, arg->vectEnd);
	// M ahora contiene el segundo y ultimo termino
	pthread_barrier_wait(&threadBarrier);

	// Sumar el primer termino haciendo asi que M tenga el resultado final
	sumarMatrices(M, ulTAAC, M, arg->vectStart, arg->vectEnd);
	//M ahora contiene el resutlado

}