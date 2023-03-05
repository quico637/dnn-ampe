#include "../include/practica2.h"
#include <iostream>
#include <immintrin.h>

void practica2LinearSeq(float* MK_matrix, float* KN_matrix, float* output_matrix, int M, int K, int N) {
     std::cout << "Running the code for optimized matrix multiplication" << std::endl;
     for(int i=0; i<M; i++) {
         for(int j=0; j<N; j++) {
	     float suma=0.0;
             for(int k=0; k<K; k++) {
	         suma+=MK_matrix[i*K+k]*KN_matrix[j*K+k];
	     }

	     output_matrix[i*N+j]=suma;
         }
     }
}

void practica2Linear(float* MK_matrix, float* KN_matrix, float* output_matrix, int M, int K, int N) {
    std::cout << "Running the code for matrix multiplication" << std::endl;

    // Número de elementos en cada vector
    const int vector_size = 8;
    const int vector_rest = K % vector_size;
    const int last_k = K - vector_size;

    //std::cout << "Elementos restantes: "<<vector_rest<<std::endl;

    // Iterar sobre las filas de la matriz MK
    for (int i = 0; i < M; i++) {

        // Iterar sobre las columnas de la matriz KN
        for (int j = 0; j < N; j++) {

            __m256 acumulador = _mm256_setzero_ps(); // Inicializar el acumulador
            int k = 0;

            // Iterar sobre las columnas de la matriz MK y filas de la matriz KN
            //std::cout << "k: "<<k<<" vector_sice: "<<vector_size<<std::endl;
            for (k = 0; k <= last_k; k += vector_size) {

                // Cargar los datos en vectores de 256 bits
                __m256 a = _mm256_loadu_ps(MK_matrix + i * K + k);
                __m256 b = _mm256_loadu_ps(KN_matrix + j * K + k);

                // Multiplicar y acumular usando FMA
                acumulador = _mm256_fmadd_ps(a, b, acumulador);
            }
            // Sumar los elementos del acumulador y almacenar el resultado en la matriz de salida
            float temp[8];
            _mm256_storeu_ps(temp, acumulador);
            output_matrix[i * N + j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            float suma=0.0;
            k=K-vector_rest;
            for(; k<K; k++) 
            {
                //std::cout<<"elemento 1: "<<MK_matrix[i*K+k]<<" elemento 2: "<<KN_matrix[j*K+k]<<std::endl;
	            suma=MK_matrix[i*K+k]*KN_matrix[j*K+k];
                //std::cout<<"suma: "<<suma<<" out: "<<output_matrix[i * N + j]<<std::endl;
                output_matrix[i * N + j] += suma;
	        }
        }
    }
}


