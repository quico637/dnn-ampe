#include "../include/practica2.h"
#include <iostream>
#include <immintrin.h>
#include <omp.h>

void practica2LinearSeq(float *MK_matrix, float *KN_matrix, float *output_matrix, int M, int K, int N)
{
    std::cout << "Running the code for optimized matrix multiplication" << std::endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float suma = 0.0;
            for (int k = 0; k < K; k++)
            {
                suma += MK_matrix[i * K + k] * KN_matrix[j * K + k];
            }

            output_matrix[i * N + j] = suma;
        }
    }
}

void practica2Linear(float *MK_matrix, float *KN_matrix, float *output_matrix, int M, int K, int N)
{
    //omp_set_num_threads(2);
    std::cout << "Running the code for matrix multiplication" << std::endl;

    const int vector_size = 8;
    const int vector_rest = K % vector_size;
    const int last_k = K - vector_size;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {

            __m256 acumulador = _mm256_setzero_ps();
            int k = 0;
            
            for (k = 0; k <= last_k; k += vector_size)
            {
           
                __m256 a = _mm256_loadu_ps(MK_matrix + i * K + k);
                __m256 b = _mm256_loadu_ps(KN_matrix + j * K + k);
            
                acumulador = _mm256_fmadd_ps(a, b, acumulador);
            }

            
            float temp[vector_size];
            _mm256_storeu_ps(temp, acumulador);

            output_matrix[i * N + j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            float suma = 0.0;
            k = K - vector_rest;
            for (; k < K; k++)
            {
                
                suma = MK_matrix[i * K + k] * KN_matrix[j * K + k];
                output_matrix[i * N + j] += suma;
            }
        }
    }
}
