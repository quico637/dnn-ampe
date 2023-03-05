#include "../include/practica2.h"
#include <iostream>
#include <stdio.h>
#include <immintrin.h>
#include <cstring>

// Commit nueva rama

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
    std::cout << "Running the code for matrix multiplication" << std::endl;

    // Número de elementos en cada vector
    const int vector_size = 8;
    const int rest = K % vector_size;

    int k;
    int mask[8];

    for (int i = 0; i < vector_size; i++)
    {
        if (i < rest)
        {
            mask[i] = -20;
        }
        else
        {
            mask[i] = 3;
        }
    }

    // Iterar sobre las filas de la matriz MK
    for (int i = 0; i < M; i++)
    {

        // Iterar sobre las columnas de la matriz KN
        for (int j = 0; j < N; j++)
        {

            __m256 acumulador = _mm256_setzero_ps(); // Inicializar el acumulador

            // Iterar sobre las columnas de la matriz MK y filas de la matriz KN

            for (k = 0; k < K; k += vector_size)
            {

                // Cargar los datos en vectores de 256 bits
                __m256 a = _mm256_loadu_ps(MK_matrix + i * K + k); // recorre por columnas
                __m256 b = _mm256_loadu_ps(KN_matrix + j * K + k);

                // Multiplicar y acumular usando FMA
                acumulador = _mm256_fmadd_ps(a, b, acumulador);
            }

            // Sumar los elementos del acumulador y almacenar el resultado en la matriz de salida
            
            float temp[vector_size];

            if (j < N - 1)
            {
                
                _mm256_storeu_ps(temp, acumulador);

                output_matrix[i * N + j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            }
            else
            {
                if (K % vector_size != 0)
                {
                    k += vector_size;
                    __m256 acumulador = _mm256_setzero_ps(); // Inicializar el acumulador

                    __m256 a = _mm256_loadu_ps(MK_matrix + i * K + k); // recorre por columnas
                    __m256 b = _mm256_loadu_ps(KN_matrix + j * K + k);

                    acumulador = _mm256_fmadd_ps(a, b, acumulador);

                    memset(temp, 0., vector_size);

                    _mm256_storeu_ps(temp, acumulador);

                    __m256i mask = _mm256_setr_epi32(mask[0], mask[1], mask[2], mask[3], mask[4], mask[5], mask[6], mask[7]);
                    __m256 result = _mm256_maskload_ps(temp, mask);
                    output_matrix[i * N + j] += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
                }else {

                    _mm256_storeu_ps(temp, acumulador);

                    output_matrix[i * N + j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
                }
            }

            // RESTO
        }
    }
}
