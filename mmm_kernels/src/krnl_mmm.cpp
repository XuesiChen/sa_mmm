/****************************************************************
 * Copyright (c) 2017~2022, 18-643 Course Staff, CMU
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of the FreeBSD Project.
 ****************************************************************/

//------------------------------------------------------------------------------
//
// kernel:  mmm
//
// Purpose: Demonstrate Matrix Multiplication Kernel
//

#define BUFFER_SIZE 512
#include <iostream>

/*
    Matrix Multiplication Kernel Implementation
    Arguments:
        in1   (input)     --> Input Matrix1
        in2   (input)     --> Input Matrix2
        out_r (output)    --> Output Matrix
        size  (input)     --> Dimension of Matrix in Integer
 */

extern "C" {
void krnl_mmm(const float *in1,  // Read-Only Matrix 1
        const float *in2,      // Read-Only Matrix 2
        float *out_r,          // Output Result
        int size                      // Dimension in integer
) {
    // std::cout << "TODO: Replace Vector-Add Implementation with MMM Implementation" << std::endl;

	for (int i = 0; i < size * size; i++)
		out_r[i] = 0;


	float in2_buffer[BUFFER_SIZE];

    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
        	float in1_broadcast = in1[i*size+k];
        	for (int j = 0; j < size; j+=BUFFER_SIZE) {

        		int chunk_size = BUFFER_SIZE;
				//boundary checks
				if ((j + BUFFER_SIZE) > size)
					chunk_size = size - j;

				for (int a = 0; a < chunk_size; a++) {
					in2_buffer[a] = in2[k*size+j + a];
				}

				for (int a = 0; a < chunk_size; a++) {
					out_r[i*size + j + a] += in1_broadcast * in2_buffer[a];
				}
        	}
        }
    }

}
}
