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
#define TR (4)
#define TC (4)
#define TX (4)
#define ARRAY_LAYOUT(ptr, iR,iC,dR,dC) \
((ptr)[iR*dC + iC])
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

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

	float BufA[TR][TX];
	float BufB[TX][TC];
	float BufO[TR][TC];


	for (int row = 0; row < size; row += TR) {
		for (int index_X = 0; index_X < size; index_X += TX) {
			for (int col = 0; col < size; col += TC) {

				// load A
				int external_row;
				int external_X;
				int external_col;
				int internal_row;
				int internal_X;
				int internal_col;

				// loop bound
				int external_row_max = MIN(row + TR, size);
				int external_X_max = MIN(index_X + TX, size);
				int external_col_max = MIN(col + TC, size);

				for (external_row = row, internal_row = 0; external_row < external_row_max; external_row++, internal_row++)
					for (external_X = index_X, internal_X = 0; external_X < external_X_max ; external_X++, internal_X++)
						BufA[internal_row][internal_X] = ARRAY_LAYOUT(in1, external_row, external_X, size, size);
				
				// load B
				for (external_X = index_X, internal_X = 0; external_X < external_X_max ; external_X++, internal_X++)
					for (external_col = col, internal_col = 0; external_col < external_col_max; external_col++, internal_col++)	
						BufB[internal_X][internal_col] = ARRAY_LAYOUT(in2, external_X, external_col, size, size);

				// load O
				for (external_row = row, internal_row = 0; external_row < external_row_max; external_row++, internal_row++)
					for (external_col = col, internal_col = 0; external_col < external_col_max; external_col++, internal_col++)	
						BufO[internal_row][internal_col] = ARRAY_LAYOUT(out_r, external_row, external_X, size, size);

				// call mmm_blocked_kernel
				mmm_blocked_kernel(BufA, BufB, BufO);
				
				// write back O
				for (external_row = row, internal_row = 0; external_row < external_row_max; external_row++, internal_row++)
					for (external_col = col, internal_col = 0; external_col < external_col_max; external_col++, internal_col++)	
						ARRAY_LAYOUT(out_r, external_row, external_col, size, size) = BufO[internal_row][internal_col];
				
			}
		}
	}
}

void mmm_blocked_kernel (float BufA[TR][TX], float BufB[TX][TC], float BufO[TR][TC]) {

	for(int i = 0; i < TR; i++) 
		for (int j = 0; j < TX; j++)
			for (int k = 0; k < TC; k++)
				BufO[i][j] += BufA[i][k]*BufB[k][j];
	
}



}

    // std::cout << "TODO: Replace Vector-Add Implementation with MMM Implementation" << std::endl;

	// for (int i = 0; i < size * size; i++)
	// 	out_r[i] = 0;


	// float in2_buffer[BUFFER_SIZE];

    // for (int k = 0; k < size; k++) {
    //     for (int i = 0; i < size; i++) {
    //     	float in1_broadcast = in1[i*size+k];
    //     	for (int j = 0; j < size; j+=BUFFER_SIZE) {

    //     		int chunk_size = BUFFER_SIZE;
	// 			//boundary checks
	// 			if ((j + BUFFER_SIZE) > size)
	// 				chunk_size = size - j;

	// 			for (int a = 0; a < chunk_size; a++) {
	// 				in2_buffer[a] = in2[k*size+j + a];
	// 			}

	// 			for (int a = 0; a < chunk_size; a++) {
	// 				out_r[i*size + j + a] += in1_broadcast * in2_buffer[a];
	// 			}
    //     	}
    //     }
    // }
