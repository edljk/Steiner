/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : 
 * Module      : 
 * Class       : none
 * Language    : C++
 * Description : 
 *
 * Author     : Thomas Pock
 * EMail      : pock@icg.tugraz.at
 *
 */

#include <mex.h>
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>

///////////////////////////////////////////////////////////
// compile with: mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" shrink_osc_mex.cpp
///////////////////////////////////////////////////////////


typedef double ElementType;


void Swap( ElementType *Lhs, ElementType *Rhs )
{
  ElementType Tmp = *Lhs;
  *Lhs = *Rhs;
  *Rhs = Tmp;
}
ElementType Median3( ElementType A[ ], int Left, int Right )
{
  int Center = ( Left + Right ) / 2;
  
  if( A[ Left ] > A[ Center ] )
    Swap( &A[ Left ], &A[ Center ] );
  if( A[ Left ] > A[ Right ] )
    Swap( &A[ Left ], &A[ Right ] );
  if( A[ Center ] > A[ Right ] )
    Swap( &A[ Center ], &A[ Right ] );
  
  /* Invariant: A[ Left ] <= A[ Center ] <= A[ Right ] */
  
  Swap( &A[ Center ], &A[ Right - 1 ] );  /* Hide pivot */
  return A[ Right - 1 ];                /* Return pivot */
}
void InsertionSort( ElementType A[ ], int N )
{
  int j, P;
  ElementType Tmp;
  
  /* 1*/      for( P = 1; P < N; P++ ) {
  /* 2*/          Tmp = A[ P ];
  /* 3*/          for( j = P; j > 0 && A[ j - 1 ] > Tmp; j-- )
  /* 4*/              A[ j ] = A[ j - 1 ];
  /* 5*/          A[ j ] = Tmp;
  }
}

#define Cutoff ( 3 )

void Qsort( ElementType A[ ], int Left, int Right )
{
  int i, j;
  ElementType Pivot;
  if( Left + Cutoff <= Right )
  {
    Pivot = Median3( A, Left, Right );
    i = Left; j = Right - 1;
    for( ; ; )
    {
      while( A[ ++i ] < Pivot ){ }
      while( A[ --j ] > Pivot ){ }
      if( i < j )
        Swap( &A[ i ], &A[ j ] );
      else
        break;
    }
    Swap( &A[ i ], &A[ Right - 1 ] );  /* Restore pivot */
    
    Qsort( A, Left, i - 1 );
    Qsort( A, i + 1, Right );
  }
  else  /* Do an insertion sort on the subarray */
    InsertionSort( A + Left, Right - Left + 1 );
}


void Quicksort( ElementType A[ ], int N )
{
  Qsort( A, 0, N - 1 );
}


void project_simplex(double* v, int q)
{
  double sumResult = -1, tmpValue, tmax;
  bool bget = false;
  double s[q];
  for (int j; j < q; ++j)
    s[j] = v[j];
  
  Quicksort(s,q);
  for(int j = q-1; j >= 1; j--){
    sumResult = sumResult + s[j];
    tmax = sumResult/(q-j);
    if(tmax >= s[j-1]){
      bget = true;
      break;
    }
  }
  
  /* if t is less than s[0] */
  if(!bget){
    sumResult = sumResult + s[0];
    tmax = sumResult/q;
  }
  
  // store the data
  for (int j=0; j<q; j++) {
    v[j] = fmax(0, v[j]-tmax);
  }
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) 
{
  omp_set_num_threads(4);

  double* u =  (double*)mxGetPr(prhs[0]);  
  const mwSize* dims = mxGetDimensions(prhs[0]);
  int n = dims[0];
  int q = dims[1];
  
  double* w =  (double*)mxGetPr(prhs[1]);
 
  // Output
  plhs[0] = mxCreateDoubleMatrix(n, q, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double* out =  (double*)mxGetPr(plhs[0]);
  double sum_iter = 0;
  
  //std::cout << "n = " << n << ", q = " << q << std::endl;  
  
#pragma omp parallel for reduction(+ : sum_iter)
  for (int i=0; i<n; i++) {
    // Dykstra
    double phi[q];
    double psi[q];
    double phi_old[q];
    double psi_old[q];
    double v[q];
    double error;
    for (int j=0; j<q; j++) {
        v[j] = u[i + j*n]/w[i];
    }
    int iter;
    for (iter=1; iter < 1000; ++iter) {
      // project phi
      error = 0;
      for (int j=0; j<q; j++) {
        phi_old[j] = phi[j];
        phi[j] = psi[j] - v[j];        
      }
      project_simplex(phi, q);
      for (int j=0; j<q; j++) {
        error  = fmax(error, fabs(phi_old[j]-phi[j]));
      }      
      
      // project psi
      for (int j=0; j<q; j++) {
        psi_old[j] = psi[j];
        psi[j] = phi[j] + v[j];        
      }
      project_simplex(psi, q);      
      for (int j=0; j<q; j++) {
        error  = fmax(error, fabs(psi_old[j]-psi[j]));
      }      
      
      if (error < 1e-15)
        break;
    }   
    // write back solution
    for (int j=0; j<q; j++) {
      out[i + j*n] = u[i+j*n] + w[i]*(phi[j]-psi[j]);
    }
    sum_iter +=iter;
  }  
  //std::cout << "projection: avg_iter = " << (double)sum_iter/ (double) n << std::endl;
  mxGetPr(plhs[1])[0] = sum_iter/n;  
}