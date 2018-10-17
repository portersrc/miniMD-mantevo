/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "mpi.h"
#include "atom.h"
#include "neighbor.h"
#include "openacc.h"
#define DELTA 20000

void Atom_init(Atom *atom)
{
  atom->natoms = 0;
  atom->nlocal = 0;
  atom->nghost = 0;
  atom->nmax = 0;
  atom->copy_size = 0;

  atom->x = NULL;
  atom->v = NULL;
  atom->f = NULL;
  atom->xold = NULL;
  atom->x_copy = NULL;
  atom->v_copy = NULL;
  atom->d_x = NULL;
  atom->d_v = NULL;
  atom->d_f = NULL;
  atom->comm_size = 3;
  atom->reverse_size = 3;
  atom->border_size = 3;

  atom->mass = 1;
}

void Atom_destroy(Atom *atom)
{
  if(atom->nmax){
    Atom_destroy_2d_MMD_float_array(atom, atom->x);
    Atom_destroy_2d_MMD_float_array(atom, atom->v);
    Atom_destroy_2d_MMD_float_array(atom, atom->f);
    Atom_destroy_2d_MMD_float_array(atom, atom->xold);
    acc_free(atom->d_x);
    acc_free(atom->d_v);
    acc_free(atom->d_f);
  }
}

void Atom_sync_host(Atom *atom, void* h_ptr_, void* d_ptr_, int bytes) {
  const int* const restrict d_ptr = (int*) d_ptr_;
  int* const restrict h_ptr = (int*) h_ptr_;
  const int n = bytes/4;

  #pragma acc kernels copyout(h_ptr[0:n]) deviceptr(d_ptr)
  for(int i = 0; i<n; i++)
    h_ptr[i] = d_ptr[i];
}

void Atom_sync_device(Atom *atom, void* d_ptr_, void* h_ptr_, int bytes) {
  int* const restrict d_ptr = (int*) d_ptr_;
  const int* const restrict h_ptr = (int*) h_ptr_;
  const int n = bytes/4;

  #pragma acc kernels copyin(h_ptr[0:n]) deviceptr(d_ptr)
  for(int i = 0; i<n; i++)
    d_ptr[i] = h_ptr[i];
}

void Atom_growarray(Atom *atom)
{
  int nold = atom->nmax;
  atom->nmax += DELTA;
  atom->x = (MMD_float**) Atom_realloc_2d_MMD_float_array(atom, atom->x, atom->nmax, PAD, PAD * nold);
  atom->v = (MMD_float**) Atom_realloc_2d_MMD_float_array(atom, atom->v, atom->nmax, PAD, PAD * nold);
  atom->f = (MMD_float**) Atom_realloc_2d_MMD_float_array(atom, atom->f, atom->nmax, PAD, PAD * nold);
  atom->xold = (MMD_float**) Atom_realloc_2d_MMD_float_array(atom, atom->xold, atom->nmax, PAD, PAD * nold);
 
  acc_free(atom->d_x);
  acc_free(atom->d_v);
  acc_free(atom->d_f);
  atom->d_x = (MMD_float*) acc_malloc(atom->nmax*PAD*sizeof(MMD_float));
  atom->d_v = (MMD_float*) acc_malloc(atom->nmax*PAD*sizeof(MMD_float));
  atom->d_f = (MMD_float*) acc_malloc(atom->nmax*PAD*sizeof(MMD_float));
  if(atom->x == NULL || atom->v == NULL || atom->f == NULL || atom->xold == NULL) {
    printf("ERROR: No memory for atoms\n");
  }
}

void Atom_addatom(Atom *atom, MMD_float x_in, MMD_float y_in, MMD_float z_in,
                   MMD_float vx_in, MMD_float vy_in, MMD_float vz_in)
{
  if(atom->nlocal == atom->nmax) Atom_growarray(atom);

  atom->x[atom->nlocal][0] = x_in;
  atom->x[atom->nlocal][1] = y_in;
  atom->x[atom->nlocal][2] = z_in;
  atom->v[atom->nlocal][0] = vx_in;
  atom->v[atom->nlocal][1] = vy_in;
  atom->v[atom->nlocal][2] = vz_in;

  atom->nlocal++;
}

/* enforce PBC
   order of 2 tests is important to insure lo-bound <= coord < hi-bound
   even with round-off errors where (coord +/- epsilon) +/- period = bound */

void Atom_pbc(Atom *atom)
{
  #pragma omp parallel for
  for(int i = 0; i < atom->nlocal; i++) {
    if(atom->x[i][0] < 0.0) atom->x[i][0] += atom->box.xprd;

    if(atom->x[i][0] >= atom->box.xprd) atom->x[i][0] -= atom->box.xprd;

    if(atom->x[i][1] < 0.0) atom->x[i][1] += atom->box.yprd;

    if(atom->x[i][1] >= atom->box.yprd) atom->x[i][1] -= atom->box.yprd;

    if(atom->x[i][2] < 0.0) atom->x[i][2] += atom->box.zprd;

    if(atom->x[i][2] >= atom->box.zprd) atom->x[i][2] -= atom->box.zprd;
  }
}

void Atom_copy(Atom *atom, int i, int j)
{
  atom->x[j][0] = atom->x[i][0];
  atom->x[j][1] = atom->x[i][1];
  atom->x[j][2] = atom->x[i][2];
  atom->v[j][0] = atom->v[i][0];
  atom->v[j][1] = atom->v[i][1];
  atom->v[j][2] = atom->v[i][2];
}

void Atom_pack_comm(Atom *atom, int n_, int* list_, MMD_float* buf_, int* pbc_flags)
{
  const int n = n_;
  const int* const restrict list = list_;
  MMD_float* const restrict buf = buf_;
  const MMD_float* const restrict x_ = atom->d_x;
  const int pbc_flags1 = pbc_flags[1];
  const int pbc_flags2 = pbc_flags[2];
  const int pbc_flags3 = pbc_flags[3];
  const MMD_float xprd = atom->box.xprd;
  const MMD_float yprd = atom->box.yprd;
  const MMD_float zprd = atom->box.zprd;

  if(pbc_flags[0] == 0) {
    #pragma acc kernels deviceptr(x_,buf) copyin(list[0:n])
    for(int i = 0; i < n; i++) {
      const int j = list[i];
      buf[3 * i] = x_[j*PAD+0];
      buf[3 * i + 1] = x_[j*PAD+1];
      buf[3 * i + 2] = x_[j*PAD+2];
    }
  } else {
    #pragma acc kernels deviceptr(x_,buf) copyin(list[0:n])
    for(int i = 0; i < n; i++) {
      const int j = list[i];
      buf[3 * i] = x_[j*PAD+0] + pbc_flags1 * xprd;
      buf[3 * i + 1] = x_[j*PAD+1] + pbc_flags2 * yprd;
      buf[3 * i + 2] = x_[j*PAD+2] + pbc_flags3 * zprd;
    }
  }
}

void Atom_unpack_comm(Atom *atom, int n_, int first_, MMD_float* buf_)
{
  const int n = 3*n_;
  const int first = first_*PAD;
  MMD_float* const restrict x_ = atom->d_x;
  const MMD_float* const restrict buf = buf_;

  #pragma acc kernels deviceptr(x_,buf)
  for(int i = 0; i < n; i++) {
    x_[first + i] = buf[i];
//    x[first + i][0] = buf[3 * i + 0];
//    x[first + i][1] = buf[3 * i + 1];
//    x[first + i][2] = buf[3 * i + 2];
  }
}

void Atom_pack_reverse(Atom *atom, int n, int first, MMD_float* buf)
{
  int i;

  for(i = 0; i < n; i++) {
    buf[3 * i] = atom->f[first + i][0];
    buf[3 * i + 1] = atom->f[first + i][1];
    buf[3 * i + 2] = atom->f[first + i][2];
  }
}

void Atom_unpack_reverse(Atom *atom, int n, int* list, MMD_float* buf)
{
  int i, j;

  for(i = 0; i < n; i++) {
    j = list[i];
    atom->f[j][0] += buf[3 * i];
    atom->f[j][1] += buf[3 * i + 1];
    atom->f[j][2] += buf[3 * i + 2];
  }
}

int Atom_pack_border(Atom *atom, int i, MMD_float* buf, int* pbc_flags)
{
  int m = 0;

  if(pbc_flags[0] == 0) {
    buf[m++] = atom->x[i][0];
    buf[m++] = atom->x[i][1];
    buf[m++] = atom->x[i][2];
  } else {
    buf[m++] = atom->x[i][0] + pbc_flags[1] * atom->box.xprd;
    buf[m++] = atom->x[i][1] + pbc_flags[2] * atom->box.yprd;
    buf[m++] = atom->x[i][2] + pbc_flags[3] * atom->box.zprd;
  }

  return m;
}

int Atom_unpack_border(Atom *atom, int i, MMD_float* buf)
{
  if(i == atom->nmax) Atom_growarray(atom);

  int m = 0;
  atom->x[i][0] = buf[m++];
  atom->x[i][1] = buf[m++];
  atom->x[i][2] = buf[m++];
  return m;
}

int Atom_pack_exchange(Atom *atom, int i, MMD_float* buf)
{
  int m = 0;
  buf[m++] = atom->x[i][0];
  buf[m++] = atom->x[i][1];
  buf[m++] = atom->x[i][2];
  buf[m++] = atom->v[i][0];
  buf[m++] = atom->v[i][1];
  buf[m++] = atom->v[i][2];
  return m;
}

int Atom_unpack_exchange(Atom *atom, int i, MMD_float* buf)
{
  if(i == atom->nmax) Atom_growarray(atom);

  int m = 0;
  atom->x[i][0] = buf[m++];
  atom->x[i][1] = buf[m++];
  atom->x[i][2] = buf[m++];
  atom->v[i][0] = buf[m++];
  atom->v[i][1] = buf[m++];
  atom->v[i][2] = buf[m++];
  return m;
}

int Atom_skip_exchange(Atom *atom, MMD_float* buf)
{
  return 6;
}

/* realloc a 2-d MMD_float array */

MMD_float** Atom_realloc_2d_MMD_float_array(Atom *atom, MMD_float** array,
    int n1, int n2, int nold)

{
  MMD_float** newarray;

  newarray = Atom_create_2d_MMD_float_array(atom, n1, n2);

  if(nold) memcpy(newarray[0], array[0], nold * sizeof(MMD_float));

  Atom_destroy_2d_MMD_float_array(atom, array);

  return newarray;
}

/* create a 2-d MMD_float array */

MMD_float** Atom_create_2d_MMD_float_array(Atom *atom, int n1, int n2)
{
  int ALIGN = 16;
  MMD_float** array;
  MMD_float* data;
  int i, n;

  if(n1 * n2 == 0) return NULL;

  #ifdef ALIGNMALLOC
    array = (MMD_float**) _mm_malloc(n1 * sizeof(MMD_float*), ALIGNMALLOC);
    data = (MMD_float*) _mm_malloc((n1 * n2 + 1024 + 1) * sizeof(MMD_float), ALIGNMALLOC);
  #else
    array = (MMD_float**) malloc(n1 * sizeof(MMD_float*));
    data = (MMD_float*) malloc((n1 * n2 + 1024 + 1) * sizeof(MMD_float));
    long mask64 = 0;

    for(int j = 0, k = 1; j < 8; j++, k *= 2) {
      mask64 = mask64 | k;
    }

    while((long)data & mask64) data++;
  #endif

  n = 0;

  for(i = 0; i < n1; i++) {
    array[i] = &data[n];
    n += n2;
  }

  return array;
}

/* free memory of a 2-d MMD_float array */

void Atom_destroy_2d_MMD_float_array(Atom *atom, MMD_float** array)
{
  if(array != NULL) {
  #ifdef ALIGNMALLOC
	_mm_free(&array[0][0]);
	_mm_free(array);
  #else
      //free(array[0]);
      free(array);
  #endif
  }
}

void Atom_sort(Atom *atom, Neighbor *neighbor)
{

  Neighbor_binatoms(neighbor, atom, atom->nlocal);

  atom->binpos = neighbor->bincount;
  atom->bins = neighbor->bins;

  const int mbins = neighbor->mbins;
  const int atoms_per_bin = neighbor->atoms_per_bin;

  {
    for(int i=1; i<mbins; i++)
	  atom->binpos[i] += atom->binpos[i-1];
    if(atom->copy_size < atom->nmax) {
	  Atom_destroy_2d_MMD_float_array(atom, atom->x_copy);
	  Atom_destroy_2d_MMD_float_array(atom, atom->v_copy);
      atom->x_copy = (MMD_float**) Atom_create_2d_MMD_float_array(atom, atom->nmax, PAD);
      atom->v_copy = (MMD_float**) Atom_create_2d_MMD_float_array(atom, atom->nmax, PAD);
      atom->copy_size = atom->nmax;
    }
  }

  MMD_float* new_x = &atom->x_copy[0][0];
  MMD_float* new_v = &atom->v_copy[0][0];
  MMD_float* old_x = &atom->x[0][0];
  MMD_float* old_v = &atom->v[0][0];

  for(int mybin = 0; mybin<mbins; mybin++) {
    const int start = mybin>0 ? atom->binpos[mybin-1]:0;
    const int count = atom->binpos[mybin] - start;
    for(int k=0; k<count; k++) {
	  const int new_i = start+k;
	  const int old_i = atom->bins[mybin*atoms_per_bin+k];
	  new_x[new_i*PAD+0] = old_x[old_i*PAD+0];
	  new_x[new_i*PAD+1] = old_x[old_i*PAD+1];
	  new_x[new_i*PAD+2] = old_x[old_i*PAD+2];
	  new_v[new_i*PAD+0] = old_v[old_i*PAD+0];
	  new_v[new_i*PAD+1] = old_v[old_i*PAD+1];
	  new_v[new_i*PAD+2] = old_v[old_i*PAD+2];
    }
  }

  {
    MMD_float** x_tmp = atom->x;
    MMD_float** v_tmp = atom->v;

    atom->x = atom->x_copy;
    atom->v = atom->v_copy;
    atom->x_copy = x_tmp;
    atom->v_copy = v_tmp;
  }
}
