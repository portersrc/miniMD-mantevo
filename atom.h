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

#ifndef ATOM_H
#define ATOM_H

#include "threadData.h"
#include "types.h"



static inline
int DL(const int idx0, const int idx1, const int n0, const int n1) {
#ifdef USELAYOUTLEFT
  return idx0+idx1*n0;
#else
  return idx0*n1+idx1;
#endif
}
/*
template<typename Scalar,class traits>
struct DualView{
  Scalar *ptr;
  int size;
  int mod_h,mod_d;
  DualView(int size_) {
    size = size_;
    mod_h = 0;
    mod_d = 0;
    ptr = (Scalar*) malloc(size*sizeof(Scalar));
    const Scalar* const restrict ptr_ = ptr;
    #pragma acc enter data create(ptr_[0:size])
  }
  ~DualView() {
    const int size_ = size;
    const Scalar* const restrict ptr- = ptr;
    #pragma acc exit data delete(ptr_[0:size_])
    
    free(ptr);
  } 
  void sync_host(int n) {
    if(mod_d<=mod_h) return;
    const Scalar* const restrict ptr_ = ptr;
    const int size_ = n;
    #pragma acc update self(ptr_[0:size_])
    mod_d = mod_h = 0;
  }
  void sync_device(int n) {
    if(mod_d>=mod_h) return;
    const Scalar* const restrict ptr_ = ptr;
    const int size_ = n;
    #pragma acc update device(ptr_[0:size_])
    mod_d = mod_h = 0;
  }

  void modify_host() {
    assert(mod_d==0)
    mod_h = mod_d>mod_h?mod_d+1:mod_h;
  }
  void modify_device() {
    assert(mod_h==0)
    mod_d = mod_h>mod_d?mod_h+1:mod_d;
  }
  Scalar* ptr_on_device() {
    return acc_deviceptr(ptr);
  }
  inline Scalar operator[] (int i) const { return ptr[i];}
}

template<typename Scalar>
struct DualView<Scalar,RandomRead> {
  texture_object tex;
  Scalar operator[] (int i) {
    return texture_fetch(tex,i);
  }
  
}*/
#ifdef USELAYOUTLEFT
#define DS0(a,b) 1
#define DS1(a,b) a
#else
#define DS0(a,b) b
#define DS1(a,b) 1
#endif

/*static inline int DS1(const int &n0, const int &n1) {
#ifdef USELAYOUTLEFT
  return n1;
#else
  return 1;
#endif
}


static inline int DS0(const int &n0, const int &n1) {
#ifdef USELAYOUTLEFT
  return 1;
#else
  return n0;
#endif 
}*/

struct Neighbor_s;
struct Box {
  MMD_float xprd, yprd, zprd;
  MMD_float xlo, xhi;
  MMD_float ylo, yhi;
  MMD_float zlo, zhi;
};

typedef struct
{
    int natoms;
    int nlocal, nghost;
    int nmax;

    MMD_float** x;
    MMD_float** v;
    MMD_float** f;

    MMD_float* d_x,*d_v,*d_f;
    MMD_float** xold;

    ThreadData* threads;
    MMD_float virial, mass;

    int comm_size, reverse_size, border_size;

    struct Box box;

    int* binpos;
    int* bins;
    MMD_float** x_copy;
    MMD_float** v_copy;
    int copy_size;
}Atom;

void Atom_init(Atom *);
void Atom_destroy(Atom *);
void Atom_addatom(Atom *, MMD_float, MMD_float, MMD_float, MMD_float, MMD_float, MMD_float);
void Atom_pbc(Atom *);
void Atom_growarray(Atom *);

void Atom_copy(Atom *, int, int);

void Atom_pack_comm(Atom *, int, int*, MMD_float*, int*);
void Atom_unpack_comm(Atom *, int, int, MMD_float*);
void Atom_pack_reverse(Atom *, int, int, MMD_float*);
void Atom_unpack_reverse(Atom *, int, int*, MMD_float*);

int Atom_pack_border(Atom *, int, MMD_float*, int*);
int Atom_unpack_border(Atom *, int, MMD_float*);
int Atom_pack_exchange(Atom *, int, MMD_float*);
int Atom_unpack_exchange(Atom *, int, MMD_float*);
int Atom_skip_exchange(Atom *, MMD_float*);

MMD_float** Atom_realloc_2d_MMD_float_array(Atom *, MMD_float**, int, int, int);
MMD_float** Atom_create_2d_MMD_float_array(Atom *, int, int);
void Atom_destroy_2d_MMD_float_array(Atom *, MMD_float**);

void Atom_sort(Atom *atom, struct Neighbor_s *neighbor);

void Atom_sync_device(Atom *, void* d_ptr, void* h_ptr,int bytes);
void Atom_sync_host(Atom *, void* h_ptr, void* d_ptr,int bytes);

#endif
