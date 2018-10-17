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
#include "math.h"
#include "force_lj.h"
#include "openmp.h"

#ifndef VECTORLENGTH
#define VECTORLENGTH 4
#endif

ForceLJ *ForceLJ_alloc()
{
  ForceLJ *forceLJ = (ForceLJ *) malloc(sizeof(ForceLJ));
  forceLJ->cutforce = 0.0;
  forceLJ->cutforcesq = 0.0;
  forceLJ->use_oldcompute = 0;
  forceLJ->reneigh = 1;
  forceLJ->style = FORCELJ;

  forceLJ->epsilon = 1.0;
  forceLJ->sigma6 = 1.0;
  forceLJ->sigma = 1.0;
  return forceLJ;
}
void ForceLJ_free(ForceLJ *f)
{
    free(f);
}

void ForceLJ_setup(ForceLJ *force_lj, Atom *atom)
{
  force_lj->cutforcesq = force_lj->cutforce *force_lj->cutforce;
}


void ForceLJ_compute(ForceLJ *force_lj, Atom *atom, Neighbor *neighbor, Comm *comm, int me)
{
  force_lj->eng_vdwl = 0;
  force_lj->virial = 0;

  if(force_lj->evflag) {
    if(force_lj->use_oldcompute) {
      ForceLJ_compute_original(force_lj, atom, neighbor, me, 1);
      return;
    }

    if(neighbor->halfneigh) {
      if(neighbor->ghost_newton) {
        if(force_lj->threads->omp_num_threads > 1) {
          ForceLJ_compute_halfneigh_threaded(force_lj, atom, neighbor, me, 1, 1);
          return;
        }else{
          ForceLJ_compute_halfneigh(force_lj, atom, neighbor, me, 1, 1);
          return;
        }
      } else {
        if(force_lj->threads->omp_num_threads > 1) {
          ForceLJ_compute_halfneigh_threaded(force_lj, atom, neighbor, me, 1, 0);
          return;
        }else{
          ForceLJ_compute_halfneigh(force_lj, atom, neighbor, me, 1, 0);
          return;
        }
      }
    } else {
      ForceLJ_compute_fullneigh(force_lj, atom, neighbor, me, 1);
      return;
    }
  } else {
    if(force_lj->use_oldcompute) {
      ForceLJ_compute_original(force_lj, atom, neighbor, me, 0);
      return;
    }

    if(neighbor->halfneigh) {
      if(neighbor->ghost_newton) {
        if(force_lj->threads->omp_num_threads > 1) {
          ForceLJ_compute_halfneigh_threaded(force_lj, atom, neighbor, me, 0, 1);
          return;
        } else {
          ForceLJ_compute_halfneigh(force_lj, atom, neighbor, me, 0, 1);
          return;
        }
      } else {
        if(force_lj->threads->omp_num_threads > 1) {
          ForceLJ_compute_halfneigh_threaded(force_lj, atom, neighbor, me, 0, 0);
          return;
        } else {
          ForceLJ_compute_halfneigh(force_lj, atom, neighbor, me, 0, 0);
          return;
        }
      }
    } else {
      ForceLJ_compute_fullneigh(force_lj, atom, neighbor, me, 0);
      return;
    }

  }
}

//original version of force compute in miniMD
//  -MPI only
//  -not vectorizable
//template<int EVFLAG>
void ForceLJ_compute_original(ForceLJ *force_lj, Atom *atom, Neighbor *neighbor, int me, int EVFLAG)
{
  int i, j, k, nlocal, nall, numneigh;
  MMD_float xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  MMD_float sr2, sr6, force;
  int* neighs;
  MMD_float** x, **f;

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  x = atom->x;
  f = atom->f;

  force_lj->eng_vdwl = 0;
  force_lj->virial = 0;
  // clear force on own and ghost atoms

  for(i = 0; i < nall; i++) {
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    f[i][2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  for(i = 0; i < nlocal; i++) {
    neighs = &neighbor->neighbors[i * neighbor->maxneighs];
    numneigh = neighbor->numneigh[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    for(k = 0; k < numneigh; k++) {
      j = neighs[k];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < force_lj->cutforcesq) {
        sr2 = 1.0 / rsq;
        sr6 = sr2 * sr2 * sr2 * force_lj->sigma6;
        force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * force_lj->epsilon;
        f[i][0] += delx * force;
        f[i][1] += dely * force;
        f[i][2] += delz * force;
        f[j][0] -= delx * force;
        f[j][1] -= dely * force;
        f[j][2] -= delz * force;

        if(EVFLAG) {
          force_lj->eng_vdwl += (4.0 * sr6 * (sr6 - 1.0)) * force_lj->epsilon;
          force_lj->virial += (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }
  }
}


//optimised version of compute
//  -MPI only
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//     -getting rid of 2d pointers
//     -use pragma simd to force vectorization of inner loop
//template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ_compute_halfneigh(ForceLJ *force_lj, Atom *atom, Neighbor *neighbor, int me, int EVFLAG, int GHOST_NEWTON)
{
  int* neighs;
  int tid = omp_get_thread_num();

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  MMD_float* x = &atom->x[0][0];
  MMD_float* f = &atom->f[0][0];

  // clear force on own and ghost atoms
  for(int i = 0; i < nall; i++) {
    f[i * PAD + 0] = 0.0;
    f[i * PAD + 1] = 0.0;
    f[i * PAD + 2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j
  MMD_float t_energy = 0;
  MMD_float t_virial = 0;

  for(int i = 0; i < nlocal; i++) {
    neighs = &neighbor->neighbors[i * neighbor->maxneighs];
    const int numneighs = neighbor->numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];

    MMD_float fix = 0.0;
    MMD_float fiy = 0.0;
    MMD_float fiz = 0.0;

#ifdef USE_SIMD
    #pragma simd reduction (+: fix,fiy,fiz)
#endif
    for(int k = 0; k < numneighs; k++) {
      const int j = neighs[k];
      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < force_lj->cutforcesq) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * force_lj->sigma6;
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * force_lj->epsilon;

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(GHOST_NEWTON || j < nlocal) {
          f[j * PAD + 0] -= delx * force;
          f[j * PAD + 1] -= dely * force;
          f[j * PAD + 2] -= delz * force;
        }

        if(EVFLAG) {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
          t_energy += scale * (4.0 * sr6 * (sr6 - 1.0)) * force_lj->epsilon;
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }

      }
    }

    f[i * PAD + 0] += fix;
    f[i * PAD + 1] += fiy;
    f[i * PAD + 2] += fiz;

  }

  force_lj->eng_vdwl += t_energy;
  force_lj->virial += t_virial;

}

//optimised version of compute
//  -MPI + OpenMP (atomics for fj update)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -getting rid of 2d pointers
//    -use pragma simd to force vectorization of inner loop (not currently supported due to OpenMP atomics
//template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ_compute_halfneigh_threaded(ForceLJ *force_lj, Atom *atom, Neighbor *neighbor, int me, int EVFLAG, int GHOST_NEWTON)
{
  int nlocal, nall;
  int* neighs;
  MMD_float* x, *f;
  int tid = omp_get_thread_num();

  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial = 0;

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  x = &atom->x[0][0];
  f = &atom->f[0][0];

  
  // clear force on own and ghost atoms

  
  for(int i = 0; i < nall; i++) {
    f[i * PAD + 0] = 0.0;
    f[i * PAD + 1] = 0.0;
    f[i * PAD + 2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  
  for(int i = 0; i < nlocal; i++) {
    neighs = &neighbor->neighbors[i * neighbor->maxneighs];
    const int numneighs = neighbor->numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    MMD_float fix = 0.0;
    MMD_float fiy = 0.0;
    MMD_float fiz = 0.0;

    for(int k = 0; k < numneighs; k++) {
      const int j = neighs[k];
      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < force_lj->cutforcesq) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * force_lj->sigma6;
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * force_lj->epsilon;

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(GHOST_NEWTON || j < nlocal) {
          
          f[j * PAD + 0] -= delx * force;
          
          f[j * PAD + 1] -= dely * force;
          
          f[j * PAD + 2] -= delz * force;
        }

        if(EVFLAG) {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
          t_eng_vdwl += scale * (4.0 * sr6 * (sr6 - 1.0)) * force_lj->epsilon;
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }

    
    f[i * PAD + 0] += fix;
    
    f[i * PAD + 1] += fiy;
    
    f[i * PAD + 2] += fiz;
  }

  
  force_lj->eng_vdwl += t_eng_vdwl;
  
  force_lj->virial += t_virial;

  
}

//optimised version of compute
//  -MPI + OpenMP (using full neighborlists)
//  -gets rid of fj update (read/write to memory)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -get rid of 2d pointers
//    -use pragma simd to force vectorization of inner loop
//template<int EVFLAG>
void ForceLJ_compute_fullneigh(ForceLJ *force_lj, Atom *atom, Neighbor *neighbor, int me, int EVFLAG)
{
  //int tid = omp_get_thread_num();

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  const MMD_float* const restrict x = atom->d_x; //&atom.x[0][0];
  //MMD_float* const restrict f = &atom.f[0][0];
  MMD_float* const restrict f = atom->d_f;
  const int* const restrict neighbors = neighbor->d_neighbors;
  const int* const restrict numneigh = neighbor->d_numneigh;
  const int maxneighs = neighbor->maxneighs;
  const MMD_float sigma6_ = force_lj->sigma6;
  const MMD_float epsilon_ = force_lj->epsilon;
  const MMD_float cutforcesq_ = force_lj->cutforcesq;
  const int nmax = neighbor->nmax;

  // clear force on own and ghost atoms

  
#pragma acc data deviceptr(x,neighbors,numneigh,f) //copyout(f[0:nall*3]) 
{
  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial = 0;
  #pragma acc kernels
  for(int i = 0; i < nlocal; i++) {
    f[i * PAD + 0] = 0.0;
    f[i * PAD + 1] = 0.0;
    f[i * PAD + 2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on atom i

  
  #pragma acc kernels
  for(int i = 0; i < nlocal; i++) {
    const int* const neighs = &neighbors[i * DS0(nmax,maxneighs)];
    const int numneighs = numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    MMD_float fix = 0;
    MMD_float fiy = 0;
    MMD_float fiz = 0;

    //pragma simd forces vectorization (ignoring the performance objections of the compiler)
    //also give hint to use certain vectorlength for MIC, Sandy Bridge and WESTMERE this should be be 8 here
    //give hint to compiler that fix, fiy and fiz are used for reduction only

    for(int k = 0; k < numneighs; k++) {
      const int j = neighs[k*DS1(nmax,maxneighs)];
      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;
      if(rsq < cutforcesq_) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6_;
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon_;
        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;
        #ifdef ENABLE_EV_CALCULATION //crashes with PGI 13.9
        if(EVFLAG) {
          t_eng_vdwl += sr6 * (sr6 - 1.0) * epsilon;
          t_virial += (delx * delx + dely * dely + delz * delz) * force;
        }
        #endif //ENABLE_EV_CALCULATION
      }
    }

    f[i * PAD + 0] += fix;
    f[i * PAD + 1] += fiy;
    f[i * PAD + 2] += fiz;

  }
  t_eng_vdwl *= 4.0;
  t_virial *= 0.5;
  force_lj->eng_vdwl += t_eng_vdwl;
  force_lj->virial += t_virial;
}

  
  
  
}


