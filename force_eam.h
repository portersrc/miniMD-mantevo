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


#ifndef FORCEEAM_H
#define FORCEEAM_H

#include "stdio.h"
#include "atom.h"
#include "neighbor.h"
#include "threadData.h"
#include "types.h"
#include "mpi.h"
#include "comm.h"
#include "force.h"

struct Funcfl {
  char* file;
  MMD_int nrho, nr;
  double drho, dr, cut, mass;
  MMD_float* frho, *rhor, *zr;
};

typedef struct
{

    // FIXME
    // This is really dirty and needs to be fixed. Notice that the C code
    // actually assumes these members are at the top of this struct and
    // in this order (e.g. force->threads). 
    // begin copy-paste of Force struct
        MMD_float cutforce;
        MMD_float cutforcesq;
        MMD_float eng_vdwl;
        MMD_float mass;
        MMD_int evflag;
        MMD_float virial;


        int use_sse;
        int use_oldcompute;
        ThreadData* threads;
        MMD_int reneigh;
        Timer* timer;

        MMD_float epsilon, sigma6, sigma; //Parameters for LJ only

        ForceStyle style;

        MMD_int me;
    // end copy-paste of Force struct



    // public variables so USER-ATC package can access them

    MMD_float cutmax;

    // potentials as array data

    MMD_int nrho, nr;
    MMD_float* frho, *rhor, *z2r;

    // potentials in spline form used for force computation

    MMD_float dr, rdr, drho, rdrho;
    MMD_float* rhor_spline, *frho_spline, *z2r_spline;
    MMD_float* d_rhor_spline, *d_frho_spline, *d_z2r_spline;


    // per-atom arrays

    MMD_float* rho, *fp, *d_fp;

    MMD_int nmax;

    // potentials as file data

    MMD_int* map;                   // which element each atom type maps to

    struct Funcfl funcfl;

}ForceEAM;


ForceEAM *ForceEAM_alloc();
void ForceEAM_free(ForceEAM *);

void ForceEAM_compute(ForceEAM *, Atom *atom, Neighbor *neighbor, Comm *comm, int me);
void ForceEAM_coeff(ForceEAM *, char*);
void ForceEAM_setup(ForceEAM *, Atom *atom);
void ForceEAM_init_style(ForceEAM *, Atom * atom);
MMD_float ForceEAM_single(ForceEAM *, MMD_int, MMD_int, MMD_int, MMD_int, MMD_float, MMD_float, MMD_float, MMD_float *);

MMD_int ForceEAM_pack_comm(ForceEAM *, int n, int iswap, MMD_float* buf, MMD_int** asendlist);
void ForceEAM_unpack_comm(ForceEAM *, int n, int first, MMD_float* buf);
MMD_int ForceEAM_pack_reverse_comm(ForceEAM *, MMD_int, MMD_int, MMD_float*);
void ForceEAM_unpack_reverse_comm(ForceEAM *, MMD_int, MMD_int*, MMD_float*);
MMD_float ForceEAM_memory_usage(ForceEAM *);

void ForceEAM_compute_halfneigh(ForceEAM *, Atom *atom, Neighbor *neighbor, Comm *comm, int me);
void ForceEAM_compute_fullneigh(ForceEAM *, Atom *atom, Neighbor *neighbor, Comm *comm, int me);

void ForceEAM_array2spline(ForceEAM *, Atom * atom);
void ForceEAM_interpolate(ForceEAM *, MMD_int n, MMD_float delta, MMD_float* f, MMD_float* spline);
void ForceEAM_grab(ForceEAM *, FILE*, MMD_int, MMD_float*);

void ForceEAM_read_file(ForceEAM *, char*);
void ForceEAM_file2array(ForceEAM *);

void ForceEAM_bounds(ForceEAM *, char* str, int nmax, int *nlo, int *nhi);

void ForceEAM_communicate(ForceEAM *, Atom *atom, Comm *comm);

#endif

