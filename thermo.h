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

#ifndef THERMO_H
#define THERMO_H

enum units {LJ, METAL};
#include "atom.h"
#include "neighbor.h"
#include "force.h"
#include "timer.h"
#include "comm.h"
#include "threadData.h"
#include "types.h"

struct Integrate_s;

typedef struct
{
    MMD_int nstat;
    MMD_int mstat;
    MMD_int ntimes;
    MMD_int* steparr;
    MMD_float* tmparr;
    MMD_float* engarr;
    MMD_float* prsarr;


    MMD_float t_act, p_act, e_act;
    MMD_float t_scale, e_scale, p_scale, mvv2e, dof_boltz;

    ThreadData* threads;
    MMD_float rho;
}Thermo;

void Thermo_init(Thermo *);
void Thermo_destroy(Thermo *);
void Thermo_setup(Thermo *, MMD_float, struct Integrate_s *integrate, Atom *atom, MMD_int);
MMD_float Thermo_temperature(Thermo *, Atom *);
MMD_float Thermo_energy(Thermo *, Atom *, Neighbor *, Force*);
MMD_float Thermo_pressure(Thermo *, MMD_float, Force*);
void Thermo_compute(Thermo *, MMD_int, Atom *, Neighbor *, Force*, Timer *, Comm *);

#endif
