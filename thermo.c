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
#include "stdlib.h"
#include "mpi.h"
#include "force_lj.h"
#include "integrate.h"
#include "thermo.h"

void Thermo_init(Thermo *t)
{
}
//void Thermo_destroy(Thermo *thermo)
//{
//}

void Thermo_setup(Thermo *thermo, MMD_float rho_in, Integrate *integrate, Atom *atom, int units)
{
  thermo->rho = rho_in;
  thermo->ntimes = integrate->ntimes;

  MMD_int maxstat;

  if(thermo->nstat == 0) maxstat = 2;
  else maxstat = thermo->ntimes / thermo->nstat + 2;

  thermo->steparr = (MMD_int*) malloc(maxstat * sizeof(MMD_int));
  thermo->tmparr = (MMD_float*) malloc(maxstat * sizeof(MMD_float));
  thermo->engarr = (MMD_float*) malloc(maxstat * sizeof(MMD_float));
  thermo->prsarr = (MMD_float*) malloc(maxstat * sizeof(MMD_float));

  if(units == LJ) {
    thermo->mvv2e = 1.0;
    thermo->dof_boltz = (atom->natoms * 3 - 3);
    thermo->t_scale = thermo->mvv2e / thermo->dof_boltz;
    thermo->p_scale = 1.0 / 3 / atom->box.xprd / atom->box.yprd / atom->box.zprd;
    thermo->e_scale = 0.5;
  } else if(units == METAL) {
    thermo->mvv2e = 1.036427e-04;
    thermo->dof_boltz = (atom->natoms * 3 - 3) * 8.617343e-05;
    thermo->t_scale = thermo->mvv2e / thermo->dof_boltz;
    thermo->p_scale = 1.602176e+06 / 3 / atom->box.xprd / atom->box.yprd / atom->box.zprd;
    thermo->e_scale = 524287.985533;//16.0;
    integrate->dtforce /= thermo->mvv2e;

  }
}

void Thermo_compute(Thermo *thermo, MMD_int iflag, Atom *atom, Neighbor *neighbor, Force* force, Timer *timer, Comm *comm)
{
  MMD_float t, eng, p;

  if(iflag > 0 && iflag % thermo->nstat) return;

  if(iflag == -1 && thermo->nstat > 0 && thermo->ntimes % thermo->nstat == 0) return;

  thermo->t_act = 0;
  thermo->e_act = 0;
  thermo->p_act = 0;
  
  t = Thermo_temperature(thermo, atom);
  
  {
    eng = Thermo_energy(thermo, atom, neighbor, force);

    p = Thermo_pressure(thermo, t, force);

    MMD_int istep = iflag;

    if(iflag == -1) istep = thermo->ntimes;

    if(iflag == 0) thermo->mstat = 0;

    thermo->steparr[thermo->mstat] = istep;
    thermo->tmparr[thermo->mstat] = t;
    thermo->engarr[thermo->mstat] = eng;
    thermo->prsarr[thermo->mstat] = p;

    thermo->mstat++;

    double oldtime = timer->array[TIME_TOTAL];
    Timer_barrier_stop(timer, TIME_TOTAL);

    if(thermo->threads->mpi_me == 0) {
      fprintf(stdout, "%i %e %e %e %6.3lf\n", istep, t, eng, p, istep == 0 ? 0.0 : timer->array[TIME_TOTAL]);
    }

    timer->array[TIME_TOTAL] = oldtime;
  }
}

/* reduced potential energy */

MMD_float Thermo_energy(Thermo *thermo, Atom *atom, Neighbor *neighbor, Force* force)
{
  thermo->e_act = force->eng_vdwl;

  if(neighbor->halfneigh) {
    thermo->e_act *= 2.0;
  }

  thermo->e_act *= thermo->e_scale;
  MMD_float eng;

  if(sizeof(MMD_float) == 4)
    MPI_Allreduce(&thermo->e_act, &eng, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  else
    MPI_Allreduce(&thermo->e_act, &eng, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return eng / atom->natoms;
}

/*  reduced temperature */

MMD_float Thermo_temperature(Thermo *thermo, Atom *atom)
{
  MMD_int i;
  MMD_float vx, vy, vz;

  MMD_float t = 0.0;
  thermo->t_act = 0;
  

  MMD_float* v = &atom->v[0][0];

  
  for(i = 0; i < atom->nlocal; i++) {
    vx = v[i * PAD + 0];
    vy = v[i * PAD + 1];
    vz = v[i * PAD + 2];
    t += (vx * vx + vy * vy + vz * vz) * atom->mass;
  }

  
  thermo->t_act += t;

  

  MMD_float t1;
  
  {
    if(sizeof(MMD_float) == 4)
      MPI_Allreduce(&thermo->t_act, &t1, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    else
      MPI_Allreduce(&thermo->t_act, &t1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  }
  return t1 * thermo->t_scale;
}


/* reduced pressure from virial
   virial = Fi dot Ri summed over own and ghost atoms, since PBC info is
   stored correctly in force array before reverse_communicate is performed */

MMD_float Thermo_pressure(Thermo *thermo, MMD_float t, Force* force)
{
  thermo->p_act = force->virial;

  MMD_float virial = 0;

  if(sizeof(MMD_float) == 4)
    MPI_Allreduce(&thermo->p_act, &virial, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  else
    MPI_Allreduce(&thermo->p_act, &virial, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  //printf("Pres: %e %e %e %e\n",t,dof_boltz,virial,p_scale);
  return (t * thermo->dof_boltz + virial) * thermo->p_scale;
}


