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
//#define PRINTDEBUG(a) a
#define PRINTDEBUG(a)
#include <assert.h>
#include "stdio.h"
#include "integrate.h"
#include "openmp.h"
#include "math.h"
#include "comm.h"
#include "force.h"

void Integrate_init(Integrate *ig)
{
    ig->sort_every = 20;
}
//void Integrate_destroy(Integrate *ig)
//{
//}

void Integrate_setup(Integrate *ig)
{
    ig->dtforce = 0.5 * ig->dt;
}

void Integrate_initialIntegrate(Integrate *ig)
{
  const int nlocal_ = ig->nlocal;
  const MMD_float dt_ = ig->dt;
  const MMD_float dtforce_ = ig->dtforce;
  const MMD_float* const restrict f_ = ig->f;
  MMD_float* const restrict v_ = ig->v;
  MMD_float* const restrict x_ = ig->x; 
  
  #pragma acc kernels deviceptr(x_,v_,f_)
  for(MMD_int i = 0; i < nlocal_; i++) {
    v_[i * PAD + 0] += dtforce_ * f_[i * PAD + 0];
    v_[i * PAD + 1] += dtforce_ * f_[i * PAD + 1];
    v_[i * PAD + 2] += dtforce_ * f_[i * PAD + 2];
    x_[i * PAD + 0] += dt_ * v_[i * PAD + 0];
    x_[i * PAD + 1] += dt_ * v_[i * PAD + 1];
    x_[i * PAD + 2] += dt_ * v_[i * PAD + 2];
  }
}

void Integrate_finalIntegrate(Integrate *ig)
{
  const int nlocal_ = ig->nlocal;
  const MMD_float dtforce_ = ig->dtforce;
  const MMD_float* const restrict f_ = ig->f;
  MMD_float* const restrict v_ = ig->v;
  #pragma acc kernels deviceptr(v_,f_)
  for(MMD_int i = 0; i < nlocal_; i++) {
    v_[i * PAD + 0] += dtforce_ * f_[i * PAD + 0];
    v_[i * PAD + 1] += dtforce_ * f_[i * PAD + 1];
    v_[i * PAD + 2] += dtforce_ * f_[i * PAD + 2];
  }
}

void Integrate_run(Integrate *ig, Atom *atom, Force *force, Neighbor *neighbor,
                    Comm *comm, Thermo *thermo, Timer *timer)
{
  int i, n;

  comm->timer = timer;
  timer->array[TIME_TEST] = 0.0;

  int check_safeexchange = comm->check_safeexchange;

  ig->mass = atom->mass;
  ig->dtforce = ig->dtforce / ig->mass;
  //Use OpenMP threads only within the following loop containing the main loop.
  //Do not use OpenMP for setup and postprocessing.
  {
    int next_sort = ig->sort_every > 0 ? ig->sort_every : ig->ntimes+1;

    Atom_sync_device(atom, atom->d_x, &atom->x[0][0], atom->nmax*3*sizeof(MMD_float));
    Atom_sync_device(atom, atom->d_v, &atom->v[0][0], atom->nmax*3*sizeof(MMD_float));

    for(n = 0; n < ig->ntimes; n++) {

      //x = &atom.x[0][0];
      //v = &atom.v[0][0];
      //f = &atom.f[0][0];
      ig->x = atom->d_x;
      ig->v = atom->d_v;
      ig->f = atom->d_f;
      ig->xold = &atom->xold[0][0];
      ig->nlocal = atom->nlocal;
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("A %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);
      Integrate_initialIntegrate(ig);
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("B %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);

      
      Timer_stamp(timer);

      if((n + 1) % neighbor->every) {
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));

        Comm_communicate(comm, atom);
        //atom.sync_device(atom.d_x,&atom.x[0][0],atom.nmax*3*sizeof(MMD_float));
        
        Timer_stamp_int(timer, TIME_COMM);

      } else {
        Atom_sync_host(atom, (void *) &atom->x[0][0], (void *)atom->d_x, atom->nmax*3*sizeof(MMD_float));
        Atom_sync_host(atom, (void *) &atom->v[0][0], (void *)atom->d_v, atom->nmax*3*sizeof(MMD_float));
        {

          Timer_stamp_extra_start(timer);
          Comm_exchange(comm, atom);
          if(n+1>=next_sort) {
            //atom.sort(neighbor);
            next_sort +=  ig->sort_every;
          }
          Comm_borders(comm, atom);
          
          {
            Timer_stamp_extra_stop(timer, TIME_TEST);
            Timer_stamp_int(timer, TIME_COMM);
          }

        }

        Neighbor_build(neighbor, atom);

        //atom.sync_device(atom.d_x,&atom.x[0][0],atom.nmax*3*sizeof(MMD_float));
        Atom_sync_device(atom, atom->d_v, &atom->v[0][0], atom->nmax*3*sizeof(MMD_float));
        Timer_stamp_int(timer, TIME_NEIGH);
      }
      Timer_stamp_int(timer, TIME_TEST);
      force->evflag = (n + 1) % thermo->nstat == 0;
      if(force->style == FORCELJ) {
        ForceLJ_compute(force, atom, neighbor, comm, comm->me);
      } else if(force->style == FORCELJ) {
        ForceEAM_compute(force, atom, neighbor, comm, comm->me);
      } else{
        assert(0);
      }
//        atom.sync_host(&atom.f[0][0],atom.d_f,atom.nmax*3*sizeof(MMD_float));

      
      Timer_stamp_int(timer, TIME_FORCE);

      if(neighbor->halfneigh && neighbor->ghost_newton) {
        Atom_sync_host(atom, &atom->f[0][0], atom->d_f, atom->nmax*3*sizeof(MMD_float));
        Comm_reverse_communicate(comm, atom);

        
        Timer_stamp_int(timer, TIME_COMM);
      }

      //v = &atom.v[0][0];
      //f = &atom.f[0][0];
      ig->v = atom->d_v;
      ig->f = atom->d_f;
      ig->nlocal = atom->nlocal;

      
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("G %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);
      Integrate_finalIntegrate(ig);
        //atom.sync_host(&atom.x[0][0],atom.d_x,atom.nmax*3*sizeof(MMD_float));
//for(int i = 0; i<nlocal;i++) printf("H %i %lf %lf %lf\n",i,atom.x[i][0],atom.x[i][1],atom.x[i][2]);
      if(thermo->nstat) {
        Atom_sync_host(atom, &atom->v[0][0], atom->d_v, atom->nmax*3*sizeof(MMD_float));
        Thermo_compute(thermo, n + 1, atom, neighbor, force, timer, comm);
      }
    }
  } //end OpenMP parallel
        Atom_sync_host(atom, &atom->v[0][0], atom->d_v, atom->nmax*3*sizeof(MMD_float));
        Atom_sync_host(atom, &atom->x[0][0], atom->d_x, atom->nmax*3*sizeof(MMD_float));
}
