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

#include "stdlib.h"
#include "mpi.h"
#include "timer.h"

void Timer_init(Timer *t)
{
  t->array = (double*) malloc(TIME_N * sizeof(double));

  for(int i = 0; i < TIME_N; i++) t->array[i] = 0.0;
}

void Timer_destroy(Timer *t)
{
  if(t->array) free(t->array);
}

void Timer_stamp(Timer *timer)
{
#ifdef PREC_TIMER
  clock_gettime(CLOCK_REALTIME, &timer->previous_time);
#else
  timer->previous_time_d = MPI_Wtime();
#endif
}

void Timer_stamp_int(Timer *timer, int which)
{
#ifdef PREC_TIMER
  timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  timer->array[which] += (current_time.tv_sec - timer->previous_time.tv_sec + 1.0 *
                   (current_time.tv_nsec - timer->previous_time.tv_nsec) / 1000000000);
  timer->previous_time = current_time;
#else
  double current_time = MPI_Wtime();
  timer->array[which] += current_time - timer->previous_time_d;
  timer->previous_time_d = current_time;
#endif
}

void Timer_stamp_extra_start(Timer *timer)
{
#ifdef PREC_TIMER
  clock_gettime(CLOCK_REALTIME, &timer->previous_time_extra);
#else
  timer->previous_time_extra_d = MPI_Wtime();
#endif
}

void Timer_stamp_extra_stop(Timer *timer, int which)
{
#ifdef PREC_TIMER
  timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  timer->array[which] += (current_time.tv_sec - timer->previous_time_extra.tv_sec + 1.0 *
                   (current_time.tv_nsec - timer->previous_time_extra.tv_nsec) / 1000000000);
  timer->previous_time_extra = current_time;
#else
  double current_time = MPI_Wtime();
  timer->array[which] += current_time - timer->previous_time_extra_d;
  timer->previous_time_extra_d = current_time;
#endif
}

void Timer_barrier_start(Timer *timer, int which)
{
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef PREC_TIMER
  timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  timer->array[which] = current_time.tv_sec + 1.0e-9 * current_time.tv_nsec;
#else
  timer->array[which] = MPI_Wtime();
#endif
}

void Timer_barrier_stop(Timer *timer, int which)
{
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef PREC_TIMER
  timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  timer->array[which] = current_time.tv_sec + 1.0e-9 * current_time.tv_nsec - timer->array[which];
#else
  double current_time = MPI_Wtime();
  timer->array[which] = current_time - timer->array[which];
#endif
}
