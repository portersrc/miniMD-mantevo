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
#include "comm.h"
#include "openmp.h"
#include "openacc.h"

#define BUFFACTOR 1.5
#define BUFMIN 1000
#define BUFEXTRA 100
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void Comm_init(Comm *c)
{
  c->maxsend = BUFMIN;
  c->buf_send = (MMD_float*) malloc((c->maxsend + BUFMIN) * sizeof(MMD_float));
  c->d_buf_send = (MMD_float*) acc_malloc((c->maxsend + BUFMIN) * sizeof(MMD_float));
  c->maxrecv = BUFMIN;
  c->buf_recv = (MMD_float*) malloc(c->maxrecv * sizeof(MMD_float));
  c->d_buf_recv = (MMD_float*) acc_malloc(c->maxrecv * sizeof(MMD_float));
  c->check_safeexchange = 0;
  c->do_safeexchange = 0;
  c->maxthreads = 0;
  c->maxnlocal = 0;
}

//void Comm_destroy(Comm *c)
//{
//}

/* setup spatial-decomposition communication patterns */

int Comm_setup(Comm *comm, MMD_float cutneigh, Atom *atom)
{
  int i;
  int nprocs;
  int periods[3];
  MMD_float prd[3];
  int myloc[3];
  MPI_Comm cartesian;
  MMD_float lo, hi;
  int ineed, idim, nbox;

  prd[0] = atom->box.xprd;
  prd[1] = atom->box.yprd;
  prd[2] = atom->box.zprd;

  /* setup 3-d grid of procs */

  MPI_Comm_rank(MPI_COMM_WORLD, &comm->me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MMD_float area[3];

  area[0] = prd[0] * prd[1];
  area[1] = prd[0] * prd[2];
  area[2] = prd[1] * prd[2];

  MMD_float bestsurf = 2.0 * (area[0] + area[1] + area[2]);

  // loop thru all possible factorizations of nprocs
  // surf = surface area of a proc sub-domain
  // for 2d, insure ipz = 1

  int ipx, ipy, ipz, nremain;
  MMD_float surf;

  ipx = 1;

  while(ipx <= nprocs) {
    if(nprocs % ipx == 0) {
      nremain = nprocs / ipx;
      ipy = 1;

      while(ipy <= nremain) {
        if(nremain % ipy == 0) {
          ipz = nremain / ipy;
          surf = area[0] / ipx / ipy + area[1] / ipx / ipz + area[2] / ipy / ipz;

          if(surf < bestsurf) {
            bestsurf = surf;
            comm->procgrid[0] = ipx;
            comm->procgrid[1] = ipy;
            comm->procgrid[2] = ipz;
          }
        }

        ipy++;
      }
    }

    ipx++;
  }

  if(comm->procgrid[0]*comm->procgrid[1]*comm->procgrid[2] != nprocs) {
    if(comm->me == 0) printf("ERROR: Bad grid of processors\n");

    return 1;
  }

  /* determine where I am and my neighboring procs in 3d grid of procs */

  int reorder = 0;
  periods[0] = periods[1] = periods[2] = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 3, comm->procgrid, periods, reorder, &cartesian);
  MPI_Cart_get(cartesian, 3, comm->procgrid, periods, myloc);
  MPI_Cart_shift(cartesian, 0, 1, &comm->procneigh[0][0], &comm->procneigh[0][1]);
  MPI_Cart_shift(cartesian, 1, 1, &comm->procneigh[1][0], &comm->procneigh[1][1]);
  MPI_Cart_shift(cartesian, 2, 1, &comm->procneigh[2][0], &comm->procneigh[2][1]);

  /* lo/hi = my local box bounds */

  atom->box.xlo = myloc[0] * prd[0] / comm->procgrid[0];
  atom->box.xhi = (myloc[0] + 1) * prd[0] / comm->procgrid[0];
  atom->box.ylo = myloc[1] * prd[1] / comm->procgrid[1];
  atom->box.yhi = (myloc[1] + 1) * prd[1] / comm->procgrid[1];
  atom->box.zlo = myloc[2] * prd[2] / comm->procgrid[2];
  atom->box.zhi = (myloc[2] + 1) * prd[2] / comm->procgrid[2];

  /* need = # of boxes I need atoms from in each dimension */

  comm->need[0] = (int)(cutneigh * comm->procgrid[0] / prd[0] + 1);
  comm->need[1] = (int)(cutneigh * comm->procgrid[1] / prd[1] + 1);
  comm->need[2] = (int)(cutneigh * comm->procgrid[2] / prd[2] + 1);

  /* alloc comm memory */

  int maxswap = 2 * (comm->need[0] + comm->need[1] + comm->need[2]);

  comm->slablo = (MMD_float*) malloc(maxswap * sizeof(MMD_float));
  comm->slabhi = (MMD_float*) malloc(maxswap * sizeof(MMD_float));
  comm->pbc_any = (int*) malloc(maxswap * sizeof(int));
  comm->pbc_flagx = (int*) malloc(maxswap * sizeof(int));
  comm->pbc_flagy = (int*) malloc(maxswap * sizeof(int));
  comm->pbc_flagz = (int*) malloc(maxswap * sizeof(int));
  comm->sendproc = (int*) malloc(maxswap * sizeof(int));
  comm->recvproc = (int*) malloc(maxswap * sizeof(int));
  comm->sendproc_exc = (int*) malloc(maxswap * sizeof(int));
  comm->recvproc_exc = (int*) malloc(maxswap * sizeof(int));
  comm->sendnum = (int*) malloc(maxswap * sizeof(int));
  comm->recvnum = (int*) malloc(maxswap * sizeof(int));
  comm->comm_send_size = (int*) malloc(maxswap * sizeof(int));
  comm->comm_recv_size = (int*) malloc(maxswap * sizeof(int));
  comm->reverse_send_size = (int*) malloc(maxswap * sizeof(int));
  comm->reverse_recv_size = (int*) malloc(maxswap * sizeof(int));
  int iswap = 0;

  for(int idim = 0; idim < 3; idim++)
    for(int i = 1; i <= comm->need[idim]; i++, iswap += 2) {
      MPI_Cart_shift(cartesian, idim, i, &comm->sendproc_exc[iswap], &comm->sendproc_exc[iswap + 1]);
      MPI_Cart_shift(cartesian, idim, i, &comm->recvproc_exc[iswap + 1], &comm->recvproc_exc[iswap]);
    }

  MPI_Comm_free(&cartesian);

  comm->firstrecv = (int*) malloc(maxswap * sizeof(int));
  comm->maxsendlist = (int*) malloc(maxswap * sizeof(int));

  for(i = 0; i < maxswap; i++) comm->maxsendlist[i] = BUFMIN;

  comm->sendlist = (int**) malloc(maxswap * sizeof(int*));
  
  for(i = 0; i < maxswap; i++)
    comm->sendlist[i] = (int*) malloc(BUFMIN * sizeof(int));

  /* setup 4 parameters for each exchange: (spart,rpart,slablo,slabhi)
     sendproc(nswap) = proc to send to at each swap
     recvproc(nswap) = proc to recv from at each swap
     slablo/slabhi(nswap) = slab boundaries (in correct dimension) of atoms
                            to send at each swap
     1st part of if statement is sending to the west/south/down
     2nd part of if statement is sending to the east/north/up
     nbox = atoms I send originated in this box */

  /* set commflag if atoms are being exchanged across a box boundary
     commflag(idim,nswap) =  0 -> not across a boundary
                          =  1 -> add box-length to position when sending
                          = -1 -> subtract box-length from pos when sending */

  comm->nswap = 0;

  for(idim = 0; idim < 3; idim++) {
    for(ineed = 0; ineed < 2 * comm->need[idim]; ineed++) {
      comm->pbc_any[comm->nswap] = 0;
      comm->pbc_flagx[comm->nswap] = 0;
      comm->pbc_flagy[comm->nswap] = 0;
      comm->pbc_flagz[comm->nswap] = 0;

      if(ineed % 2 == 0) {
        comm->sendproc[comm->nswap] = comm->procneigh[idim][0];
        comm->recvproc[comm->nswap] = comm->procneigh[idim][1];
        nbox = myloc[idim] + ineed / 2;
        lo = nbox * prd[idim] / comm->procgrid[idim];

        if(idim == 0) hi = atom->box.xlo + cutneigh;

        if(idim == 1) hi = atom->box.ylo + cutneigh;

        if(idim == 2) hi = atom->box.zlo + cutneigh;

        hi = MIN(hi, (nbox + 1) * prd[idim] / comm->procgrid[idim]);

        if(myloc[idim] == 0) {
          comm->pbc_any[comm->nswap] = 1;

          if(idim == 0) comm->pbc_flagx[comm->nswap] = 1;

          if(idim == 1) comm->pbc_flagy[comm->nswap] = 1;

          if(idim == 2) comm->pbc_flagz[comm->nswap] = 1;
        }
      } else {
        comm->sendproc[comm->nswap] = comm->procneigh[idim][1];
        comm->recvproc[comm->nswap] = comm->procneigh[idim][0];
        nbox = myloc[idim] - ineed / 2;
        hi = (nbox + 1) * prd[idim] / comm->procgrid[idim];

        if(idim == 0) lo = atom->box.xhi - cutneigh;

        if(idim == 1) lo = atom->box.yhi - cutneigh;

        if(idim == 2) lo = atom->box.zhi - cutneigh;

        lo = MAX(lo, nbox * prd[idim] / comm->procgrid[idim]);

        if(myloc[idim] == comm->procgrid[idim] - 1) {
          comm->pbc_any[comm->nswap] = 1;

          if(idim == 0) comm->pbc_flagx[comm->nswap] = -1;

          if(idim == 1) comm->pbc_flagy[comm->nswap] = -1;

          if(idim == 2) comm->pbc_flagz[comm->nswap] = -1;
        }
      }

      comm->slablo[comm->nswap] = lo;
      comm->slabhi[comm->nswap] = hi;
      comm->nswap++;
    }
  }

  return 0;
}

/* communication of atom info every timestep */

void Comm_communicate(Comm *comm, Atom *atom)
{

  int iswap;
  int pbc_flags[4];
  MMD_float* buf;
  MPI_Request request;
  MPI_Status status;

  for(iswap = 0; iswap < comm->nswap; iswap++) {

    /* pack buffer */

    pbc_flags[0] = comm->pbc_any[iswap];
    pbc_flags[1] = comm->pbc_flagx[iswap];
    pbc_flags[2] = comm->pbc_flagy[iswap];
    pbc_flags[3] = comm->pbc_flagz[iswap];

    //
    //printf("C1\n");
    Atom_pack_comm(atom, comm->sendnum[iswap], comm->sendlist[iswap], comm->d_buf_send, pbc_flags);
    //printf("C2\n");

    //

    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(comm->sendproc[iswap] != comm->me) {
      
      {
        if(sizeof(MMD_float) == 4) {
          MPI_Irecv(comm->d_buf_recv, comm->comm_recv_size[iswap], MPI_FLOAT,
          comm->recvproc[iswap], 0, MPI_COMM_WORLD, &request);
          MPI_Send(comm->d_buf_send, comm->comm_send_size[iswap], MPI_FLOAT,
          comm->sendproc[iswap], 0, MPI_COMM_WORLD);
        } else {
          MPI_Irecv(comm->d_buf_recv, comm->comm_recv_size[iswap], MPI_DOUBLE,
          comm->recvproc[iswap], 0, MPI_COMM_WORLD, &request);
          MPI_Send(comm->d_buf_send, comm->comm_send_size[iswap], MPI_DOUBLE,
          comm->sendproc[iswap], 0, MPI_COMM_WORLD);
        }

        MPI_Wait(&request, &status);
      }
      buf = comm->d_buf_recv;
    } else buf = comm->d_buf_send;

    
    /* unpack buffer */

    //printf("C3\n");
    Atom_unpack_comm(atom, comm->recvnum[iswap], comm->firstrecv[iswap], buf);
    //printf("C4\n");
    //
  }
}

/* reverse communication of atom info every timestep */

void Comm_reverse_communicate(Comm *comm, Atom *atom)
{
  int iswap;
  MMD_float* buf;
  MPI_Request request;
  MPI_Status status;

  for(iswap = comm->nswap - 1; iswap >= 0; iswap--) {

    /* pack buffer */

    // 
    Atom_pack_reverse(atom, comm->recvnum[iswap], comm->firstrecv[iswap], comm->buf_send);

    // 
    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(comm->sendproc[iswap] != comm->me) {

      
      {
        if(sizeof(MMD_float) == 4) {
          MPI_Irecv(comm->buf_recv, comm->reverse_recv_size[iswap], MPI_FLOAT,
          comm->sendproc[iswap], 0, MPI_COMM_WORLD, &request);
          MPI_Send(comm->buf_send, comm->reverse_send_size[iswap], MPI_FLOAT,
          comm->recvproc[iswap], 0, MPI_COMM_WORLD);
        } else {
          MPI_Irecv(comm->buf_recv, comm->reverse_recv_size[iswap], MPI_DOUBLE,
          comm->sendproc[iswap], 0, MPI_COMM_WORLD, &request);
          MPI_Send(comm->buf_send, comm->reverse_send_size[iswap], MPI_DOUBLE,
          comm->recvproc[iswap], 0, MPI_COMM_WORLD);
        }
        MPI_Wait(&request, &status);
      }
      buf = comm->buf_recv;
    } else buf = comm->buf_send;

    /* unpack buffer */

    
    Atom_unpack_reverse(atom, comm->sendnum[iswap], comm->sendlist[iswap], buf);
    // 
  }
}

/* exchange:
   move atoms to correct proc boxes
   send out atoms that have left my box, receive ones entering my box
   this routine called before every reneighboring
   atoms exchanged with all 6 stencil neighbors
*/

void Comm_exchange(Comm *comm, Atom *atom)
{
  if(comm->do_safeexchange) {
    Comm_exchange_all(comm, atom);
    return;
  }

  int i, m, n, idim, nsend, nrecv, nrecv1, nrecv2, nlocal;
  MMD_float lo, hi, value;
  MMD_float** x;

  MPI_Request request;
  MPI_Status status;

  /* enforce PBC */

  Atom_pbc(atom);

  /* loop over dimensions */
  #pragma omp parallel 
  {
  int tid = omp_get_thread_num();

  for(idim = 0; idim < 3; idim++) {

    /* only exchange if more than one proc in this dimension */

    if(comm->procgrid[idim] == 1) continue;

    /* fill buffer with atoms leaving my box
       when atom is deleted, fill it in with last atom */

    i = nsend = 0;

    if(idim == 0) {
      lo = atom->box.xlo;
      hi = atom->box.xhi;
    } else if(idim == 1) {
      lo = atom->box.ylo;
      hi = atom->box.yhi;
    } else {
      lo = atom->box.zlo;
      hi = atom->box.zhi;
    }

    x = atom->x;

    nlocal = atom->nlocal;

    #pragma omp master
    {
      if(nlocal > comm->maxnlocal) {
        comm->send_flag = (int *) malloc(sizeof(int) * nlocal);
        comm->maxnlocal = nlocal;
      }

      if(comm->maxthreads < comm->threads->omp_num_threads) {
        comm->maxthreads = comm->threads->omp_num_threads;
        comm->nsend_thread   = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->nrecv_thread   = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->nholes_thread  = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->maxsend_thread = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->exc_sendlist_thread = (int **) malloc(sizeof(int *) * (comm->maxthreads));

        for(int i = 0; i < comm->maxthreads; i++) {
          comm->maxsend_thread[i] = comm->maxsend;
          comm->exc_sendlist_thread[i] = (int*) malloc(comm->maxsend * sizeof(int));
        }
      }
    }

    #pragma omp barrier

    nsend = 0;
    
    comm->nsend_thread[tid] = 0;
    comm->nholes_thread[tid] = 0;

    
    #pragma omp for
    for(int i = 0; i < nlocal; i++) {
      if(x[i][idim] < lo || x[i][idim] >= hi) {
        if(nsend >= comm->maxsend_thread[tid])  {
          comm->maxsend_thread[tid] = nsend + 100;
          comm->exc_sendlist_thread[tid] = (int*) realloc(comm->exc_sendlist_thread[tid], (nsend + 100) * sizeof(int));
        }

        comm->exc_sendlist_thread[tid][nsend++] = i;
        comm->send_flag[i] = 0;
      } else
        comm->send_flag[i] = 1;
    }

    comm->nsend_thread[tid] = nsend;
  
    #pragma omp barrier

    #pragma omp master
    {
      int total_nsend = 0;

      for(int i = 0; i < comm->threads->omp_num_threads; i++) {
        total_nsend += comm->nsend_thread[i];
        comm->nsend_thread[i] = total_nsend;
      }

      if(total_nsend * 6 > comm->maxsend) Comm_growsend(comm, total_nsend * 6);
    }
    #pragma omp barrier

    int total_nsend = comm->nsend_thread[comm->threads->omp_num_threads - 1];
    int nholes = 0;

    for(int i = 0; i < nsend; i++)
      if(comm->exc_sendlist_thread[tid][i] < nlocal - total_nsend)
        nholes++;

    comm->nholes_thread[tid] = nholes;
    #pragma omp barrier

    #pragma omp master    
    {
      int total_nholes = 0;

      for(int i = 0; i < comm->threads->omp_num_threads; i++) {
        total_nholes += comm->nholes_thread[i];
        comm->nholes_thread[i] = total_nholes;
      }
    }
    #pragma omp barrier


    int j = nlocal;
    int holes = 0;

    while(holes < comm->nholes_thread[tid]) {
      j--;

      if(comm->send_flag[j]) holes++;

    }


    for(int k = 0; k < nsend; k++) {
      Atom_pack_exchange(atom, comm->exc_sendlist_thread[tid][k], &comm->buf_send[(k + comm->nsend_thread[tid] - nsend) * 6]);

      if(comm->exc_sendlist_thread[tid][k] < nlocal - total_nsend) {
        while(!comm->send_flag[j]) j++;

        Atom_copy(atom, j++, comm->exc_sendlist_thread[tid][k]);
      }
    }

    nsend *= 6;
    
    #pragma omp barrier
    #pragma omp master    
    {
      atom->nlocal = nlocal - total_nsend;
      nsend = total_nsend * 6;

      /* send/recv atoms in both directions
         only if neighboring procs are different */

      MPI_Send(&nsend, 1, MPI_INT, comm->procneigh[idim][0], 0, MPI_COMM_WORLD);
      MPI_Recv(&nrecv1, 1, MPI_INT, comm->procneigh[idim][1], 0, MPI_COMM_WORLD, &status);
      nrecv = nrecv1;

      if(comm->procgrid[idim] > 2) {
        MPI_Send(&nsend, 1, MPI_INT, comm->procneigh[idim][1], 0, MPI_COMM_WORLD);
        MPI_Recv(&nrecv2, 1, MPI_INT, comm->procneigh[idim][0], 0, MPI_COMM_WORLD, &status);
        nrecv += nrecv2;
      }

      if(nrecv > comm->maxrecv) Comm_growrecv(comm, nrecv);

      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(comm->buf_recv, nrecv1, MPI_FLOAT, comm->procneigh[idim][1], 0,
                  MPI_COMM_WORLD, &request);
        MPI_Send(comm->buf_send, nsend, MPI_FLOAT, comm->procneigh[idim][0], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(comm->buf_recv, nrecv1, MPI_DOUBLE, comm->procneigh[idim][1], 0,
                  MPI_COMM_WORLD, &request);
        MPI_Send(comm->buf_send, nsend, MPI_DOUBLE, comm->procneigh[idim][0], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);

      if(comm->procgrid[idim] > 2) {
        if(sizeof(MMD_float) == 4) {
          MPI_Irecv(&comm->buf_recv[nrecv1], nrecv2, MPI_FLOAT, comm->procneigh[idim][0], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(comm->buf_send, nsend, MPI_FLOAT, comm->procneigh[idim][1], 0, MPI_COMM_WORLD);
        } else {
          MPI_Irecv(&comm->buf_recv[nrecv1], nrecv2, MPI_DOUBLE, comm->procneigh[idim][0], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(comm->buf_send, nsend, MPI_DOUBLE, comm->procneigh[idim][1], 0, MPI_COMM_WORLD);
        }

        MPI_Wait(&request, &status);
      }

      comm->nrecv_atoms = nrecv / 6;

      for(int i = 0; i < comm->threads->omp_num_threads; i++)
        comm->nrecv_thread[i] = 0;

    }
    /* check incoming atoms to see if they are in my box
       if they are, add to my list */
    #pragma omp barrier
    

    nrecv = 0;

    
    #pragma omp for
    for(int i = 0; i < comm->nrecv_atoms; i++) {
      value = comm->buf_recv[i * 6 + idim];

      if(value >= lo && value < hi)
        nrecv++;
    }

    comm->nrecv_thread[tid] = nrecv;
    nlocal = atom->nlocal;
    
    #pragma omp barrier

    #pragma omp master
    {
      int total_nrecv = 0;

      for(int i = 0; i < comm->threads->omp_num_threads; i++) {
        total_nrecv += comm->nrecv_thread[i];
        comm->nrecv_thread[i] = total_nrecv;
      }

      atom->nlocal += total_nrecv;
    }
    #pragma omp barrier

    int copyinpos = nlocal + comm->nrecv_thread[tid] - nrecv;
    
    #pragma omp for
    for(int i = 0; i < comm->nrecv_atoms; i++) {
      value = comm->buf_recv[i * 6 + idim];

      if(value >= lo && value < hi)
        Atom_unpack_exchange(atom, copyinpos++, &comm->buf_recv[i * 6]);
    }

    // 
   }
  }
}

void Comm_exchange_all(Comm *comm, Atom *atom)
{
  int i, m, n, idim, nsend, nrecv, nrecv1, nrecv2, nlocal;
  MMD_float lo, hi, value;
  MMD_float** x;

  MPI_Request request;
  MPI_Status status;

  /* enforce PBC */

  Atom_pbc(atom);

  /* loop over dimensions */
  int iswap = 0;

  for(idim = 0; idim < 3; idim++) {

    /* only exchange if more than one proc in this dimension */

    if(comm->procgrid[idim] == 1) {
      iswap += 2 * comm->need[idim];
      continue;
    }

    /* fill buffer with atoms leaving my box
    *        when atom is deleted, fill it in with last atom */

    i = nsend = 0;

    if(idim == 0) {
      lo = atom->box.xlo;
      hi = atom->box.xhi;
    } else if(idim == 1) {
      lo = atom->box.ylo;
      hi = atom->box.yhi;
    } else {
      lo = atom->box.zlo;
      hi = atom->box.zhi;
    }

    x = atom->x;

    nlocal = atom->nlocal;

    while(i < nlocal) {
      if(x[i][idim] < lo || x[i][idim] >= hi) {
        if(nsend > comm->maxsend) Comm_growsend(comm, nsend);

        nsend += Atom_pack_exchange(atom, i, &comm->buf_send[nsend]);
        Atom_copy(atom, nlocal - 1, i);
        nlocal--;
      } else i++;
    }

    atom->nlocal = nlocal;

    /* send/recv atoms in both directions
    *        only if neighboring procs are different */
    for(int ineed = 0; ineed < 2 * comm->need[idim]; ineed += 1) {
      if(ineed < comm->procgrid[idim] - 1) {
        MPI_Send(&nsend, 1, MPI_INT, comm->sendproc_exc[iswap], 0, MPI_COMM_WORLD);
        MPI_Recv(&nrecv, 1, MPI_INT, comm->recvproc_exc[iswap], 0, MPI_COMM_WORLD, &status);

        if(nrecv > comm->maxrecv) Comm_growrecv(comm, nrecv);

        if(sizeof(MMD_float) == 4) {
          MPI_Irecv(comm->buf_recv, nrecv, MPI_FLOAT, comm->recvproc_exc[iswap], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(comm->buf_send, nsend, MPI_FLOAT, comm->sendproc_exc[iswap], 0, MPI_COMM_WORLD);
        } else {
          MPI_Irecv(comm->buf_recv, nrecv, MPI_DOUBLE, comm->recvproc_exc[iswap], 0,
                    MPI_COMM_WORLD, &request);
          MPI_Send(comm->buf_send, nsend, MPI_DOUBLE, comm->sendproc_exc[iswap], 0, MPI_COMM_WORLD);
        }

        MPI_Wait(&request, &status);

        /* check incoming atoms to see if they are in my box
        *        if they are, add to my list */

        n = atom->nlocal;
        m = 0;

        while(m < nrecv) {
          value = comm->buf_recv[m + idim];

          if(value >= lo && value < hi)
            m += Atom_unpack_exchange(atom, n++, &comm->buf_recv[m]);
          else m += Atom_skip_exchange(atom, &comm->buf_recv[m]);
        }

        atom->nlocal = n;
      }

      iswap += 1;

    }
  }
}

/* borders:
   make lists of nearby atoms to send to neighboring procs at every timestep
   one list is created for every swap that will be made
   as list is made, actually do swaps
   this does equivalent of a communicate (so don't need to explicitly
     call communicate routine on reneighboring timestep)
   this routine is called before every reneighboring
*/

void Comm_borders(Comm *comm, Atom *atom)
{
  int i, m, n, iswap, idim, ineed, nsend, nrecv, nall, nfirst, nlast;
  MMD_float lo, hi;
  int pbc_flags[4];
  MMD_float** x;
  MPI_Request request;
  MPI_Status status;

  /* erase all ghost atoms */

  atom->nghost = 0;

  /* do swaps over all 3 dimensions */

  iswap = 0;

  int tid = omp_get_thread_num();

    
    {
      if(atom->nlocal > comm->maxnlocal) {
        comm->send_flag = (int *) malloc(sizeof(int) * (atom->nlocal));
        comm->maxnlocal = atom->nlocal;
      }

      if(comm->maxthreads < comm->threads->omp_num_threads) {
        comm->maxthreads = comm->threads->omp_num_threads;
        comm->nsend_thread   = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->nrecv_thread   = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->nholes_thread  = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->maxsend_thread = (int *) malloc(sizeof(int) * comm->maxthreads);
        comm->exc_sendlist_thread = (int **) malloc(sizeof(int *) * (comm->maxthreads));

        for(int i = 0; i < comm->maxthreads; i++) {
          comm->maxsend_thread[i] = comm->maxsend;
          comm->exc_sendlist_thread[i] = (int*) malloc(comm->maxsend * sizeof(int));
        }
      }
    }

  for(idim = 0; idim < 3; idim++) {
    nlast = 0;

    for(ineed = 0; ineed < 2 * comm->need[idim]; ineed++) {

      // find atoms within slab boundaries lo/hi using <= and >=
      // check atoms between nfirst and nlast
      //   for first swaps in a dim, check owned and ghost
      //   for later swaps in a dim, only check newly arrived ghosts
      // store sent atom indices in list for use in future timesteps

      lo = comm->slablo[iswap];
      hi = comm->slabhi[iswap];
      pbc_flags[0] = comm->pbc_any[iswap];
      pbc_flags[1] = comm->pbc_flagx[iswap];
      pbc_flags[2] = comm->pbc_flagy[iswap];
      pbc_flags[3] = comm->pbc_flagz[iswap];

      x = atom->x;

      if(ineed % 2 == 0) {
        nfirst = nlast;
        nlast = atom->nlocal + atom->nghost;
      }

      

      for(int i = 0; i < comm->threads->omp_num_threads; i++) {
        comm->nsend_thread[i] = 0;
      }

      //
      nsend = 0;
      m = 0;

      

      for(int i = nfirst; i < nlast; i++) {
        if(x[i][idim] >= lo && x[i][idim] <= hi) {
          if(nsend >= comm->maxsend_thread[tid])  {
            comm->maxsend_thread[tid] = nsend + 100;
            comm->exc_sendlist_thread[tid] = (int*) realloc(comm->exc_sendlist_thread[tid], (nsend + 100) * sizeof(int));
          }

          comm->exc_sendlist_thread[tid][nsend++] = i;
        }
      }

      comm->nsend_thread[tid] = nsend;

      

      
      {
        int total_nsend = 0;

        for(int i = 0; i < comm->threads->omp_num_threads; i++) {
          total_nsend += comm->nsend_thread[i];
          comm->nsend_thread[i] = total_nsend;
        }

        if(total_nsend > comm->maxsendlist[iswap]) Comm_growlist(comm, iswap, total_nsend);

        if(total_nsend * 3 > comm->maxsend) Comm_growsend(comm, total_nsend * 3);
      }
      

      for(int k = 0; k < nsend; k++) {
        Atom_pack_border(atom, comm->exc_sendlist_thread[tid][k], &comm->buf_send[(k + comm->nsend_thread[tid] - nsend) * 3], pbc_flags);
        comm->sendlist[iswap][k + comm->nsend_thread[tid] - nsend] = comm->exc_sendlist_thread[tid][k];
      }

      


      /* swap atoms with other proc
      put incoming ghosts at end of my atom arrays
      if swapping with self, simply copy, no messages */

      
      {
        nsend = comm->nsend_thread[comm->threads->omp_num_threads - 1];

        if(comm->sendproc[iswap] != comm->me) {
          MPI_Send(&nsend, 1, MPI_INT, comm->sendproc[iswap], 0, MPI_COMM_WORLD);
          MPI_Recv(&nrecv, 1, MPI_INT, comm->recvproc[iswap], 0, MPI_COMM_WORLD, &status);

          if(nrecv * atom->border_size > comm->maxrecv) Comm_growrecv(comm, nrecv * atom->border_size);

          if(sizeof(MMD_float) == 4) {
            MPI_Irecv(comm->buf_recv, nrecv * atom->border_size, MPI_FLOAT,
                      comm->recvproc[iswap], 0, MPI_COMM_WORLD, &request);
            MPI_Send(comm->buf_send, nsend * atom->border_size, MPI_FLOAT,
                     comm->sendproc[iswap], 0, MPI_COMM_WORLD);
          } else {
            MPI_Irecv(comm->buf_recv, nrecv * atom->border_size, MPI_DOUBLE,
                      comm->recvproc[iswap], 0, MPI_COMM_WORLD, &request);
            MPI_Send(comm->buf_send, nsend * atom->border_size, MPI_DOUBLE,
                     comm->sendproc[iswap], 0, MPI_COMM_WORLD);
          }

          MPI_Wait(&request, &status);
          comm->buf = comm->buf_recv;
        } else {
          nrecv = nsend;
          comm->buf = comm->buf_send;
        }

        comm->nrecv_atoms = nrecv;
      }
      /* unpack buffer */

      
      n = atom->nlocal + atom->nghost;
      nrecv = comm->nrecv_atoms;

      

      for(int i = 0; i < nrecv; i++)
        Atom_unpack_border(atom, n + i, &comm->buf[i * 3]);

      // 

      /* set all pointers & counters */

      
      {
        comm->sendnum[iswap] = nsend;
        comm->recvnum[iswap] = nrecv;
        comm->comm_send_size[iswap] = nsend * atom->comm_size;
        comm->comm_recv_size[iswap] = nrecv * atom->comm_size;
        comm->reverse_send_size[iswap] = nrecv * atom->reverse_size;
        comm->reverse_recv_size[iswap] = nsend * atom->reverse_size;
        comm->firstrecv[iswap] = atom->nlocal + atom->nghost;
        atom->nghost += nrecv;
      }
      
      iswap++;
    }
  }

  /* insure buffers are large enough for reverse comm */

  int max1, max2;
  max1 = max2 = 0;

  for(iswap = 0; iswap < comm->nswap; iswap++) {
    max1 = MAX(max1, comm->reverse_send_size[iswap]);
    max2 = MAX(max2, comm->reverse_recv_size[iswap]);
  }

  if(max1 > comm->maxsend) Comm_growsend(comm, max1);

  if(max2 > comm->maxrecv) Comm_growrecv(comm, max2);
}

/* realloc the size of the send buffer as needed with BUFFACTOR & BUFEXTRA */

void Comm_growsend(Comm *comm, int n)
{
  comm->maxsend = (int)(BUFFACTOR * n);
  comm->buf_send = (MMD_float*) realloc(comm->buf_send, (comm->maxsend + BUFEXTRA) * sizeof(MMD_float));
  acc_free(comm->d_buf_send);
  comm->d_buf_send = (MMD_float*) acc_malloc((comm->maxsend+BUFEXTRA)*sizeof(MMD_float));
}

/* free/malloc the size of the recv buffer as needed with BUFFACTOR */

void Comm_growrecv(Comm *comm, int n)
{
  comm->maxrecv = (int)(BUFFACTOR * n);
  free(comm->buf_recv);
  acc_free(comm->d_buf_recv);
  comm->buf_recv = (MMD_float*) malloc(comm->maxrecv * sizeof(MMD_float));
  comm->d_buf_recv = (MMD_float*) acc_malloc(comm->maxrecv * sizeof(MMD_float));
}

/* realloc the size of the iswap sendlist as needed with BUFFACTOR */

void Comm_growlist(Comm *comm, int iswap, int n)
{
  comm->maxsendlist[iswap] = (int)(BUFFACTOR * n);
  comm->sendlist[iswap] =
    (int*) realloc(comm->sendlist[iswap], comm->maxsendlist[iswap] * sizeof(int));
}
