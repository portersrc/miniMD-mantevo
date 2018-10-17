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

#include "neighbor.h"
#include "openmp.h"
#include "openacc.h"
#define FACTOR 0.999
#define SMALL 1.0e-6

void Neighbor_init(Neighbor *n)
{
  n->ncalls = 0;
  n->max_totalneigh = 0;
  n->d_numneigh = n->numneigh = NULL;
  n->d_neighbors = n->neighbors = NULL;
  n->maxneighs = 100;
  n->nmax = 0;
  n->bincount = NULL;
  n->bins = NULL;
  n->atoms_per_bin = 8;
  n->stencil = NULL;
  n->threads = NULL;
  n->halfneigh = 0;
  n->ghost_newton = 1;
}

void Neighbor_destroy(Neighbor *n)
{
#ifdef ALIGNMALLOC
  if(n->numneigh) _mm_free(n->numneigh);
  if(n->neighbors) _mm_free(n->neighbors);
#else 
  if(n->numneigh) free(n->numneigh);
  if(n->neighbors) free(n->neighbors);
  acc_free(n->d_numneigh);
  acc_free(n->d_neighbors);
#endif
  
  if(n->bincount) free(n->bincount);

  if(n->bins) free(n->bins);
}

/* binned neighbor list construction with full Newton's 3rd law
   every pair stored exactly once by some processor
   each owned atom i checks its own bin and other bins in Newton stencil */

void Neighbor_build(Neighbor *neighbor, Atom *atom)
{
  neighbor->ncalls++;
  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  /* extend atom arrays if necessary */

  

  if(nall > neighbor->nmax) {
    neighbor->nmax = nall;
#ifdef ALIGNMALLOC
    if(numneigh) _mm_free(numneigh);
    numneigh = (int*) _mm_malloc(neighbor->nmax * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
    if(neighbors) _mm_free(neighbors);	
    neighbors = (int*) _mm_malloc(neighbor->nmax * maxneighs * sizeof(int*) + ALIGNMALLOC, ALIGNMALLOC);	
#else

    if(neighbor->numneigh) free(neighbor->numneigh);
    if(neighbor->neighbors) free(neighbor->neighbors);
    acc_free(neighbor->d_numneigh);
    acc_free(neighbor->d_neighbors);   

    neighbor->numneigh = (int*) malloc(neighbor->nmax * sizeof(int));
    neighbor->neighbors = (int*) malloc(neighbor->nmax * neighbor->maxneighs * sizeof(int*));
    neighbor->d_numneigh = (int*) acc_malloc(neighbor->nmax * sizeof(int));
    neighbor->d_neighbors = (int*) acc_malloc(neighbor->nmax * neighbor->maxneighs * sizeof(int*));
#endif
  }

  int omp_me = 0 ;//omp_get_thread_num();
  int num_omp_threads = 1;//threads->omp_num_threads;
  int master = -1;

  
  master = omp_me;

  
  /* bin local & ghost atoms */
  Neighbor_binatoms(neighbor, atom, -1);
  neighbor->count = 0;
  /* loop over each atom, storing neighbors */


  int resize[1];
  resize[0] = 1;
  MMD_float* const restrict x_ = atom->d_x;
  Atom_sync_device(atom, x_, &atom->x[0][0], atom->nmax*PAD*sizeof(MMD_float));
  

  while(resize[0]) {
    
    int new_maxneighs = neighbor->maxneighs;
    resize[0] = 0;
    

    int* const restrict neighbors_ = neighbor->d_neighbors;
    int* const restrict numneigh_ = neighbor->d_numneigh;
    const int* const restrict bins_ = neighbor->bins;
    const int* const restrict bincount_ = neighbor->bincount;
    const int* const restrict stencil_ = neighbor->stencil;
    const int nlocal_ = nlocal;
    const int nmax_ = neighbor->nmax;
    const int maxneighs_ = neighbor->maxneighs;
    const MMD_float xprd_ = neighbor->xprd;
    const MMD_float yprd_ = neighbor->yprd;
    const MMD_float zprd_ = neighbor->zprd;
    const int nbinx_ = neighbor->nbinx;
    const int nbiny_ = neighbor->nbiny;
    const int nbinz_ = neighbor->nbinz;
    const int mbinx_ = neighbor->mbinx;
    const int mbiny_ = neighbor->mbiny;
    const int mbinz_ = neighbor->mbinz;
    const MMD_float mbinxlo_ = neighbor->mbinxlo;
    const MMD_float mbinylo_ = neighbor->mbinylo;
    const MMD_float mbinzlo_ = neighbor->mbinzlo;
    const MMD_float bininvx_ = neighbor->bininvx;
    const MMD_float bininvy_ = neighbor->bininvy;
    const MMD_float bininvz_ = neighbor->bininvz;
    const int nstencil_ = neighbor->nstencil; 
    const int mbins_ = neighbor->mbins;
    const int atoms_per_bin_ = neighbor->atoms_per_bin;
    const int halfneigh_ = neighbor->halfneigh;
    const int ghost_newton_ = neighbor->ghost_newton;
    const MMD_float cutneighsq_ = neighbor->cutneighsq;

    #pragma acc kernels deviceptr(x_,numneigh_,neighbors_) copyin(bins_[0:mbins_*atoms_per_bin_],bincount_[0:mbins_],stencil_[0:nstencil_]) copy(resize[0:1])
    for(int i = 0; i < nlocal_; i++) {
      int* const restrict neighptr = &neighbors_[i * DS0(nmax_,maxneighs_)];
      //int* neighptr = &neighbors[i * maxneighs];
      /* if necessary, goto next page and add pages */

      int n = 0;

      const MMD_float xtmp = x_[i * PAD + 0];
      const MMD_float ytmp = x_[i * PAD + 1];
      const MMD_float ztmp = x_[i * PAD + 2];

      /* loop over atoms in i's bin,
      */

      int ibin; 
      {
        int ix, iy, iz;

        if(xtmp >= xprd_)
          ix = (int)((xtmp - xprd_) * bininvx_) + nbinx_ - mbinxlo_;
        else if(xtmp >= 0.0)
          ix = (int)(xtmp * bininvx_) - mbinxlo_;
        else
          ix = (int)(xtmp * bininvx_) - mbinxlo_ - 1;

        if(ytmp >= yprd_)
          iy = (int)((ytmp - yprd_) * bininvy_) + nbiny_ - mbinylo_;
        else if(ytmp >= 0.0)
          iy = (int)(ytmp * bininvy_) - mbinylo_;
        else
          iy = (int)(ytmp * bininvy_) - mbinylo_ - 1;

        if(ztmp >= zprd_)
          iz = (int)((ztmp - zprd_) * bininvz_) + nbinz_ - mbinzlo_;
        else if(ztmp >= 0.0)
          iz = (int)(ztmp * bininvz_) - mbinzlo_;
        else
          iz = (int)(ztmp * bininvz_) - mbinzlo_ - 1;
        ibin = (iz * mbiny_ * mbinx_ + iy * mbinx_ + ix + 1);
      }

      for(int k = 0; k < nstencil_; k++) {
        const int jbin = ibin + stencil_[k];

        const int* restrict loc_bin = &bins_[jbin * atoms_per_bin_];

        if(ibin == jbin)
          for(int m = 0; m < bincount_[jbin]; m++) {
            const int j = loc_bin[m];

            //for same bin as atom i skip j if i==j and skip atoms "below and to the left" if using halfneighborlists
            if(((j == i) || (halfneigh_ && !ghost_newton_ && (j < i)) ||
                (halfneigh_ && ghost_newton_ && ((j < i) || ((j >= nlocal_) &&
                                               ((x_[j * PAD + 2] < ztmp) || (x_[j * PAD + 2] == ztmp && x_[j * PAD + 1] < ytmp) ||
                                                (x_[j * PAD + 2] == ztmp && x_[j * PAD + 1]  == ytmp && x_[j * PAD + 0] < xtmp))))))) continue;

            const MMD_float delx = xtmp - x_[j * PAD + 0];
            const MMD_float dely = ytmp - x_[j * PAD + 1];
            const MMD_float delz = ztmp - x_[j * PAD + 2];
            const MMD_float rsq = delx * delx + dely * dely + delz * delz;

            if((rsq <= cutneighsq_)) {neighptr[n*DS1(nmax_,maxneighs_)] = j; n++;}
            //if((rsq <= cutneighsq)) neighptr[n++] = j;
          }
        else {
          for(int m = 0; m < bincount_[jbin]; m++) {
            const int j = loc_bin[m];

            if(halfneigh_ && !ghost_newton_ && (j < i)) continue;

            const MMD_float delx = xtmp - x_[j * PAD + 0];
            const MMD_float dely = ytmp - x_[j * PAD + 1];
            const MMD_float delz = ztmp - x_[j * PAD + 2];
            const MMD_float rsq = delx * delx + dely * dely + delz * delz;

            if((rsq <= cutneighsq_)) {neighptr[n*DS1(nmax_,maxneighs_)] = j; n++;}
            //if((rsq <= cutneighsq)) neighptr[n++] = j;
          }
        }
      }
      numneigh_[i] = n;

      if(n >= maxneighs_) {
        resize[0] = 1;

        if(n >= new_maxneighs) new_maxneighs = n;
      }
    }

    // 

    if(resize[0]) {
      
      {
        neighbor->maxneighs = new_maxneighs * 1.2;
#ifdef ALIGNMALLOC
  		_mm_free(neighbors);
  		neighbors = (int*) _mm_malloc(nmax* maxneighs * sizeof(int) + ALIGNMALLOC, ALIGNMALLOC);
#else
        free(neighbor->neighbors);
        neighbor->neighbors = (int*) malloc(neighbor->nmax* neighbor->maxneighs * sizeof(int));
        acc_free(neighbor->d_neighbors);
        neighbor->d_neighbors = (int*) acc_malloc(neighbor->nmax* neighbor->maxneighs * sizeof(int));
#endif
      }
      
    }
  }
}

void Neighbor_binatoms(Neighbor *neighbor, Atom *atom, int count)
{
  
  const int omp_me = omp_get_thread_num();
  const int num_omp_threads = neighbor->threads->omp_num_threads;

  const int nlocal = atom->nlocal;
  const int nall = count < 0 ? atom->nlocal + atom->nghost:count;
  const MMD_float* x = &atom->x[0][0];

  neighbor->xprd = atom->box.xprd;
  neighbor->yprd = atom->box.yprd;
  neighbor->zprd = atom->box.zprd;

  neighbor->resize = 1; 

  while(neighbor->resize > 0) {
    
    neighbor->resize = 0;
    
    
    for(int i = 0; i < neighbor->mbins; i++) neighbor->bincount[i] = 0;


    #pragma omp parallel for 
    for(int i = 0; i < nall; i++) {
      const int ibin = Neighbor_coord2bin(neighbor, x[i * PAD + 0], x[i * PAD + 1], x[i * PAD + 2]);
      //printf("%i %i %lf %lf %lf\n",i,ibin,x[i * PAD + 0], x[i * PAD + 1], x[i * PAD + 2]);
      if(neighbor->bincount[ibin] < neighbor->atoms_per_bin) {
        int ac;
#ifdef OpenMP31
        ac = neighbor->bincount[ibin]++;
#else
        ac = __sync_fetch_and_add(neighbor->bincount + ibin, 1);
#endif
        neighbor->bins[ibin * neighbor->atoms_per_bin + ac] = i;
      } else neighbor->resize = 1;
    }

    // 

    

    if(neighbor->resize) {
      free(neighbor->bins);
      neighbor->atoms_per_bin *= 2;
      neighbor->bins = (int*) malloc(neighbor->mbins * neighbor->atoms_per_bin * sizeof(int));
    }

    // 
  }

  

}

/* convert xyz atom coords into local bin #
   take special care to insure ghost atoms with
   coord >= prd or coord < 0.0 are put in correct bins */

inline int Neighbor_coord2bin(Neighbor *neighbor, MMD_float x, MMD_float y, MMD_float z)
{
  int ix, iy, iz;

  if(x >= neighbor->xprd)
    ix = (int)((x - neighbor->xprd) * neighbor->bininvx) + neighbor->nbinx - neighbor->mbinxlo;
  else if(x >= 0.0)
    ix = (int)(x * neighbor->bininvx) - neighbor->mbinxlo;
  else
    ix = (int)(x * neighbor->bininvx) - neighbor->mbinxlo - 1;

  if(y >= neighbor->yprd)
    iy = (int)((y - neighbor->yprd) * neighbor->bininvy) + neighbor->nbiny - neighbor->mbinylo;
  else if(y >= 0.0)
    iy = (int)(y * neighbor->bininvy) - neighbor->mbinylo;
  else
    iy = (int)(y * neighbor->bininvy) - neighbor->mbinylo - 1;

  if(z >= neighbor->zprd)
    iz = (int)((z - neighbor->zprd) * neighbor->bininvz) + neighbor->nbinz - neighbor->mbinzlo;
  else if(z >= 0.0)
    iz = (int)(z * neighbor->bininvz) - neighbor->mbinzlo;
  else
    iz = (int)(z * neighbor->bininvz) - neighbor->mbinzlo - 1;

  return (iz * neighbor->mbiny * neighbor->mbinx + iy * neighbor->mbinx + ix + 1);
}


/*
setup neighbor binning parameters
bin numbering is global: 0 = 0.0 to binsize
                         1 = binsize to 2*binsize
                         nbin-1 = prd-binsize to binsize
                         nbin = prd to prd+binsize
                         -1 = -binsize to 0.0
coord = lowest and highest values of ghost atom coords I will have
        add in "small" for round-off safety
mbinlo = lowest global bin any of my ghost atoms could fall into
mbinhi = highest global bin any of my ghost atoms could fall into
mbin = number of bins I need in a dimension
stencil() = bin offsets in 1-d sense for stencil of surrounding bins
*/

int Neighbor_setup(Neighbor *neighbor, Atom *atom)
{
  int i, j, k, nmax;
  MMD_float coord;
  int mbinxhi, mbinyhi, mbinzhi;
  int nextx, nexty, nextz;
  int num_omp_threads = neighbor->threads->omp_num_threads;

  neighbor->cutneighsq = neighbor->cutneigh * neighbor->cutneigh;

  neighbor->xprd = atom->box.xprd;
  neighbor->yprd = atom->box.yprd;
  neighbor->zprd = atom->box.zprd;

  /*
  c bins must evenly divide into box size,
  c   becoming larger than cutneigh if necessary
  c binsize = 1/2 of cutoff is near optimal

  if (flag == 0) {
    nbinx = 2.0 * xprd / cutneigh;
    nbiny = 2.0 * yprd / cutneigh;
    nbinz = 2.0 * zprd / cutneigh;
    if (nbinx == 0) nbinx = 1;
    if (nbiny == 0) nbiny = 1;
    if (nbinz == 0) nbinz = 1;
  }
  */

  neighbor->binsizex = neighbor->xprd / neighbor->nbinx;
  neighbor->binsizey = neighbor->yprd / neighbor->nbiny;
  neighbor->binsizez = neighbor->zprd / neighbor->nbinz;
  neighbor->bininvx = 1.0 / neighbor->binsizex;
  neighbor->bininvy = 1.0 / neighbor->binsizey;
  neighbor->bininvz = 1.0 / neighbor->binsizez;

  coord = atom->box.xlo - neighbor->cutneigh - SMALL * neighbor->xprd;
  neighbor->mbinxlo = (int)(coord * neighbor->bininvx);

  if(coord < 0.0) neighbor->mbinxlo = neighbor->mbinxlo - 1;

  coord = atom->box.xhi + neighbor->cutneigh + SMALL * neighbor->xprd;
  mbinxhi = (int)(coord * neighbor->bininvx);

  coord = atom->box.ylo - neighbor->cutneigh - SMALL * neighbor->yprd;
  neighbor->mbinylo = (int)(coord * neighbor->bininvy);

  if(coord < 0.0) neighbor->mbinylo = neighbor->mbinylo - 1;

  coord = atom->box.yhi + neighbor->cutneigh + SMALL * neighbor->yprd;
  mbinyhi = (int)(coord * neighbor->bininvy);

  coord = atom->box.zlo - neighbor->cutneigh - SMALL * neighbor->zprd;
  neighbor->mbinzlo = (int)(coord * neighbor->bininvz);

  if(coord < 0.0) neighbor->mbinzlo = neighbor->mbinzlo - 1;

  coord = atom->box.zhi + neighbor->cutneigh + SMALL * neighbor->zprd;
  mbinzhi = (int)(coord * neighbor->bininvz);

  /* extend bins by 1 in each direction to insure stencil coverage */

  neighbor->mbinxlo = neighbor->mbinxlo - 1;
  mbinxhi = mbinxhi + 1;
  neighbor->mbinx = mbinxhi - neighbor->mbinxlo + 1;

  neighbor->mbinylo = neighbor->mbinylo - 1;
  mbinyhi = mbinyhi + 1;
  neighbor->mbiny = mbinyhi - neighbor->mbinylo + 1;

  neighbor->mbinzlo = neighbor->mbinzlo - 1;
  mbinzhi = mbinzhi + 1;
  neighbor->mbinz = mbinzhi - neighbor->mbinzlo + 1;

  /*
  compute bin stencil of all bins whose closest corner to central bin
  is within neighbor cutoff
  for partial Newton (newton = 0),
  stencil is all surrounding bins including self
  for full Newton (newton = 1),
  stencil is bins to the "upper right" of central bin, does NOT include self
  next(xyz) = how far the stencil could possibly extend
  factor < 1.0 for special case of LJ benchmark so code will create
  correct-size stencil when there are 3 bins for every 5 lattice spacings
  */

  nextx = (int)(neighbor->cutneigh * neighbor->bininvx);

  if(nextx * neighbor->binsizex < FACTOR * neighbor->cutneigh) nextx++;

  nexty = (int)(neighbor->cutneigh * neighbor->bininvy);

  if(nexty * neighbor->binsizey < FACTOR * neighbor->cutneigh) nexty++;

  nextz = (int)(neighbor->cutneigh * neighbor->bininvz);

  if(nextz * neighbor->binsizez < FACTOR * neighbor->cutneigh) nextz++;

  nmax = (2 * nextz + 1) * (2 * nexty + 1) * (2 * nextx + 1);

  if(neighbor->stencil) free(neighbor->stencil);

  neighbor->stencil = (int*) malloc(nmax * sizeof(int));

  neighbor->nstencil = 0;
  int kstart = -nextz;

  if(neighbor->halfneigh && neighbor->ghost_newton) {
    kstart = 0;
    neighbor->stencil[neighbor->nstencil++] = 0;
  }

  for(k = kstart; k <= nextz; k++) {
    for(j = -nexty; j <= nexty; j++) {
      for(i = -nextx; i <= nextx; i++) {
        if(!neighbor->ghost_newton || !neighbor->halfneigh || (k > 0 || j > 0 || (j == 0 && i > 0)))
          if(Neighbor_bindist(neighbor, i, j, k) < neighbor->cutneighsq) {
            neighbor->stencil[neighbor->nstencil++] = k * neighbor->mbiny * neighbor->mbinx + j * neighbor->mbinx + i;
          }
      }
    }
  }

  neighbor->mbins = neighbor->mbinx * neighbor->mbiny * neighbor->mbinz;

  if(neighbor->bincount) free(neighbor->bincount);

  neighbor->bincount = (int*) malloc(neighbor->mbins * num_omp_threads * sizeof(int));

  if(neighbor->bins) free(neighbor->bins);

  neighbor->bins = (int*) malloc(neighbor->mbins * num_omp_threads * neighbor->atoms_per_bin * sizeof(int));
  return 0;
}

/* compute closest distance between central bin (0,0,0) and bin (i,j,k) */

MMD_float Neighbor_bindist(Neighbor *neighbor, int i, int j, int k)
{
  MMD_float delx, dely, delz;

  if(i > 0)
    delx = (i - 1) * neighbor->binsizex;
  else if(i == 0)
    delx = 0.0;
  else
    delx = (i + 1) * neighbor->binsizex;

  if(j > 0)
    dely = (j - 1) * neighbor->binsizey;
  else if(j == 0)
    dely = 0.0;
  else
    dely = (j + 1) * neighbor->binsizey;

  if(k > 0)
    delz = (k - 1) * neighbor->binsizez;
  else if(k == 0)
    delz = 0.0;
  else
    delz = (k + 1) * neighbor->binsizez;

  return (delx * delx + dely * dely + delz * delz);
}
