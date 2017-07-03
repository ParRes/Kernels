/*
Copyright (c) 2016, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
* Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    PIC

PURPOSE: This program tests the efficiency with which a cloud of
         charged particles can be moved through a spatially fixed
         collection of charges located at the vertices of a square
         equi-spaced grid. It is a proxy for a component of a
         particle-in-cell method

USAGE:   <progname> <#simulation steps> <grid size> <#particles> \
                    <horizontal velocity> <vertical velocity>    \
                    <init mode> <init parameters>

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than standard C functions, the following functions are used in
         this program:
         initializeGrid()
         initializeGeometric()
         initializeSinusoidal()
         initializeLinear()
         initializePatch()
         finishParticlesInitialization()
         find_owner()
         computeCoulomb()
         computeTotalForce()
         verifyParticle()
         add_particle_to_buffer()
         attach_particles()
         attach_received_particles()
         resize_buffer()
         bad_patch()
         contain()
         wtime()
         random_draw()

HISTORY: - Written by Evangelos Georganas, August 2015.
         - RvdW: Refactored to make the code PRK conforming, March 2016

**********************************************************************************/
#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>
#include <random_draw.h>

/* M_PI is not defined in strict C99 */
#ifdef M_PI
#define PRK_M_PI M_PI
#else
#define PRK_M_PI 3.14159265358979323846264338327950288419716939937510
#endif

#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>

#define MASS_INV 1.0
#define Q 1.0
#define epsilon 0.000001
#define DT 1.0
#define MEMORYSLACK 10

#define REL_X 0.5
#define REL_Y 0.5

#define GEOMETRIC  10
#define SINUSOIDAL 11
#define LINEAR     12
#define PATCH      13
#define UNDEFINED  14

typedef struct {
  uint64_t left;
  uint64_t right;
  uint64_t bottom;
  uint64_t top;
} bbox_t;

/* Particle data structure */
typedef struct particle_t {
  double   x;    // x coordinate of particle
  double   y;    // y coordinate of particle
  double   v_x;  // component of velocity in x direction
  double   v_y;  // component of velocity in y direction
  double   q;    // charge of the particle
  /* The following variables are used only for verification/debug purposes */
  double   x0;   // initial position in x
  double   y0;   // initial position in y
  double   k;
  double   m;
  double   ID;   // ID of particle; use double to create homogeneous type
} particle_t;

int bad_patch(bbox_t *patch, bbox_t *patch_contain) {
  if (patch->left>=patch->right || patch->bottom>=patch->top) return(1);
  if (patch_contain) {
    if (patch->left  <patch_contain->left   || patch->right>=patch_contain->right) return(2);
    if (patch->bottom<patch_contain->bottom || patch->top  >=patch_contain->top)   return(3);
  }
  return(0);
}

int contain(uint64_t x, uint64_t y, bbox_t patch) {
  if (x<patch.left || x>patch.right || y<patch.bottom || y>patch.top) return 0;
  return 1;
}

/* Initializes the grid of charges */
double *initializeGrid(bbox_t tile) {
  double   *grid;
  uint64_t x, y, n_columns, n_rows;
  int      error=0, my_ID;

  n_columns = tile.right-tile.left+1;
  n_rows = tile.top-tile.bottom+1;

  grid = (double*) prk_malloc(n_columns*n_rows*sizeof(double));
  if (grid == NULL) {
    MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
    printf("ERROR: Process %d could not allocate space for grid\n", my_ID);
    error = 1;
  }
  bail_out(error);

  /* So far supporting only initialization with dipoles */
  for (y=tile.bottom; y<=tile.top; y++) {
    for (x=tile.left; x<=tile.right; x++) {
      grid[y-tile.bottom+(x-tile.left)*n_rows] = (x%2 == 0) ? Q : -Q;
    }
  }
  return grid;
}

/* Completes particle distribution */
void finishParticlesInitialization(uint64_t n, particle_t *p) {
  double x_coord, y_coord, rel_x, rel_y, cos_theta, cos_phi, r1_sq, r2_sq, base_charge, ID;
  uint64_t x, pi, cumulative_count;

  MPI_Scan(&n, &cumulative_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  ID = (double) (cumulative_count - n + 1);
  int my_ID;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);

  for (pi=0; pi<n; pi++) {
    x_coord = p[pi].x;
    y_coord = p[pi].y;
    rel_x = fmod(x_coord,1.0);
    rel_y = fmod(y_coord,1.0);
    x = (uint64_t) x_coord;
    r1_sq = rel_y * rel_y + rel_x * rel_x;
    r2_sq = rel_y * rel_y + (1.0-rel_x) * (1.0-rel_x);
    cos_theta = rel_x/sqrt(r1_sq);
    cos_phi = (1.0-rel_x)/sqrt(r2_sq);
    base_charge = 1.0 / ((DT*DT) * Q * (cos_theta/r1_sq + cos_phi/r2_sq));

    p[pi].v_x = 0.0;
    p[pi].v_y = ((double) p[pi].m) / DT;
    /* this particle charge assures movement in positive x-direction */
    p[pi].q = (x%2 == 0) ? (2*p[pi].k+1)*base_charge : -1.0 * (2*p[pi].k+1)*base_charge ;
    p[pi].x0 = x_coord;
    p[pi].y0 = y_coord;
    p[pi].ID = ID;
    ID += 1.0;
  }
}

/* Initializes the particles following the geometric distribution as described in the spec */
particle_t *initializeGeometric(uint64_t n_input, uint64_t L, double rho,
                                bbox_t tile, double k, double m,
		                uint64_t *n_placed, uint64_t *n_size,
                                random_draw_t *parm) {
  particle_t  *particles;
  double      A;
  uint64_t    x, y, p, pi, actual_particles, start_index;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them               */
  /* Each cell in the i-th column of cells contains p(i) = A * rho^i particles */
  A = n_input * ((1.0-rho) / (1.0-pow(rho, L))) / (double) L;

  for (*n_placed=0,x=tile.left; x<tile.right; x++) {
    /* at start of each grid column we jump into sequence of random numbers */
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index, 0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      (*n_placed) += random_draw(A * pow(rho, x), parm);
    }
  }

  /* use some slack in allocating memory to avoid fine-grain memory management */
  (*n_size) = ((*n_placed)*(1+MEMORYSLACK))/MEMORYSLACK;
  particles = (particle_t*) prk_malloc((*n_size) * sizeof(particle_t));
  if (particles == NULL) return(particles);

  for (pi=0,x=tile.left; x<tile.right; x++) {
    /* at start of each grid column we jump into sequence of random numbers */
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index, 0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      actual_particles = random_draw(A * pow(rho, x), parm);
      for (p=0; p<actual_particles; p++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
        pi++;
      }
    }
  }
  finishParticlesInitialization((*n_placed), particles);

  return particles;
}

/* Initialize with a sinusodial particle distribution */
particle_t *initializeSinusoidal(uint64_t n_input, uint64_t L,
                                 bbox_t tile, double k, double m,
                                 uint64_t *n_placed, uint64_t *n_size,
                                 random_draw_t *parm) {
  particle_t  *particles;
  double      step;
  uint64_t     x, y, pi, p, actual_particles, start_index;

  /* initialize random number generator */
  LCG_init(parm);

  step = PRK_M_PI/L;
  /* Place number of particles to each cell to form distribution decribed in spec.         */
  for ((*n_placed)=0,x=tile.left; x<tile.right; x++) {
    /* at start of each grid column we jump into sequence of random numbers */
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index, 0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      (*n_placed) += random_draw(2.0*cos(x*step)*cos(x*step)*n_input/(L*L), parm);
    }
  }

  /* use some slack in allocating memory to avoid fine-grain memory management */
  (*n_size) = ((*n_placed)*(1+MEMORYSLACK))/MEMORYSLACK;
  particles = (particle_t*) prk_malloc((*n_size) * sizeof(particle_t));
  if (particles == NULL) return(particles);

  for (pi=0,x=tile.left; x<tile.right; x++) {
    /* at start of each grid column we jump into sequence of random numbers */
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index, 0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      actual_particles = random_draw(2.0*cos(x*step)*cos(x*step)*n_input/(L*L), parm);
      for (p=0; p<actual_particles; p++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
        pi++;
      }
    }
  }
  finishParticlesInitialization((*n_placed), particles);
  return particles;
}

/* Initialize particles with "linearly-decreasing" distribution */
/* The linear function is f(x) = -alpha * x + beta , x in [0,1]*/
particle_t *initializeLinear(uint64_t n_input, uint64_t L, double alpha, double beta,
                             bbox_t tile, double k, double m,
                             uint64_t *n_placed, uint64_t *n_size,
                             random_draw_t *parm) {
  particle_t  *particles;
  double      total_weight, step, current_weight;
  uint64_t     x, y, p, pi, actual_particles, start_index;

  /* initialize random number generator */
  LCG_init(parm);

  /* First, find sum of all weights in order to normalize the number of particles */
  step         = 1.0/(L-1);
  total_weight = beta*L-alpha*0.5*step*L*(L-1);

  /* Loop over columns of cells and assign number of particles proportional linear weight */
  for (*n_placed=0,x=tile.left; x<tile.right; x++) {
    current_weight = (beta - alpha * step * ((double) x));
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index, 0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      (*n_placed) += random_draw(n_input*(current_weight/total_weight)/L, parm);
    }
  }

  /* use some slack in allocating memory to avoid fine-grain memory management */
  (*n_size) = ((*n_placed)*(1+MEMORYSLACK))/MEMORYSLACK;
  particles = (particle_t*) prk_malloc((*n_size) * sizeof(particle_t));
  if (particles == NULL) return(particles);

  for (pi=0,x=tile.left; x<tile.right; x++) {
    current_weight = (beta - alpha * step * ((double) x));
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index,0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      actual_particles = random_draw(n_input*(current_weight/total_weight)/L, parm);
      for (p=0; p<actual_particles; p++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
        pi++;
      }
    }
  }
  finishParticlesInitialization((*n_placed), particles);
  return particles;
}

/* Initialize uniformly particles within a "patch" */
particle_t *initializePatch(uint64_t n_input, uint64_t L, bbox_t patch,
                            bbox_t tile, double k, double m,
                            uint64_t *n_placed, uint64_t *n_size,
                            random_draw_t *parm) {
  particle_t *particles;
  uint64_t   x, y, total_cells, pi, p, actual_particles, start_index;
  double     particles_per_cell;

  /* initialize random number generator */
  LCG_init(parm);

  total_cells  = (patch.right - patch.left+1)*(patch.top - patch.bottom+1);
  particles_per_cell = (double) n_input/total_cells;

  /* Loop over columns of cells and assign number of particles if inside patch */
  for (*n_placed=0,x=tile.left; x<tile.right; x++) {
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index, 0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      if (contain(x,y,patch)) (*n_placed) += random_draw(particles_per_cell, parm);
      else                    (*n_placed) += random_draw(0.0, parm);
    }
  }

  /* use some slack in allocating memory to avoid fine-grain memory management */
  (*n_size) = ((*n_placed)*(1+MEMORYSLACK))/MEMORYSLACK;
  particles = (particle_t*) prk_malloc((*n_size) * sizeof(particle_t));
  if (particles == NULL) return(particles);

  for (pi=0,x=tile.left; x<tile.right; x++) {
    start_index = tile.bottom+x*L;
    LCG_jump(2*start_index,0, parm);
    for (y=tile.bottom; y<tile.top; y++) {
      actual_particles = random_draw(particles_per_cell, parm);
      if (!contain(x,y,patch)) actual_particles = 0;
      for (p=0; p<actual_particles; p++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
        pi++;
      }
    }
  }
  finishParticlesInitialization((*n_placed), particles);
  return particles;
}

/* introduce function pointer type to be able to switch between search functions */
typedef int (*find_owner_type)(particle_t p, int width, int height, int icrit, int jcrit,
                               int ileftover, int jleftover, int Num_procsx);

/* Finds the owner of particle (2D decomposition of grid to ranks) */
int find_owner_general(particle_t p, int width, int height, int Num_procsx,
                       int icrit, int jcrit, int ileftover, int jleftover) {
  int IDx, IDy, x, y;

  x = (int) floor(p.x);
  y = (int) floor(p.y);
  if (x<icrit) IDx = x/(width+1);
  else         IDx = ileftover + (x-icrit)/width;
  if (y<jcrit) IDy = y/(height+1);
  else         IDy = jleftover + (y-jcrit)/height;
  int proc_id = IDy * Num_procsx + IDx;

  return proc_id;
}

/* Finds the owner of particle (2D decomposition of grid to ranks) */
int find_owner_simple(particle_t p, int width, int height, int Num_procsx,
                       int icrit, int jcrit, int ileftover, int jleftover) {
  int IDx, IDy, x, y;

  x = (int) floor(p.x);
  y = (int) floor(p.y);
  IDx = x/width;
  IDy = y/height;
  int proc_id = IDy * Num_procsx + IDx;

  return proc_id;
}

/* Computes the Coulomb force among two charges q1 and q2 */
int computeCoulomb(double x_dist, double y_dist, double q1, double q2, double *fx, double *fy)
{
  double r, r2, f_coulomb;

  r2 = x_dist * x_dist + y_dist * y_dist;
  r = sqrt(r2);
  f_coulomb = q1 * q2 / r2;

  (*fx) = f_coulomb * x_dist/r; // f_coulomb * cos_theta
  (*fy) = f_coulomb * y_dist/r; // f_coulomb * sin_theta

  return 0;
}

/* Computes the total Coulomb force on a particle exerted from the charges of the corresponding cell */
void computeTotalForce(particle_t p, bbox_t tile, double *grid, double *fx, double *fy)
{
  uint64_t  x, y, n_rows;
  double   tmp_fx, tmp_fy, rel_y, rel_x, tmp_res_x, tmp_res_y;

  n_rows = tile.top-tile.bottom+1;

  /* Coordinates of the cell containing the particle */
  y = (uint64_t) floor(p.y);
  x = (uint64_t) floor(p.x);

  rel_x = p.x - x;
  rel_y = p.y - y;

  x = x - tile.left;
  y = y - tile.bottom;

  computeCoulomb(rel_x, rel_y, p.q, grid[y+x*n_rows], &tmp_fx, &tmp_fy);

  tmp_res_x = tmp_fx;
  tmp_res_y = tmp_fy;

  /* Coulomb force from bottom-left charge */
  computeCoulomb(rel_x, 1.0-rel_y, p.q, grid[(y+1)+x*n_rows], &tmp_fx, &tmp_fy);
  tmp_res_x += tmp_fx;
  tmp_res_y -= tmp_fy;

  /* Coulomb force from top-right charge */
  computeCoulomb(1.0-rel_x, rel_y, p.q, grid[y+(x+1)*n_rows], &tmp_fx, &tmp_fy);
  tmp_res_x -= tmp_fx;
  tmp_res_y += tmp_fy;

  /* Coulomb force from bottom-right charge */
  computeCoulomb(1.0-rel_x, 1.0-rel_y, p.q, grid[(y+1)+(x+1)*n_rows], &tmp_fx, &tmp_fy);
  tmp_res_x -= tmp_fx;
  tmp_res_y -= tmp_fy;

  (*fx) = tmp_res_x;
  (*fy) = tmp_res_y;
}

/* Verifies the final position of a particle */
int verifyParticle(particle_t p, double L, uint64_t iterations)
{
   double   x_final, y_final, x_periodic, y_periodic;

   x_final = p.x0 + (double) (iterations+1) * (2.0*p.k+1);
   y_final = p.y0 + (double) (iterations+1) * p.m;

   x_periodic = (x_final >= 0.0) ? fmod(x_final, L) : L + fmod(x_final, L);
   y_periodic = (y_final >= 0.0) ? fmod(y_final, L) : L + fmod(y_final, L);

   if ( fabs(p.x - x_periodic) > epsilon || fabs(p.y - y_periodic) > epsilon) {
     return(0);
   }
   return(1);
}

/* Adds a particle to a buffer. Resizes buffer if need be. */
void add_particle_to_buffer(particle_t p, particle_t **buffer, uint64_t *position, uint64_t *buffer_size)
{
   uint64_t cur_pos = (*position);
   uint64_t cur_buf_size = (*buffer_size);
   particle_t *cur_buffer = (*buffer);
   particle_t *temp_buf;

   if (cur_pos == cur_buf_size) {
      /* Have to resize buffer */
      temp_buf = (particle_t*) prk_malloc(2 * cur_buf_size * sizeof(particle_t));
      if (!temp_buf) {
        printf("Could not increase particle buffer size\n");
        /* do not attempt graceful exit; just allow code to abort */
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      memcpy(temp_buf, cur_buffer, cur_buf_size*sizeof(particle_t));
      prk_free(cur_buffer);
      cur_buffer = temp_buf;
      (*buffer) = temp_buf;
      (*buffer_size) = cur_buf_size * 2;
   }

   cur_buffer[cur_pos] = p;
   (*position)++;
}

/* Attaches src buffer of particles to destination buffer. Resizes destination buffer if need be. */
void attach_particles(particle_t **dst_buffer, uint64_t *position, uint64_t *buffer_size,
                      particle_t *src_buffer, uint64_t n_src_particles) {
   uint64_t cur_pos = (*position);
   uint64_t cur_buf_size = (*buffer_size);
   particle_t *cur_buffer = (*dst_buffer);
   particle_t *temp_buf;

   if ((cur_pos + n_src_particles) > cur_buf_size) {
      /* Have to resize buffer */
      temp_buf = (particle_t*) prk_malloc(2 *(cur_buf_size + n_src_particles) * sizeof(particle_t));
      if (!temp_buf) {
        printf("Could not increase particle buffer size\n");
        /* do not attempt graceful exit; just allow code to abort */
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      memcpy(temp_buf, cur_buffer, cur_pos*sizeof(particle_t));
      prk_free(cur_buffer);
      cur_buffer = temp_buf;
      (*dst_buffer) = temp_buf;
      (*buffer_size) = 2*(cur_buf_size + n_src_particles);
   }

   memcpy(&cur_buffer[cur_pos], src_buffer, n_src_particles * sizeof(particle_t));
   (*position) += n_src_particles;
}

void attach_received_particles(particle_t **dst_buffer, uint64_t *position, uint64_t *buffer_size, particle_t *src_buffer,
                               uint64_t n_src_particles, particle_t *src_buffer2, uint64_t n_src_particles2)
{
   uint64_t cur_pos = (*position);
   uint64_t cur_buf_size = (*buffer_size);
   particle_t *cur_buffer = (*dst_buffer);
   particle_t *temp_buf;

   if ((cur_pos + n_src_particles + n_src_particles2 ) > cur_buf_size) {
      /* Have to resize buffer */
      temp_buf = (particle_t*) prk_malloc((cur_buf_size + 2*(n_src_particles + n_src_particles2)) * sizeof(particle_t));
      if (!temp_buf) {
        printf("Could not increase particle buffer size\n");
        /* do not attempt graceful exit; just allow code to abort */
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      memcpy(temp_buf, cur_buffer, cur_pos*sizeof(particle_t));
      prk_free(cur_buffer);
      cur_buffer = temp_buf;
      (*dst_buffer) = temp_buf;
      (*buffer_size) = cur_buf_size + 2*(n_src_particles + n_src_particles2);
   }

   memcpy(&cur_buffer[cur_pos], src_buffer, n_src_particles * sizeof(particle_t));
   (*position) += n_src_particles;
   memcpy(&cur_buffer[*position], src_buffer2, n_src_particles2 * sizeof(particle_t));
   (*position) += n_src_particles2;
}

/* Resizes a buffer if need be */
void resize_buffer(particle_t **buffer, uint64_t *size, uint64_t new_size)
{
   uint64_t cur_size = (*size);

   if (new_size > cur_size) {
      prk_free(*buffer);
      (*buffer) = (particle_t*) prk_malloc(2*new_size*sizeof(particle_t));
      if (!(*buffer)) {
        printf("Could not increase particle buffer size\n");
        /* do not attempt graceful exit; just allow code to abort */
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
      (*size) = 2*new_size;
   }

}

int main(int argc, char ** argv) {

  int             Num_procs;         // number of ranks
  int             Num_procsx,
                  Num_procsy;        // number of ranks in each coord direction
  int             args_used = 1;     // keeps track of # consumed arguments
  int             my_ID;             // MPI rank
  int             my_IDx, my_IDy;    // coordinates of rank in rank grid
  int             root = 0;          // master rank
  uint64_t        L;                 // dimension of grid in cells
  uint64_t        iterations ;       // total number of simulation steps
  uint64_t        n;                 // total number of particles requested in the simulation
  uint64_t        actual_particles,  // actual number of particles owned by my rank
                  total_particles;   // total number of generated particles
  char            *init_mode;        // particle initialization mode (char)
  double          rho ;              // attenuation factor for geometric particle distribution
  uint64_t        k, m;              // determine initial horizontal and vertical velocity of
                                     // particles-- (2*k)+1 cells per time step
  double          *grid;             // the grid is represented as an array of charges
  uint64_t        iter, i;           // dummies
  double          fx, fy, ax, ay;    // particle forces and accelerations
  int             error=0;           // used for graceful exit after error
  uint64_t        correctness=0;     // boolean indicating correct particle displacements
  uint64_t        istart, jstart, iend, jend, particles_size, particles_count;
  bbox_t          grid_patch,        // whole grid
                  init_patch,        // subset of grid used for localized initialization
                  my_tile;           // subset of grid owner by my rank
  particle_t      *particles, *p;    // array of particles owned by my rank
  uint64_t        sendbuf_size[8], recvbuf_size[8];
  particle_t      *sendbuf[8], *recvbuf[8]; // particle communication buffers
  uint64_t        ptr_my;            //
  uint64_t        owner;             // owner (rank) of a particular particle
  double          pic_time, local_pic_time, avg_time;
  uint64_t        my_checksum = 0, tot_checksum = 0, correctness_checksum = 0;
  uint64_t        width, height;     // minimum dimensions of grid tile owned by my rank
  int             particle_mode;     // type of initialization
  double          alpha, beta;       // negative slope and offset for linear initialization
  int             nbr[8];            // topological neighbor ranks
  int             icrit, jcrit;      // global grid indices where tile size drops to minimum
  find_owner_type find_owner;
  int             ileftover, jleftover;// excess grid points divided among "fat" tiles
  uint64_t        to_send[8], to_recv[8];//
  MPI_Request requests[16];
  random_draw_t   dice;

  /* Initialize the MPI environment */
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

  /* FIXME: This can be further improved */
  /* Create MPI data type for particle_t */
  MPI_Datatype PARTICLE;
  MPI_Type_contiguous(sizeof(particle_t)/sizeof(double), MPI_DOUBLE, &PARTICLE);
  MPI_Type_commit( &PARTICLE );

  if (my_ID==root) {
    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("MPI Particle-in-Cell execution on 2D grid\n");

    if (argc<6) {
      printf("Usage: %s <#simulation steps> <grid size> <#particles> <k (particle charge semi-increment)> ", argv[0]);
      printf("<m (vertical particle velocity)>\n");
      printf("          <init mode> <init parameters>]\n");
      printf("   init mode \"GEOMETRIC\"  parameters: <attenuation factor>\n");
      printf("             \"SINUSOIDAL\" parameters: none\n");
      printf("             \"LINEAR\"     parameters: <negative slope> <constant offset>\n");
      printf("             \"PATCH\"      parameters: <xleft> <xright>  <ybottom> <ytop>\n");
      error = 1;
      goto ENDOFTESTS;
    }

    iterations = atol(*++argv);  args_used++;
    if (iterations<1) {
      printf("ERROR: Number of time steps must be positive: %llu\n", iterations);
      error = 1;
      goto ENDOFTESTS;
    }

    L = atol(*++argv);  args_used++;
    if (L<1 || L%2) {
      printf("ERROR: Number of grid cells must be positive and even: %llu\n", L);
      error = 1;
      goto ENDOFTESTS;
    }
    n = atol(*++argv);  args_used++;
    if (n<1) {
      printf("ERROR: Number of particles must be positive: %llu\n", n);
      error = 1;
      goto ENDOFTESTS;
    }

    particle_mode  = UNDEFINED;
    k = atoi(*++argv);   args_used++;
    m = atoi(*++argv);   args_used++;
    init_mode = *++argv; args_used++;

    ENDOFTESTS:;

  } // done with standard initialization parameters
  bail_out(error);

  MPI_Bcast(&iterations, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
  MPI_Bcast(&L,          1, MPI_UINT64_T, root, MPI_COMM_WORLD);
  MPI_Bcast(&n,          1, MPI_UINT64_T, root, MPI_COMM_WORLD);
  MPI_Bcast(&k,          1, MPI_UINT64_T, root, MPI_COMM_WORLD);
  MPI_Bcast(&m,          1, MPI_UINT64_T, root, MPI_COMM_WORLD);

  grid_patch = (bbox_t){0, L+1, 0, L+1};

  if (my_ID==root) { // process initialization parameters
    /* Initialize particles with geometric distribution */
    if (strcmp(init_mode, "GEOMETRIC") == 0) {
      if (argc<args_used+1) {
        printf("ERROR: Not enough arguments for GEOMETRIC\n");
        error = 1;
        goto ENDOFTESTS2;
      }
      particle_mode = GEOMETRIC;
      rho = atof(*++argv);   args_used++;
    }

    /* Initialize with a sinusoidal particle distribution (single period) */
    if (strcmp(init_mode, "SINUSOIDAL") == 0) {
      particle_mode = SINUSOIDAL;
    }

    /* Initialize particles with linear distribution */
    /* The linear function is f(x) = -alpha * x + beta , x in [0,1]*/
    if (strcmp(init_mode, "LINEAR") == 0) {
      if (argc<args_used+2) {
        printf("ERROR: Not enough arguments for LINEAR initialization\n");
        error = 1;
        goto ENDOFTESTS2;
        exit(EXIT_FAILURE);
      }
      particle_mode = LINEAR;
      alpha = atof(*++argv); args_used++;
      beta  = atof(*++argv); args_used++;
      if (beta <0 || beta<alpha) {
        printf("ERROR: linear profile gives negative particle density\n");
        error = 1;
        goto ENDOFTESTS2;
      }
    }

    /* Initialize particles uniformly within a "patch" */
    if (strcmp(init_mode, "PATCH") == 0) {
      if (argc<args_used+4) {
        printf("ERROR: Not enough arguments for PATCH initialization\n");
        error = 1;
        goto ENDOFTESTS2;
      }
      particle_mode = PATCH;
      init_patch.left   = atoi(*++argv); args_used++;
      init_patch.right  = atoi(*++argv); args_used++;
      init_patch.bottom = atoi(*++argv); args_used++;
      init_patch.top    = atoi(*++argv); args_used++;
      if (bad_patch(&init_patch, &grid_patch)) {
        printf("ERROR: inconsistent initial patch\n");
        error = 1;
        goto ENDOFTESTS2;
      }
    }
    ENDOFTESTS2:;

  } //done with processing initializaton parameters, now broadcast

  bail_out(error);

  MPI_Bcast(&particle_mode, 1, MPI_INT, root, MPI_COMM_WORLD);
  switch (particle_mode) {
  case GEOMETRIC:  MPI_Bcast(&rho,               1, MPI_DOUBLE,  root, MPI_COMM_WORLD);
                   break;
  case SINUSOIDAL: break;
  case LINEAR:     MPI_Bcast(&alpha,             1, MPI_DOUBLE,  root, MPI_COMM_WORLD);
                   MPI_Bcast(&beta,              1, MPI_DOUBLE,  root, MPI_COMM_WORLD);
                   break;
  case PATCH:      MPI_Bcast(&init_patch.left,   1, MPI_UINT64_T, root, MPI_COMM_WORLD);
                   MPI_Bcast(&init_patch.right,  1, MPI_UINT64_T, root, MPI_COMM_WORLD);
                   MPI_Bcast(&init_patch.bottom, 1, MPI_UINT64_T, root, MPI_COMM_WORLD);
                   MPI_Bcast(&init_patch.top,    1, MPI_UINT64_T, root, MPI_COMM_WORLD);
                   break;
  }

  /* determine best way to create a 2D grid of ranks (closest to square, for
     best surface/volume ratio); we do this brute force for now                        */

  for (Num_procsx=(int) (sqrt(Num_procs+1)); Num_procsx>0; Num_procsx--) {
    if (!(Num_procs%Num_procsx)) {
      Num_procsy = Num_procs/Num_procsx;
      break;
    }
  }
  my_IDx = my_ID%Num_procsx;
  my_IDy = my_ID/Num_procsx;

  if (my_ID == root) {
    printf("Number of ranks                    = %d\n", Num_procs);
    printf("Load balancing                     = None\n");
    printf("Grid size                          = %llu\n", L);
    printf("Tiles in x/y-direction             = %d/%d\n", Num_procsx, Num_procsy);
    printf("Number of particles requested      = %llu\n", n);
    printf("Number of time steps               = %llu\n", iterations);
    printf("Initialization mode                = %s\n",   init_mode);
    switch(particle_mode) {
    case GEOMETRIC: printf("  Attenuation factor               = %lf\n", rho);    break;
    case SINUSOIDAL:                                                              break;
    case LINEAR:    printf("  Negative slope                   = %lf\n", alpha);
                    printf("  Offset                           = %lf\n", beta);   break;
    case PATCH:     printf("  Bounding box                     = %llu, %llu, %llu, %llu\n",
                           init_patch.left, init_patch.right,
                           init_patch.bottom, init_patch.top);                    break;
    default:        printf("ERROR: Unsupported particle initializating mode\n");
                    error = 1;
    }
    printf("Particle charge semi-increment (k) = %llu\n", k);
    printf("Vertical velocity              (m) = %llu\n", m);
  }
  bail_out(error);

  /* The processes collectively create the underlying grid following a 2D block decomposition;
     unlike in the stencil code, successive blocks share an overlap vertex                   */
  width = L/Num_procsx;
  if (width < 2*k) {
    if (my_ID==0) printf("k-value too large: %llu, must be no greater than %llu\n", k, width/2);
    bail_out(1);
  }
  ileftover = L%Num_procsx;
  if (my_IDx<ileftover) {
    istart = (width+1) * my_IDx;
    iend = istart + width + 1;
  }
  else {
    istart = (width+1) * ileftover + width * (my_IDx-ileftover);
    iend = istart + width;
  }
  icrit = (width+1) * ileftover;

  height = L/Num_procsy;
  if (height < m) {
    if (my_ID==0) printf("m-value too large: %llu, must be no greater than %llu\n", m, height);
    bail_out(1);
  }

  jleftover = L%Num_procsy;
  if (my_IDy<jleftover) {
    jstart = (height+1) * my_IDy;
    jend = jstart + height + 1;
  }
  else {
    jstart = (height+1) * jleftover + height * (my_IDy-jleftover);
    jend = jstart + height;
  }
  jcrit = (height+1) * jleftover;

  /* if the problem can be divided evenly among ranks, use the simple owner finding function */
  if (icrit==0 && jcrit==0) {
    find_owner = find_owner_simple;
    if (my_ID==root) printf("Rank search mode used              = simple\n");
  }
  else {
    find_owner = find_owner_general;
    if (my_ID==root) printf("Rank search mode used              = general\n");
  }

  /* define bounding box for tile owned by my rank for convenience */
  my_tile = (bbox_t){istart,iend,jstart,jend};

  /* Find neighbors. Indexing: left=0, right=1, bottom=2, top=3,
                               bottom-left=4, bottom-right=5, top-left=6, top-right=7 */

  /* These are IDs in the global communicator */
  nbr[0] = (my_IDx == 0           ) ? my_ID  + Num_procsx - 1         : my_ID  - 1;
  nbr[1] = (my_IDx == Num_procsx-1) ? my_ID  - Num_procsx + 1         : my_ID  + 1;
  nbr[2] = (my_IDy == Num_procsy-1) ? my_ID  + Num_procsx - Num_procs : my_ID  + Num_procsx;
  nbr[3] = (my_IDy == 0           ) ? my_ID  - Num_procsx + Num_procs : my_ID  - Num_procsx;
  nbr[4] = (my_IDy == Num_procsy-1) ? nbr[0] + Num_procsx - Num_procs : nbr[0] + Num_procsx;
  nbr[5] = (my_IDy == Num_procsy-1) ? nbr[1] + Num_procsx - Num_procs : nbr[1] + Num_procsx;
  nbr[6] = (my_IDy == 0           ) ? nbr[0] - Num_procsx + Num_procs : nbr[0] - Num_procsx;
  nbr[7] = (my_IDy == 0           ) ? nbr[1] - Num_procsx + Num_procs : nbr[1] - Num_procsx;

  grid = initializeGrid(my_tile);

  LCG_init(&dice);
  switch(particle_mode){
  case GEOMETRIC:
    particles = initializeGeometric(n, L, rho, my_tile, k, m,
				    &particles_count, &particles_size, &dice);
    break;
  case LINEAR:
    particles = initializeLinear(n, L, alpha, beta, my_tile, k, m,
                                    &particles_count, &particles_size, &dice);
    break;
  case SINUSOIDAL:
    particles = initializeSinusoidal(n, L, my_tile, k, m,
                                    &particles_count, &particles_size, &dice);
    break;
  case PATCH:
    particles = initializePatch(n, L, init_patch, my_tile, k, m,
                                    &particles_count, &particles_size, &dice);
  }

  if (!particles) {
    printf("ERROR: Rank %d could not allocate space for %llu particles\n", my_ID, particles_size);
    error=1;
  }
  bail_out(error);

#if VERBOSE
  for (i=0; i<Num_procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == my_ID)  printf("Rank %d has %llu particles\n", my_ID, particles_count);
  }
#endif
  if (my_ID==root) {
    MPI_Reduce(&particles_count, &total_particles, 1, MPI_UINT64_T, MPI_SUM, root, MPI_COMM_WORLD);
    printf("Number of particles placed         = %llu\n", total_particles);
  }
  else {
    MPI_Reduce(&particles_count, &total_particles, 1, MPI_UINT64_T, MPI_SUM, root, MPI_COMM_WORLD);
  }

  /* Allocate space for communication buffers. Adjust appropriately as the simulation proceeds */

  error=0;
  for (i=0; i<8; i++) {
    sendbuf_size[i] = MAX(1,n/(MEMORYSLACK*Num_procs));
    recvbuf_size[i] = MAX(1,n/(MEMORYSLACK*Num_procs));
    sendbuf[i] = (particle_t*) prk_malloc(sendbuf_size[i] * sizeof(particle_t));
    recvbuf[i] = (particle_t*) prk_malloc(recvbuf_size[i] * sizeof(particle_t));
    if (!sendbuf[i] || !recvbuf[i]) error++;
  }
  if (error) printf("Rank %d could not allocate communication buffers\n", my_ID);
  bail_out(error);

  /* Run the simulation */
  for (iter=0; iter<=iterations; iter++) {

    /* start timer after a warmup iteration */
    if (iter == 1) {
      MPI_Barrier(MPI_COMM_WORLD);
      local_pic_time = wtime();
    }

    ptr_my = 0;
    for (i=0; i<8; i++) to_send[i]=0;

    /* Process own particles */
    p = particles;

    for (i=0; i < particles_count; i++) {
      fx = 0.0;
      fy = 0.0;
      computeTotalForce(p[i], my_tile, grid, &fx, &fy);

      ax = fx * MASS_INV;
      ay = fy * MASS_INV;

      /* Update particle positions, taking into account periodic boundaries */
      p[i].x = fmod(p[i].x + p[i].v_x*DT + 0.5*ax*DT*DT + L, L);
      p[i].y = fmod(p[i].y + p[i].v_y*DT + 0.5*ay*DT*DT + L, L);

      /* Update velocities */
      p[i].v_x += ax * DT;
      p[i].v_y += ay * DT;

      /* Check if particle stayed in same subdomain or moved to another */
      owner = find_owner(p[i], width, height, Num_procsx, icrit, jcrit, ileftover, jleftover);
      if (owner==my_ID) {
        add_particle_to_buffer(p[i], &p, &ptr_my, &particles_size);
      /* Add particle to the appropriate communication buffer */
      } else if (owner == nbr[0]) {
        add_particle_to_buffer(p[i], &sendbuf[0], &to_send[0], &sendbuf_size[0]);
      } else if (owner == nbr[1]) {
        add_particle_to_buffer(p[i], &sendbuf[1], &to_send[1], &sendbuf_size[1]);
      } else if (owner == nbr[2]) {
        add_particle_to_buffer(p[i], &sendbuf[2], &to_send[2], &sendbuf_size[2]);
      } else if (owner == nbr[3]) {
        add_particle_to_buffer(p[i], &sendbuf[3], &to_send[3], &sendbuf_size[3]);
      } else if (owner == nbr[4]) {
        add_particle_to_buffer(p[i], &sendbuf[4], &to_send[4], &sendbuf_size[4]);
      } else if (owner == nbr[5]) {
        add_particle_to_buffer(p[i], &sendbuf[5], &to_send[5], &sendbuf_size[5]);
      } else if (owner == nbr[6]) {
        add_particle_to_buffer(p[i], &sendbuf[6], &to_send[6], &sendbuf_size[6]);
      } else if (owner == nbr[7]) {
        add_particle_to_buffer(p[i], &sendbuf[7], &to_send[7], &sendbuf_size[7]);
      } else {
        printf("Could not find neighbor owner of particle %llu in tile %llu\n",
        i, owner);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
    }

    /* Communicate the number of particles to be sent/received */
    for (i=0; i<8; i++) {
      MPI_Isend(&to_send[i], 1, MPI_UINT64_T, nbr[i], 0, MPI_COMM_WORLD, &requests[i]);
      MPI_Irecv(&to_recv[i], 1, MPI_UINT64_T, nbr[i], 0, MPI_COMM_WORLD, &requests[8+i]);
    }
    MPI_Waitall(16, requests, MPI_STATUSES_IGNORE);

    /* Resize receive buffers if need be */
    for (i=0; i<8; i++) {
      resize_buffer(&recvbuf[i], &recvbuf_size[i], to_recv[i]);
    }

    /* Communicate the particles */
    for (i=0; i<8; i++) {
      MPI_Isend(sendbuf[i], to_send[i], PARTICLE, nbr[i], 0, MPI_COMM_WORLD, &requests[i]);
      MPI_Irecv(recvbuf[i], to_recv[i], PARTICLE, nbr[i], 0, MPI_COMM_WORLD, &requests[8+i]);
    }
    MPI_Waitall(16, requests, MPI_STATUSES_IGNORE);

    /* Attach received particles to particles buffer */
    for (i=0; i<4; i++) {
      attach_received_particles(&particles, &ptr_my, &particles_size, recvbuf[2*i], to_recv[2*i],
                                recvbuf[2*i+1], to_recv[2*i+1]);
    }
    particles_count = ptr_my;
  } // end of iterations

  local_pic_time = wtime() - local_pic_time;
  MPI_Reduce(&local_pic_time, &pic_time, 1, MPI_DOUBLE, MPI_MAX, root,
             MPI_COMM_WORLD);

  /* Run the verification test */
  /* First verify own particles */
  for (i=0; i < particles_count; i++) {
    correctness += verifyParticle(particles[i], (double)L, iterations);
    my_checksum += (uint64_t)particles[i].ID;
  }

  /* Gather total checksum of particles */
  MPI_Reduce(&my_checksum, &tot_checksum, 1, MPI_UINT64_T, MPI_SUM, root, MPI_COMM_WORLD);
  /* Gather total checksum of correctness flags */
  MPI_Reduce(&correctness, &correctness_checksum, 1, MPI_UINT64_T, MPI_SUM, root, MPI_COMM_WORLD);

  if ( my_ID == root) {
    if (correctness_checksum != total_particles ) {
      printf("ERROR: there are %llu miscalculated locations\n", total_particles-correctness_checksum);
    }
    else {
      if (tot_checksum != (total_particles*(total_particles+1))/2) {
        printf("ERROR: Particle checksum incorrect\n");
      }
      else {
        avg_time = total_particles*iterations/pic_time;
        printf("Solution validates\n");
        printf("Rate (Mparticles_moved/s): %lf\n", 1.0e-6*avg_time);
      }
    }
  }

#if VERBOSE
  for (i=0; i<Num_procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == my_ID)  printf("Rank %d has %llu particles\n", my_ID, particles_count);
  }
#endif

  MPI_Finalize();

  return 0;
}
