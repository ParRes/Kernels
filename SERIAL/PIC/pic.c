/*
Copyright (c) 2015, Intel Corporation

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
         wtime()
         bad_patch()
         random_draw()

HISTORY: - Written by Evangelos Georganas, August 2015.
         - RvdW: Refactored to make the code PRK conforming, December 2015

**********************************************************************************/

#include <par-res-kern_general.h>
#include <random_draw.h>

#include <math.h>
/* M_PI is not defined in strict C99 */
#ifdef M_PI
#define PRK_M_PI M_PI
#else
#define PRK_M_PI 3.14159265358979323846264338327950288419716939937510
#endif

#include <stdint.h>
#include <inttypes.h>

#define QG(i,j,L) Qgrid[(j)*(L+1)+i]
#define MASS_INV 1.0
#define Q 1.0
#define epsilon 0.000001
#define DT 1.0

#define SUCCESS 1
#define FAILURE 0

#define REL_X 0.5
#define REL_Y 0.5

#define GEOMETRIC  0
#define SINUSOIDAL 1
#define LINEAR     2
#define PATCH      3
#define UNDEFINED  4

typedef struct {
  uint64_t left;
  uint64_t right;
  uint64_t bottom;
  uint64_t top;
} bbox_t;

/* Particle data structure */
typedef struct particle_t {
  double   x;
  double   y;
  double   v_x;
  double   v_y;
  double   q;
  /* The following variables are used only for verification/debug purposes */
  double   x0;
  double   y0;
  int64_t  k; //  determines how many cells particles move per time step in the x direction
  int64_t  m; //  determines how many cells particles move per time step in the y direction
} particle_t;

/* Initializes the grid of charges
  We follow a column major format for the grid. Note that this may affect cache performance, depending on access pattern of particles. */

/* The grid is indexed in this way:

   y  ^
      |
      |
      |
      |
      |
      |
   (0,0)--------------> x                           */

double *initializeGrid(uint64_t L) {
  double   *Qgrid;
  uint64_t  x, y;

  Qgrid = (double*) prk_malloc((L+1)*(L+1)*sizeof(double));
  if (Qgrid == NULL) {
    printf("ERROR: Could not allocate space for grid\n");
    exit(EXIT_FAILURE);
  }

  /* initialization with dipoles */
  for (x=0; x<=L; x++) {
    for (y=0; y<=L; y++) {
      QG(y,x,L) = (x%2 == 0) ? Q : -Q;
    }
  }
  return Qgrid;
}

/* Completes particle distribution */
void finish_distribution(uint64_t n, particle_t *p) {
  double x_coord, y_coord, rel_x, rel_y, cos_theta, cos_phi, r1_sq, r2_sq, base_charge;
  uint64_t x, pi;

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
    p[pi].q = (x%2 == 0) ? (2*p[pi].k+1) * base_charge : -1.0 * (2*p[pi].k+1) * base_charge ;
    p[pi].x0 = x_coord;
    p[pi].y0 = y_coord;
  }
}

/* Initializes  particles with geometric distribution */
particle_t *initializeGeometric(uint64_t n_input, uint64_t L, double rho,
                                double k, double m, uint64_t *n_placed, 
                                random_draw_t *parm){
  particle_t  *particles;
  uint64_t    x, y, p, pi, actual_particles;
  double      A;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */

  /* Each cell in the i-th column of cells contains p(i) = A * rho^i particles */
  A = n_input * ((1.0-rho) / (1.0-pow(rho,L))) / (double)L;
  for (*n_placed=0,x=0; x<L; x++) {
    for (y=0; y<L; y++) {
      (*n_placed) += random_draw(A * pow(rho, x), parm);
    }
  }

  particles = (particle_t*) prk_malloc((*n_placed) * sizeof(particle_t));
  if (particles == NULL) {
    printf("ERROR: Could not allocate space for particles\n");
    exit(EXIT_FAILURE);
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  A = n_input * ((1.0-rho) / (1.0-pow(rho,L))) / (double)L;
  for (pi=0,x=0; x<L; x++) {
    for (y=0; y<L; y++) {
      actual_particles = random_draw(A * pow(rho, x), parm);
      for (p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution((*n_placed), particles);

  return particles;
}

/* Initialize particles with a sinusoidal distribution */
particle_t *initializeSinusoidal(uint64_t n_input, uint64_t L,
                                 double k, double m, uint64_t *n_placed,
                                 random_draw_t *parm){
  particle_t  *particles;
  double      step = PRK_M_PI/L;
  uint64_t    x, y, p, pi, actual_particles;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */

  /* Loop over columns of cells and assign number of particles proportional to sinusodial weight */
  for (*n_placed=0,x=0; x<L; x++) {
    for (y=0; y<L; y++) {
      (*n_placed) += random_draw(2.0*cos(x*step)*cos(x*step)*n_input/(L*L), parm);
    }
  }

  particles = (particle_t*) prk_malloc((*n_placed) * sizeof(particle_t));
  if (particles == NULL) {
    printf("ERROR: Could not allocate space for particles\n");
    exit(EXIT_FAILURE);
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  for (pi=0,x=0; x<L; x++) {
    for (y=0; y<L; y++) {
      actual_particles = random_draw(2.0*cos(x*step)*cos(x*step)*n_input/(L*L), parm);
      for (p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution((*n_placed), particles);

  return particles;
}

/* Initialize particles with linearly decreasing distribution */
/* The linear function is f(x) = -alpha * x + beta , x in [0,1]*/
particle_t *initializeLinear(uint64_t n_input, uint64_t L, double alpha, double beta,
                             double k, double m, uint64_t *n_placed, random_draw_t *parm){
  particle_t  *particles;
  uint64_t    x, y, p, pi, actual_particles;
  double      total_weight, step = 1.0/L, current_weight;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */

  /* Find sum of all weights to normalize the number of particles */
  total_weight = beta*L-alpha*0.5*step*L*(L-1);

  /* Loop over columns of cells and assign number of particles proportional linear weight */
  for ((*n_placed)=0,x=0; x<L; x++) {
    current_weight = (beta - alpha * step * ((double) x));
    for (y=0; y<L; y++) {
      (*n_placed) += random_draw(n_input * (current_weight/total_weight)/L, parm);
    }
  }

  particles = (particle_t*) prk_malloc((*n_placed) * sizeof(particle_t));
  if (particles == NULL) {
    printf("ERROR: Could not allocate space for particles\n");
    exit(EXIT_FAILURE);
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  /* Loop over columns of cells and assign number of particles proportional linear weight */
  for (pi=0,x=0; x<L; x++) {
    current_weight = (beta - alpha * step * ((double) x));
    for (y=0; y<L; y++) {
      actual_particles = random_draw(n_input * (current_weight/total_weight)/L, parm);
      for (p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution((*n_placed), particles);

  return particles;
}

/* Initialize uniformly particles within a "patch" */
particle_t *initializePatch(uint64_t n_input, uint64_t L, bbox_t patch,
                            double k, double m, uint64_t *n_placed,
                            random_draw_t *parm){
  particle_t  *particles;
  uint64_t    x, y, p, pi, total_cells, actual_particles;
  double      particles_per_cell;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */
  total_cells  = (patch.right - patch.left+1)*(patch.top - patch.bottom+1);
  particles_per_cell = (double) n_input/total_cells;

  /* Iterate over the columns of cells and assign uniform number of particles */
  for ((*n_placed)=0,x=0; x<L; x++) {
    for (y=0; y<L; y++) {
      actual_particles = random_draw(particles_per_cell, parm);
      if (x<patch.left || x>patch.right || y<patch.bottom || y>patch.top)
        actual_particles = 0;
      (*n_placed) += actual_particles;
    }
  }

  particles = (particle_t*) prk_malloc((*n_placed) * sizeof(particle_t));
  if (particles == NULL) {
    printf("ERROR: Could not allocate space for particles\n");
    exit(EXIT_FAILURE);
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  /* Iterate over the columns of cells and assign uniform number of particles */
  for (pi=0,x=0; x<L; x++) {
    for (y=0; y<L; y++) {
      actual_particles = random_draw(particles_per_cell, parm);
      if (x<patch.left || x>patch.right || y<patch.bottom || y>patch.top)
        actual_particles = 0;
      for (p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution((*n_placed), particles);

  return particles;
}

/* Verifies the final position of a particle */
int verifyParticle(particle_t p, uint64_t iterations, double *Qgrid, uint64_t L){
  uint64_t x, y;
  double   x_final, y_final, x_periodic, y_periodic, disp;

  /* Coordinates of the cell containing the particle initially */
  y = (uint64_t) p.y0;
  x = (uint64_t) p.x0;

  /* According to initial location and charge determine the direction of displacements */
  disp = (double)(iterations+1)*(2*p.k+1);
  x_final = ( (p.q * QG(y,x,L)) > 0) ? p.x0+disp : p.x0-disp;
  y_final = p.y0 + p.m * (double)(iterations+1);

  /* apply periodicity, making sure we never mod a negative value */
  x_periodic = fmod(x_final+(double)(iterations+1) *(2*p.k+1)*L, L);
  y_periodic = fmod(y_final+(double)(iterations+1) *llabs(p.m)*L, L);

  if ( fabs(p.x - x_periodic) > epsilon || fabs(p.y - y_periodic) > epsilon) {
    return FAILURE;
  }
  return SUCCESS;
}

/* Computes the Coulomb force among two charges q1 and q2 */
void computeCoulomb(double x_dist, double y_dist, double q1, double q2, double *fx, double *fy){
  double   r2 = x_dist * x_dist + y_dist * y_dist;
  double   r = sqrt(r2);
  double   f_coulomb = q1 * q2 / r2;

  (*fx) = f_coulomb * x_dist/r; // f_coulomb * cos_theta
  (*fy) = f_coulomb * y_dist/r; // f_coulomb * sin_theta
  return;
}

/* Computes the total Coulomb force on a particle exerted from the charges of the corresponding cell */
void computeTotalForce(particle_t p, uint64_t L, double *Qgrid, double *fx, double *fy){
  uint64_t  y, x;
  double   tmp_fx, tmp_fy, rel_y, rel_x, tmp_res_x = 0.0, tmp_res_y = 0.0;

  /* Coordinates of the cell containing the particle */
  y = (uint64_t) floor(p.y);
  x = (uint64_t) floor(p.x);
  rel_x = p.x -  x;
  rel_y = p.y -  y;

  /* Coulomb force from top-left charge */
  computeCoulomb(rel_x, rel_y, p.q, QG(y,x,L), &tmp_fx, &tmp_fy);
  tmp_res_x += tmp_fx;
  tmp_res_y += tmp_fy;

  /* Coulomb force from bottom-left charge */
  computeCoulomb(rel_x, 1.0-rel_y, p.q, QG(y+1,x,L), &tmp_fx, &tmp_fy);
  tmp_res_x += tmp_fx;
  tmp_res_y -= tmp_fy;

  /* Coulomb force from top-right charge */
  computeCoulomb(1.0-rel_x, rel_y, p.q, QG(y,x+1,L), &tmp_fx, &tmp_fy);
  tmp_res_x -= tmp_fx;
  tmp_res_y += tmp_fy;

  /* Coulomb force from bottom-right charge */
  computeCoulomb(1.0-rel_x, 1.0-rel_y, p.q, QG(y+1,x+1,L), &tmp_fx, &tmp_fy);
  tmp_res_x -= tmp_fx;
  tmp_res_y -= tmp_fy;

  (*fx) = tmp_res_x;
  (*fy) = tmp_res_y;
}

int bad_patch(bbox_t *patch, bbox_t *patch_contain) {
  if (patch->left>=patch->right || patch->bottom>=patch->top) return(1);
  if (patch_contain) {
    if (patch->left  <patch_contain->left   || patch->right>patch_contain->right) return(2);
    if (patch->bottom<patch_contain->bottom || patch->top  >patch_contain->top)   return(3);
  }
  return(0);
}

int main(int argc, char ** argv) {

  int         args_used = 1;     // keeps track of # consumed arguments
  uint64_t    L;                 // dimension of grid in cells
  uint64_t    iterations;        // total number of simulation steps
  uint64_t    n;                 // total number of particles in the simulation
  char        *init_mode;        // particle initialization mode (char)
  uint64_t    particle_mode;     // particle initialization mode (int)
  double      rho;               // attenuation factor for geometric particle distribution
  int64_t     k, m;              // determine initial horizontal and vertical velocity of
                                 // particles-- (2*k)+1 cells per time step
  double      alpha, beta;       // slope and offset values for linear particle distribution
  bbox_t      grid_patch,        // whole grid
              init_patch;        // subset of grid used for localized initialization
  int         correctness = 1;   // determines whether simulation was correct
  double      *Qgrid;            // field of fixed charges
  particle_t  *particles, *p;    // the particles array
  uint64_t    iter, i;           // dummies
  double      fx, fy, ax, ay;    // forces and accelerations
#if UNUSED
  int         particles_per_cell;// number of particles per cell to be injected
  int         error=0;           // used for graceful exit after error
#endif
  double      avg_time, pic_time;// timing parameters
  random_draw_t dice;

  printf("Parallel Research Kernels Version %s\n", PRKVERSION);
  printf("Serial Particle-in-Cell execution on 2D grid\n");

  /*******************************************************************************
  ** process and test input parameters
  ********************************************************************************/

  if (argc<6) {
    printf("Usage: %s <#simulation steps> <grid size> <#particles> <k (particle charge semi-increment)> ", argv[0]);
    printf("<m (vertical particle velocity)>\n");
    printf("          <init mode> <init parameters>]\n");
    printf("   init mode \"GEOMETRIC\"  parameters: <attenuation factor>\n");
    printf("             \"SINUSOIDAL\" parameters: none\n");
    printf("             \"LINEAR\"     parameters: <negative slope> <constant offset>\n");
    printf("             \"PATCH\"      parameters: <xleft> <xright>  <ybottom> <ytop>\n");
    exit(SUCCESS);
  }

  iterations = atol(*++argv);  args_used++;
  if (iterations<1) {
    printf("ERROR: Number of time steps must be positive: %" PRIu64 "\n", iterations);
    exit(FAILURE);
  }
  L = atol(*++argv);  args_used++;
  if (L<1 || L%2) {
    printf("ERROR: Number of grid cells must be positive and even: %" PRIu64 "\n", L);
    exit(FAILURE);
  }

  grid_patch = (bbox_t){0, L+1, 0, L+1};
  n = atol(*++argv);  args_used++;
  if (n<1) {
    printf("ERROR: Number of particles must be positive: %" PRIu64 "\n", n);
    exit(FAILURE);
  }

  particle_mode  = UNDEFINED;
  k = atoi(*++argv);   args_used++;
  if (k<0) {
    printf("ERROR: Particle semi-charge must be non-negative: %" PRIu64 "\n", k);
    exit(FAILURE);
  }
  m = atoi(*++argv);   args_used++;
  init_mode = *++argv; args_used++;

  /* Initialize particles with geometric distribution */
  if (strcmp(init_mode, "GEOMETRIC") == 0) {
    if (argc<args_used+1) {
      printf("ERROR: Not enough arguments for GEOMETRIC\n");
      exit(FAILURE);
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
      exit(EXIT_FAILURE);
    }
    particle_mode = LINEAR;
    alpha = atof(*++argv); args_used++;
    beta  = atof(*++argv); args_used++;
    if (beta <0 || beta<alpha) {
      printf("ERROR: linear profile gives negative particle density\n");
      exit(EXIT_FAILURE);
    }
  }

  /* Initialize particles uniformly within a "patch" */
  if (strcmp(init_mode, "PATCH") == 0) {
    if (argc<args_used+4) {
      printf("ERROR: Not enough arguments for PATCH initialization\n");
      exit(FAILURE);
    }
    particle_mode = PATCH;
    init_patch.left   = atoi(*++argv); args_used++;
    init_patch.right  = atoi(*++argv); args_used++;
    init_patch.bottom = atoi(*++argv); args_used++;
    init_patch.top    = atoi(*++argv); args_used++;
    if (bad_patch(&init_patch, &grid_patch)) {
      printf("ERROR: inconsistent initial patch\n");
      exit(FAILURE);
    }
  }

  printf("Grid size                      = %lld\n", L);
  printf("Number of particles requested  = %lld\n", n);
  printf("Number of time steps           = %lld\n", iterations);
  printf("Initialization mode            = %s\n", init_mode);
  switch(particle_mode) {
  case GEOMETRIC: printf("  Attenuation factor           = %lf\n", rho);    break;
  case SINUSOIDAL:                                                          break;
  case LINEAR:    printf("  Negative slope               = %lf\n", alpha);
                  printf("  Offset                       = %lf\n", beta);   break;
  case PATCH:     printf("  Bounding box                 = %" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "\n",
                         init_patch.left, init_patch.right,
                         init_patch.bottom, init_patch.top);                break;
  default:        printf("ERROR: Unsupported particle initializating mode\n");
                   exit(FAILURE);
  }
  printf("Particle charge semi-increment = %" PRIu64 "\n", k);
  printf("Vertical velocity              = %" PRIu64 "\n", m);

  /* Initialize grid of charges and particles */
  Qgrid = initializeGrid(L);

  LCG_init(&dice);
  switch(particle_mode) {
  case GEOMETRIC:  particles = initializeGeometric(n, L, rho, k, m, &n, &dice);      break;
  case SINUSOIDAL: particles = initializeSinusoidal(n, L, k, m, &n, &dice);          break;
  case LINEAR:     particles = initializeLinear(n, L, alpha, beta, k, m, &n, &dice); break;
  case PATCH:      particles = initializePatch(n, L, init_patch, k, m, &n, &dice);   break;
  default:         printf("ERROR: Unsupported particle distribution\n");  exit(FAILURE);
  }

  printf("Number of particles placed     = %lld\n", n);

  for (iter=0; iter<=iterations; iter++) {

    /* start the timer after one warm-up time step */
    if (iter==1) pic_time = wtime();

    /* Calculate forces on particles and update positions */
    for (i=0; i<n; i++) {
      p = particles;
      fx = 0.0;
      fy = 0.0;
      computeTotalForce(p[i], L, Qgrid, &fx, &fy);
      ax = fx * MASS_INV;
      ay = fy * MASS_INV;

      /* Update particle positions, taking into account periodic boundaries */
      p[i].x = fmod(p[i].x + p[i].v_x*DT + 0.5*ax*DT*DT + L, L);
      p[i].y = fmod(p[i].y + p[i].v_y*DT + 0.5*ay*DT*DT + L, L);

      /* Update velocities */
      p[i].v_x += ax * DT;
      p[i].v_y += ay * DT;
    }
  }

  pic_time = wtime() - pic_time;

  /* Run the verification test */
  for (i=0; i<n; i++) {
    correctness *= verifyParticle(particles[i], iterations, Qgrid, L);
  }

  if (correctness) {
    printf("Solution validates\n");
#if VERBOSE
    printf("Simulation time is %lf seconds\n", pic_time);
#endif
    avg_time = n*iterations/pic_time;
    printf("Rate (Mparticles_moved/s): %lf\n", 1.0e-6*avg_time);
  } else {
    printf("Solution does not validate\n");
  }

  return(EXIT_SUCCESS);
}
