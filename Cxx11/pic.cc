///
/// Copyright (c) 2020, Intel Corporation
/// Copyright (c) 2020, Argonne National Laboratory
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////////////////
///
///NAME:    PIC
///
///PURPOSE: This program tests the efficiency with which a cloud of
///         charged particles can be moved through a spatially fixed
///         collection of charges located at the vertices of a square
///         equi-spaced grid. It is a proxy for a component of a
///         particle-in-cell method
///
///USAGE:   <progname> <#simulation steps> <grid size> <#particles> \
///                    <horizontal velocity> <vertical velocity>    \
///                    <init mode> <init parameters>
///
///         The output consists of diagnostics to make sure the
///         algorithm worked, and of timing statistics.
///
///FUNCTIONS CALLED:
///
///         Other than standard C functions, the following functions are used in
///         this program:
///         bad_patch()
///         random_draw()
///
///HISTORY: - Written by Evangelos Georganas, August 2015.
///         - RvdW: Refactored to make the code PRK conforming, December 2015.
///         - Ported to DPC++ by Zheming Jin, August 2020.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"

#include "random_draw.h"

static const double Q = 1.0;
static const double epsilon = 0.000001;
static const double DT = 1.0;

static const double REL_X = 0.5;
static const double REL_Y = 0.5;

enum geometry { GEOMETRIC, SINUSOIDAL, LINEAR, PATCH, UNDEFINED };

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

double * initializeGrid(uint64_t L)
{
  double * Qgrid = prk::malloc<double>((L+1)*(L+1));
  if (Qgrid == NULL) {
    throw "ERROR: Could not allocate space for grid.";
  }

  /* initialization with dipoles */
  for (uint64_t x=0; x<=L; x++) {
    for (uint64_t y=0; y<=L; y++) {
      Qgrid[x*(L+1)+y] = (x%2 == 0) ? Q : -Q;
    }
  }
  return Qgrid;
}

/* Completes particle distribution */
void finish_distribution(const uint64_t n, particle_t * RESTRICT p)
{
  for (uint64_t pi=0; pi<n; pi++) {
    double x_coord = p[pi].x;
    double y_coord = p[pi].y;
    double rel_x = fmod(x_coord,1.0);
    double rel_y = fmod(y_coord,1.0);
    uint64_t x = (uint64_t) x_coord;
    double r1_sq = rel_y * rel_y + rel_x * rel_x;
    double r2_sq = rel_y * rel_y + (1.0-rel_x) * (1.0-rel_x);
    double cos_theta = rel_x/sqrt(r1_sq);
    double cos_phi = (1.0-rel_x)/sqrt(r2_sq);
    double base_charge = 1.0 / ((DT*DT) * Q * (cos_theta/r1_sq + cos_phi/r2_sq));

    p[pi].v_x = 0.0;
    p[pi].v_y = ((double) p[pi].m) / DT;
    /* this particle charge assures movement in positive x-direction */
    p[pi].q = (x%2 == 0) ? (2*p[pi].k+1) * base_charge : -1.0 * (2*p[pi].k+1) * base_charge ;
    p[pi].x0 = x_coord;
    p[pi].y0 = y_coord;
  }
}

/* Initializes  particles with geometric distribution */
particle_t * initializeGeometric(uint64_t n_input, uint64_t L, double rho, double k, double m, uint64_t & n_placed, random_draw_t *parm)
{
  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */

  /* Each cell in the i-th column of cells contains p(i) = A * rho^i particles */
  double A = n_input * ((1.0-rho) / (1.0-pow(rho,L))) / (double)L;
  n_placed = 0;
  for (uint64_t x=0; x<L; x++) {
    for (uint64_t y=0; y<L; y++) {
      n_placed += random_draw(A * pow(rho, x), parm);
    }
  }

  particle_t * particles = prk::malloc<particle_t>(n_placed);
  if (particles == NULL) {
      throw "ERROR: Could not allocate space for particles";
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  A = n_input * ((1.0-rho) / (1.0-pow(rho,L))) / (double)L;
  for (uint64_t pi=0, x=0; x<L; x++) {
    for (uint64_t y=0; y<L; y++) {
      uint64_t actual_particles = random_draw(A * pow(rho, x), parm);
      for (uint64_t p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution(n_placed, particles);

  return particles;
}

/* Initialize particles with a sinusoidal distribution */
particle_t *initializeSinusoidal(uint64_t n_input, uint64_t L, double k, double m, uint64_t & n_placed, random_draw_t *parm)
{
  const double step = prk::constants::pi() / L;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */

  /* Loop over columns of cells and assign number of particles proportional to sinusodial weight */
  n_placed = 0;
  for (uint64_t x=0; x<L; x++) {
    for (uint64_t y=0; y<L; y++) {
      n_placed += random_draw(2.0*cos(x*step)*cos(x*step)*n_input/(L*L), parm);
    }
  }

  particle_t * particles = prk::malloc<particle_t>(n_placed);
  if (particles == NULL) {
      throw "ERROR: Could not allocate space for particles.";
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  for (uint64_t pi=0,x=0; x<L; x++) {
    for (uint64_t y=0; y<L; y++) {
      const uint64_t actual_particles = random_draw(2.0*cos(x*step)*cos(x*step)*n_input/(L*L), parm);
      for (uint64_t p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution(n_placed, particles);

  return particles;
}

/* Initialize particles with linearly decreasing distribution */
/* The linear function is f(x) = -alpha * x + beta , x in [0,1]*/
particle_t *initializeLinear(uint64_t n_input, uint64_t L, double alpha, double beta, double k, double m, uint64_t & n_placed, random_draw_t *parm)
{
  double step = 1.0/L;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */

  /* Find sum of all weights to normalize the number of particles */
  double total_weight = beta*L-alpha*0.5*step*L*(L-1);

  /* Loop over columns of cells and assign number of particles proportional linear weight */
  n_placed = 0;
  for (uint64_t x=0; x<L; x++) {
    const double current_weight = (beta - alpha * step * ((double) x));
    for (uint64_t y=0; y<L; y++) {
      n_placed += random_draw(n_input * (current_weight/total_weight)/L, parm);
    }
  }

  particle_t * particles = prk::malloc<particle_t>(n_placed);
  if (particles == NULL) {
      throw "ERROR: Could not allocate space for particles.";
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  /* Loop over columns of cells and assign number of particles proportional linear weight */
  for (uint64_t pi=0,x=0; x<L; x++) {
    const double current_weight = (beta - alpha * step * ((double) x));
    for (uint64_t y=0; y<L; y++) {
      uint64_t actual_particles = random_draw(n_input * (current_weight/total_weight)/L, parm);
      for (uint64_t p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution(n_placed, particles);

  return particles;
}

/* Initialize uniformly particles within a "patch" */
particle_t *initializePatch(uint64_t n_input, uint64_t L, bbox_t patch, double k, double m, uint64_t & n_placed, random_draw_t *parm)
{
  uint64_t total_cells, actual_particles;

  /* initialize random number generator */
  LCG_init(parm);

  /* first determine total number of particles, then allocate and place them   */
  total_cells  = (patch.right - patch.left+1)*(patch.top - patch.bottom+1);
  double particles_per_cell = (double) n_input/total_cells;

  /* Iterate over the columns of cells and assign uniform number of particles */
  n_placed = 0;
  for (uint64_t x=0; x<L; x++) {
    for (uint64_t y=0; y<L; y++) {
      uint64_t actual_particles = random_draw(particles_per_cell, parm);
      if (x<patch.left || x>patch.right || y<patch.bottom || y>patch.top) actual_particles = 0;
      n_placed += actual_particles;
    }
  }

  particle_t * particles = prk::malloc<particle_t>(n_placed);
  if (particles == NULL) {
      throw "ERROR: Could not allocate space for particles";
  }

  /* Re-initialize random number generator */
  LCG_init(parm);

  /* Iterate over the columns of cells and assign uniform number of particles */
  for (uint64_t pi=0,x=0; x<L; x++) {
    for (uint64_t y=0; y<L; y++) {
      actual_particles = random_draw(particles_per_cell, parm);
      if (x<patch.left || x>patch.right || y<patch.bottom || y>patch.top) actual_particles = 0;
      for (uint64_t p=0; p<actual_particles; p++,pi++) {
        particles[pi].x = x + REL_X;
        particles[pi].y = y + REL_Y;
        particles[pi].k = k;
        particles[pi].m = m;
      }
    }
  }
  finish_distribution(n_placed, particles);

  return particles;
}

/* Verifies the final position of a particle */
bool verifyParticle(particle_t p, int iterations, double *Qgrid, uint64_t L)
{
  /* Coordinates of the cell containing the particle initially */
  uint64_t y = (uint64_t) p.y0;
  uint64_t x = (uint64_t) p.x0;

  /* According to initial location and charge determine the direction of displacements */
  double disp = (double)(iterations+1)*(2*p.k+1);
  double x_final = ( (p.q * Qgrid[x*(L+1)+y]) > 0) ? p.x0+disp : p.x0-disp;
  double y_final = p.y0 + p.m * (double)(iterations+1);

  /* apply periodicity, making sure we never mod a negative value */
  double x_periodic = fmod(x_final+(double)(iterations+1) *(2*p.k+1)*L, L);
  double y_periodic = fmod(y_final+(double)(iterations+1) *llabs(p.m)*L, L);

  if ( fabs(p.x - x_periodic) > epsilon || fabs(p.y - y_periodic) > epsilon) {
    return false;
  }
  return true;
}

/* Computes the Coulomb force among two charges q1 and q2 */
inline void computeCoulomb(double x_dist, double y_dist, double q1, double q2, double & fx, double & fy)
{
  const double r2 = x_dist * x_dist + y_dist * y_dist;
  const double r = std::sqrt(r2);
  const double f_coulomb = q1 * q2 / r2;

  fx = f_coulomb * x_dist / r; // f_coulomb * cos_theta
  fy = f_coulomb * y_dist / r; // f_coulomb * sin_theta
  return;
}

/* Computes the total Coulomb force on a particle exerted from the charges of the corresponding cell */
inline void computeTotalForce(const particle_t p, const uint64_t L, const double * const Qgrid, double & fx, double & fy)
{
  double tmp_fx, tmp_fy;
  double tmp_res_x{0}, tmp_res_y{0};

  /* Coordinates of the cell containing the particle */
  uint64_t y = (uint64_t) std::floor(p.y);
  uint64_t x = (uint64_t) std::floor(p.x);
  double rel_x = p.x -  x;
  double rel_y = p.y -  y;

  /* Coulomb force from top-left charge */
  computeCoulomb(rel_x, rel_y, p.q, Qgrid[x*(L+1)+y], tmp_fx, tmp_fy);
  tmp_res_x += tmp_fx;
  tmp_res_y += tmp_fy;

  /* Coulomb force from bottom-left charge */
  computeCoulomb(rel_x, 1.0-rel_y, p.q, Qgrid[x*(L+1)+y+1], tmp_fx, tmp_fy);
  tmp_res_x += tmp_fx;
  tmp_res_y -= tmp_fy;

  /* Coulomb force from top-right charge */
  computeCoulomb(1.0-rel_x, rel_y, p.q, Qgrid[(x+1)*(L+1)+y], tmp_fx, tmp_fy);
  tmp_res_x -= tmp_fx;
  tmp_res_y += tmp_fy;

  /* Coulomb force from bottom-right charge */
  computeCoulomb(1.0-rel_x, 1.0-rel_y, p.q, Qgrid[(x+1)*(L+1)+y+1], tmp_fx, tmp_fy);
  tmp_res_x -= tmp_fx;
  tmp_res_y -= tmp_fy;

  fx = tmp_res_x;
  fy = tmp_res_y;
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

  std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
  std::cout << "C++11 Particle-in-Cell execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  uint64_t    L;                 // dimension of grid in cells
  uint64_t    n;                 // total number of particles in the simulation
  geometry particle_mode;        // particle initialization mode (int)
  std::string init_mode;

  double      rho;               // attenuation factor for geometric particle distribution
  double      alpha, beta;       // slope and offset values for linear particle distribution
  int64_t     k, m;              // determine initial horizontal and vertical velocity of
                                 // particles-- (2*k)+1 cells per time step
  bbox_t      grid_patch,        // whole grid
              init_patch;        // subset of grid used for localized initialization


  try {
    if (argc<6) {
      std::cout << "Usage: " << argv[0]
                << " <#simulation steps> <grid size> <#particles>"
                << " <k (particle charge semi-increment)> " << std::endl;
      std::cout << "<m (vertical particle velocity)>" << std::endl;
      std::cout << "          <init mode> <init parameters>]" << std::endl;
      std::cout << "   init mode \"GEOMETRIC\"  parameters: <attenuation factor>" << std::endl;
      std::cout << "             \"SINUSOIDAL\" parameters: none" << std::endl;
      std::cout << "             \"LINEAR\"     parameters: <negative slope> <constant offset>" << std::endl;
      std::cout << "             \"PATCH\"      parameters: <xleft> <xright>  <ybottom> <ytop>" << std::endl;
      throw "";
    }

    iterations = std::atol(argv[1]);
    if (iterations<1) {
      throw "ERROR: Number of time steps must be positive.";
    }
    L = std::atol(argv[2]);
    if (L<1 || L%2) {
      throw "ERROR: Number of grid cells must be positive and even.";
    }

    grid_patch = (bbox_t){0, L+1, 0, L+1};
    n = std::atol(argv[3]);
    if (n<1) {
      throw "ERROR: Number of particles must be positive.";
    }

    k = std::atoi(argv[4]);
    if (k<0) {
      throw "ERROR: Particle semi-charge must be non-negative.";
    }
    m = std::atoi(argv[5]);

    init_mode = std::string(argv[6]);

    /* Initialize particles with geometric distribution */
    if (init_mode.find("GEOMETRIC") != std::string::npos) {
      if (argc<7) {
        throw "ERROR: Not enough arguments for GEOMETRIC.";
      }
      particle_mode = GEOMETRIC;
      rho = std::atof(argv[7]);
    }

    /* Initialize with a sinusoidal particle distribution (single period) */
    if (init_mode.find("SINUSOIDAL") != std::string::npos) {
      particle_mode = SINUSOIDAL;
    }

    /* Initialize particles with linear distribution */
    /* The linear function is f(x) = -alpha * x + beta , x in [0,1]*/
    if (init_mode.find("LINEAR") != std::string::npos) {
      if (argc<8) {
        throw "ERROR: Not enough arguments for LINEAR initialization.";
      }
      particle_mode = LINEAR;
      alpha = std::atof(argv[7]);
      beta  = std::atof(argv[8]);
      if (beta <0 || beta<alpha) {
        throw "ERROR: linear profile gives negative particle density.";
      }
    }

    /* Initialize particles uniformly within a "patch" */
    if (init_mode.find("PATCH") != std::string::npos) {
      if (argc<10) {
        throw "ERROR: Not enough arguments for PATCH initialization.";
      }
      particle_mode = PATCH;
      init_patch.left   = std::atoi(argv[7]);
      init_patch.right  = std::atoi(argv[8]);
      init_patch.bottom = std::atoi(argv[9]);
      init_patch.top    = std::atoi(argv[10]);
      if (bad_patch(&init_patch, &grid_patch)) {
        throw "ERROR: inconsistent initial patch.";
      }
    }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Grid size                      = " << L << std::endl;
  std::cout << "Number of particles requested  = " << n << std::endl;
  std::cout << "Number of time steps           = " << iterations << std::endl;
  std::cout << "Initialization mode            = " << init_mode << std::endl;

  switch (particle_mode)
  {
      case GEOMETRIC:
          std::cout << "  Attenuation factor           = " << rho << std::endl;
          break;
      case SINUSOIDAL:
          break;
      case LINEAR:
          std::cout << "  Negative slope               = " << alpha << std::endl;
          std::cout << "  Offset                       = " << beta << std::endl;
          break;
      case PATCH:
          std::cout << "  Bounding box                 = "
                    << init_patch.left   << ", " << init_patch.right << ", "
                    << init_patch.bottom << ", " << init_patch.top   << std::endl;
          break;
      default:
          throw "ERROR: Unsupported particle initializating mode";
  }
  std::cout << "Particle charge semi-increment = " << k << std::endl;
  std::cout << "Vertical velocity              = " << m << std::endl;

  /* Initialize grid of charges and particles */
  double * Qgrid = initializeGrid(L);

  random_draw_t dice;
  LCG_init(&dice);

  particle_t * particles; // the particles array
  switch (particle_mode)
  {
      case GEOMETRIC:
          particles = initializeGeometric(n, L, rho, k, m, n, &dice);
          break;
      case SINUSOIDAL:
          particles = initializeSinusoidal(n, L, k, m, n, &dice);
          break;
      case LINEAR:
          particles = initializeLinear(n, L, alpha, beta, k, m, n, &dice);
          break;
      case PATCH:
          particles = initializePatch(n, L, init_patch, k, m, n, &dice);
          break;
      default:
          throw "ERROR: Unsupported particle distribution";
  }

  std::cout << "Number of particles placed     = " << n << std::endl;

  double pic_time;
  {
      for (int iter=0; iter<=iterations; iter++) {

          if (iter==1) pic_time = prk::wtime();

          for (uint64_t i = 0; i < n; ++i) {
              double fx = 0.0;
              double fy = 0.0;
              computeTotalForce(particles[i], L, Qgrid, fx, fy);
              const double MASS_INV = 1.0;
              const double ax = fx * MASS_INV;
              const double ay = fy * MASS_INV;

              /* Update particle positions, taking into account periodic boundaries */
              particles[i].x = std::fmod(particles[i].x + particles[i].v_x*DT + 0.5*ax*DT*DT + L, (double)L);
              particles[i].y = std::fmod(particles[i].y + particles[i].v_y*DT + 0.5*ay*DT*DT + L, (double)L);

              /* Update velocities */
              particles[i].v_x += ax * DT;
              particles[i].v_y += ay * DT;
          }
      }
  }

  pic_time = prk::wtime() - pic_time;

  /* Run the verification test */
  int correct = true;
  for (uint64_t i=0; i<n; i++) {
    correct &= verifyParticle(particles[i], iterations, Qgrid, L);
  }

  if (correct) {
      std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
    std::cout << "Simulation time is" << pic_time << "seconds" << std::endl;
#endif
    double avg_time = n*iterations/pic_time;
    std::cout << "Rate (Mparticles_moved/s): " << 1.0e-6*avg_time << std::endl;
  } else {
    std::cout << "Solution does not validate" << std::endl;
  }

  return 0;
}
