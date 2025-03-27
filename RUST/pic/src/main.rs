/*
Copyright (c) 2015, Intel Corporation
Copyright (c) 2025, Christian Asch

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

USAGE:   <progname> -s <#simulation steps> -g <grid size> -t <#particles> \
                    -p <horizontal velocity> -v <vertical velocity>    \
                    <init mode> <init parameters>

         The output consists of diagnostics to make sure the
         algorithm worked, and of timing statistics.

HISTORY: - Written by Evangelos Georganas, August 2015.
         - RvdW: Refactored to make the code PRK conforming, December 2015
         - Ported to Rust by Christian Asch, March 2025

**********************************************************************************/

use clap::{Parser, Subcommand};
use std::f64::consts::PI;
use std::time::Instant;

const Q: f64 = 1.0;
const DT: f64 = 1.0;
const MASS_INV: f64 = 1.0;
const REL_X: f64 = 0.5;
const REL_Y: f64 = 0.5;
const EPSILON: f64 = 0.000001;

/// Particle initialization mode
#[derive(Subcommand, Debug, Clone)]
enum InitStyle {
    Geometric {
        /// Attenuation Factor
        #[arg(short, long, value_name = "rho")]
        attenuation_factor: f64,
    },
    Sinusoidal,
    Linear {
        /// Negative slope
        #[arg(short, long, value_name = "alpha")]
        negative_slope: f64,
        /// Constant offset
        #[arg(short, long, value_name = "beta")]
        constant_offset: f64,
    },
    Patch {
        #[arg(short, long)]
        xleft: u64,
        #[arg(short, long)]
        xright: u64,
        #[arg(short, long)]
        ybottom: u64,
        #[arg(short, long)]
        ytop: u64,
    },
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    /// Total number of simulation steps
    pub iterations: u64,
    #[arg(short, long, value_name = "L")]
    /// Dimension of grid in cells
    pub grid_size: u64,
    /// Total number of generated particles
    #[arg(short, long)]
    pub total_particles: u64,
    /// Initial horizontal velocity of particles
    #[arg(short, long, value_name = "k")]
    pub particle_charge_semi_increment: u64,
    /// Initial vertical velocity of particles
    #[arg(short, long, value_name = "m")]
    pub vertical_particle_velocity: u64,
    #[command(subcommand)]
    init_style: InitStyle,
}

struct BoundingBox {
    left: u64,
    right: u64,
    bottom: u64,
    top: u64,
}

fn bad_patch(patch: &BoundingBox, patch_contain: &BoundingBox) -> bool {
    if patch.left >= patch.right || patch.bottom >= patch.top {
        return true;
    }
    if patch.left < patch_contain.left
        || patch.right > patch_contain.right
        || patch.bottom < patch_contain.bottom
        || patch.top > patch_contain.top
    {
        return true;
    }
    return false;
}

#[derive(Default, Debug)]
struct Particle {
    x: f64,
    y: f64,
    v_x: f64,
    v_y: f64,
    q: f64,
    x0: f64,
    y0: f64,
    k: f64,
    m: f64,
}

#[derive(Debug)]
struct Grid<T> {
    data: Vec<T>,
    cols: usize,
}

impl<T: Copy> Grid<T> {
    fn new(default_value: T, dimensions: usize) -> Self {
        Grid::<T> {
            data: vec![default_value; dimensions * dimensions],
            cols: dimensions,
        }
    }

    fn get(&self, col_idx: usize, row_idx: usize) -> T {
        let index = col_idx * self.cols + row_idx;
        self.data[index]
    }

    fn set(&mut self, col_idx: usize, row_idx: usize, value: T) {
        let index = col_idx * self.cols + row_idx;
        self.data[index] = value;
    }
}

fn initialize_grid(grid_size: usize) -> Grid<f64> {
    let mut grid = Grid::<f64>::new(0f64, grid_size + 1);
    for col_idx in 0..(grid_size + 1) {
        for row_idx in 0..(grid_size + 1) {
            let value = match col_idx % 2 {
                0 => Q,
                _ => -Q,
            };
            grid.set(col_idx, row_idx, value);
        }
    }
    grid
}

fn finalize_distribution(particles: &mut [Particle]) {
    for particle in particles {
        let x_coord = particle.x;
        let y_coord = particle.y;
        let rel_x = x_coord % 1.0;
        let rel_y = y_coord % 1.0;
        let x = x_coord as u64;
        let r1_sq = rel_y * rel_y + rel_x * rel_x;
        let r2_sq = rel_y * rel_y + (1.0 - rel_x) * (1.0 - rel_x);
        let cos_theta = rel_x / r1_sq.sqrt();
        let cos_phi = (1.0 - rel_x) / r2_sq.sqrt();
        let base_charge = 1.0 / ((DT * DT) * Q * (cos_theta / r1_sq + cos_phi / r2_sq));

        particle.v_x = 0.0;
        particle.v_y = particle.m / DT;

        let q_val = (2.0 * particle.k + 1.0) * base_charge;

        particle.q = match x % 2 {
            0 => q_val,
            _ => -q_val,
        };
        particle.x0 = x_coord;
        particle.y0 = y_coord;
    }
}

fn initialize_linear(
    n_input: u64,
    grid_size: u64,
    alpha: f64,
    beta: f64,
    horizontal_speed: f64,
    vertical_speed: f64,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();
    let mut particles = Vec::<Particle>::new();

    let step = 1.0 / (grid_size as f64);
    let total_weight =
        beta * grid_size as f64 - alpha * 0.5 * step * grid_size as f64 * (grid_size as f64 - 1.0);
    dice.lcg_init();
    for x in 0..grid_size {
        let current_weight = beta - alpha * step * (x as f64);
        for y in 0..grid_size {
            let part_num = dice
                .random_draw(n_input as f64 * (current_weight / total_weight) / grid_size as f64);
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }
    finalize_distribution(particles.as_mut_slice());
    particles
}

fn initialize_patch(
    n_input: u64,
    grid_size: u64,
    patch: &BoundingBox,
    horizontal_speed: f64,
    vertical_speed: f64,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();

    let total_cells = (patch.right - patch.left + 1) * (patch.top - patch.bottom + 1);
    let particles_per_cell = (n_input as f64) / (total_cells as f64);

    let mut particles = Vec::<Particle>::new();

    dice.lcg_init();

    for x in 0..grid_size {
        for y in 0..grid_size {
            let mut part_num = dice.random_draw(particles_per_cell);
            if x < patch.left || x > patch.right || y < patch.bottom || y > patch.top {
                part_num = 0;
            }
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }
    finalize_distribution(particles.as_mut_slice());

    particles
}

fn initialize_geometric(
    n_input: u64,
    grid_size: u64,
    rho: f64,
    horizontal_speed: f64,
    vertical_speed: f64,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();

    let mut particles = Vec::<Particle>::new();
    dice.lcg_init();

    let factor =
        n_input as f64 * ((1.0 - rho) / (1.0 - rho.powf(grid_size as f64))) / (grid_size as f64);

    for x in 0..grid_size {
        for y in 0..grid_size {
            let part_num = dice.random_draw(factor * rho.powf(x as f64));
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }

    finalize_distribution(particles.as_mut_slice());
    particles
}

fn initialize_sinusoidal(
    n_input: u64,
    grid_size: u64,
    horizontal_speed: f64,
    vertical_speed: f64,
) -> Vec<Particle> {
    let mut dice = common::RandomDraw::new();
    let step = PI / (grid_size as f64);
    dice.lcg_init();

    let mut particles = Vec::<Particle>::new();

    for x in 0..grid_size {
        for y in 0..grid_size {
            let val = (x as f64 * step).cos();
            let part_num =
                dice.random_draw(2.0 * n_input as f64 * val * val / (grid_size * grid_size) as f64);
            for _p in 0..part_num {
                let mut particle = Particle::default();
                particle.x = x as f64 + REL_X;
                particle.y = y as f64 + REL_Y;
                particle.k = horizontal_speed;
                particle.m = vertical_speed;
                particles.push(particle);
            }
        }
    }

    finalize_distribution(particles.as_mut_slice());
    particles
}

fn compute_coulomb(x_dist: f64, y_dist: f64, q1: f64, q2: f64) -> (f64, f64) {
    let r2 = x_dist.powi(2) + y_dist.powi(2);
    let r = r2.sqrt();
    let f_coulomb = q1 * q2 / r2;

    let fx = f_coulomb * x_dist / r;
    let fy = f_coulomb * y_dist / r;

    (fx, fy)
}

fn compute_total_force(particle: &mut Particle, grid: &Grid<f64>) -> (f64, f64) {
    let x = particle.x.floor() as usize;
    let y = particle.y.floor() as usize;
    let rel_x = particle.x - particle.x.floor();
    let rel_y = particle.y - particle.y.floor();
    let mut temp_res_x = 0.0;
    let mut temp_res_y = 0.0;

    let (temp_fx, temp_fy) = compute_coulomb(rel_x, rel_y, particle.q, grid.get(x, y));
    temp_res_x += temp_fx;
    temp_res_y += temp_fy;

    let (temp_fx, temp_fy) = compute_coulomb(rel_x, 1.0 - rel_y, particle.q, grid.get(x, y + 1));
    temp_res_x += temp_fx;
    temp_res_y -= temp_fy;

    let (temp_fx, temp_fy) = compute_coulomb(1.0 - rel_x, rel_y, particle.q, grid.get(x + 1, y));
    temp_res_x -= temp_fx;
    temp_res_y += temp_fy;

    let (temp_fx, temp_fy) =
        compute_coulomb(1.0 - rel_x, 1.0 - rel_y, particle.q, grid.get(x + 1, y + 1));
    temp_res_x -= temp_fx;
    temp_res_y -= temp_fy;

    let fx = temp_res_x;
    let fy = temp_res_y;
    (fx, fy)
}

enum Status {
    Failure,
    Success,
}

fn verify_particle(
    particle: &Particle,
    iterations: u64,
    grid: &Grid<f64>,
    grid_size: u64,
) -> Status {
    let disp = (iterations + 1) as f64 * (2.0 * particle.k + 1.0);
    let x_final = match particle.q * grid.get(particle.x0 as usize, particle.y0 as usize) > 0.0 {
        true => particle.x0 + disp,
        false => particle.x0 - disp,
    };
    let y_final = particle.y0 + particle.m * (iterations + 1) as f64;
    let grid_size_f = grid_size as f64;
    let total_it = iterations as f64;

    let x_periodic = (x_final + total_it * (2.0 * particle.k + 1.0) * grid_size_f) % grid_size_f;
    let y_periodic = (y_final + total_it * (particle.m.abs()) * grid_size_f) % grid_size_f;

    if (particle.x - x_periodic).abs() > EPSILON || (particle.y - y_periodic).abs() > EPSILON {
        Status::Failure
    } else {
        Status::Success
    }
}

fn main() {
    let args = Args::parse();
    println!("Parallel Research Kernels");
    println!("Serial Particle-in-Cell execution on 2D grid");

    let grid_size = args.grid_size;
    let grid_patch = BoundingBox {
        left: 0,
        right: grid_size + 1,
        bottom: 0,
        top: grid_size,
    };
    let grid = initialize_grid(grid_size as usize);
    println!("Grid size                      = {}", args.grid_size);
    println!("Number of particles requested  = {}", args.total_particles);
    println!("Number of time steps           = {}", args.iterations);
    print!("Initialization mode");
    match args.init_style {
        InitStyle::Sinusoidal => println!("            = SINUSOIDAL"),
        InitStyle::Geometric { attenuation_factor } => {
            println!("            = GEOMETRIC");
            println!("  Attenuation factor           = {:.6}", attenuation_factor)
        }
        InitStyle::Linear {
            negative_slope,
            constant_offset,
        } => {
            println!("            = LINEAR");
            println!("  Negative slope               = {:.6}", negative_slope);
            println!("  Offset                       = {:.6}", constant_offset);
        }
        InitStyle::Patch {
            xleft,
            xright,
            ybottom,
            ytop,
        } => {
            println!("            = PATCH");
            println!(
                "  Bounding box                 = {}, {}, {}, {}",
                xleft, xright, ybottom, ytop
            );
        }
    };
    println!(
        "Particle charge semi-increment = {}",
        args.particle_charge_semi_increment
    );
    println!(
        "Vertical velocity              = {}",
        args.vertical_particle_velocity
    );

    let k = args.particle_charge_semi_increment as f64;
    let m = args.vertical_particle_velocity as f64;
    let mut particles = match args.init_style {
        InitStyle::Geometric { attenuation_factor } => {
            initialize_geometric(args.total_particles, grid_size, attenuation_factor, k, m)
        }
        InitStyle::Sinusoidal => initialize_sinusoidal(args.total_particles, grid_size, k, m),
        InitStyle::Linear {
            negative_slope,
            constant_offset,
        } => {
            if constant_offset < 0.0 || constant_offset < negative_slope {
                panic!("ERROR: linear profile gives negative density");
            }
            initialize_linear(
                args.total_particles,
                grid_size,
                negative_slope,
                constant_offset,
                k,
                m,
            )
        }
        InitStyle::Patch {
            xleft,
            xright,
            ybottom,
            ytop,
        } => {
            let patch = BoundingBox {
                left: xleft,
                right: xright,
                bottom: ybottom,
                top: ytop,
            };
            if bad_patch(&patch, &grid_patch) {
                panic!("ERROR: inconsistent initial patch");
            };
            initialize_patch(args.total_particles, grid_size, &patch, k, m)
        }
    };
    println!("Number of particles placed     = {}", particles.len());
    let timer = Instant::now();
    let mut t0 = timer.elapsed();

    for it in 0..args.iterations + 1 {
        if it == 1 {
            t0 = timer.elapsed();
        }
        for particle in particles.iter_mut() {
            let (fx, fy) = compute_total_force(particle, &grid);
            let ax = fx * MASS_INV;
            let ay = fy * MASS_INV;
            let x_disp = particle.x + particle.v_x * DT + 0.5 * ax * DT.powi(2) + grid_size as f64;
            let y_disp = particle.y + particle.v_y * DT + 0.5 * ay * DT.powi(2) + grid_size as f64;
            particle.x = x_disp % grid_size as f64;
            particle.y = y_disp % grid_size as f64;

            particle.v_x += ax * DT;
            particle.v_y += ay * DT;
        }
    }
    let t1 = timer.elapsed();
    let dt = (t1.checked_sub(t0)).unwrap();
    let pic_time = dt.as_secs_f64();

    let mut result = true;
    for particle in particles.iter() {
        match verify_particle(particle, args.iterations, &grid, grid_size) {
            Status::Failure => {
                result = false;
                break;
            }
            _ => (),
        };
    }

    match result {
        true => {
            let average_time = (args.iterations * args.total_particles) as f64 / pic_time;
            println!("Solution validates");
            println!("Rate (Mparticles_moved/s): {:.6}", 1e-6 * average_time);
        }
        false => println!("Solution does not validate"),
    };
}
