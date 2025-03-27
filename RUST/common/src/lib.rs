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
use std::f64::consts::PI;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;
const LCG_SEED: u64 = 27182818285;

#[derive(Debug)]
pub struct RandomDraw {
    pub lcg_seed: u64,
}

impl RandomDraw {
    pub fn new() -> Self {
        RandomDraw { lcg_seed: LCG_SEED }
    }

    pub fn lcg_init(&mut self) {
        self.lcg_seed = LCG_SEED;
    }

    pub fn lcg_next(&mut self, bound: u64) -> u64 {
        let a = LCG_A;
        let c = LCG_C;
        self.lcg_seed = self.lcg_seed.wrapping_mul(a).wrapping_add(c);
        self.lcg_seed % bound
    }

    pub fn random_draw(&mut self, mu: f64) -> u64 {
        let rand_max = u64::MAX;
        let rand_div = 1.0 / rand_max as f64;
        let denominator = u32::MAX as u64;
        let two_pi = 2.0 * PI;

        if mu >= 1.0 {
            let sigma = mu * 0.15;
            let u0 = self.lcg_next(rand_max) as f64 * rand_div;
            let u1 = self.lcg_next(rand_max) as f64 * rand_div;
            let z0 = (-2.0 * u0.ln()).sqrt() * (two_pi * u1).cos();
            return (z0 * sigma + mu + 0.5) as u64;
        } else {
            let numerator = (mu * denominator as f64) as u64;
            self.lcg_next(denominator); // Called but result ignored
            let i1 = self.lcg_next(denominator);
            return (i1 <= numerator) as u64;
        }
    }
}
