from numpy._core import fromnumeric
import torch
import numpy as np

from utils import forward_kinematics_particles, particle_fitness


class PSO_final:
    def __init__(
        self,
        nominal_dh,
        lower_bounds,
        upper_bounds,
        joint_types=None,
        num_particles=32,
        w=0.7,
        c1=1.5,
        c2=1.5,
        position_weight=1.0,
        orientation_weight=1.0,
        device=None,
        dtype=torch.float32,
        vmax_scale=0.1,
        topology="ring",
        neighborhood_size=1,
        
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.nominal_dh = nominal_dh.to(self.device, self.dtype)
        self.lower_bounds = lower_bounds.to(self.device, self.dtype)
        self.upper_bounds = upper_bounds.to(self.device, self.dtype)

        self.N = self.nominal_dh.shape[0]
        self.P = num_particles

        self.joint_types = joint_types or ["R"] * self.N

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

        self.particles = self.initialize_particles()
        self.velocities = torch.zeros_like(self.particles)

        self.pbest_particles = self.particles.clone()
        self.pbest_fitness = torch.full(
            (self.P,),
            float("inf"),
            device=self.device,
            dtype=self.dtype,
        )
        self.pbest_age = torch.zeros(
            self.P,
            device=self.device,
            dtype=self.dtype,
        )

        self.gbest_particle = None
        self.gbest_fitness = torch.tensor(
            float("inf"),
            device=self.device,
            dtype=self.dtype,
        )
        self.vmax_scale = vmax_scale
        self.vmax = self.vmax_scale * (self.upper_bounds - self.lower_bounds)

        self.diversity_history = []
        self.velocity_diversity_history = []

        self.topology = topology
        self.neighborhood_size = neighborhood_size

        self.w_min = 0.4
        self.w_max = 0.9
        self.age_lambda = 0.1

        self.reference_dx = 0.015
        self.reference_dv = 0.0015

        self.current_w = self.w
        self.current_c1 = self.c1
        self.current_c2 = self.c2
        self.current_vmax_scale = self.vmax_scale

    def initialize_particles(self):
        """
        Particle = teljes DH paraméterhalmaz.
        shape: (P, N, 4)
        """

        random_values = torch.rand(
            self.P,
            self.N,
            4,
            device=self.device,
            dtype=self.dtype,
        )

        particles = self.lower_bounds + random_values * (
            self.upper_bounds - self.lower_bounds
        )

        return particles

    def evaluate_particles(self, particles, joint_values, T_measured):
        joint_values = joint_values.to(self.device, self.dtype)
        T_measured = T_measured.to(self.device, self.dtype)

        T_particles = forward_kinematics_particles(
            joint_values=joint_values,
            particles=particles,
            joint_types=self.joint_types,
        )

        fitness = particle_fitness(
            T_measured=T_measured,
            T_particles=T_particles,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
            reduction="sum",
        )

        return fitness

    def evaluate(self, joint_values, T_measured):
        return self.evaluate_particles(
            self.particles,
            joint_values,
            T_measured
        )

    def update_best(self, fitness, joint_values, T_measured):

        pbest_current_fitness = self.evaluate_particles(
            self.pbest_particles,
            joint_values,
            T_measured
        )

        improved = fitness < pbest_current_fitness

        self.pbest_age += 1

        self.pbest_age[improved] = 0

        self.pbest_particles[improved] = self.particles[improved]

        self.pbest_fitness = pbest_current_fitness
        self.pbest_fitness[improved] = fitness[improved]

        best_idx = torch.argmin(self.pbest_fitness)

        self.gbest_fitness = self.pbest_fitness[best_idx].clone()
        self.gbest_particle = self.pbest_particles[best_idx].clone()
    
    def update_particles(self):

        self.update_fuzzy_parameters()

        if self.gbest_particle is None:
            return

        r1 = torch.rand_like(self.particles)
        r2 = torch.rand_like(self.particles)

        cognitive = (
            self.current_c1
            * r1
            * (self.pbest_particles - self.particles)
        )

        social = torch.zeros_like(self.particles)

        for i in range(self.P):

            local_best = self.compute_local_best(i)

            social[i] = (
                self.current_c2
                * r2[i]
                * (
                    local_best
                    - self.particles[i]
                )
            )

        self.velocities = (
            self.current_w * self.velocities
            + cognitive
            + social
        )

        self.velocities = torch.clamp(
            self.velocities,
            min=-self.vmax,
            max=self.vmax
        )

        self.particles = self.particles + self.velocities

        self.particles = torch.maximum(
            torch.minimum(self.particles, self.upper_bounds),
            self.lower_bounds,
        )

    def step(self, joint_values, T_measured):
        fitness = self.evaluate(joint_values, T_measured)

        self.update_best(
            fitness=fitness,
            joint_values=joint_values,
            T_measured=T_measured
        )

        self.update_particles()

        position_diversity, velocity_diversity = self.compute_diversity()

        self.diversity_history.append(position_diversity.item())
        self.velocity_diversity_history.append(velocity_diversity.item())

        return self.gbest_particle, self.gbest_fitness

    def optimize(self, joint_values, T_measured, iterations=100, print_every=10):
        history = []

        for it in range(iterations):
            _, best_fitness = self.step(joint_values, T_measured)
            history.append(best_fitness.item())

            if it % print_every == 0:
                print(f"Iter {it:04d} | Best fitness: {best_fitness.item():.8f}")

        return self.gbest_particle, self.gbest_fitness, history

    def compute_diversity(self):
        centroid = torch.mean(self.particles, dim=0, keepdim=True)
        position_diversity = torch.mean(
            torch.norm(self.particles - centroid, dim=(1, 2))
        )

        velocity_centroid = torch.mean(self.velocities, dim=0, keepdim=True)

        velocity_diversity = torch.mean(
            torch.norm(self.velocities - velocity_centroid, dim=(1, 2))
        )

        return position_diversity, velocity_diversity


    def get_neighbors(self, idx):

        if self.topology == "global":
            return torch.arange(
                self.P,
                device=self.device
            )

        elif self.topology == "ring":

            neighbors = []

            for offset in range(
                -self.neighborhood_size,
                self.neighborhood_size + 1
            ):

                neighbor_idx = (idx + offset) % self.P
                neighbors.append(neighbor_idx)

            return torch.tensor(
                neighbors,
                device=self.device
            )

        else:
            raise ValueError(
                f"Unknown topology: {self.topology}"
            )


    def compute_local_best(self, particle_idx):

        neighbors = self.get_neighbors(particle_idx)

        neighbor_particles = self.pbest_particles[neighbors]
        neighbor_fitness = self.pbest_fitness[neighbors]
        neighbor_ages = self.pbest_age[neighbors]

        age_weights = torch.exp(-(self.age_lambda) * neighbor_ages)

        scores = neighbor_fitness * age_weights

        best_local_idx = torch.argmin(scores)

        local_best = neighbor_particles[best_local_idx]

        return local_best

    def update_fuzzy_parameters(self):

        if len(self.diversity_history) == 0:
            return

        dx = self.diversity_history[-1]
        dv = self.velocity_diversity_history[-1]

        dx_n = min(dx / self.reference_dx, 1.0)
        dv_n = min(dv / self.reference_dv, 1.0)

        collapse = (1.0 - dx_n) * (1.0 - dv_n)
        active_exploration = dx_n * dv_n
        frozen_spread = dx_n * (1.0 - dv_n)
        unstable_motion = (1.0 - dx_n) * dv_n

        self.current_w = (
            collapse * self.w_max +
            active_exploration * self.w_min +
            frozen_spread * 0.75 +
            unstable_motion * 0.55
        )

        self.current_c1 = (
            collapse * 2.0 +
            active_exploration * 1.4 +
            frozen_spread * 1.2 +
            unstable_motion * 1.5
        )

        self.current_c2 = (
            collapse * 1.0 +
            active_exploration * 1.8 +
            frozen_spread * 2.0 +
            unstable_motion * 1.2
        )

        self.current_vmax_scale = (
            collapse * 0.15 +
            active_exploration * 0.06 +
            frozen_spread * 0.08 +
            unstable_motion * 0.05
        )

        self.vmax = self.current_vmax_scale * (
            self.upper_bounds - self.lower_bounds
        )