import torch

from utils import forward_kinematics_particles, particle_fitness


class PSO:
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
        vmax_scale=0.1
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

        self.pbest_particles[improved] = self.particles[improved]
        self.pbest_fitness = pbest_current_fitness
        self.pbest_fitness[improved] = fitness[improved]

        best_idx = torch.argmin(self.pbest_fitness)

        self.gbest_fitness = self.pbest_fitness[best_idx].clone()
        self.gbest_particle = self.pbest_particles[best_idx].clone()
    
    def update_particles(self):
        if self.gbest_particle is None:
            return

        r1 = torch.rand_like(self.particles)
        r2 = torch.rand_like(self.particles)

        cognitive = self.c1 * r1 * (self.pbest_particles - self.particles)
        social = self.c2 * r2 * (self.gbest_particle.unsqueeze(0) - self.particles)

        self.velocities = self.w * self.velocities + cognitive + social

        self.velocities = torch.clamp(
            self.velocities,
            min=-self.vmax,
            max=self.vmax
        )

        self.particles = self.particles + self.velocities

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
