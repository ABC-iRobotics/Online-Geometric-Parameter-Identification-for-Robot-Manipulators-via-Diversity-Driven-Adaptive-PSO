import torch

from utils import forward_kinematics_particles, particle_fitness


class APSO:
    def __init__(
        self,
        nominal_dh,
        lower_bounds,
        upper_bounds,
        joint_types=None,
        num_particles=256,
        w=0.7,
        c1=1.5,
        c2=1.5,
        position_weight=1.0,
        orientation_weight=0.1,
        device=None,
        dtype=torch.float32,
        vmax_scale=0.1,
        c_min=1.5,
        c_max=2.5,
        delta_c=0.05,
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

        self.c_min = c_min
        self.c_max = c_max
        self.delta_c = delta_c

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
        self.vmax = self.vmax_scale * (
            self.upper_bounds - self.lower_bounds
        )

        self.diversity_history = []
        self.velocity_diversity_history = []

        self.inertia_history = []
        self.c1_history = []
        self.c2_history = []
        self.evolutionary_factor_history = []
        self.state_history = []

    def initialize_particles(self):
        random_values = torch.rand(
            self.P,
            self.N,
            4,
            device=self.device,
            dtype=self.dtype,
        )

        return self.lower_bounds + random_values * (
            self.upper_bounds - self.lower_bounds
        )

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
            T_measured,
        )

    def update_best(self, fitness, joint_values, T_measured):
        pbest_current_fitness = self.evaluate_particles(
            self.pbest_particles,
            joint_values,
            T_measured,
        )

        improved = fitness < pbest_current_fitness

        self.pbest_particles[improved] = self.particles[improved]
        self.pbest_fitness = pbest_current_fitness
        self.pbest_fitness[improved] = fitness[improved]

        best_idx = torch.argmin(self.pbest_fitness)

        self.gbest_fitness = self.pbest_fitness[best_idx].clone()
        self.gbest_particle = self.pbest_particles[best_idx].clone()
        self.gbest_index = best_idx

    def compute_evolutionary_factor(self):
        """
        APSO evolutionary factor based on population distribution.

        For each particle, compute the mean distance to all other particles.
        Then use the distance of the global-best particle to estimate the
        evolutionary state of the swarm.
        """

        flat_particles = self.particles.view(self.P, -1)

        distances = torch.cdist(
            flat_particles,
            flat_particles,
            p=2,
        )

        mean_distances = torch.sum(distances, dim=1) / (self.P - 1)

        d_min = torch.min(mean_distances)
        d_max = torch.max(mean_distances)

        gbest_idx = torch.argmin(self.pbest_fitness)
        d_g = mean_distances[gbest_idx]

        eps = torch.tensor(
            1e-12,
            device=self.device,
            dtype=self.dtype,
        )

        f = (d_g - d_min) / (d_max - d_min + eps)

        f = torch.clamp(f, 0.0, 1.0)

        return f.item()

    def compute_adaptive_inertia(self, evolutionary_factor):
        """
        APSO nonlinear inertia adaptation.

        This follows the commonly used APSO mapping:
        low f  -> lower inertia
        high f -> higher inertia
        """

        f = torch.tensor(
            evolutionary_factor,
            device=self.device,
            dtype=self.dtype,
        )

        w = 1.0 / (
            1.0 + 1.5 * torch.exp(-2.6 * f)
        )

        return float(w.item())

    def estimate_state(self, f):
        """
        Simplified state estimation based on APSO evolutionary factor.

        The original paper uses fuzzy classification of four evolutionary
        states. For a compact benchmark implementation, crisp intervals are
        used to approximate these states.
        """

        if f < 0.25:
            return "convergence"
        elif f < 0.50:
            return "exploitation"
        elif f < 0.75:
            return "exploration"
        else:
            return "jumping_out"

    def adapt_acceleration_coefficients(self, state):
        """
        Adaptive c1/c2 control inspired by APSO state-dependent behavior.
        """

        dc = self.delta_c

        if state == "exploration":
            self.c1 += dc
            self.c2 -= dc

        elif state == "exploitation":
            self.c1 += 0.5 * dc
            self.c2 -= 0.5 * dc

        elif state == "convergence":
            self.c1 += 0.5 * dc
            self.c2 += 0.5 * dc

        elif state == "jumping_out":
            self.c1 -= dc
            self.c2 += dc

        self.c1 = float(
            max(self.c_min, min(self.c1, self.c_max))
        )

        self.c2 = float(
            max(self.c_min, min(self.c2, self.c_max))
        )

        total = self.c1 + self.c2

        if total > 4.0:
            scale = 4.0 / total
            self.c1 *= scale
            self.c2 *= scale

    def update_particles(self):
        if self.gbest_particle is None:
            return

        evolutionary_factor = self.compute_evolutionary_factor()
        state = self.estimate_state(evolutionary_factor)

        self.w = self.compute_adaptive_inertia(evolutionary_factor)
        self.adapt_acceleration_coefficients(state)

        self.evolutionary_factor_history.append(evolutionary_factor)
        self.state_history.append(state)
        self.inertia_history.append(self.w)
        self.c1_history.append(self.c1)
        self.c2_history.append(self.c2)

        r1 = torch.rand_like(self.particles)
        r2 = torch.rand_like(self.particles)

        cognitive = self.c1 * r1 * (
            self.pbest_particles - self.particles
        )

        social = self.c2 * r2 * (
            self.gbest_particle.unsqueeze(0) - self.particles
        )

        self.velocities = (
            self.w * self.velocities
            + cognitive
            + social
        )

        self.velocities = torch.clamp(
            self.velocities,
            min=-self.vmax,
            max=self.vmax,
        )

        self.particles = self.particles + self.velocities

        self.particles = torch.maximum(
            torch.minimum(self.particles, self.upper_bounds),
            self.lower_bounds,
        )

    def compute_diversity(self):
        centroid = torch.mean(self.particles, dim=0, keepdim=True)

        position_diversity = torch.mean(
            torch.norm(self.particles - centroid, dim=(1, 2))
        )

        velocity_centroid = torch.mean(
            self.velocities,
            dim=0,
            keepdim=True,
        )

        velocity_diversity = torch.mean(
            torch.norm(self.velocities - velocity_centroid, dim=(1, 2))
        )

        return position_diversity, velocity_diversity

    def step(self, joint_values, T_measured):
        fitness = self.evaluate(joint_values, T_measured)

        self.update_best(
            fitness=fitness,
            joint_values=joint_values,
            T_measured=T_measured,
        )

        self.update_particles()

        position_diversity, velocity_diversity = self.compute_diversity()

        self.diversity_history.append(position_diversity.item())
        self.velocity_diversity_history.append(velocity_diversity.item())

        return self.gbest_particle, self.gbest_fitness

    def optimize(
        self,
        joint_values,
        T_measured,
        iterations=100,
        print_every=10,
    ):
        history = []

        for it in range(iterations):
            _, best_fitness = self.step(joint_values, T_measured)
            history.append(best_fitness.item())

            if it % print_every == 0:
                print(
                    f"Iter {it:04d} | "
                    f"Best fitness: {best_fitness.item():.8f} | "
                    f"w: {self.w:.4f} | "
                    f"c1: {self.c1:.3f} | "
                    f"c2: {self.c2:.3f}"
                )

        return self.gbest_particle, self.gbest_fitness, history