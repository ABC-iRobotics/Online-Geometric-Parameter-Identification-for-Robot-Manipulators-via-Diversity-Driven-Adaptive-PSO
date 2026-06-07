import torch

from utils import forward_kinematics_particles, particle_fitness


class DEOptimizer:
    def __init__(
        self,
        nominal_dh,
        lower_bounds,
        upper_bounds,
        joint_types=None,
        num_particles=32,
        w=0.7,      # benchmark kompatibilitás miatt marad
        c1=1.5,     # nem használjuk
        c2=1.5,     # nem használjuk
        position_weight=1.0,
        orientation_weight=1.0,
        device=None,
        dtype=torch.float32,
        vmax_scale=0.1,  # nem használjuk
        F=0.7,
        CR=0.7,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.nominal_dh = nominal_dh.to(self.device, self.dtype)
        self.lower_bounds = lower_bounds.to(self.device, self.dtype)
        self.upper_bounds = upper_bounds.to(self.device, self.dtype)

        self.N = self.nominal_dh.shape[0]
        self.P = num_particles

        self.joint_types = joint_types or ["R"] * self.N

        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

        self.F = F
        self.CR = CR

        self.population = self.initialize_population()

        self.best_particle = None
        self.best_fitness = torch.tensor(
            float("inf"),
            device=self.device,
            dtype=self.dtype,
        )

        self.diversity_history = []
        self.velocity_diversity_history = []

        self.previous_population = self.population.clone()

    def initialize_population(self):
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

    def mutate(self):
        mutants = torch.zeros_like(self.population)

        for i in range(self.P):
            candidates = torch.randperm(
                self.P,
                device=self.device
            )

            candidates = candidates[candidates != i]

            r1, r2, r3 = candidates[:3]

            mutant = (
                self.population[r1]
                + self.F * (
                    self.population[r2]
                    - self.population[r3]
                )
            )

            mutant = torch.maximum(
                torch.minimum(mutant, self.upper_bounds),
                self.lower_bounds,
            )

            mutants[i] = mutant

        return mutants

    def crossover(self, mutants):
        random_mask = torch.rand_like(self.population) < self.CR

        trial = torch.where(
            random_mask,
            mutants,
            self.population
        )

        return trial

    def step(self, joint_values, T_measured):
        self.previous_population = self.population.clone()

        current_fitness = self.evaluate_particles(
            self.population,
            joint_values,
            T_measured,
        )

        mutants = self.mutate()
        trial_population = self.crossover(mutants)

        trial_fitness = self.evaluate_particles(
            trial_population,
            joint_values,
            T_measured,
        )

        improved = trial_fitness < current_fitness

        self.population[improved] = trial_population[improved]
        current_fitness[improved] = trial_fitness[improved]

        best_idx = torch.argmin(current_fitness)

        self.best_fitness = current_fitness[best_idx].clone()
        self.best_particle = self.population[best_idx].clone()

        position_diversity, velocity_diversity = self.compute_diversity()

        self.diversity_history.append(position_diversity.item())
        self.velocity_diversity_history.append(velocity_diversity.item())

        return self.best_particle, self.best_fitness

    def compute_diversity(self):
        centroid = torch.mean(self.population, dim=0, keepdim=True)

        position_diversity = torch.mean(
            torch.norm(self.population - centroid, dim=(1, 2))
        )

        pseudo_velocity = self.population - self.previous_population

        velocity_centroid = torch.mean(
            pseudo_velocity,
            dim=0,
            keepdim=True
        )

        velocity_diversity = torch.mean(
            torch.norm(
                pseudo_velocity - velocity_centroid,
                dim=(1, 2)
            )
        )

        return position_diversity, velocity_diversity

    def optimize(self, joint_values, T_measured, iterations=100, print_every=10):
        history = []

        for it in range(iterations):
            _, best_fitness = self.step(joint_values, T_measured)
            history.append(best_fitness.item())

            if it % print_every == 0:
                print(
                    f"Iter {it:04d} | "
                    f"Best fitness: {best_fitness.item():.8f}"
                )

        return self.best_particle, self.best_fitness, history