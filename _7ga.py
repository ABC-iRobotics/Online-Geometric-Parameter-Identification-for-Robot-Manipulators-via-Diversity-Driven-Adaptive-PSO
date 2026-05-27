import torch

from utils import forward_kinematics_particles, particle_fitness


class GAOptimizer:
    def __init__(
        self,
        nominal_dh,
        lower_bounds,
        upper_bounds,
        joint_types=None,
        num_particles=256,
        w=0.7,              # benchmark kompatibilitás miatt
        c1=1.5,             # nem használjuk
        c2=1.5,             # nem használjuk
        position_weight=1.0,
        orientation_weight=0.1,
        device=None,
        dtype=torch.float32,
        vmax_scale=0.1,     # mutation skálához használjuk
        crossover_rate=0.9,
        mutation_rate=0.2,
        mutation_scale=0.05,
        elite_count=2,
        tournament_size=3,
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

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite_count = elite_count
        self.tournament_size = tournament_size

        self.population = self.initialize_population()
        self.previous_population = self.population.clone()

        self.best_particle = None
        self.best_fitness = torch.tensor(
            float("inf"),
            device=self.device,
            dtype=self.dtype,
        )

        self.diversity_history = []
        self.velocity_diversity_history = []

        self.param_range = self.upper_bounds - self.lower_bounds
        self.mutation_std = self.mutation_scale * self.param_range

    def initialize_population(self):
        random_values = torch.rand(
            self.P,
            self.N,
            4,
            device=self.device,
            dtype=self.dtype,
        )

        population = self.lower_bounds + random_values * (
            self.upper_bounds - self.lower_bounds
        )

        return population

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

    def tournament_selection(self, fitness):
        candidate_idx = torch.randint(
            low=0,
            high=self.P,
            size=(self.P, self.tournament_size),
            device=self.device,
        )

        candidate_fitness = fitness[candidate_idx]

        best_local_idx = torch.argmin(candidate_fitness, dim=1)

        selected_idx = candidate_idx[
            torch.arange(self.P, device=self.device),
            best_local_idx,
        ]

        return selected_idx

    def arithmetic_crossover(self, parents_a, parents_b):
        alpha = torch.rand(
            self.P,
            1,
            1,
            device=self.device,
            dtype=self.dtype,
        )

        children = alpha * parents_a + (1.0 - alpha) * parents_b

        crossover_mask = (
            torch.rand(
                self.P,
                1,
                1,
                device=self.device,
                dtype=self.dtype,
            )
            < self.crossover_rate
        )

        children = torch.where(
            crossover_mask,
            children,
            parents_a,
        )

        return children

    def mutate(self, children):
        mutation_mask = (
            torch.rand_like(children)
            < self.mutation_rate
        )

        noise = self.mutation_std.unsqueeze(0) * torch.randn_like(children)

        mutated = torch.where(
            mutation_mask,
            children + noise,
            children,
        )

        mutated = torch.maximum(
            torch.minimum(mutated, self.upper_bounds),
            self.lower_bounds,
        )

        return mutated

    def step(self, joint_values, T_measured):
        self.previous_population = self.population.clone()

        fitness = self.evaluate_particles(
            self.population,
            joint_values,
            T_measured,
        )

        sorted_idx = torch.argsort(fitness)
        elite_idx = sorted_idx[:self.elite_count]

        elites = self.population[elite_idx].clone()
        elite_fitness = fitness[elite_idx].clone()

        best_idx = elite_idx[0]

        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx].clone()
            self.best_particle = self.population[best_idx].clone()

        selected_a = self.tournament_selection(fitness)
        selected_b = self.tournament_selection(fitness)

        parents_a = self.population[selected_a]
        parents_b = self.population[selected_b]

        children = self.arithmetic_crossover(parents_a, parents_b)
        children = self.mutate(children)

        children_fitness = self.evaluate_particles(
            children,
            joint_values,
            T_measured,
        )

        new_population = children.clone()

        replacement_idx = torch.argsort(children_fitness)[-self.elite_count:]

        new_population[replacement_idx] = elites

        self.population = new_population

        # Friss best az új populáció alapján is
        new_fitness = self.evaluate_particles(
            self.population,
            joint_values,
            T_measured,
        )

        new_best_idx = torch.argmin(new_fitness)

        if new_fitness[new_best_idx] < self.best_fitness:
            self.best_fitness = new_fitness[new_best_idx].clone()
            self.best_particle = self.population[new_best_idx].clone()

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
            keepdim=True,
        )

        velocity_diversity = torch.mean(
            torch.norm(
                pseudo_velocity - velocity_centroid,
                dim=(1, 2),
            )
        )

        return position_diversity, velocity_diversity

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
                    f"Best fitness: {best_fitness.item():.8f}"
                )

        return self.best_particle, self.best_fitness, history