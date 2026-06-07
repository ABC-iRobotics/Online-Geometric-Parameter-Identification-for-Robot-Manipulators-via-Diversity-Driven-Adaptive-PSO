from numpy import dtype
import numpy as np
import time
import torch
import csv

from utils import *
from _1pso import PSO
from _2pso import PSO_topology
from _3pso import PSO_aging
from _4pso import PSO_final
from _6de  import DEOptimizer
from _7ga  import GAOptimizer
from _8apso import APSO

def generate_joint_trajectory(
    trajectory_id,
    K,
    N,
    device,
    dtype,
    seed=None
):
    if seed is not None:
        torch.manual_seed(seed)

    t = torch.linspace(0, 2 * torch.pi, K, device=device, dtype=dtype)
    joint_values = torch.zeros(K, N, device=device, dtype=dtype)

    if trajectory_id == "smooth_sine":
        for i in range(N):
            joint_values[:, i] = 0.5 * torch.sin(t + i * 0.3)

    elif trajectory_id == "multi_frequency":
        for i in range(N):
            joint_values[:, i] = (
                0.35 * torch.sin(t + i * 0.3)
                + 0.20 * torch.sin(2.7 * t + i * 0.5)
                + 0.10 * torch.sin(5.0 * t + i * 0.2)
            )

    elif trajectory_id == "slow_sine":
        for i in range(N):
            joint_values[:, i] = 0.35 * torch.sin(0.4 * t + i * 0.2)

    elif trajectory_id == "aggressive_sine":
        for i in range(N):
            joint_values[:, i] = (
                0.75 * torch.sin(1.5 * t + i * 0.4)
                + 0.25 * torch.sin(4.0 * t + i * 0.3)
            )

    elif trajectory_id == "random_smooth":
        raw = torch.randn(K, N, device=device, dtype=dtype)

        window = 10
        kernel = torch.ones(window, device=device, dtype=dtype) / window

        for i in range(N):
            signal = raw[:, i].view(1, 1, -1)
            padded = torch.nn.functional.pad(
                signal,
                (window // 2, window // 2),
                mode="reflect"
            )
            smoothed = torch.nn.functional.conv1d(
                padded,
                kernel.view(1, 1, -1)
            ).view(-1)

            smoothed = smoothed[:K]
            smoothed = smoothed / (torch.max(torch.abs(smoothed)) + 1e-8)
            joint_values[:, i] = 0.6 * smoothed

    else:
        raise ValueError(f"Unknown trajectory_id: {trajectory_id}")

    return joint_values


def run_single_experiment(
    method_name,
    trajectory_id,
    seed,
    optimizer_class,
    K=500,
    P=32,
    pso_iterations_per_measurement=20,
    device=None,
    dtype=torch.float32,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    N = 6
    joint_types = ["R"] * N

    nominal_dh = torch.tensor([
        [0.0, 0.3, 0.0, 1.5708],
        [0.0, 0.0, 0.4, 0.0],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.2, 0.0, 1.5708],
        [0.0, 0.0, 0.0, -1.5708],
        [0.0, 0.1, 0.0, 0.0],
    ], device=device, dtype=dtype)

    joint_values = generate_joint_trajectory(
        trajectory_id=trajectory_id,
        K=K,
        N=N,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    true_dh = nominal_dh.clone()
    true_dh[:, 0] += 0.01 * torch.randn(N, device=device, dtype=dtype)
    true_dh[:, 1] += 0.002 * torch.randn(N, device=device, dtype=dtype)
    true_dh[:, 2] += 0.002 * torch.randn(N, device=device, dtype=dtype)
    true_dh[:, 3] += 0.01 * torch.randn(N, device=device, dtype=dtype)

    T_measured = simulate_measurements_from_dh(
        joint_values=joint_values,
        true_dh_params=true_dh,
        position_noise_std=0.003,
        orientation_noise_std=0.002,
        joint_types=joint_types,
    )

    lower_bounds = nominal_dh.clone()
    upper_bounds = nominal_dh.clone()

    lower_bounds[:, 0] -= 0.03
    upper_bounds[:, 0] += 0.03
    lower_bounds[:, 1] -= 0.02
    upper_bounds[:, 1] += 0.02
    lower_bounds[:, 2] -= 0.02
    upper_bounds[:, 2] += 0.02
    lower_bounds[:, 3] -= 0.03
    upper_bounds[:, 3] += 0.03

    optimizer = optimizer_class(
        nominal_dh=nominal_dh,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        joint_types=joint_types,
        num_particles=P,
        w=0.7,
        c1=1.5,
        c2=1.5,
        position_weight=1.0,
        orientation_weight=1.0,
        device=device,
        dtype=dtype,
        vmax_scale=0.1,
    )

    history = []
    measurement_times = []

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for k in range(K):
        measurement_start = time.time()

        joint_values_so_far = joint_values[:k + 1]
        T_measured_so_far = T_measured[:k + 1]

        for _ in range(pso_iterations_per_measurement):
            best_dh, best_fitness = optimizer.step(
                joint_values=joint_values_so_far,
                T_measured=T_measured_so_far,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        measurement_times.append(time.time() - measurement_start)
        history.append(best_fitness.item())

    if device == "cuda":
        torch.cuda.synchronize()

    total_runtime = time.time() - start

    with torch.no_grad():
        nominal_particle = nominal_dh.unsqueeze(0)
        T_nominal = forward_kinematics_particles(
            joint_values=joint_values,
            particles=nominal_particle,
            joint_types=joint_types,
        )
        nominal_fitness = particle_fitness(
            T_measured=T_measured,
            T_particles=T_nominal,
            position_weight=1.0,
            orientation_weight=1.0,
            reduction="mean",
        )[0]

        best_particle = best_dh.unsqueeze(0)
        T_best = forward_kinematics_particles(
            joint_values=joint_values,
            particles=best_particle,
            joint_types=joint_types,
        )
        optimized_fitness = particle_fitness(
            T_measured=T_measured,
            T_particles=T_best,
            position_weight=1.0,
            orientation_weight=1.0,
            reduction="mean",
        )[0]

    dh_error = best_dh - true_dh

    result = {
        "method": method_name,
        "trajectory_id": trajectory_id,
        "seed": seed,

        "final_best_fitness": best_fitness.item(),

        "nominal_fitness": nominal_fitness.item(),
        "optimized_fitness": optimized_fitness.item(),

        "improvement_ratio":
            nominal_fitness.item() / optimized_fitness.item(),

        "absolute_DH_error_mean":
            torch.mean(torch.abs(dh_error)).item(),

        "theta_error":
            torch.mean(torch.abs(dh_error[:, 0])).item(),

        "d_error":
            torch.mean(torch.abs(dh_error[:, 1])).item(),

        "a_error":
            torch.mean(torch.abs(dh_error[:, 2])).item(),

        "alpha_error":
            torch.mean(torch.abs(dh_error[:, 3])).item(),

        "runtime_total": total_runtime,

        "runtime_per_measurement":
            sum(measurement_times) / len(measurement_times),

        "mean_Dx":
            float(np.mean(optimizer.diversity_history)),

        "min_Dx":
            float(np.min(optimizer.diversity_history)),

        "mean_Dv":
            float(np.mean(optimizer.velocity_diversity_history)),

        "min_Dv":
            float(np.min(optimizer.velocity_diversity_history)),

        "max_Dv":
            float(np.max(optimizer.velocity_diversity_history)),

        # ------------------------------------------------
        # FUZZY PARAMETER STATES
        # ------------------------------------------------

        "final_w":
            getattr(optimizer, "current_w", np.nan),

        "final_c1":
            getattr(optimizer, "current_c1", np.nan),

        "final_c2":
            getattr(optimizer, "current_c2", np.nan),

        "final_vmax_scale":
            getattr(optimizer, "current_vmax_scale", np.nan),
    }

    return result

def main():

    trajectory_ids = [
        "smooth_sine",
        "multi_frequency",
        "slow_sine",
        "aggressive_sine",
        "random_smooth",
    ]

    seeds = range(20)

    methods = {
        "PSO": PSO,
        "PSO_topology": PSO_topology,
        "PSO_topology_aging": PSO_aging,
        "DE": DEOptimizer,
        "GA": GAOptimizer,
        "APSO": APSO,
        "PSO_final": PSO_final,
    }

    results = []

    for method_name, optimizer_class in methods.items():
        for trajectory_id in trajectory_ids:
            for seed in seeds:
                print(method_name, trajectory_id, seed)

                result = run_single_experiment(
                    method_name=method_name,
                    trajectory_id=trajectory_id,
                    seed=seed,
                    optimizer_class=optimizer_class,
                )

                results.append(result)

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    
if __name__ == "__main__":
    main()
