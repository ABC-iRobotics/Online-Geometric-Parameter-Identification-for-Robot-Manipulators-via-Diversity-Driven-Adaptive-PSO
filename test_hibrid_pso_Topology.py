import time
import torch

from utils import *
from Hibrid_pso_Topology import DHParticleSwarmOptimizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print("Device:", device)

    # ------------------------------------------------------------
    # Robot / trajectory settings
    # ------------------------------------------------------------

    N = 6
    K = 500
    P = 128

    joint_types = ["R"] * N

    nominal_dh = torch.tensor([
        [0.0, 0.3, 0.0, 1.5708],
        [0.0, 0.0, 0.4, 0.0],
        [0.0, 0.0, 0.3, 0.0],
        [0.0, 0.2, 0.0, 1.5708],
        [0.0, 0.0, 0.0, -1.5708],
        [0.0, 0.1, 0.0, 0.0],
    ], device=device, dtype=dtype)

    # ------------------------------------------------------------
    # Joint trajectory
    # ------------------------------------------------------------

    t = torch.linspace(0, 2 * torch.pi, K, device=device, dtype=dtype)

    joint_values = torch.zeros(K, N, device=device, dtype=dtype)

    for i in range(N):
        joint_values[:, i] = 0.5 * torch.sin(t + i * 0.3)

    # ------------------------------------------------------------
    # True DH model
    # ------------------------------------------------------------

    true_dh = nominal_dh.clone()

    true_dh[:, 0] += 0.01 * torch.randn(N, device=device, dtype=dtype)
    true_dh[:, 1] += 0.002 * torch.randn(N, device=device, dtype=dtype)
    true_dh[:, 2] += 0.002 * torch.randn(N, device=device, dtype=dtype)
    true_dh[:, 3] += 0.01 * torch.randn(N, device=device, dtype=dtype)

    true_particle = true_dh.unsqueeze(0)

    # ------------------------------------------------------------
    # Generate measured poses
    # ------------------------------------------------------------
    T_measured = simulate_measurements_from_dh(
        joint_values=joint_values,
        true_dh_params=true_dh,
        position_noise_std=0.003,
        orientation_noise_std=0.002,
        joint_types=joint_types
    )

    # ------------------------------------------------------------
    # DH search bounds
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # PSO
    # ------------------------------------------------------------

    optimizer = DHParticleSwarmOptimizer(
        nominal_dh=nominal_dh,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        joint_types=joint_types,
        num_particles=P,
        w=0.7,
        c1=1.5,
        c2=1.5,
        position_weight=1.0,
        orientation_weight=0.1,
        device=device,
        dtype=dtype,
        vmax_scale=0.1
    )

    # ------------------------------------------------------------
    # ONLINE / STREAMING PSO
    # ------------------------------------------------------------

    pso_iterations_per_measurement = 5
    history = []

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
                T_measured=T_measured_so_far
            )
        if device == "cuda":
            torch.cuda.synchronize()

        measurement_time = time.time() - measurement_start
        history.append(best_fitness.item())
        
        if k % 10 == 0:
            current_measurements = joint_values_so_far.shape[0]
            fk_evaluations = P * current_measurements * pso_iterations_per_measurement
            fk_per_second = fk_evaluations / measurement_time

            print(
                f"Measurement {k + 1:04d}/{K} | "
                f"Best fitness: {best_fitness.item():.8f} | "
                f"Time: {measurement_time:.4f}s | "
                f"FK/s: {fk_per_second:,.0f} | "
                f"Dx: {optimizer.diversity_history[-1]:.6f} | "
                f"Dv: {optimizer.velocity_diversity_history[-1]:.6f}"
            )

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    # ------------------------------------------------------------
    # NOMINAL DH FITNESS
    # ------------------------------------------------------------

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
            orientation_weight=0.1,
            reduction="mean",
        )[0]

    # ------------------------------------------------------------
    # OPTIMIZED DH FITNESS
    # ------------------------------------------------------------

    with torch.no_grad():

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
            orientation_weight=0.01,
            reduction="mean",
        )[0]

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("\nOptimization finished.")
    print("Time:", end - start, "s")
    print("Best fitness:", best_fitness.item())

    print("\nTrue DH:")
    print(true_dh.detach().cpu())

    print("\nEstimated DH:")
    print(best_dh.detach().cpu())

    print("\nDH error:")
    print((best_dh - true_dh).detach().cpu())

    print("\nAbsolute DH error mean:")
    print(torch.mean(torch.abs(best_dh - true_dh)).item())

    print("Mean theta error:")
    print(torch.mean(torch.abs(best_dh[:, 0] - true_dh[:, 0])).item())

    print("Mean d error:")
    print(torch.mean(torch.abs(best_dh[:, 1] - true_dh[:, 1])).item())

    print("Mean a error:")
    print(torch.mean(torch.abs(best_dh[:, 2] - true_dh[:, 2])).item())

    print("Mean alpha error:")
    print(torch.mean(torch.abs(best_dh[:, 3] - true_dh[:, 3])).item())

    print("\nNominal DH fitness:")
    print(nominal_fitness.item())

    print("\nOptimized DH fitness:")
    print(optimized_fitness.item())

    print("\nImprovement ratio:")
    print(nominal_fitness.item() / optimized_fitness.item())

    


if __name__ == "__main__":
    main()
