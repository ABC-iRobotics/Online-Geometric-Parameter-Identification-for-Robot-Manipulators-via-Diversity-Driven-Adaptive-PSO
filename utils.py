import torch

def skew(v):
    """
    Skew-symmetric mátrix rotációs vektorhoz.

    v shape: (K, 3)

    return:
        shape: (K, 3, 3)
    """

    K = v.shape[0]
    device = v.device
    dtype = v.dtype

    S = torch.zeros(K, 3, 3, device=device, dtype=dtype)

    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] =  v[:, 1]
    S[:, 1, 0] =  v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] =  v[:, 0]

    return S

def rotation_vector_to_matrix(rotvec):
    """
    Rotációs vektor -> rotációs mátrix Rodrigues-formulával.

    rotvec shape: (K, 3)

    return:
        R shape: (K, 3, 3)
    """

    K = rotvec.shape[0]
    device = rotvec.device
    dtype = rotvec.dtype

    angle = torch.norm(rotvec, dim=1, keepdim=True)  # (K, 1)

    axis = rotvec / (angle + 1e-8)

    K_mat = skew(axis)

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(K, 1, 1)

    angle = angle.view(K, 1, 1)

    R = (
        eye
        + torch.sin(angle) * K_mat
        + (1.0 - torch.cos(angle)) * torch.matmul(K_mat, K_mat)
    )

    return R

def dh_transform(theta, d, a, alpha):
    """
    Standard DH transzformációs mátrix.

    theta, d, a, alpha shape: (...,)

    return:
        T shape: (..., 4, 4)
    """

    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)

    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)

    T = torch.stack([
        torch.stack([ct, -st * ca,  st * sa, a * ct], dim=-1),
        torch.stack([st,  ct * ca, -ct * sa, a * st], dim=-1),
        torch.stack([zeros, sa, ca, d], dim=-1),
        torch.stack([zeros, zeros, zeros, ones], dim=-1),
    ], dim=-2)

    return T

def simulate_measurements_from_dh(
    joint_values,
    true_dh_params,
    position_noise_std=0.001,
    orientation_noise_std=0.0,
    joint_types=None
):
    """
    Zajos mért homogén transzformációk szimulálása egy true DH modellből.

    joint_values:
        shape: (K, N)

    true_dh_params:
        shape: (1, N, 4) vagy (N, 4)

    position_noise_std:
        pozíció zaj szórása méterben

    orientation_noise_std:
        orientációs zaj szórása radiánban

    return:
        T_measured:
            shape: (K, 4, 4)
    """

    if true_dh_params.ndim == 2:
        true_dh_params = true_dh_params.unsqueeze(0)

    T_true = forward_kinematics_particles(
        joint_values=joint_values,
        particles=true_dh_params,
        joint_types=joint_types
    )[0].clone()

    K = T_true.shape[0]
    device = T_true.device
    dtype = T_true.dtype

    # Pozíció zaj
    if position_noise_std > 0:
        T_true[:, :3, 3] += (
            position_noise_std
            * torch.randn_like(T_true[:, :3, 3])
        )

    # Orientáció zaj
    if orientation_noise_std > 0:
        rotvec_noise = (
            orientation_noise_std
            * torch.randn(K, 3, device=device, dtype=dtype)
        )

        R_noise = rotation_vector_to_matrix(rotvec_noise)

        T_true[:, :3, :3] = torch.matmul(
            R_noise,
            T_true[:, :3, :3]
        )

    return T_true

def forward_kinematics_particles(
    joint_values,
    particles,
    joint_types=None
):
    """
    Batch forward kinematics particle-ökre.

    Bemenetek:
    --------------------------------------------------------
    joint_values:
        shape: (K, N)

        K = mérési pontok száma
        N = jointok száma

    particles:
        shape: (P, N, 4)

        P = particle-ök száma

        Minden particle:
            teljes DH paraméter készlet

        DH sorrend:
            [theta, d, a, alpha]

    joint_types:
        pl:
            ["R", "R", "R", "R", "R", "R"]

    Kimenet:
    --------------------------------------------------------

    T_total:
        shape: (P, K, 4, 4)

        P particle
        K mérési pont
        mindenre homogén transzformáció
    """

    K, N = joint_values.shape
    P = particles.shape[0]

    device = joint_values.device
    dtype = joint_values.dtype

    if joint_types is None:
        joint_types = ["R"] * N

    if len(joint_types) != N:
        raise ValueError("joint_types length mismatch.")

    # --------------------------------------------------------
    # DH paraméterek
    # --------------------------------------------------------

    theta = particles[:, :, 0].unsqueeze(1).expand(P, K, N).clone()
    d = particles[:, :, 1].unsqueeze(1).expand(P, K, N).clone()
    a = particles[:, :, 2].unsqueeze(1).expand(P, K, N).clone()
    alpha = particles[:, :, 3].unsqueeze(1).expand(P, K, N).clone()

    q = joint_values.unsqueeze(0).expand(P, K, N)

    # --------------------------------------------------------
    # Revolute / Prismatic joint kezelés
    # --------------------------------------------------------

    for i, jt in enumerate(joint_types):

        jt = jt.upper()

        if jt == "R":
            theta[:, :, i] += q[:, :, i]

        elif jt == "P":
            d[:, :, i] += q[:, :, i]

        else:
            raise ValueError(f"Unknown joint type: {jt}")

    # --------------------------------------------------------
    # Forward kinematics
    # --------------------------------------------------------

    T_total = torch.eye(
        4,
        device=device,
        dtype=dtype
    ).repeat(P, K, 1, 1)

    for i in range(N):

        T_i = dh_transform(
            theta[:, :, i],
            d[:, :, i],
            a[:, :, i],
            alpha[:, :, i]
        )

        T_total = torch.matmul(T_total, T_i)

    return T_total

def homogeneous_from_measurements(positions, rotations):
    """
    Mért pozíciókból és rotációkból homogén transzformációs mátrixokat készít.

    positions:
        shape: (K, 3)

    rotations:
        shape: (K, 3, 3)

    return:
        T_measured:
            shape: (K, 4, 4)
    """

    K = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    if rotations.shape != (K, 3, 3):
        raise ValueError("rotations shape must be (K, 3, 3).")

    T = torch.eye(4, device=device, dtype=dtype).repeat(K, 1, 1)

    T[:, :3, :3] = rotations
    T[:, :3, 3] = positions

    return T

def particle_fitness(
    T_measured,
    T_particles,
    position_weight=1.0,
    orientation_weight=1.0,
    reduction="mean"
):
    """
    Fitness számítás particle-ökre homogén transzformációs mátrixok alapján.

    Bemenetek:
    ----------------------------------------------------------

    T_measured:
        shape: (1, K, 4, 4)
        vagy
        shape: (K, 4, 4)

        K = mérési pontok száma

    T_particles:
        shape: (P, K, 4, 4)

        P = particle-ök száma

    Hiba:
    ----------------------------------------------------------

    pozíció:
        e_pos = ||p_measured - p_particle||

    orientáció:
        e_ori = relatív rotáció szöge
              = rotációs vektor hossza

    teljes hiba:
        e_total =
            position_weight * e_pos
            +
            orientation_weight * e_ori

    reduction:
    ----------------------------------------------------------

    "mean":
        átlagolt fitness particle-önként

    "sum":
        összegzett fitness particle-önként

    "none":
        teljes (P, K) hibamátrix

    Kimenet:
    ----------------------------------------------------------

    fitness:
        reduction="mean" vagy "sum":
            shape: (P,)

        reduction="none":
            shape: (P, K)
    """

    # ----------------------------------------------------------
    # Shape kezelés
    # ----------------------------------------------------------

    if T_measured.ndim == 3:
        T_measured = T_measured.unsqueeze(0)

    if T_measured.ndim != 4:
        raise ValueError("T_measured must have shape (K,4,4) or (1,K,4,4)")

    if T_particles.ndim != 4:
        raise ValueError("T_particles must have shape (P,K,4,4)")

    P = T_particles.shape[0]
    K = T_particles.shape[1]

    if T_measured.shape[1] != K:
        raise ValueError("Measurement count mismatch.")

    # ----------------------------------------------------------
    # Pozíció
    # ----------------------------------------------------------

    p_measured = T_measured[..., :3, 3]
    p_particles = T_particles[..., :3, 3]

    position_error = torch.norm(
        p_measured - p_particles,
        dim=-1
    )  # (P, K)

    # ----------------------------------------------------------
    # Orientáció
    # ----------------------------------------------------------

    R_measured = T_measured[..., :3, :3]
    R_particles = T_particles[..., :3, :3]

    R_rel = torch.matmul(
        R_measured,
        R_particles.transpose(-1, -2)
    )

    trace = (
        R_rel[..., 0, 0] +
        R_rel[..., 1, 1] +
        R_rel[..., 2, 2]
    )

    cos_angle = (trace - 1.0) / 2.0

    cos_angle = torch.clamp(
        cos_angle,
        -1.0 + 1e-7,
        1.0 - 1e-7
    )

    orientation_error = torch.acos(cos_angle)

    # ----------------------------------------------------------
    # Teljes hiba
    # ----------------------------------------------------------

    total_error = (
        position_weight * position_error +
        orientation_weight * orientation_error
    )

    # ----------------------------------------------------------
    # Reduction
    # ----------------------------------------------------------

    if reduction == "none":
        return total_error

    elif reduction == "mean":
        return torch.mean(total_error, dim=1)

    elif reduction == "sum":
        return torch.sum(total_error, dim=1)

    else:
        raise ValueError("Unknown reduction mode.")