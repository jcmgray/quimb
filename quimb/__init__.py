"""
QUIMB
-----

Quantum Information for Many-Body calculations.
"""

# Accelerated basic linalg operations
from .accel import (
    prod,
    isket,
    isbra,
    isop,
    isvec,
    issparse,
    isherm,
    mul,
    dot,
    vdot,
    rdot,
    ldmul,
    rdmul,
    outer,
    explt,
    kron,
    kronpow,
)

# Core functions
from .core import (
    normalize,
    chop,
    quimbify,
    qu,
    ket,
    bra,
    dop,
    sparse,
    infer_size,
    trace,
    identity,
    eye,
    speye,
    dim_map,
    dim_compress,
    eyepad,
    perm_eyepad,
    permute,
    itrace,
    partial_trace,
    overlap,
    nmlz,
    tr,
    ptr,
)

# Solving functions
from .solve.base_solver import (
    eigsys,
    eigvals,
    eigvecs,
    seigsys,
    seigvals,
    seigvecs,
    groundstate,
    groundenergy,
    bound_spectrum,
    eigsys_window,
    eigvals_window,
    eigvecs_window,
    svd,
    svds,
    norm,
)

# Generating objects
from .gen.operators import (
    sig,
    controlled,
    ham_heis,
    ham_j1j2,
    zspin_projector,
    swap,
)
from .gen.states import (
    basis_vec,
    up,
    zplus,
    down,
    zminus,
    plus,
    xplus,
    minus,
    xminus,
    yplus,
    yminus,
    bloch_state,
    bell_state,
    singlet,
    thermal_state,
    neel_state,
    singlet_pairs,
    werner_state,
    ghz_state,
    w_state,
    levi_civita,
    perm_state,
    graph_state_1d,
)
from .gen.rand import (
    rand_matrix,
    rand_herm,
    rand_pos,
    rand_rho,
    rand_ket,
    rand_uni,
    rand_haar_state,
    gen_rand_haar_states,
    rand_mix,
    rand_product_state,
    rand_matrix_product_state,
    rand_mps,
    rand_seperable,
)

# Functions for calculating properties
from .calc import (
    expm,
    sqrtm,
    fidelity,
    purify,
    entropy,
    mutual_information,
    mutinf,
    partial_transpose,
    negativity,
    logarithmic_negativity,
    logneg,
    concurrence,
    one_way_classical_information,
    quantum_discord,
    trace_distance,
    decomp,
    pauli_decomp,
    bell_decomp,
    correlation,
    pauli_correlations,
    ent_cross_matrix,
    qid,
    is_degenerate,
    page_entropy,
)

# Evolution class and methods
from .evo import QuEvo


__all__ = [
    # Accel ----------------------------------------------------------------- #
    'prod',
    'isket',
    'isbra',
    'isop',
    'isvec',
    'issparse',
    'isherm',
    'mul',
    'dot',
    'vdot',
    'rdot',
    'ldmul',
    'rdmul',
    'outer',
    'explt',
    'kron',
    'kronpow',
    # Core ------------------------------------------------------------------ #
    'normalize',
    'chop',
    'quimbify',
    'qu',
    'ket',
    'bra',
    'dop',
    'sparse',
    'infer_size',
    'trace',
    'identity',
    'eye',
    'speye',
    'dim_map',
    'dim_compress',
    'eyepad',
    'perm_eyepad',
    'permute',
    'itrace',
    'partial_trace',
    'overlap',
    'nmlz',
    'tr',
    'ptr',
    # Solve ----------------------------------------------------------------- #
    'eigsys',
    'eigvals',
    'eigvecs',
    'seigsys',
    'seigvals',
    'seigvecs',
    'groundstate',
    'groundenergy',
    'bound_spectrum',
    'eigsys_window',
    'eigvals_window',
    'eigvecs_window',
    'svd',
    'svds',
    'norm',
    # Gen ------------------------------------------------------------------- #
    'sig',
    'controlled',
    'ham_heis',
    'ham_j1j2',
    'zspin_projector',
    'swap',
    'basis_vec',
    'up',
    'zplus',
    'down',
    'zminus',
    'plus',
    'xplus',
    'minus',
    'xminus',
    'yplus',
    'yminus',
    'bloch_state',
    'bell_state',
    'singlet',
    'thermal_state',
    'neel_state',
    'singlet_pairs',
    'werner_state',
    'ghz_state',
    'w_state',
    'levi_civita',
    'perm_state',
    'graph_state_1d',
    'rand_matrix',
    'rand_herm',
    'rand_pos',
    'rand_rho',
    'rand_ket',
    'rand_uni',
    'rand_haar_state',
    'gen_rand_haar_states',
    'rand_mix',
    'rand_product_state',
    'rand_matrix_product_state',
    'rand_mps',
    'rand_seperable',
    # Calc ------------------------------------------------------------------ #
    'expm',
    'sqrtm',
    'fidelity',
    'purify',
    'entropy',
    'mutual_information',
    'mutinf',
    'partial_transpose',
    'negativity',
    'logarithmic_negativity',
    'logneg',
    'concurrence',
    'one_way_classical_information',
    'quantum_discord',
    'trace_distance',
    'decomp',
    'pauli_decomp',
    'bell_decomp',
    'correlation',
    'pauli_correlations',
    'ent_cross_matrix',
    'qid',
    'is_degenerate',
    'page_entropy',
    # Evo ------------------------------------------------------------------- #
    'QuEvo',
]
