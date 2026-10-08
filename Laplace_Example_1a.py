import bempp.api
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
# Set the test configuration to run. Options:
# "STANDARD"          : Standard Calderón residuals convergence test.
# "INACCURATE_QUAD"   : Adapted quadrature (regular=2, singular=1).
# "ERROR_A"           : Replace diagonal of V with random numbers in [0, 10].
# "ERROR_B"           : Scale diagonal of V by a factor of 1/2.
# "ERROR_C"           : Scale diagonal of V by a factor of 10^3.
# "ERROR_D"           : Scale first half of the diagonal of V by 10^3.
# "ERROR_E"           : Scale diagonal of V by a factor of h^-1.
TEST_MODE = "ERROR_E"

# Set mesh refinement levels (e.g., 2 -> 128 elements, 3 -> 512, etc.)
REFINEMENT_LEVELS = [2, 3]

# =============================================================================
# Exact Solutions & Data Functions (Example 1a)
# =============================================================================
@bempp.api.complex_callable
def dirichlet_data(x, n, domain_index, result):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    theta = np.arccos(x[2] / r)
    phi = np.sign(x[1]) * np.arccos(x[0] / np.sqrt(x[0]**2 + x[1]**2))
    result[0] = -1 * np.exp(1j * phi) * np.sqrt(1 - np.cos(theta)**2) * (r**(-2))
    
@bempp.api.complex_callable
def neumann_data(x, n, domain_index, result):
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    theta = np.arccos(x[2] / r)
    phi = np.sign(x[1]) * np.arccos(x[0] / np.sqrt(x[0]**2 + x[1]**2))
    # Extra minus sign for outward normal
    result[0] = 2 * np.exp(1j * phi) * np.sqrt(1 - np.cos(theta)**2) * (r**(-3))

# =============================================================================
# Main Convergence Loop
# =============================================================================
def run_tests():
    N_elements, h_maxs = [], []
    residuals_D_inf, residuals_D_2 = [], []
    residuals_N_inf, residuals_N_2 = [], []

    for level in REFINEMENT_LEVELS:
        print(f"--- Processing Refinement Level {level} ---")
        
        # 1. Adapt Quadrature
        if TEST_MODE == "INACCURATE_QUAD":
            bempp.api.GLOBAL_PARAMETERS.quadrature.regular = 2
            bempp.api.GLOBAL_PARAMETERS.quadrature.singular = 1
        else:
            bempp.api.GLOBAL_PARAMETERS.quadrature.regular = 4
            bempp.api.GLOBAL_PARAMETERS.quadrature.singular = 4
            
        grid = bempp.api.shapes.regular_sphere(level)
        N = grid.number_of_elements
        h = np.min(grid.diameters)
        N_elements.append(N)
        h_maxs.append(np.max(grid.diameters))

        # 2. Setup Spaces (Lowest Order)
        space_pwc = bempp.api.function_space(grid, "DP", 0)  # Piecewise constant, H^{-1/2}
        space_pwl = bempp.api.function_space(grid, "P", 1)   # Piecewise linear, H^{1/2}

        # 3. Assemble Galerkin Matrices
        print("Assembling Galerkin matrices...")
        id_pwc_pwl = bempp.api.operators.boundary.sparse.identity(space_pwl, space_pwc, space_pwc)
        Mass_pwc_pwl = bempp.api.assembly.discrete_boundary_operator.as_matrix(id_pwc_pwl.weak_form())

        id_pwl_pwc = bempp.api.operators.boundary.sparse.identity(space_pwc, space_pwl, space_pwl)
        Mass_pwl_pwc = bempp.api.assembly.discrete_boundary_operator.as_matrix(id_pwl_pwc.weak_form())

        dlp = bempp.api.operators.boundary.laplace.double_layer(space_pwl, space_pwc, space_pwc)
        K = bempp.api.assembly.discrete_boundary_operator.as_matrix(dlp.weak_form())

        adlp = bempp.api.operators.boundary.laplace.adjoint_double_layer(space_pwc, space_pwl, space_pwl)
        K_adj = bempp.api.assembly.discrete_boundary_operator.as_matrix(adlp.weak_form())

        slp = bempp.api.operators.boundary.laplace.single_layer(space_pwc, space_pwc, space_pwc)
        V = bempp.api.assembly.discrete_boundary_operator.as_matrix(slp.weak_form())

        hlp = bempp.api.operators.boundary.laplace.hypersingular(space_pwl, space_pwl, space_pwl)
        W = bempp.api.assembly.discrete_boundary_operator.as_matrix(hlp.weak_form())
        
        print("Assembly finished.")

        # 4. Generate Dirichlet and Neumann Data Coefficients
        dirichlet_fun = bempp.api.GridFunction(space_pwl, fun=dirichlet_data)
        neumann_fun = bempp.api.GridFunction(space_pwc, fun=neumann_data)
        u_D = dirichlet_fun.coefficients
        u_N = neumann_fun.coefficients

        # 5. Apply Artificial Defective Errors
        V_e, W_e = V.copy(), W.copy()
        if TEST_MODE == "ERROR_A":
            np.fill_diagonal(V_e, np.random.uniform(0, 10, len(V_e)))
        elif TEST_MODE == "ERROR_B":
            np.fill_diagonal(V_e, 0.5 * V_e.diagonal())
        elif TEST_MODE == "ERROR_C":
            np.fill_diagonal(V_e, 10**3 * V_e.diagonal())
        elif TEST_MODE == "ERROR_D":
            d_indices = np.diag_indices_from(V_e)
            half_idx = len(d_indices[0]) // 2
            V_e[d_indices[0][:half_idx], d_indices[1][:half_idx]] *= 10**3
        elif TEST_MODE == "ERROR_E":
            np.fill_diagonal(V_e, (1.0 / h) * V_e.diagonal())

        # 6. Calderón Residuals Calculation
        Dirichlet_D = (0.5 * Mass_pwc_pwl - K) @ u_D
        Neumann_D = V_e @ u_N
        rho_D = Dirichlet_D + Neumann_D
        residuals_D_2.append(np.linalg.norm(rho_D.T, 2))
        residuals_D_inf.append(np.linalg.norm(rho_D.T, np.inf))

        # Skip rho_N for V diagonal errors
        if TEST_MODE in ["STANDARD", "INACCURATE_QUAD"]:
            Dirichlet_N = W_e @ u_D
            Neumann_N = (0.5 * Mass_pwl_pwc + K_adj) @ u_N
            rho_N = Dirichlet_N + Neumann_N
            residuals_N_2.append(np.linalg.norm(rho_N.T, 2))
            residuals_N_inf.append(np.linalg.norm(rho_N.T, np.inf))
        else:
            residuals_N_2.append(None)
            residuals_N_inf.append(None)
            
        print("Iteration complete.\n")

    # 7. Compute Experimental Order of Convergence (OOC)
    def fit_ooc(values):
        clean_vals = [v for v in values if v is not None]
        if len(clean_vals) > 1:
            b, _ = np.polyfit(np.log(h_maxs), np.log(clean_vals), 1)
            return float(b)
        return None

    rates = {
        "rho_D_inf": fit_ooc(residuals_D_inf),
        "rho_D_2": fit_ooc(residuals_D_2),
        "rho_N_inf": fit_ooc(residuals_N_inf),
        "rho_N_2": fit_ooc(residuals_N_2)
    }

    # 8. Print Output Summary
    print("="*60)
    print(f"FINAL METRICS FOR: {TEST_MODE}")
    print("="*60)
    print(f"Elements: {N_elements}")
    print(f"rho_D (inf): {residuals_D_inf} | OOC: {rates['rho_D_inf']}")
    print(f"rho_D (2):   {residuals_D_2} | OOC: {rates['rho_D_2']}")
    if rates['rho_N_inf'] is not None:
        print(f"rho_N (inf): {residuals_N_inf} | OOC: {rates['rho_N_inf']}")
        print(f"rho_N (2):   {residuals_N_2} | OOC: {rates['rho_N_2']}")
    print("="*60)

if __name__ == "__main__":
    run_tests()
