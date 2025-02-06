from multiprocessing import Pool
import time

def run(self, n_macroreps: int) -> None:
    """Run n_macroreps of the solver on the problem.

    Notes
    -----
    RNGs dedicated for random problem instances and temporarily unused.
    Under development.

    Parameters
    ----------
    n_macroreps : int
        Number of macroreplications of the solver to run on the problem.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(n_macroreps, int):
        error_msg = "Number of macroreplications must be an integer."
        raise TypeError(error_msg)
    # Value checking
    if n_macroreps <= 0:
        error_msg = "Number of macroreplications must be positive."
        raise ValueError(error_msg)

    print(
        "Running Solver",
        self.solver.name,
        "on Problem",
        self.problem.name + ".",
    )

    # Initialize variables
    self.n_macroreps = n_macroreps
    self.all_recommended_xs = [[] for _ in range(n_macroreps)]
    self.all_intermediate_budgets = [[] for _ in range(n_macroreps)]
    self.timings = [0.0 for _ in range(n_macroreps)]

    # Create, initialize, and attach random number generators
    #     Stream 0: reserved for taking post-replications
    #     Stream 1: reserved for bootstrapping
    #     Stream 2: reserved for overhead ...
    #         Substream 0: rng for random problem instance
    #         Substream 1: rng for random initial solution x0 and
    #                      restart solutions
    #         Substream 2: rng for selecting random feasible solutions
    #         Substream 3: rng for solver's internal randomness
    #     Streams 3, 4, ..., n_macroreps + 2: reserved for
    #                                         macroreplications
    # rng0 = MRG32k3a(s_ss_sss_index=[2, 0, 0])  # Currently unused.
    rng_list = [MRG32k3a(s_ss_sss_index=[2, i + 1, 0]) for i in range(3)]
    self.solver.attach_rngs(rng_list)

    # Start a timer
    function_start = time.time()

    print("Starting macroreplications in parallel")
    with Pool() as process_pool:
        # Start the macroreplications in parallel (async)
        result = process_pool.map_async(
            self.run_multithread, range(n_macroreps)
        )
        # Wait for the results to be returned (or 1 second)
        while not result.ready():
            # Update status bar here
            result.wait(1)

        print(
            f"Finished running {n_macroreps} macroreplications in {round(time.time() - function_start, 3)} seconds."
        )

        # Grab all the data out of the result
        for mrep in range(n_macroreps):
            (
                self.all_recommended_xs[mrep],
                self.all_intermediate_budgets[mrep],
                self.timings[mrep],
            ) = result.get()[mrep]

    self.has_run = True
    self.has_postreplicated = False
    self.has_postnormalized = False

    # Save ProblemSolver object to .pickle file if specified.
    if self.create_pickle:
        file_name = os.path.basename(self.file_name_path)
        self.record_experiment_results(file_name=file_name)

def run_multithread(self, mrep: int) -> tuple:
    """Run a single macroreplication of the solver on the problem.

    Parameters
    ----------
    mrep : int
        Index of the macroreplication.

    Returns
    -------
    tuple
        Tuple of recommended solutions, intermediate budgets, and runtime.

    Raises
    ------
    TypeError
    ValueError

    """
    # Type checking
    if not isinstance(mrep, int):
        error_msg = "Macroreplication index must be an integer."
        raise TypeError(error_msg)
    # Value checking
    if mrep < 0:
        error_msg = "Macroreplication index must be non-negative."
        raise ValueError(error_msg)

    print(
        f"Macroreplication {mrep + 1}: Starting Solver {self.solver.name} on Problem {self.problem.name}."
    )
    # Create, initialize, and attach RNGs used for simulating solutions.
    progenitor_rngs = [
        MRG32k3a(s_ss_sss_index=[mrep + 3, ss, 0])
        for ss in range(self.problem.model.n_rngs)
    ]
    # Create a new set of RNGs for the solver based on the current macroreplication.
    # Tried re-using the progentior RNGs, but we need to match the number needed by the solver, not the problem
    solver_rngs = [
        MRG32k3a(
            s_ss_sss_index=[
                mrep + 3,
                self.problem.model.n_rngs + rng_index,
                0,
            ]
        )
        for rng_index in range(len(self.solver.rng_list))
    ]

    # Set progenitor_rngs and rng_list for solver.
    self.solver.solution_progenitor_rngs = progenitor_rngs
    self.solver.rng_list = solver_rngs

    # print([rng.s_ss_sss_index for rng in progenitor_rngs])
    # Run the solver on the problem.
    tic = time.perf_counter()
    recommended_solns, intermediate_budgets = self.solver.solve(
        problem=self.problem
    )
    toc = time.perf_counter()
    runtime = toc - tic
    print(
        f"Macroreplication {mrep + 1}: Finished Solver {self.solver.name} on Problem {self.problem.name} in {runtime:0.4f} seconds."
    )

    # Trim the recommended solutions and intermediate budgets
    recommended_solns, intermediate_budgets = trim_solver_results(
        problem=self.problem,
        recommended_solutions=recommended_solns,
        intermediate_budgets=intermediate_budgets,
    )
    # Return tuple (rec_solns, int_budgets, runtime)
    return (
        [solution.x for solution in recommended_solns],
        intermediate_budgets,
        runtime,
    )