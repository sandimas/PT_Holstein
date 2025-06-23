using SmoQyElPhQMC

using SmoQyDQMC

using Random
using Printf
using MPI
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm

# Top-level function to run simulation.
function run_simulation(;
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    Ω, # Phonon energy.
    λ, # Electron-phonon coupling.
    μ, # Chemical potential.
    
    N_therm, # Number of thermalization updates.
    N_updates, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
    Δτ = 0.05, # Discretization in imaginary time.
    Nt = 100, # Numer of time-steps in HMC update.
    Nrv = 10, # Number of random vectors used to estimate fermionic correlation functions.
    tol = 1e-10, # CG iterations tolerance.
    maxiter = 1000, # Maximum number of CG iterations.
    seed = 1000,#abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)
    MPI.Init()
    comm = MPI.COMM_WORLD

    pID = MPI.Comm_rank(comm)
    
    try
        
        L = 14
        β = 5.0

        t_total = - time()
        t_init = -time()
        t_meas = 0.0
        t_write = 0.0
        t_therm = 0.0
        t_updates = 0.0


        
        α = Ω*sqrt(8*λ)

        filepath = "simulations/"
        mkpath(filepath)
        # Construct the foldername the data will be written to.
        datafolder_prefix = @sprintf "DQMC_2D_w%.2f_l%.2f_mu%.2f_L%d_b%.2f" Ω λ μ L β

        # Initialize simulation info.
        simulation_info = SimulationInfo(
            filepath = filepath,
            datafolder_prefix = datafolder_prefix,
            sID = sID,
            pID = pID,
        )
        MPI.Barrier(comm)
        # Initialize the directory the data will be written to.
        initialize_datafolder(simulation_info)
        MPI.Barrier(comm)
        # Initialize random number generator
        rng = Xoshiro(seed + pID)

        # Initialize additiona_info dictionary
        additional_info = Dict()

        Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

        # whether to use the checkerboard approximation
        checkerboard = false
    
        # whether to use symmetric propagator defintion 
        symmetric = false
    
        # initial stabilization frequency
        n_stab = 10
    
        # max allowed error in green's function
        δG_max = 1e-6
    
    
        # bin size
        bin_size = div(N_updates, N_bins)
    
        # hyrbid/hamiltonian monte carlo (HMC) update time-step
        Δt = 0.05
    
        # number of fermionic time-steps in HMC trajecotry
        Nt = 20
    
        # number of bosonic time-steps per fermionic time-step
        nt = 10
    
        # mass regularization in fourier acceleration
        reg = 1.0


        # Record simulation parameters.
        additional_info["N_therm"]   = N_therm    # Number of thermalization updates
        additional_info["N_updates"] = N_updates  # Total number of measurements and measurement updates
        additional_info["N_bins"]    = N_bins     # Number of times bin-averaged measurements are written to file
        additional_info["maxiter"]   = maxiter    # Maximum number of conjugate gradient iterations
        additional_info["tol"]       = tol        # Tolerance used for conjugate gradient solves
        additional_info["Nt"]        = Nt         # Number of time-steps in HMC update
        additional_info["Nrv"]       = Nrv        # Number of random vectors used to estimate fermionic correlation functions
        additional_info["seed"]      = seed       # Random seed used to initialize random number generator in simulation

        # Define the unit cell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [[1.0,0.0], [0.0,1.0]],
            basis_vecs   = [[0.0,0.0]]
        )

        # Define finite lattice with periodic boundary conditions.
        lattice = lu.Lattice(
            L = [L, L],
            periodic = [true, true]
        )

        # Initialize model geometry.
        model_geometry = ModelGeometry(unit_cell, lattice)

        # calculate number of orbitals in the lattice
        N = lu.nsites(unit_cell, lattice)

        # define first nearest-neighbor bond
        bond_x = lu.Bond(orbitals = (1,1), displacement = [1,0])
        bond_x_id = add_bond!(model_geometry, bond_x)

        # define second nearest-neighbor bond
        bond_y = lu.Bond(orbitals = (1,1), displacement = [0,1])
        bond_y_id = add_bond!(model_geometry, bond_y)


        # Set neartest-neighbor hopping amplitude to unity,
        # setting the energy scale in the model.
        t = 1.0

        # define non-interacting tight binding model
        tight_binding_model = TightBindingModel(
            model_geometry = model_geometry,
            t_bonds = [bond_x, bond_y],
            t_mean = [t, t],
            μ = μ,
            ϵ_mean = [0.]
        )
        
        # Initialize a null electron-phonon model.
        electron_phonon_model = ElectronPhononModel(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model
        )

        # Define a dispersionless electron-phonon mode to live on each site in the lattice.
        phonon_1 = PhononMode(orbital = 1, Ω_mean = Ω)

        # Add the phonon mode definition to the electron-phonon model.
        phonon_1_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_1
        )


        # Define first local Holstein coupling for first phonon mode.
        holstein_coupling_1 = HolsteinCoupling(
            model_geometry = model_geometry,
            phonon_mode = phonon_1_id,
            # Couple the first phonon mode to first orbital in the unit cell.
            bond = lu.Bond(orbitals = (1,1), displacement = [0, 0]),
            α_mean = α
        )

        # Add the first local Holstein coupling definition to the model.
        holstein_coupling_1_id = add_holstein_coupling!(
            electron_phonon_model = electron_phonon_model,
            holstein_coupling = holstein_coupling_1,
            model_geometry = model_geometry
        )

        # Write model summary TOML file specifying Hamiltonian that will be simulated.
        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (electron_phonon_model,)
        )

        # Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize electron-phonon parameters.
        electron_phonon_parameters = ElectronPhononParameters(
            β = β, Δτ = Δτ,
            electron_phonon_model = electron_phonon_model,
            tight_binding_parameters = tight_binding_parameters,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        # Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        # Initialize the electron-phonon interaction related measurements.
        initialize_measurements!(measurement_container, electron_phonon_model)

        # Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens",
            time_displaced = true,
            pairs = [
                # Measure green's functions for all pairs or orbitals.
                (1, 1),
            ]
        )

        # Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "phonon_greens",
            time_displaced = true,
            pairs = [
                # Measure green's functions for all pairs or orbitals.
                (1, 1),
            ]
        )

        # Initialize density correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "density",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1), 
            ]
        )

        # Initialize the pair correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "pair",
            time_displaced = false,
            integrated = true,
            pairs = [
                # Measure local s-wave pair susceptibility associated with
                # each orbital in the unit cell.
                (1, 1), 
            ]
        )

        # Initialize the spin-z correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "spin_z",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1), 
            ]
        )

        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "current",
            time_displaced = true,
            integrated = false,
            pairs = [(1, 1)] # hopping ID pair for y-direction hopping
        )

        # Initialize the sub-directories to which the various measurements will be written.
        initialize_measurement_directories(simulation_info, measurement_container)

        # Allocate a single FermionPathIntegral for both spin-up and down electrons.
        fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

        # Initialize FermionPathIntegral type to account for electron-phonon interaction.
        initialize!(fermion_path_integral, electron_phonon_parameters)

        # allocate and initialize propagators for each imaginary time slice
        B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

        # initialize fermion greens calculator
        fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

        # initialize alternate fermion greens calculator required for performing various global updates
        fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

        # calculate/initialize equal-time green's function matrix
        G = zeros(eltype(B[1]), size(B[1]))
        logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

        # initialize G(τ,τ), G(τ,0) and G(0,τ) Green's function matrices for both spin species
        G_ττ = similar(G)
        G_τ0 = similar(G)
        G_0τ = similar(G)

        # initialize hamitlonian/hybrid monte carlo (HMC) updater
        hmc_updater = HMCUpdater(
            electron_phonon_parameters = electron_phonon_parameters,
            G = G, Nt = Nt, Δt = Δt, nt = nt, reg = reg
        )
        # intialize errors corrected by numerical stabilization to zero
        δG = zero(typeof(logdetG))
        δθ = zero(typeof(sgndetG))

        # Initialize variables to record acceptance rates for various udpates.
        additional_info["hmc_acceptance_rate"] = 0.0
        additional_info["reflection_acceptance_rate"] = 0.0
        additional_info["swap_acceptance_rate"] = 0.0

        # Initialize variables to record the average number of CG iterations
        # for each type of update and measurements.
        additional_info["hmc_iters"] = 0.0
        additional_info["reflection_iters"] = 0.0
        additional_info["swap_iters"] = 0.0
        additional_info["measurement_iters"] = 0.0

        t_init += time()

        t_therm = - time()
        # Iterate over number of thermalization updates to perform.
        # println("Thermalizing system...\n")
        for n in 1:N_therm
            # perform hmc update
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng, initialize_force = true
            )
            
    
            # record accept/reject outcome
            additional_info["hmc_acceptance_rate"] += accepted
    
            # # perform swap update
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng,
                phonon_type_pairs = ((phonon_1_id, phonon_1_id))
            )
    
            # record accept/reject outcome
            additional_info["swap_acceptance_rate"] += accepted
    
            # perform reflection update
            (accepted, logdetG, sgndetG) = reflection_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng, phonon_types = (phonon_1_id,)
            )
    
        end
        t_therm += time() 
        # Calculate the bin size.
        bin_size = N_updates ÷ N_bins

        t_updates -= time()
        # Iterate over bins.
        for bin in 1:N_bins
            # println("Bin $bin of $N_bins")
            # Iterate over update sweeps and measurements in bin.
            for n in 1:bin_size
                
                
                # perform hmc update
                (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                    G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                    B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng, initialize_force = true
                )

                # record accept/reject outcome
                additional_info["hmc_acceptance_rate"] += accepted

                # # perform swap update
                (accepted, logdetG, sgndetG) = swap_update!(
                    G, logdetG, sgndetG, electron_phonon_parameters,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                    B = B, rng = rng,
                    phonon_type_pairs = ((phonon_1_id, phonon_1_id))
                )

                # record accept/reject outcome
                additional_info["swap_acceptance_rate"] += accepted

                # perform reflection update
                (accepted, logdetG, sgndetG) = reflection_update!(
                    G, logdetG, sgndetG, electron_phonon_parameters,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                    B = B, rng = rng, phonon_types = (phonon_1_id,)
                )

                t_meas -= time()
                # Make measurements.
                (logdetG, sgndetG, δG, δθ) = make_measurements!(
                    measurement_container,
                    logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
                    fermion_path_integral = fermion_path_integral,
                    fermion_greens_calculator = fermion_greens_calculator,
                    B = B, δG_max = δG_max, δG = δG, δθ = δθ,
                    model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                    coupling_parameters = (electron_phonon_parameters,)
                )
                t_meas +=  time()
                # Record the average number of iterations per CG solve for measurements.
                # additional_info["measurement_iters"] += iters
            end

            t_write -= time()
            # Write the bin-averaged measurements to file.
            write_measurements!(
                measurement_container = measurement_container,
                simulation_info = simulation_info,
                model_geometry = model_geometry,
                bin = bin,
                bin_size = bin_size,
                Δτ = Δτ
            )
            t_write += time()
        end
        MPI.Barrier(comm)
        t_updates += time()
        t_total += time()
        println("Finished binning measurements.\n")
        # Calculate acceptance rates.
        additional_info["hmc_acceptance_rate"] /= (N_updates + N_therm)
        additional_info["reflection_acceptance_rate"] /= (N_updates + N_therm)
        additional_info["swap_acceptance_rate"] /= (N_updates + N_therm)

        # Calculate average number of CG iterations.
        additional_info["hmc_iters"] /= (N_updates + N_therm)
        additional_info["reflection_iters"] /= (N_updates + N_therm)
        additional_info["swap_iters"] /= (N_updates + N_therm)
        additional_info["measurement_iters"] /= N_updates
        additional_info["t_total"] = t_total
        additional_info["t_init"] = t_init
        additional_info["t_meas"] = t_meas
        additional_info["t_write"] = t_write
        additional_info["t_therm"] = t_therm
        additional_info["t_updates"] = t_updates
        # Write simulation metadata to simulation_info.toml file.
        save_simulation_info(simulation_info, additional_info)

        MPI.Barrier(comm)
        # Process the simulation results, calculating final error bars for all measurements,
        # writing final statisitics to CSV files.
        process_measurements(comm, simulation_info.datafolder, N_bins , time_displaced = true)

        # # Merge binary files containing binned data into a single file.
        # compress_jld2_bins(folder = simulation_info.datafolder)
    catch e
        println("Error: $(pID) $(e)")
    end
    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end # end of run_simulation function

# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Run the simulation.
    run_simulation(
        sID       = 1,
        Ω         = 0.5,
        λ         = 0.375,
        μ         = -0.38129762,
        
        N_therm   = 1500,
        N_updates = 1000,
        N_bins    = 10,
    )
end
