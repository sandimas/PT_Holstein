using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework   as dqmcf
import MuTuner                    as mt
using Random
using Printf
using MPI
using LinearAlgebra

# Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    Ω, # Phonon energy.
    λ_min,
    λ_max, # Electron-phonon coupling.
    N_λ, # Total number of λ values to simulate.
    n, # Filling
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_updates, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
    checkpoint_freq, # Frequency with which checkpoint files are written in hours.
    runtime_limit = Inf, # Simulation runtime limit in hours.
    n_procs = 1, # Number of MPI processes.
    pID = 0, # Process ID.
    Δτ = 0.1, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    filepath = ".", # Filepath to where data folder will be created.
    N_retherm = 0, # Number of re-thermalization updates to perform.
    N_temper_freq = 25, # Frequency with which swap updates are performed.
    do_temper = false # Whether to perform swap updates.

)
    mkpath(filepath) # Create the data folder if it does not exist.
    BLAS.set_num_threads(1)
    # set seed
    seed = 1000 + (sID * n_procs) + pID
    μ0 = -.15 # Chemical potential.
    λs = range(λ_min, λ_max, length = N_λ) # λ values to simulate.

if pID == 0
@printf "Running simulation with parameters:\n"
@printf "sID: %d, Ω: %.2f, λ_max: %.3f, N_λ: %d, n: %.2f, L: %d, β: %.2f\n" sID Ω λ_max N_λ n L β
@printf "N_therm: %d, N_updates: %d, N_bins: %d, checkpoint_freq: %.2f hours\n" N_therm N_updates N_bins checkpoint_freq
@printf "runtime_limit: %.2f hours\n" runtime_limit
@printf "N_retherm: %d, N_temper_freq: %d, do_temper: %s\n" N_retherm N_temper_freq do_temper
println("λs: ", collect(λs))
end
MPI.Barrier(comm) # Synchronize processes before proceeding.
    # exit()

    n_tier = length(λs)
    n_walker_per_tier = div(n_procs,n_tier)
    mpi_rank_tier = div(pID, n_walker_per_tier)
    mpi_rank_pID = pID % n_walker_per_tier

    λi = 1 + mpi_rank_tier # λ index for this process.
    λ = λs[λi] # λ value for this process.
    α = Ω * sqrt(λ*8.0)   # Holstein coupling strength.

    shift_val = 0 # Shift value for tempering λ.

    # Record when the simulation began.
    start_timestamp = time()

    # Convert runtime limit from hours to seconds.
    runtime_limit = runtime_limit * 60.0^2

    # Convert checkpoint frequency from hours to seconds.
    checkpoint_freq = checkpoint_freq * 60.0^2

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_2D_w%.2f_l%.3f_n%.2f_L%d_b%.2f" Ω λ n L β
    datafolder_prefix = datafolder_prefix * (do_temper ? "_temp" : "_reg")

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID,
        pID = mpi_rank_pID
    )

    resume = isfile(joinpath(simulation_info.datafolder, "checkpoint_pID-$(mpi_rank_pID).jld2"))

    if mpi_rank_pID == 0
        initialize_datafolder(simulation_info)
        
    end
    MPI.Barrier(comm) # Synchronize processes before proceeding.
    # Initialize the directory the data will be written to.

    # If starting a new simulation i.e. not resuming a previous simulation.
    if !resume

        # Begin thermalization updates from start.
        n_therm = 1

        # Begin measurement updates from start.
        n_updates = 1

        # Initialize random number generator
        rng = Xoshiro(seed)

        # Initialize additiona_info dictionary
        metadata = Dict()

        # Record simulation parameters.
        metadata["N_therm"] = N_therm
        metadata["N_updates"] = N_updates
        metadata["N_bins"] = N_bins
        metadata["n_stab"] = n_stab
        metadata["dG_max"] = δG_max
        metadata["symmetric"] = symmetric
        metadata["checkerboard"] = checkerboard
        metadata["seed"] = seed
        metadata["hmc_acceptance_rate"] = 0.0
        metadata["reflection_acceptance_rate"] = 0.0
        metadata["swap_acceptance_rate"] = 0.0
        metadata["N_swap_up"] = 0.0
        metadata["N_swap_dn"] = 0.0
        metadata["N_swap_attempts"] = 0.0
        # Define the unit cell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [[1.0,0.0],
                            [0.0,1.0]],
            basis_vecs   = [[0.,0.]]
        )

        # Define finite lattice with periodic boundary conditions.
        lattice = lu.Lattice(
            L = [L, L],
            periodic = [true, true]
        )

        # Initialize model geometry.
        model_geometry = ModelGeometry(unit_cell, lattice)

        # define first nearest-neighbor bond
        bond_x = lu.Bond(orbitals = (1,1), displacement = [1,0])
        bond_x_id = add_bond!(model_geometry, bond_x)

        # define second nearest-neighbor bond
        bond_y = lu.Bond(orbitals = (1,1), displacement = [0,1])
        bond_y_id = add_bond!(model_geometry, bond_y)


        # Set neartest-neighbor hopping amplitude to unity,
        # setting the energy scale in the model.
        t = 1.0

        # Define the honeycomb tight-binding model.
        tight_binding_model = TightBindingModel(
            model_geometry = model_geometry,
            t_bonds        = [bond_x, bond_y], # defines hopping
            t_mean         = [t, t], # defines corresponding hopping amplitude
            μ              = μ0, # set chemical potential
            ϵ_mean         = [0.0] # set the (mean) on-site energy
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

        chemical_potential_tuner = mt.init_mutunerlogger(
            target_density = n,
            inverse_temperature = β,
            system_size = lu.nsites(unit_cell, lattice),
            initial_chemical_potential = μ0,
            complex_sign_problem = false,
            memory_fraction = 0.5, # fraction of memory to use for storing measurements
            intensive_energy_scale = 1.0#α^2/Ω^2,
        )
        # chemical_potential_tuner = mt.MuTunerLogger(n₀ = n, β = β, V =L*L, u₀ = α^2/Ω^2, μ₀ = μ0, c = 0.5)


        # Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        # Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        # Initialize the electron-phonon interaction related measurements.
        initialize_measurements!(measurement_container, electron_phonon_model)

        # measure time-displaced green's function
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens", # Gup = Gdn, so just measure Gup
            time_displaced = true,
            pairs = [(1, 1)]
        )

        # measure time-displaced phonon green's function
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "density",
            time_displaced = true,
            pairs = [(1,1)]
        )

        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "spin_z",
            time_displaced = true,
            pairs = [(1, 1)]
        )

        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "current",
            time_displaced = true,
            pairs = [(1, 1)] # hopping ID pair for y-direction hopping
        )
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "pair",
            time_displaced = true,
            pairs = [(1, 1)] # hopping ID pair for y-direction hopping
        )

        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "phonon_greens",
            time_displaced = true,
            pairs = [(1, 1)] # hopping ID pair for y-direction hopping
        )
       

        # Initialize the sub-directories to which the various measurements will be written.
        initialize_measurement_directories(comm, simulation_info, measurement_container)

        # Write initial checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm, n_updates,
            tight_binding_parameters, electron_phonon_parameters, 
            chemical_potential_tuner,
            measurement_container, model_geometry, metadata, rng
        )

    # If resuming a previous simulation.
    else

        # Load the checkpoint file.
        checkpoint, checkpoint_timestamp = read_jld2_checkpoint(simulation_info)

        # Unpack contents of checkpoint dictionary.
        tight_binding_parameters    = checkpoint["tight_binding_parameters"]
        electron_phonon_parameters  = checkpoint["electron_phonon_parameters"]
        measurement_container       = checkpoint["measurement_container"]
        model_geometry              = checkpoint["model_geometry"]
        metadata                    = checkpoint["metadata"]
        rng                         = checkpoint["rng"]
        n_therm                     = checkpoint["n_therm"]
        n_updates                   = checkpoint["n_updates"]
        chemical_potential_tuner    = checkpoint["chemical_potential_tuner"]
        if pID == 0
            @printf "Resuming simulation from checkpoint file.\n"
            @printf "n_therm: %d / %d, n_updates: %d / %d\n" n_therm N_therm  n_updates N_updates
        end
    end

    # Allocate a single FermionPathIntegral for both spin-up and down electrons.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize FermionPathIntegral type to account for electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Initialize imaginary-time propagators for all imaginary-time slices.
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator type.
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

    # Initialize alternate fermion greens calculator required for performing EFA-HMC, reflection and swap updates below.
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

    # Allcoate equal-time electron Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize electron Green's function matrx, also calculating the matrix determinant as the same time.
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)


    # Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    # Initialize diagonostic parameters to asses numerical stability.
    δG = zero(logdetG)
    δθ = zero(sgndetG)

    # Number of fermionic time-steps in HMC update.
    Nt = 20

    # Fermionic time-step used in HMC update.
    Δt = 0.05 # π/(2*Ω*Nt)
# if pID == 0; println("Δt: $(Δt)"); end

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = HMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt, nt = 10, reg = 1.0
    )

    MPI.Barrier(comm) # Synchronize processes before proceeding.
    if pID == 0; println("begin warms"); end
    
    # Iterate over number of thermalization updates to perform.
    for update in n_therm:N_therm

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        # Perform a swap update.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        # Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted
# if( pID == 29 ); println("$(pID) $(update) $(metadata["hmc_acceptance_rate"]) $(δG) $(δθ)"); end
        # Update the chemical potential to achieve the target density.
        (logdetG, sgndetG) = update_chemical_potential!(
            G, logdetG, sgndetG;
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B
        )

        # Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm  = update + 1,
            n_updates = 1,
            tight_binding_parameters, electron_phonon_parameters, 
            chemical_potential_tuner,
            measurement_container, model_geometry, metadata, rng
        )
    end
# MPI.Barrier(comm) # Synchronize processes before proceeding.
# if (pID == 29 ); println("$(pID) $(metadata["hmc_acceptance_rate"]) $(δG) $(δθ)"); end
# MPI.Barrier(comm) # Synchronize processes before proceeding.
# if ( pID == 0); println("$(pID) $(metadata["hmc_acceptance_rate"]) $(δG) $(δθ)"); end

# MPI.Barrier(comm) # Synchronize processes before proceeding.
# exit()
    # Reset diagonostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetG)
    δθ = zero(sgndetG)

    # Calculate the bin size.
    bin_size = N_updates ÷ N_bins
    MPI.Barrier(comm) # Synchronize processes before proceeding.
    if pID == 0; println("begin updates"); end
    
    # Iterate over updates and measurements.
    for update in n_updates:N_updates

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        # Perform a swap update.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        # Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

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

        # Write the bin-averaged measurements to file if update ÷ bin_size == 0.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            update = update,
            bin_size = bin_size,
            Δτ = Δτ
        )

        # Update the chemical potential to achieve the target density.
        (logdetG, sgndetG) = update_chemical_potential!(
            G, logdetG, sgndetG;
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B
        )

        if do_temper && (update % N_temper_freq == 0) && update != N_updates
            # Perform a tempering update.
if pID == 0; 
    println("tempering $(metadata["N_swap_attempts"] + 1.0) $(pID) $(update) $(metadata["N_swap_dn"])"); 
end


            (logdetG, sgndetG, shift_val) = temper_λ!(
                comm,
                G=G, 
                B=B, 
                logdetG =logdetG, 
                sgndetG=sgndetG,
                fermion_greens_calculator=fermion_greens_calculator,
                fermion_greens_calculator_alt=fermion_greens_calculator_alt,
                electron_phonon_parameters=electron_phonon_parameters,
                fermion_path_integral=fermion_path_integral,
                holstein=true, 
                ssh = false,
                n_tier = n_tier,
                n_walker_per_tier=n_walker_per_tier,
                mpi_rank_tier=mpi_rank_tier,
                mpi_rank_pID=mpi_rank_pID,
                shift_val=shift_val,
                rng=rng,
                additional_info=metadata
            )

           
            

        end
        # Rethermalize the system after tempering update.
        for rewarm in 1:N_retherm
            # Perform a reflection update.
            (accepted, logdetG, sgndetG) = reflection_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng
            )

            # Perform a swap update.
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng
            )

            # Perform an HMC update.
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )


            # Update the chemical potential to achieve the target density.
            (logdetG, sgndetG) = update_chemical_potential!(
                G, logdetG, sgndetG;
                chemical_potential_tuner = chemical_potential_tuner,
                tight_binding_parameters = tight_binding_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B
            )
        end
        # Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm  = N_therm + 1,
            n_updates = update + 1,
            tight_binding_parameters, electron_phonon_parameters,
            chemical_potential_tuner,
            measurement_container, model_geometry, metadata, rng
        )

    end

    # Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_updates + N_therm)
    metadata["reflection_acceptance_rate"] /= (N_updates + N_therm)
    metadata["swap_acceptance_rate"] /= (N_updates + N_therm)
    
    # Record largest numerical error encountered during simulation.
    metadata["dG"] = δG

    # Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    MPI.Barrier(comm) # Synchronize processes before proceeding.
    if pID == 0
        @printf "\nSimulation complete. Processing results...\n"
    end
    if mpi_rank_pID == 0
        process_measurements(simulation_info.datafolder, N_bins, time_displaced = true)
        correlations = [
            "greens",  
            "density",
            "spin_z",
            "current",
            "pair",
            "phonon_greens"
        ]
        spaces = [
            "momentum",
            "position"
        ]

        for correlation in correlations
            for space in spaces
                correlation_bins_to_csv(
                    folder = simulation_info.datafolder,
                    correlation = correlation,
                    type = "time-displaced",
                    space = space,
                    write_index_key = true
                )
            end
        end 
    end
    MPI.Barrier(comm) # Synchronize processes before proceeding.

    # Merge binary files containing binned data into a single file.
    # compress_jld2_bins(comm, folder = simulation_info.datafolder)

    # Rename the data folder to indicate the simulation is complete.
    simulation_info = rename_complete_simulation(
        comm, simulation_info,
        delete_jld2_checkpoints = true
    )
    MPI.Barrier(comm) # Synchronize processes before proceeding.

    return nothing
end # end of run_simulation function





# This function probablistically swaps states between λ tiers
# It does so in a "waterfall" manner. 
function temper_λ!(
    mpi_comm;
    G, 
    B,
    logdetG::Real,
    sgndetG::Real,
    fermion_greens_calculator,
    fermion_greens_calculator_alt,
    electron_phonon_parameters,
    fermion_path_integral,
    holstein::Bool,
    ssh::Bool,
    n_tier::Int,
    n_walker_per_tier::Int,
    mpi_rank_tier::Int,
    mpi_rank_pID::Int,
    shift_val::Int,
    rng,
    additional_info 
)
    # Temporary storage for state swaps
    x_tmp = similar(electron_phonon_parameters.x)
    G_tmp = fermion_greens_calculator_alt.G′

    # Array to pass new and old values of phonon action, logdetG, and a status code between MPI ranks
    weights = zeros(Float64,5)

    additional_info["N_swap_attempts"] += 1.0
    # mpi_comm = MPI.COMM_WORLD

    # Loop over UPPER tiers
    # In this we use the convention that lower tiers have higher λ values
    for tier ∈ 0:n_tier-2

        ## sync across MPI ranks
        MPI.Barrier(mpi_comm)

        ## store current phonon fields
        x_old = copy(electron_phonon_parameters.x)

        ## UPPER tier
        ## this tier has a lower λ value in the pair swapping
        if tier == mpi_rank_tier
            ## 0 - Initialization
            
            ## duplicate state of fermion greens calculator
            copyto!(fermion_greens_calculator_alt,fermion_greens_calculator)

            ## ensure variables are declared outside of try statements
            logdetG_upper_old = logdetG
            logdetG_upper_new = 0.0
            sgndetG_upper_new = sgndetG
            good_upper = true

            ## MPI rank of LOWER to swap with 
            lower = n_walker_per_tier*(tier+1)  + ((mpi_rank_pID + shift_val) % n_walker_per_tier)
            
            ## 1 - Exchange X fields with LOWER
            MPI.Send(electron_phonon_parameters.x, mpi_comm, dest=lower)
            MPI.Recv!(x_tmp, mpi_comm)

            ## 2 - using X fields from LOWER, calculate state for UPPER
            ## Old bosonic action
            Sb_upper_old = SmoQyDQMC.bosonic_action(electron_phonon_parameters)

            ## update path integral with new X fields
            SmoQyDQMC.update!(fermion_path_integral, electron_phonon_parameters, x_tmp, x_old)
            copyto!(electron_phonon_parameters.x, x_tmp)

            ## New bosonic action
            Sb_upper_new = SmoQyDQMC.bosonic_action(electron_phonon_parameters)

            ## Update propagators (B matrices)
            calculate_propagators!(B, fermion_path_integral, calculate_exp_K = ssh, calculate_exp_V = holstein)

            ## Calculate Greens functions, catch failure
            try
                logdetG_upper_new, sgndetG_upper_new = dqmcf.calculate_equaltime_greens!(G_tmp, fermion_greens_calculator_alt,B)
            catch
                good_upper = false
            end

            ## 3 - Receive weights from LOWER
            MPI.Recv!(weights, mpi_comm)
            (logdetG_lower_old,Sb_lower_old,logdetG_lower_new,Sb_lower_new,good_lower) = weights

            ## 4 - Calculate if we should exchange
            ## log of the probability. Using logs of detG and not exponentiating the action helps with numerical stability

            S_new = Sb_upper_new + Sb_lower_new + 2.0*(logdetG_upper_new + logdetG_lower_new)
            S_old = Sb_upper_old + Sb_lower_old + 2.0*(logdetG_upper_old + logdetG_lower_old)

            ΔS = S_new - S_old
            w = exp(-ΔS)
            r_P = rand(rng,Float64)
# if pID == 0
#     @printf "new Sb0: %.2f, Sb1: %.2f, detG0: %.2f, detG1: %.2f\n" Sb_upper_new Sb_lower_new logdetG_upper_new logdetG_lower_new
#     @printf "old Sb0: %.2f, Sb1: %.2f, detG0: %.2f, detG1: %.2f\n" Sb_upper_old Sb_lower_old logdetG_upper_old logdetG_lower_old
#     @printf "ΔS: %.2f, w: %.2f, r_P %.2f\n" ΔS w r_P
#     # sleep(5.0)
# end

            if isfinite(w) && good_upper && good_lower == 1.0
                P = [w,r_P]
            else
                P = [0.0,1.0]
            end

            ## 5 - Send P to LOWER
            MPI.Isend(P,mpi_comm,dest=lower)

            ## 6 - Finalize acceptance of update or revert back on rejection
            if P[1] > P[2] # accept
                additional_info["N_swap_dn"] += 1.0
                logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator,B)
                
            else ## reject
                SmoQyDQMC.update!(fermion_path_integral,electron_phonon_parameters, x_old,electron_phonon_parameters.x)
                copyto!(electron_phonon_parameters.x,x_old)
                calculate_propagators!(B,fermion_path_integral, calculate_exp_K = ssh, calculate_exp_V = holstein)
            end
        end ## UPPER tier



        ## LOWER tier
        if tier+1 == mpi_rank_tier
            ## 0 - Initialization
            upper = n_walker_per_tier*tier  + ((mpi_rank_pID - shift_val + n_walker_per_tier) % n_walker_per_tier)

            ## duplicate state of fermion greens calculator
            copyto!(fermion_greens_calculator_alt , fermion_greens_calculator)

            ## ensure variables are declared outside of try statements
            sgndetG_lower_new = sgndetG

            ## 1 - Exchange X fields with UPPER
            MPI.Recv!(x_tmp,mpi_comm)
            MPI.Send(electron_phonon_parameters.x,mpi_comm,dest=upper)

            ## 2 - Calculate updated state 
            
            ## Old logdetG, Sb
            weights[1] = logdetG
            weights[2] = SmoQyDQMC.bosonic_action(electron_phonon_parameters)

            ## Update phonon fields to UPPER
            SmoQyDQMC.update!(fermion_path_integral, electron_phonon_parameters, x_tmp, x_old)
            copyto!(electron_phonon_parameters.x, x_tmp)

            ## New bosonic action
            weights[4] = SmoQyDQMC.bosonic_action(electron_phonon_parameters)

            ## Update propagators (B matrices)
            calculate_propagators!(B,fermion_path_integral, calculate_exp_K = ssh, calculate_exp_V = holstein)

            ## Calculate Greens functions, catch failure
            try 
                weights[3], sgndetG_lower_new = dqmcf.calculate_equaltime_greens!(G_tmp, fermion_greens_calculator_alt,B)
                weights[5] = 1.0
            catch
                weights[3] = 0.0
                weights[5] = 0.0
            end

            ## 3 - Send weights to UPPER
            MPI.Send(weights,mpi_comm,dest=upper)

            ## 4 - Nothing, UPPER calculating update acceptance

            ## 5 - Receiv accept/reject
            P = zeros(Float64, 2) 
            MPI.Recv!(P, mpi_comm)

            ## 6 - Finalize acceptance / rejection
            if P[1] > P[2] # accept
                additional_info["N_swap_up"] += 1.0
                logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator,B)
            else ## reject
                SmoQyDQMC.update!(fermion_path_integral,electron_phonon_parameters, x_old,electron_phonon_parameters.x)
                copyto!(electron_phonon_parameters.x,x_old)
                calculate_propagators!(B,fermion_path_integral, calculate_exp_K = ssh, calculate_exp_V = holstein)
            end

        end ## LOWER tier

        ## Handle 

    end ## tier ∈ 0:n_tier-2

    ## Update shift value
    shift_val = (shift_val + 1) % n_walker_per_tier

    return (logdetG, sgndetG, shift_val)

end ## function temper_λ!()




# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Initialize MPI
    MPI.Init()

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Get the number of processes.
    n_procs = MPI.Comm_size(comm)

    # Get the process ID.
    pID = MPI.Comm_rank(comm)

    # Run the simulation.
    run_simulation(
        comm;
        sID             = parse(Int,     ARGS[1]),  # Simulation ID.
        Ω               = parse(Float64, ARGS[2]),  # Phonon energy.
        λ_min           = parse(Float64, ARGS[3]),  # Minimum electron-phonon coupling.
        λ_max           = parse(Float64, ARGS[4]),  # Maximum electron-phonon coupling.
        N_λ             = parse(Int,     ARGS[5]),  # Number of λ values to simulate.
        n               = parse(Float64, ARGS[6]),  # Number of λ values to simulate.
        L               = parse(Int,     ARGS[7]),  # System size.
        β               = parse(Float64, ARGS[8]),  # Inverse temperature.
        N_therm         = parse(Int,     ARGS[9]),  # Number of thermalization updates.
        N_updates       = parse(Int,     ARGS[10]), # Total number of measurements and measurement updates.
        N_bins          = parse(Int,     ARGS[11]), # Number of times bin-averaged measurements are written to file.
        checkpoint_freq = parse(Float64, ARGS[12]), # Frequency with which checkpoint files are written in hours.
        # runtime_limit   = parse(Float64, ARGS[14]), # Simulation runtime limit in hours.
        n_procs         = n_procs,
        pID             = pID,
        filepath        = ARGS[13], # Filepath to where data folder will be created.
        N_retherm       = parse(Int,     ARGS[14]), # Number of re-thermalization updates.
        N_temper_freq   = parse(Int,     ARGS[15]), # Frequency with which tempering updates are performed.
        do_temper       = parse(Bool,    ARGS[16]), # Whether to perform tempering updates.
    )

    # Finalize MPI.
    MPI.Finalize()
end
