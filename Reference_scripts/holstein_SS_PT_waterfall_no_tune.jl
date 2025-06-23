# # Single site Holstein model with Parallel Tempering
# In this script we simulate a single site Holstein model with no hopping. 
# This creates a double well potential or "Mexican Hat" potential. 
# Normally reflection updates would suffice to move between the two wells. 
# However, here we use Parallel Tempering, instead. 
#
# In this example will temper across Holstein model coupling strength after each bin. 
# We will use a "waterfall" technique where each tier of λ values will be swapped with the next tier.
# So tempering will happen across tier 0+1, then 1+2, then 2+3, etc.

using LinearAlgebra
using Random
using Printf
using MPI

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
import SmoQyDQMC.MuTuner           as mt


# top level function to run simulation
function run_simulation(parameter_dict)
    
	# Load model parameters
    ## dimensionless coupling constant array
    λs = parameter_dict["lambdas"]

    ## 1D chain length
    L = parameter_dict["L"]

    ## Inverse temperature
    β = parameter_dict["beta"]

    ## Phonon Frequency
    Ω = parameter_dict["Omega"]
    
    ## nearest-neighbor hopping amplitude
    t = parameter_dict["t"]
    
    ## holstein coupling constant
    αs = @. Ω*sqrt(4*λs)
    
    ## chemical potential
    μ = 0.0

    ## target filling
    n = parameter_dict["avg_n"]

    ## discretization in imaginary time
    Δτ = parameter_dict["dtau"]

    ## evaluate length of imaginary time axis
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    ## whether to use the checkerboard approximation
    checkerboard = parameter_dict["checkerboard"]

    ## whether to use symmetric propagator defintion 
    symmetric = parameter_dict["symmetric"]

    ## initial stabilization frequency
    n_stab = parameter_dict["n_stab"]

    ## max allowed error in green's function
    δG_max = parameter_dict["dG_max"]

    ## number of thermalization/burnin updates
    N_burnin = parameter_dict["N_burnin"]

    ## number of simulation updates
    N_updates = parameter_dict["N_updates"]

    ## Parallel tempering specific parameter
    N_burnin_after_swap = parameter_dict["N_burnin_after_swap"]
    
    ## number of bins
    N_bins = parameter_dict["N_bins"]

    ## Updates per bin
    bin_size = div(N_updates,N_bins)

    ## number of fermionic time-steps in HMC trajectory
    Nt = parameter_dict["Nt"]

    swap_frequency = parameter_dict["swap_frequency"]

    ## hyrbid/hamiltonian monte carlo (HMC) update time-step
    Δt = 1/(Nt*Ω)
    
    ## calculate length of imaginary time axis
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    # MPI initialization
    MPI.Init()
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = MPI.Comm_rank(mpi_comm)
    mpi_n_rank = MPI.Comm_size(mpi_comm)

    # Calculate this MPI rank is in grid of λs
    # Note, I start our (λ,pID) grid at (0,0) as in MPI convention, not (1,1) as in Julia 
    n_tier = size(λs,1)
    n_walker_per_tier = div(mpi_n_rank,n_tier)
    mpi_rank_tier = div(mpi_rank,n_walker_per_tier)
    mpi_rank_pID = mpi_rank%n_walker_per_tier

    ## RNG setup
    seed = abs(parameter_dict["base_seed"] + mpi_rank)
    rng = Xoshiro(seed)
    
    ## Initialize a dictionary to store additional information about the simulation.
    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "hmc_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "Nt" => Nt,
        "dt" => Δt,
        "seed" => seed,
        "swap_up_rate" => 0.0,
        "swap_dn_rate" => 0.0,
        "N_swap_up" => 0.0,
        "N_swap_dn" => 0.0,
        
    )

    ## Setup output folders
    suffix = (parameter_dict["do_swaps"]) ? "_swaps_on" : "_swaps_off"
    datafolder_prefix = @sprintf "holstein_SS_w%.2f_l%.2f_n%.2f_L%d_b%.2f" Ω λs[1+mpi_rank_tier] n L β
    datafolder_prefix *= suffix
    datafolder_path = "./SS_output_new"
    mkpath(datafolder_path)
    MPI.Barrier(mpi_comm)

    ## Initialize simulation folder
    simulation_info = SimulationInfo(
            filepath = datafolder_path,
            datafolder_prefix = datafolder_prefix,
            pID = mpi_rank_pID,
            sID= 1
    
    )
    initialize_datafolder(simulation_info)

    # My personal preference: I keep a copy of the script in the data folder for reproducibility
    try
        cp(@__FILE__,joinpath(datafolder_path,datafolder_prefix*"_script.jl"),force=true)
    catch
    end

    # Set up geometries and model
    ## define singleband unit cell
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0,0.0],[0.0,1.0]],
                            basis_vecs   = [[0.0,0.0]])

    ## define size of lattice (only supports periodic b.c. for now)
    lattice = lu.Lattice(
        L = [L,L],
        periodic = [true,true] # must be true for now
    )

    ## define model geometry
    model_geometry = ModelGeometry(unit_cell, lattice)
    ## calculate number of orbitals in the lattice
    N = lu.nsites(unit_cell, lattice)

    ## 2D lattice
    ## define first nearest-neighbor bond
    bond_x = lu.Bond(orbitals = (1,1), displacement = [1,0])
    bond_x_id = add_bond!(model_geometry, bond_x)
    bond_y = lu.Bond(orbitals = (1,1), displacement = [0,1])
    bond_y_id = add_bond!(model_geometry, bond_x)
    
    ## Hopping integrals t = 0 for single site
    ## define non-interacting tight binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_x,bond_y],
        t_mean = [0.0,0.0],
        μ = μ,
        ϵ_mean = [0.]
    )
    
    ## Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    ## define phonon mode for each orbital in unit cell
    phonon = PhononMode(orbital = 1, Ω_mean = Ω)

    ## Add the phonon mode definition to the electron-phonon model.
    phonon_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon
    )

    ## Define a on-site Holstein coupling between the electron and the local dispersionless phonon mode.
    holstein_coupling = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
    	α_mean = αs[1+mpi_rank_tier]
    )
    ## Add the Holstein coupling definition to the model.
    holstein_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_coupling,
    	model_geometry = model_geometry
    )

    ## Write a model summary to file.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (electron_phonon_model,)
    )


    #################################################
    ### INITIALIZE FINITE LATTICE MODEL PARAMETERS ##
    #################################################

    ## Initialize tight-binding parameters.
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    ## Initialize electron-phonon parameters.
    electron_phonon_parameters = ElectronPhononParameters(
        β = β, Δτ = Δτ,
        electron_phonon_model = electron_phonon_model,
        tight_binding_parameters = tight_binding_parameters,
        model_geometry = model_geometry,
        rng = rng
    )
    ##############################
    ### INITIALIZE MEASUREMENTS ##
    ##############################

    ## Initialize the container that measurements will be accumulated into.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    ## Initialize the tight-binding model related measurements, like the hopping energy.
    initialize_measurements!(measurement_container, tight_binding_model)

    ## Initialize the electron-phonon interaction related measurements.
    initialize_measurements!(measurement_container, electron_phonon_model)

    ## Initialize the single-particle electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    ## Initialize time-displaced phonon Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "phonon_greens",
        time_displaced = true,
        pairs = [(phonon_id, phonon_id)]
    )

    ## Initialize density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    ## Initialize the pair correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    ## Initialize the spin-z correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    ## Initialize the sub-directories to which the various measurements will be written.
    
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    

    #############################
    ### SET-UP DQMC SIMULATION ##
    #############################

    # Note that the spin-up and spin-down electron sectors are equivalent in the Holstein model
    # without Hubbard interaction. Therefore, there is only a single Fermion determinant
    # that needs to be calculated. This fact is reflected in the code below.

    ## Allocate fermion path integral type.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    ## Initialize the fermion path integral type with respect to electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    ## Allocate and initialize propagators for each imaginary time slice.
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    ## Initialize fermion greens calculator.
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

    ## Initialize alternate fermion greens calculator required for performing various global updates.
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

    ## Allocate equal-time Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    ## Initialize equal-time Green's function matrix
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    ## Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    ## Initialize variables to keep track of the largest numerical error in the
    ## Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    ## Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt
    )

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################
    
    if (mpi_rank==0); println("Begin Burnin"); end

    ## Iterate over burnin/thermalization updates.
    for n in 1:N_burnin
        ## Note on Reflection updates
        ## For the double well potential reflection updates are usually necessary for ergodicity
        ## We do not use them as Parallel tempering allows movement across the barrier
        
        ## Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        ## Record whether the HMC update was accepted or rejected.
        additional_info["hmc_acceptance_rate"] += accepted

        ## Print warm-up progress.
        if (mpi_rank==0); 
            if (n % (div(N_burnin,10))==0)
                println(" warm ", n, " of ",N_burnin); 
            end
        end    
    end ## for n in 1:N_burnin

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    
    ## Re-initialize variables to keep track of the largest numerical error in the
    ## Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    ## Track the offset for tier swaps. Allows diagonal swapping between tiers
    shift_val = 0

    # Counter for tempering swaps every nth measurements
    swap_counter = 0
    N_swaps = 0
    
    MPI.Barrier(mpi_comm)
    if (mpi_rank==0); println("Begin measurements"); end
    
    ## Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in 1:N_bins

        ## Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            ## Again, we skip reflection updates for the same reason as above
            
            ## Perform an HMC update.
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            ## Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

            ## Make measurements.
            (logdetG, sgndetG, δG, δθ) = make_measurements!(
                measurement_container,
                logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (electron_phonon_parameters,)
            )

            ## Increment swap counter
            swap_counter += 1

            ## If swaps
            if  swap_counter % swap_frequency == 0
                MPI.Barrier(mpi_comm)
                swap_counter = 0
                N_swaps += 1
                ## swaps the states stochastically
                if parameter_dict["do_swaps"]
                    (logdetG, sgndetG, shift_val) = temper_λ!(
                        G=G, B=B, logdetG =logdetG, sgndetG=sgndetG,
                        fermion_greens_calculator=fermion_greens_calculator,
                        fermion_greens_calculator_alt=fermion_greens_calculator_alt,
                        fermion_path_integral=fermion_path_integral,
                        electron_phonon_parameters=electron_phonon_parameters,
                        holstein=true, ssh = false,
                        n_tier = n_tier,
                        n_walker_per_tier=n_walker_per_tier,
                        mpi_rank_tier=mpi_rank_tier,
                        mpi_rank_pID=mpi_rank_pID,
                        shift_val=shift_val,
                        rng=rng,
                        additional_info=additional_info
                    )
                end
                ## Rethermalize after a swap
                for n ∈ 1:N_burnin_after_swap
                    
                    ## Perform an HMC update.
                    (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                        G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                        fermion_path_integral = fermion_path_integral,
                        fermion_greens_calculator = fermion_greens_calculator,
                        fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                        B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
                    )

                   
                end ## n in 1:N_burnin_after_swap
            end ## if  swap_counter % swap_frequency == 0
        end ## for n in 1:bin_size

        ## Write the average measurements for the current bin to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )

        

        if (mpi_rank==0); println("Bin ",bin," of ", N_bins); end
    end ## for bin in 1:N_bins

    MPI.Barrier(mpi_comm)

    ## Calculate rates
    additional_info["swap_up_rate"] = additional_info["N_swap_up"] / N_swaps
    additional_info["swap_dn_rate"] = additional_info["N_swap_dn"] / N_swaps
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)

    ## Calculate reflection update acceptance rate.
    ## additional_info["reflection_acceptance_rate"] /= (N_updates + N_burnin)

    ## Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator.n_stab

    ## Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

    ## Process the simulation results, calculating final error bars for all measurements,
    ## writing final statisitics to CSV files.
    if (mpi_rank_pID==0)
        process_measurements(simulation_info.datafolder, N_bins)
        process_measurements(simulation_info.datafolder, N_bins,time_displaced=true)
    end

    MPI.Barrier(mpi_comm)

    return nothing
end





# This function probablistically swaps states between λ tiers
# It does so in a "waterfall" manner. 
function temper_λ!(;
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


    mpi_comm = MPI.COMM_WORLD

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
            logdetG_upper_new =0.0
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
            lnP = - Sb_upper_new - Sb_lower_new - 2.0 * (logdetG_upper_new + logdetG_lower_new)
            lnP += Sb_upper_old + Sb_lower_old + 2.0 * (logdetG_upper_old + logdetG_lower_old)

            ## ln(P) and log(random)
            if isfinite(lnP) && good_upper && good_lower == 1.0
                P = [lnP,log(rand(rng,Float64))]
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




# run the simulation, once without swaps and once with swaps
if abspath(PROGRAM_FILE) == @__FILE__
    ## Run with swaps turned off
    parameter_dict = Dict{String,Any}(
        "base_seed" =>1000,
        "lambdas" => collect(LinRange(0.0,0.6,4)),
        "L" => 1,
        "beta" => 10.0,
        "Omega" => 1.0,
        "t" => 0.0,
        "avg_n" => 1.0,
        "dtau" => 0.1,
        "symmetric" => false,
        "checkerboard" => false,
        "n_stab" => 10,
        "dG_max" => 1e-6,
        "N_burnin" => 5_000,
        "N_updates" => 10_000,
        "N_burnin_after_swap" => 10,
        "N_bins" => 20,
        "Nt" => 10,
        "do_swaps" => false,
        "swap_frequency" => 20
    )
     
    run_simulation(parameter_dict)

    ## Run with swaps turned on
    parameter_dict = Dict{String,Any}(
        "base_seed" =>1000,
        "lambdas" => collect(LinRange(0.0,0.6,4)),
        "L" => 1,
        "beta" => 10.0,
        "Omega" => 1.0,
        "t" => 0.0,
        "avg_n" => 1.0,
        "dtau" => 0.1,
        "symmetric" => false,
        "checkerboard" => false,
        "n_stab" => 10,
        "dG_max" => 1e-6,
        "N_burnin" => 5_000,
        "N_updates" => 10_000,
        "N_burnin_after_swap" => 10,
        "N_bins" => 20,
        "Nt" => 10,
        "do_swaps" => true,
        "swap_frequency" => 20
    )
    
     
    run_simulation(parameter_dict)
    
end
