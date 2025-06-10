using LinearAlgebra
using Random
using Printf
using MPI
using FileIO
using JLD2
using DataFrames
using CSV
include("SmoQyDQMCloader.jl")

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
import SmoQyDQMC.MuTuner           as mt

λs = [0.95] 
ns = 0.05:0.05:1.0 |> collect
Ωs = 0.5:0.5:2.0 |> collect
βs = 5.0:1.0:20.0 |> collect

# top level function to run simulation
function run_simulation(ARGS)
    
    #############################
    ## INITIALIZE MPI COMMUNICATOR ##
    #############################
    MPI.Init()
    comm = MPI.COMM_WORLD

    pID = MPI.Comm_rank(comm)

    mkpath("simulations/")
    mkpath("output_jld2")

    λi = 1
    nn = length(ns)
    ni = mod(pID,nn)+1
    Ωi = div(pID,nn)+1
    
    βi = parse(Int64,ARGS[2])
    if parse(Int64,ARGS[1]) == 2
        Ωi += 2
    end
    
    for i in 1:40

        if pID == (i-1)
            println("pID=",pID," λi=",λi," ni=",ni," Ωi=",Ωi)
        end
        MPI.Barrier(comm)
    end

# exit()
	#############################
    ## DEFINE MODEL PARAMETERS ##
    #############################

    #####
    #
    # Arguments:
    #   L BETA DENSITY OMEGA LAMBDA 
    #
    #####

    L = 14#parse(Int64,ARGS[1])
    β = βs[βi]
    n = ns[ni]#parse(Float64,ARGS[3])
    Ω = Ωs[Ωi]#parse(Float64,ARGS[4])
    λ = λs[λi]#parse(Float64,ARGS[5])
    # system size
    
    # nearest-neighbor hopping amplitude
    t = 1.0

    # if pID==0; println("BEGIN\nβ=",β,"\n","Ω=",Ω,"\n"); end
    # holstein coupling constant
    α = Ω*sqrt(4*t*λ)

    # initial chemical potential
    μ = 0.0

 
    # define simulation name
    datafolder_prefix = @sprintf "holstein_singleband_2D_w%.2f_l%.2f_n%.2f_L%d_b%.2f" Ω λ n L β

    # initialize simulation info
    simulation_info = SimulationInfo(
        filepath = "simulations/",                     
        datafolder_prefix = datafolder_prefix,
        sID = 1
    )

    # initialize data folder
    initialize_datafolder(simulation_info)

    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    # initialize random seed
    seed = abs(rand(Int))

    # initialize random number generator
    rng = Xoshiro(seed)

    # discretization in imaginary time
    Δτ = 0.10

    # evaluate length of imaginary time axis
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    # whether to use the checkerboard approximation
    checkerboard = false

    # whether to use symmetric propagator defintion 
    symmetric = false

    # initial stabilization frequency
    n_stab = 10

    # max allowed error in green's function
    δG_max = 1e-6

    # number of thermalization/burnin updates
    N_burnin = 1500
    # number of simulation updates
    N_updates = 5000

    # number of bins/number of time 
    N_bins = 100

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

    # initialize addition simulation information dictionary
    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "dt" => Δt,
        "Nt" => Nt,
        "nt" => nt,
        "reg" => reg,
        "hmc_acceptance_rate" => 0.0,
        "swap_acceptance_rate" => 0.0,
        "reflection_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "seed" => seed,
    )

    ##################
    ## DEFINE MODEL ##
    ##################

    # calculate length of imaginary time axis
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    # define kagome unit cell
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0,0.0], [0.0,1.0]],
                            basis_vecs   = [[0.0,0.0]])

   # define size of lattice (only supports periodic b.c. for now)
   lattice = lu.Lattice(
        L = [L,L],
        periodic = [true, true] # must be true for now
    )

    # define model geometry
    model_geometry = ModelGeometry(unit_cell, lattice)

    # calculate number of orbitals in the lattice
    N = lu.nsites(unit_cell, lattice)

    # define first nearest-neighbor bond
    bond_x = lu.Bond(orbitals = (1,1), displacement = [1,0])
    bond_x_id = add_bond!(model_geometry, bond_x)

    # define second nearest-neighbor bond
    bond_y = lu.Bond(orbitals = (1,1), displacement = [0,1])
    bond_y_id = add_bond!(model_geometry, bond_y)


    # define non-interacting tight binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_x, bond_y],
        t_mean = [t, t],
        μ = μ,
        ϵ_mean = [0.]
    )

    # initialize null electron-phonon model
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    # define phonon mode for each orbital in unit cell
    phonon_1 = PhononMode(orbital = 1, Ω_mean = Ω)
    

    # add the three phonon modes to the model
    phonon_1_id = add_phonon_mode!(electron_phonon_model = electron_phonon_model, phonon_mode = phonon_1)
    
    # define holstein coupling for first orbital/phonon mode in unit cell
    holstein_coupling_1 = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_1_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
    	α_mean = α
    )

    # add first holstein coupling to the model
    holstein_coupling_1_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_coupling_1,
    	model_geometry = model_geometry
    )

    # write model summary to file
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (electron_phonon_model,)
    )

    ####################################################
    ## INITIALIZE MODEL PARAMETERS FOR FINITE LATTICE ##
    ####################################################

    # define tight binding parameters for finite lattice based on tight binding model
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    # define electron-phonon parameters for finite model based on electron-phonon model
    electron_phonon_parameters = ElectronPhononParameters(
        β = β, Δτ = Δτ,
        electron_phonon_model = electron_phonon_model,
        tight_binding_parameters = tight_binding_parameters,
        model_geometry = model_geometry,
        rng = rng
    )

    ######################################
    ## DEFINE AND INIALIZE MEASUREMENTS ##
    ######################################

    # initialize measurement container
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    # initializing tight-binding model measurements
    initialize_measurements!(measurement_container, tight_binding_model)

    # initialize electron-phonon model measurements
    initialize_measurements!(measurement_container, electron_phonon_model)

    # # measure time-displaced green's function
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
        pairs = [(phonon_1_id, phonon_1_id)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = true,
        integrated = false,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "current",
        time_displaced = true,
        integrated = false,
        pairs = [(1, 1)] # hopping ID pair for y-direction hopping
    )

    # initialize measurement sub-directories
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    ###################################################
    ## SET-UP & INITIALIZE DQMC SIMULATION FRAMEWORK ##
    ###################################################

    # initialize a fermion path integral according non-interacting tight-binding model
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # initialize fermion path integral to electron-phonon interaction contribution
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

    # initialize the density/chemical potential tuner
    chemical_potential_tuner = mt.MuTunerLogger(n₀ = n, β = β, V = N, u₀ = α^2/Ω^2, μ₀ = μ, c = 0.5)

    # initialize hamitlonian/hybrid monte carlo (HMC) updater
    hmc_updater = HMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt, nt = nt, reg = reg
    )

    ############################
    ## PERFORM BURNIN UPDATES ##
    ############################

    # intialize errors corrected by numerical stabilization to zero
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    # perform thermalization/burnin updates
    for n in 1:N_burnin

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

        # record accept/reject outcome
        additional_info["reflection_acceptance_rate"] += accepted

        # update the chemical potential
        logdetG, sgndetG = update_chemical_potential!(
            G, logdetG, sgndetG,
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B
        )
    end

    ##################################################################
    ## PERFORM SIMULATION/MEASUREMENT UPDATES AND MAKE MEASUREMENTS ##
    ##################################################################
    
    # intialize errors associated with numerical instability to zero
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))
    MPI.Barrier(comm)
    if pID==0; println("Warms done, measurements"); end
    # iterate of measurement bins
    for bin in 1:N_bins

        # iterate over updates per bin
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

            # record accept/reject outcome
            additional_info["reflection_acceptance_rate"] += accepted

            # update the chemical potential
            logdetG, sgndetG = update_chemical_potential!(
                G, logdetG, sgndetG,
                chemical_potential_tuner = chemical_potential_tuner,
                tight_binding_parameters = tight_binding_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B
            )

            # make measurements
            (logdetG, sgndetG, δG, δθ) = make_measurements!(
                measurement_container,
                logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (electron_phonon_parameters,)
            )
        end

        # write measurements to file
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
        if pID==0; println("Bin ",bin," done"); end
    end
    

    
    # normalize acceptance rate measurements
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["swap_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["reflection_acceptance_rate"] /= (N_updates + N_burnin)

    # record final max stabilization error that was correct and frequency of stabilization
    additional_info["n_stab_final"] = fermion_greens_calculator.n_stab
    additional_info["dG"] = δG

    # write simulation information to file
    save_simulation_info(simulation_info, additional_info)

    # save density/chemical potential tuning profile
    save_density_tuning_profile(simulation_info, chemical_potential_tuner)

    # process measurements
    process_measurements(simulation_info.datafolder, N_bins, time_displaced = true)

    # calculate time-displaced green's function stats in momentum space
    # process_correlation_measurement(
    #     folder = simulation_info.datafolder,
    #     correlation = "greens",
    #     type = "time-displaced",
    #     space = "momentum",
    #     N_bin = 20
    # )

    MPI.Barrier(comm)
    datafolder = simulation_info.datafolder
    correlations = [
        "greens",
        
        "density",
        "spin_z",
        "current"
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
    
    MPI.Barrier(comm)
    dict = Dict{String,Any}()
    dict["global_stats"] = CSV.read(joinpath(datafolder,"global_stats.csv"),DataFrame)
    dict["local_stats"] = CSV.read(joinpath(datafolder,"local_stats.csv"),DataFrame)

    for correlation in correlations
        for space in spaces

            _,dict[correlation*"_"*space],_,_,_,_ = load_from_SmoQyDQMC(
                simulationfolder=simulation_info.datafolder,
                correlation=correlation,
                space=space,
                type="time_displaced",
                bin=true
            )
        end
    end      
    save(joinpath("output_jld2",datafolder_prefix*".jld2"),dict)
    rm(datafolder,force=true, recursive=true)
    MPI.Barrier(comm)
    MPI.Finalize()
    return nothing
end

# run the simulation
run_simulation(ARGS)
