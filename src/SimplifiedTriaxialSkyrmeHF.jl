module SimplifiedTriaxialSkyrmeHF

using Plots
using LinearAlgebra
using Parameters
using SparseArrays
#using Arpack
using Test
#using MyLibrary

include("./derivative.jl")
include("./Hamiltonian.jl")
include("./hermite.jl")
include("./Laplacian.jl")
include("./states.jl")
include("./density.jl")
include("./potential.jl")
include("./energy.jl")

@with_kw struct PhysicalParam{T} @deftype Float64
    ħc = 197.
    mc² = 938.

    Z::Int64 = 8; @assert iseven(Z) === true
    N::Int64 = Z; @assert iseven(N) === true
    A::Int64 = Z + N; @assert A === Z + N

    ħω₀ = 41A^(-1/3)

    #=
    t₀ = -1800.
    t₃ = 12871.
    α  = 1/3
    =#
    
    t₀ = -497.726 
    t₃ = 17270
    α  = 1

    a = 0.45979
    V₀ = -166.9239/a

    Nx::Int64 = 10
    Ny::Int64 = Nx
    Nz::Int64 = Nx

    Δx = 0.8
    Δy = Δx
    Δz = Δx

    xs::T = range((1-1/2)*Δx, (Nx-1/2)*Δx, length=Nx)
    ys::T = range((1-1/2)*Δy, (Ny-1/2)*Δy, length=Ny)
    zs::T = range((1-1/2)*Δz, (Nz-1/2)*Δz, length=Nz)
end

@with_kw struct QuantumNumbers @deftype Int64
    Πx = 1; @assert Πx === 1 || Πx === -1 
    Πy = 1; @assert Πy === 1 || Πy === -1
    Πz = 1; @assert Πz === 1 || Πz === -1
    q  = 1; @assert q  === 1 || q  === -1
end

@with_kw struct SingleParticleStates 
    nstates::Int64 
    ψs::Matrix{Float64}; @assert size(ψs,2) === nstates
    spEs::Vector{Float64}; @assert length(spEs) === nstates
    qnums::Vector{QuantumNumbers}; @assert length(qnums) === nstates 
    occ::Vector{Float64}; @assert length(occ) === nstates
end

@with_kw struct Densities 
    ρ::Array{Float64, 3}
    τ::Array{Float64, 3} 
end


    


function calc_norm(param, ψ)
    @unpack Δx, Δy, Δz = param 
    sqrt(dot(ψ, ψ)*2Δx*2Δy*2Δz)
end




function imaginary_time_evolution!(states, dens, vpot, Hmat, param, Lmat; Δt=0.1)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    @unpack nstates, ψs, spEs, qnums, occ = states

    calc_density!(dens, param, states)
    ϕ_yukawa = calc_yukawa_potential(param, dens, Lmat) # ~ 0.1s
    calc_potential!(vpot, param, dens, ϕ_yukawa)

    ψs2 = similar(ψs)
    ψs2[:] = ψs
    vpot2 = similar(vpot)
    Hmat2 = similar(Hmat)

    @views for istate in 1:nstates 
        make_Hamiltonian!(Hmat, param, vpot, qnums[istate]) 

        U₁ = I - 0.5Δt*Hmat
        U₂ = I + 0.5Δt*Hmat

        ψs[:,istate] = U₂\(U₁*ψs[:,istate]) # ここがボトルネック

        # gram schmidt orthogonalization 
        for jstate in 1:istate-1
            if qnums[istate] !== qnums[jstate] continue end
            ψs[:,istate] .-= ψs[:,jstate] .* (dot(ψs[:,jstate], ψs[:,istate])*2Δx*2Δy*2Δz)
        end

        # normalization 
        ψs[:,istate] ./= calc_norm(param, ψs[:,istate])
        spEs[istate] = calc_sp_energy(param, Hmat, ψs[:,istate])
    end

    return
end



function HF_calc_with_imaginary_time_step(param; Δt=0.1, iter_max=20)
    @unpack Nx, Ny, Nz, xs, ys, zs = param 
    N = Nx*Ny*Nz

    states = initial_states(param)
    sort_states!(states)
    calc_occ!(states, param)
    @unpack nstates, spEs = states

    ρ = zeros(Float64, Nx, Ny, Nz)
    τ = similar(ρ)
    dens = Densities(ρ, τ)

    vpot = similar(ρ)
    Hmat = spzeros(Float64, N, N)

    Lmat = spzeros(Float64, N, N)
    make_Laplacian!(Lmat, param)

    Etots  = Float64[]
    Etots2 = Float64[]
    spEss = zeros(Float64, nstates, iter_max)
    for iter in 1:iter_max
        imaginary_time_evolution!(states, dens, vpot, Hmat, param, Lmat; Δt=Δt)
        sort_states!(states)
        spEss[:,iter] = spEs

        ϕ_yukawa = calc_yukawa_potential(param, dens, Lmat)
        push!(Etots, calc_total_energy(param, dens, ϕ_yukawa))
        push!(Etots2, calc_total_energy_with_spEs(param, dens, states))
    end

    
    p = plot(xlabel="iter", ylabel="total energy [MeV]")
    plot!(Etots,  marker=:dot)
    plot!(Etots2, marker=:dot)
    display(p)

    p = plot(xlabel="iter", ylabel="single-particle energy [MeV]")
    plot!(spEss', marker=:dot)
    display(p)

    @show Etots[end] Etots2[end]
    
    show_states(states)
end









end # module
