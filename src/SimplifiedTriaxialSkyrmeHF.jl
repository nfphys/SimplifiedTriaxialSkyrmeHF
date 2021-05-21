module SimplifiedTriaxialSkyrmeHF

using Plots
using LinearAlgebra
using Parameters
using SparseArrays
using Arpack
using MyLibrary

@with_kw struct PhysicalParam{T} @deftype Float64
    ħc = 197.
    mc² = 938.

    Z::Int64 = 8; @assert iseven(Z) === true
    N::Int64 = Z; @assert iseven(N) === true
    A::Int64 = Z + N; @assert A === Z + N

    ħω₀ = 12A^(-1/3)

    t₀ = -1800.
    t₃ = 12871.
    α  = 1/3

    Nx::Int64 = 40
    Ny::Int64 = Nx
    Nz::Int64 = Nx

    Δx = 0.25
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
    


@inline function second_deriv_coeff(i, j, a, N, Π)
    d = 0.0
    if i === 1
        d += ifelse(j===2,    1, 0)
        d += ifelse(j===1, -2+Π, 0)
    elseif i === N
        d += ifelse(j===N,  -2, 0)
        d += ifelse(j===N-1, 1, 0)
    else
        d += ifelse(j===i+1, 1, 0)
        d += ifelse(j===i,  -2, 0)
        d += ifelse(j===i-1, 1, 0)
    end
    d /= a*a
    return d 
end

function second_deriv_coeff2(i, j, a, N, Π) 
    d = 0.0 
    if i === 1
        d += ifelse(j===3, -1/12, 0)
        d += ifelse(j===2,   4/3 + Π*(-1/12), 0)
        d += ifelse(j===1,  -5/2 + Π*(4/3), 0)
    elseif i === 2
        d += ifelse(j===4, -1/12, 0)
        d += ifelse(j===3, 4/3, 0)
        d += ifelse(j===2, -5/2, 0)
        d += ifelse(j===1, 4/3 + Π*(-1/12), 0)
    elseif i === N-1
        d += ifelse(j===N, 4/3, 0)
        d += ifelse(j===N-1, -5/2, 0)
        d += ifelse(j===N-2, 4/3, 0)
        d += ifelse(j===N-3, -1/12, 0)
    elseif i === N 
        d += ifelse(j===N, -5/2, 0)
        d += ifelse(j===N-1, 4/3, 0)
        d += ifelse(j===N-2, -1/12, 0)
    else
        d += ifelse(j===i+2, -1/12, 0)
        d += ifelse(j===i+1, 4/3, 0)
        d += ifelse(j===i, -5/2, 0)
        d += ifelse(j===i-1, 4/3, 0)
        d += ifelse(j===i-2, -1/12, 0)
    end
    d /= a*a 
    return d 
end

#=
function test_second_deriv_coeff(param) 
    @unpack Nx, Δx, xs = param 

    fs = @. xs^4 
    
    dfs = zeros(Float64, Nx)
    for ix in 1:Nx 
        for dx in -2:2
        dfs[ix] += second_deriv_coeff2(ix, jx, Δx, Nx, 1)*fs[jx]
    end

end
=#




function make_Hamiltonian(param, vpot, qnum)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    @unpack Πx, Πy, Πz = qnum 

    Hmat = spzeros(Float64, N, N)
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        Hmat[i,i] += vpot[ix, iy, iz]

        for dx in -2:2
            jx = ix + dx
            jy = iy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jx ≤ Nx) continue end

            Hmat[i,j] += -second_deriv_coeff(ix, jx, Δx, Nx, Πx)
        end

        for dy in -2:2
            jx = ix
            jy = iy + dy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jy ≤ Ny) continue end

            Hmat[i,j] += -second_deriv_coeff(iy, jy, Δy, Ny, Πy)
        end

        for dz in -2:2
            jx = ix
            jy = iy 
            jz = iz + dz 
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jz ≤ Nz) continue end 
            Hmat[i,j] += -second_deriv_coeff(iz, jz, Δz, Nz, Πz)
        end
    end

    return Hmat
end


function test_make_Hamiltonian(param; Πx=1, Πy=1, Πz=1)
    @unpack Nx, Ny, Nz, xs, ys, zs = param 
    N = Nx*Ny*Nz 

    vpot = zeros(Float64, Nx, Ny, Nz) 
    @time for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        r2 = xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz]
        vpot[ix, iy, iz] = r2
    end

    qnum = QuantumNumbers(Πx=Πx, Πy=Πy, Πz=Πz)

    @time Hmat = make_Hamiltonian(param, vpot, qnum)

    @time vals, vecs = eigs(Hmat, nev=5, which=:SM)
    vals./2
end

"""
    calc_howf(param, n, x)

Calculate harmonic oscillator wave function.
"""
function calc_howf(param, n, x)
    @unpack mc², ħc, ħω₀ = param 
    ξ = sqrt(mc²*ħω₀/(ħc*ħc)) * x 

    1/sqrt(2^n * factorial(n)) * (mc²*ħω₀/(π*ħc*ħc))^(1/4) * 
        exp(-0.5ξ*ξ) * MyLibrary.hermite(n,ξ)
end


function initial_states(param; Nmax=2)
    @unpack A, ħω₀, Nx, Ny, Nz, xs, ys, zs = param 
    N = Nx*Ny*Nz 

    nstates = div((Nmax+1)*(Nmax+2)*(Nmax+3), 6)
    ψs = zeros(Float64, N, nstates)
    spEs = zeros(Float64, nstates)
    qnums = Vector{QuantumNumbers}(undef, nstates)
    occ = zeros(Float64, nstates)

    istate = 0
    for nz in 0:Nmax, ny in 0:Nmax, nx in 0:Nmax
        if (nx + ny + nz > Nmax) continue end
        istate += 1

        for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
            i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
            ψs[i, istate] = calc_howf(param, nx, xs[ix]) * 
                            calc_howf(param, ny, ys[iy]) *
                            calc_howf(param, nz, zs[iz])
        end

        spEs[istate] = ħω₀*(nx + ny + nz + 3/2)
        qnums[istate] = QuantumNumbers(Πx=(-1)^nx, Πy=(-1)^ny, Πz=(-1)^nz)

    end
    
    return ψs, spEs, qnums
end

function sort_states(ψs, spEs, qnums)
    p = sortperm(spEs)
    return ψs[:,p], spEs[p], qnums[p]
end

function calc_occ!(occ, param)
    @unpack A = param 
    nstates = length(occ)

    fill!(occ, 0)
    occupied_states = 0
    for i in 1:nstates 
        if occupied_states + 4 ≤ A
            occ[i] = 1
            occupied_states += 4
        elseif occupied_states < A 
            occ[i] = (A - occupied_states)/4
            occupied_states = A 
        end
    end

    @assert occupied_states == A 
    return 
end


function show_states(ψs, spEs, qnums, occ)
    nstates = size(ψs, 2)
    println("")
    for i in 1:nstates
        println("i = ", i, ": ")
        @show spEs[i] occ[i] qnums[i]
    end
end

function plot_density(param, ρ)
    @unpack xs, ys, zs = param
    p = heatmap(xs, ys, ρ[:,:,1]'; xlabel="x", ylabel="y", ratio=:equal)
    display(p)

    p = heatmap(xs, zs, ρ[:,1,:]'; xlabel="x", ylabel="z", ratio=:equal)
    display(p)

    p = heatmap(ys, zs, ρ[1,:,:]'; xlabel="y", ylabel="z", ratio=:equal)
    display(p)
end

function test_initial_states(param; Nmax=2, istate=1)
    @unpack ħc, mc², Nx, Ny, Nz, Δx, Δy, Δz, ħω₀, xs, ys, zs = param 
    @show ħω₀

    ψs, spEs, qnums = initial_states(param; Nmax=Nmax) 
    ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    occ = similar(spEs)
    calc_occ!(occ, param)


    ρ = zeros(Float64, Nx, Ny, Nz)
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        ρ[ix, iy, iz] = ψs[i,istate]*ψs[i,istate]
    end
    @show sum(ρ)*2Δx*2Δy*2Δz # must be equal to one 

    plot_density(param, ρ)


    vpot = zeros(Float64, Nx, Ny, Nz) 
    @time for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        r2 = xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz]
        vpot[ix, iy, iz] = (mc²*ħω₀/ħc^2)^2*r2
    end
    Hmat = make_Hamiltonian(param, vpot, qnums[istate]) 
    @show calc_sp_energy(param, Hmat, ψs[:,istate]) spEs[istate] # must be equal to spEs[istate] 

    show_states(ψs, spEs, qnums, occ)
end


function calc_density!(ρ, param, ψs, spEs, qnums, occ)
    @unpack mc², ħc, Nx, Ny, Nz, xs, ys, zs = param 
    nstates = size(ψs, 2)

    fill!(ρ, 0)
    for istate in 1:nstates 
        @views ψ = ψs[:,istate]
        for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
            i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
            ρ[ix,iy,iz] += 4occ[istate]*ψs[i]*ψs[i]
        end
    end

end

function test_calc_density!(param) 
    @unpack A, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param

    ψs, spEs, qnums = initial_states(param)
    ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    occ = similar(spEs)
    calc_occ!(occ, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    calc_density!(ρ, param, ψs, spEs, qnums, occ)

    @show sum(ρ)*2Δx*2Δy*2Δz # must be equal to mass number A 
    @show A
    
    plot_density(param, ρ)

end


function calc_potential!(vpot, param, ρ)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, xs, ys, zs = param 

    @. vpot = (3/4)*t₀*ρ + (α+2)/16*t₃*ρ^(α+1)

    @. vpot *= 2mc²/(ħc*ħc)

    return 
end

function test_calc_potential!(param)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param

    ψs, spEs, qnums = initial_states(param)
    ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    occ = similar(spEs)
    calc_occ!(occ, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    calc_density!(ρ, param, ψs, spEs, qnums, occ)

    vpot = similar(ρ)
    @time calc_potential!(vpot, param, ρ)

    plot_density(param, vpot)
end



function calc_norm(param, ψ)
    @unpack Δx, Δy, Δz = param 
    sqrt(dot(ψ, ψ)*2Δx*2Δy*2Δz)
end

function calc_sp_energy(param, Hmat, ψ)
    @unpack mc², ħc = param 
    dot(ψ, Hmat, ψ)/dot(ψ, ψ) * (ħc*ħc/2mc²)
end

function imaginary_time_evolution!(ψs, spEs, qnums, occ, ρ, vpot, param; Δt=0.1, iter_max=20, rtol=1e-5)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    nstates = size(ψs, 2)

    Etots = Float64[] # history of total energy 

    for iter in 1:iter_max 
        calc_density!(ρ, param, ψs, spEs, qnums, occ)
        calc_potential!(vpot, param, ρ)

        for istate in 1:nstates 
            Hmat = make_Hamiltonian(param, vpot, qnums[istate])

            @views ψs[:,istate] = (I - 0.5Δt*Hmat)*ψs[:,istate]
            @views ψs[:,istate] = (I + 0.5Δt*Hmat)\ψs[:,istate]

            # gram schmidt orthogonalization 
            for jstate in 1:istate-1
                if qnums[istate] !== qnums[jstate] continue end
                @views ψs[:,istate] .-= ψs[:,jstate] .* (dot(ψs[:,jstate], ψs[:,istate])*2Δx*2Δy*2Δz)
            end

            # normalization 
            @views ψs[:,istate] ./= calc_norm(param, ψs[:,istate])
            @views spEs[istate] = calc_sp_energy(param, Hmat, ψs[:,istate])
        end

        ψs, spEs, qnums = sort_states(ψs, spEs, qnums)
    end

end

function HF_calc_with_imaginary_time_step(;Δt=0.1, iter_max=20)
    param = PhysicalParam()
    @unpack Nx, Ny, Nz, xs, ys, zs = param 

    ψs, spEs, qnums = initial_states(param)
    ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    occ = similar(spEs)
    calc_occ!(occ, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    calc_density!(ρ, param, ψs, spEs, qnums, occ)

    vpot = similar(ρ)

    @time imaginary_time_evolution!(ψs, spEs, qnums, occ, ρ, vpot, param; Δt=Δt, iter_max=iter_max)

    show_states(ψs, spEs, qnums, occ)

    plot_density(param, ρ)
end










function initial_density(param)
    @unpack Nx, Ny, Nz, xs, ys, zs = param 

    ρ = zeros(Float64, Nx, Ny, Nz) 

    r₀ = 1.2
    R = r₀*param.A^(1/3) 
    a = 0.67 
    ρ₀ = 3/(4π*r₀^3) 

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        x = xs[ix] 
        y = ys[iy] 
        z = zs[iz] 
        r = sqrt(x*x + y*y + z*z) 
        ρ[ix, iy, iz] = ρ₀/(1 + exp((r - R)/a))
    end

    return ρ
end


function test_initial_density(param) 
    ρ = initial_density(param) 

    plot_density(param, ρ)
end



function calc_states!(vpot, param, ρ; Emax=0, nev=5, nstates_max=100)
    @unpack ħc, mc², Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    N = Nx*Ny*Nz
    
    ψs = zeros(Float64, N, nstates_max) 
    spEs = zeros(Float64, nstates_max)
    qnums = Vector{QuantumNumbers}(undef, nstates_max) 

    istate = 0
    for Πz in 1:-2:-1, Πy in 1:-2:-1, Πx in 1:-2:-1 
        if !(Πx === 1 && Πy === 1 && Πz === 1) continue end 

        @show qnum = QuantumNumbers(Πx=Πx, Πy=Πy, Πz=Πz)
        calc_potential!(vpot, param, ρ) 
        Hmat = make_Hamiltonian(param, vpot, qnum) 

        vals, vecs = eigs(Hmat, nev=nev, which=:SM) 

        # normalization 
        @. vals *= ħc^2/2mc² 
        @. vecs /= sqrt(2Δx*2Δy*2Δz)

        @show vals

        for i in 1:length(vals) 
            #if vals[i] > Emax continue end 
            istate += 1
            ψs[:,istate] = vecs[:,i]
            spEs[istate] = vals[i] 
            qnums[istate] = qnum 
        end
            
    end

    return ψs[:,1:istate], spEs[1:istate], qnums[1:istate] 
end

function test_calc_states!(param; nev=5) 
    @unpack Nx, Ny, Nz = param 

    ρ = initial_density(param) 
    vpot = similar(ρ) 

    calc_potential!(vpot, param, ρ)
    #plot_density(param, vpot)

    @time ψs, spEs, qnums = calc_states!(vpot, param, ρ; nev=nev)
    ψs, spEs, qnums = sort_states(ψs, spEs, qnums) 

    occ = zeros(Float64, length(spEs))
    #calc_occ!(occ, param) 

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        ρ[ix, iy, iz] = ψs[i,1]*ψs[i,1]
    end
    plot_density(param, ρ)

    ρ = initial_density(param) 
    calc_potential!(vpot, param, ρ)
    qnum = QuantumNumbers()
    Hmat = make_Hamiltonian(param, vpot, qnum)
    @show calc_sp_energy(param, Hmat, ψs[:,1])

    show_states(ψs, spEs, qnums, occ)
end

    










end # module
