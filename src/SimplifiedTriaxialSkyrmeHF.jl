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
    Πx = 1
    Πy = 1
    Πz = 1
    q  = 1
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


function make_Hamiltonian(param, vpot, qnum)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    @unpack Πx, Πy, Πz = qnum 

    Hmat = spzeros(Float64, N, N)
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        Hmat[i,i] += vpot[ix, iy, iz]

        for dx in -1:1
            jx = ix + dx
            jy = iy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jx ≤ Nx) continue end

            Hmat[i,j] += -second_deriv_coeff(ix, jx, Δx, Nx, Πx)
        end

        for dy in -1:1
            jx = ix
            jy = iy + dy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jy ≤ Ny) continue end

            Hmat[i,j] += -second_deriv_coeff(iy, jy, Δy, Ny, Πy)
        end

        for dz in -1:1
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


function test_make_Hamiltonian(param)
    @unpack Nx, Ny, Nz, xs, ys, zs = param 
    N = Nx*Ny*Nz 

    vpot = zeros(Float64, Nx, Ny, Nz) 
    @time for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        r2 = xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz]
        vpot[ix, iy, iz] = r2
    end

    qnum = QuantumNumbers()

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
    for i in 1:nstates
        println("i = ", i, ": ")
        @show spEs[i] occ[i] qnums[i]
    end
end

function test_initial_states(param; Nmax=2, istate=1)
    @unpack Nx, Ny, Nz, ħω₀, xs, ys, zs = param 
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

    p = heatmap(xs, ys, ρ[:,:,1]'; xlabel="x", ylabel="y", ratio=:equal)
    display(p)

    p = heatmap(xs, zs, ρ[:,1,:]'; xlabel="x", ylabel="z", ratio=:equal)
    display(p)

    p = heatmap(ys, zs, ρ[1,:,:]'; xlabel="y", ylabel="z", ratio=:equal)
    display(p)

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

    p = plot(xs, ρ[:,1,1]; xlabel="x")
    display(p)
    
    p = heatmap(xs, ys, ρ[:,:,1]'; xlabel="x", ylabel="y", ratio=:equal)
    display(p)

    p = heatmap(xs, zs, ρ[:,1,:]'; xlabel="x", ylabel="z", ratio=:equal)
    display(p)

    p = heatmap(ys, zs, ρ[1,:,:]'; xlabel="y", ylabel="z", ratio=:equal)
    display(p)

end


function calc_potential!(vpot, param, ρ)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, xs, ys, zs = param 

    fill!(vpot, 0)

    # t₀ term 
    @. vpot += (3/4)*t₀*ρ 

    # t₃ term 
    @. vpot += (α+2)/16*t₃*ρ^(α+1)

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

    p = plot(xs, vpot[:,1,1]; xlabel="x")
    display(p)

    p = heatmap(xs, zs, ρ[:,1,:]'; xlabel="x", ylabel="z", ratio=:equal)
    display(p)

    p = heatmap(ys, zs, ρ[1,:,:]'; xlabel="y", ylabel="z", ratio=:equal)
    display(p)

    p = heatmap(xs, ys, vpot[:,:,1]'; xlabel="x", ylabel="y", ratio=:equal)
    display(p)

    p = heatmap(xs, zs, vpot[:,1,:]'; xlabel="x", ylabel="z", ratio=:equal)
    display(p)

    p = heatmap(ys, zs, vpot[1,:,:]'; xlabel="y", ylabel="z", ratio=:equal)
    display(p)
end





end # module
