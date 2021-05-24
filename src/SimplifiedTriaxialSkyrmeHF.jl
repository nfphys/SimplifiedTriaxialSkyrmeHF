module SimplifiedTriaxialSkyrmeHF

using Plots
using LinearAlgebra
using Parameters
using SparseArrays
using Arpack
using Test
using MyLibrary

@with_kw struct PhysicalParam{T} @deftype Float64
    ħc = 197.
    mc² = 938.

    Z::Int64 = 8; @assert iseven(Z) === true
    N::Int64 = Z; @assert iseven(N) === true
    A::Int64 = Z + N; @assert A === Z + N

    ħω₀ = 41A^(-1/3)

    t₀ = -1800.
    t₃ = 12871.
    α  = 1/3

    #=
    t₀ = -497.726 
    t₃ = 17_270
    α  = 1
    =#

    a = 0.45979
    V₀ = -166.9239/a

    Nx::Int64 = 20
    Ny::Int64 = Nx
    Nz::Int64 = Nx

    Δx = 0.5
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

@inline function second_deriv_coeff2(i, j, a, N, Π) 
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





function make_Hamiltonian!(Hmat, param, vpot, qnum)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    @unpack Πx, Πy, Πz = qnum 

    fill!(Hmat, 0)
    @inbounds @fastmath for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        Hmat[i,i] += vpot[ix, iy, iz]

        for dx in -2:2
            jx = ix + dx
            jy = iy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jx ≤ Nx) continue end

            Hmat[i,j] += -second_deriv_coeff2(ix, jx, Δx, Nx, Πx)
        end

        for dy in -2:2
            jx = ix
            jy = iy + dy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jy ≤ Ny) continue end

            Hmat[i,j] += -second_deriv_coeff2(iy, jy, Δy, Ny, Πy)
        end

        for dz in -2:2
            jx = ix
            jy = iy 
            jz = iz + dz 
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jz ≤ Nz) continue end 
            Hmat[i,j] += -second_deriv_coeff2(iz, jz, Δz, Nz, Πz)
        end
    end

    return Hmat
end


function test_make_Hamiltonian(param; Πx=1, Πy=1, Πz=1)
    @unpack Nx, Ny, Nz, xs, ys, zs = param 
    N = Nx*Ny*Nz 

    # harmonic oscillator potential 
    vpot = zeros(Float64, Nx, Ny, Nz) 
    @time for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        r2 = xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz]
        vpot[ix, iy, iz] = r2
    end

    # number of odd parity 
    nodd = 0
    nodd += ifelse(Πx === -1, 1, 0)
    nodd += ifelse(Πy === -1, 1, 0)
    nodd += ifelse(Πz === -1, 1, 0) 

    qnum = QuantumNumbers(Πx=Πx, Πy=Πy, Πz=Πz)

    Hmat = spzeros(Float64, N, N)
    @time make_Hamiltonian!(Hmat, param, vpot, qnum)
    @show typeof(Hmat) nnz(Hmat)/length(Hmat[:])*100


    @time vals, vecs = eigs(Hmat, nev=5, which=:SM)
    @. vals /= 2

    @testset "harmonic oscillator" begin
        @test vals[1] ≈ nodd + 3/2 rtol=0.1
        @test vals[2] ≈ nodd + 3/2 + 2 rtol=0.1
        @test vals[3] ≈ nodd + 3/2 + 2 rtol=0.1
        @test vals[4] ≈ nodd + 3/2 + 2 rtol=0.1
    end
    
    vals
end




function make_Laplacian!(Lmat, param)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    fill!(Lmat, 0)
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 

        for dx in -2:2
            jx = ix + dx
            jy = iy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jx ≤ Nx) continue end

            Lmat[i,j] += second_deriv_coeff2(ix, jx, Δx, Nx, 1)
        end

        for dy in -2:2
            jx = ix
            jy = iy + dy
            jz = iz
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jy ≤ Ny) continue end

            Lmat[i,j] += second_deriv_coeff2(iy, jy, Δy, Ny, 1)
        end

        for dz in -2:2
            jx = ix
            jy = iy 
            jz = iz + dz 
            j = (jz-1)*Nx*Ny + (jy-1)*Nx + jx

            if !(1 ≤ jz ≤ Nz) continue end 
            Lmat[i,j] += second_deriv_coeff2(iz, jz, Δz, Nz, 1)
        end
    end

    return Lmat
end

function test_make_Laplacian(param)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    N = Nx*Ny*Nz

    Lmat = spzeros(Float64, N, N)
    @time make_Laplacian!(Lmat, param) 
    Lmat = factorize(Lmat)

    ρ = zeros(Float64, N)
    ϕ_exact = zeros(Float64, N)
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        r = sqrt(xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz])
        if r < 1 
            ρ[i] = 1
            ϕ_exact[i] = -r*r/6 + 1/2
        else
            ϕ_exact[i] = 1/3r 
        end
    end
    @. ϕ_exact -= ϕ_exact[100]

    @time ϕ = -(Lmat\ρ)

    ρ = reshape(ρ, Nx, Ny, Nz)
    ϕ_exact = reshape(ϕ_exact, Nx, Ny, Nz)
    ϕ = reshape(ϕ, Nx, Ny, Nz)

    p = plot()
    plot!(p, xs, ϕ[:,1,1]; xlabel="x", label="ϕ")
    plot!(p, xs, ϕ_exact[:,1,1]; label="ϕ_exact")
    display(p)

    p = plot()
    plot!(p, xs, ϕ[1,:,1]; xlabel="y", label="ϕ")
    plot!(p, xs, ϕ_exact[1,:,1]; label="ϕ_exact")
    display(p)

    p = plot()
    plot!(p, xs, ϕ[1,1,:]; xlabel="z", label="ϕ")
    plot!(p, xs, ϕ_exact[1,1,:]; label="ϕ_exact")
    display(p)
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

    @time ψs, spEs, qnums = initial_states(param; Nmax=Nmax) 
    @time ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    nstates = size(ψs, 2)
    @time @testset "norm" begin 
        for i in 1:nstates 
            @test dot(ψs[:,i], ψs[:,i])*2Δx*2Δy*2Δz ≈ 1 rtol=1e-2
        end
    end


    vpot = zeros(Float64, Nx, Ny, Nz) 
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        r2 = xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz]
        vpot[ix, iy, iz] = (mc²*ħω₀/ħc^2)^2*r2
    end

    N = Nx*Ny*Nz
    Hmat = spzeros(Float64, N, N)
    @time @testset "single particle energy" begin 
        for i in 1:nstates
            make_Hamiltonian!(Hmat, param, vpot, qnums[i])
            @test calc_sp_energy(param, Hmat, ψs[:,i]) ≈ spEs[i] rtol=1e-2
        end
    end

    

    occ = similar(spEs)
    calc_occ!(occ, param)

    show_states(ψs, spEs, qnums, occ)

    ρ = zeros(Float64, Nx, Ny, Nz)
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        ρ[ix, iy, iz] = ψs[i,istate]*ψs[i,istate]
    end

    #plot_density(param, ρ)
    
end



function first_deriv_coeff(i, j, a, N, Π)
    d = 0.0
    if i === 1
        d += ifelse(j===3, -1/12           , 0)
        d += ifelse(j===2,   2/3 + Π*(1/12), 0)
        d += ifelse(j===1,     0 + Π*(-2/3), 0)
    elseif i === 2
        d += ifelse(j===4, -1/12           , 0)
        d += ifelse(j===3,   2/3           , 0)
        d += ifelse(j===1,  -2/3 + Π*(1/12), 0)
    elseif i === N-1
        d += ifelse(j===N,    2/3, 0)
        d += ifelse(j===N-2, -2/3, 0)
        d += ifelse(j===N-3, 1/12, 0)
    elseif i === N 
        d += ifelse(j===N-1, -2/3, 0)
        d += ifelse(j===N-2, 1/12, 0)
    else
        d += ifelse(j===i+2, -1/12, 0)
        d += ifelse(j===i+1,   2/3, 0)
        d += ifelse(j===i-1,  -2/3, 0)
        d += ifelse(j===i-2,  1/12, 0)
    end
    d /= a
end


function test_first_deriv_coeff(param)
    @unpack Nx, Δx, xs = param 

    fs = @. exp(-xs^2)

    dfs = zeros(Float64, Nx)
    for ix in 1:Nx, dx in -2:2
        jx = ix + dx; if !(1 ≤ jx ≤ Nx) continue end
        dfs[ix] += first_deriv_coeff(ix, jx, Δx, Nx, 1)*fs[jx]
    end

    dfs_exact = @. -2xs*exp(-xs^2)

    p = plot()
    plot!(p, xs, fs)
    plot!(p, xs, dfs)
    plot!(p, xs, dfs_exact)
    display(p)

end





function calc_density!(ρ, τ, param, ψs, spEs, qnums, occ)
    @unpack mc², ħc, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    nstates = size(ψs, 2)

    fill!(ρ, 0)
    fill!(τ, 0)

    for istate in 1:nstates 
        @views ψ = ψs[:,istate]
        @unpack Πx, Πy, Πz = qnums[istate]

        for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
            i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 

            # number density 
            ρ[ix,iy,iz] += 4occ[istate]*ψ[i]*ψ[i]

            # kinetic density 
            dxψ = 0.0 # derivative with respect to x 
            for dx in -2:2 
                jx = ix + dx; if !(1 ≤ jx ≤ Nx) continue end 
                jy = iy 
                jz = iz 
                j  = (jz-1)*Nx*Ny + (jy-1)*Nx + jx 

                dxψ += first_deriv_coeff(ix, jx, Δx, Nx, Πx)*ψ[j] 
            end
            τ[ix,iy,iz] += 4occ[istate]*dxψ*dxψ

            dyψ = 0.0 # derivative with respect to y 
            for dy in -2:2 
                jx = ix 
                jy = iy + dy; if !(1 ≤ jy ≤ Ny) continue end 
                jz = iz 
                j  = (jz-1)*Nx*Ny + (jy-1)*Nx + jx 

                dyψ += first_deriv_coeff(iy, jy, Δy, Ny, Πy)*ψ[j]
            end
            τ[ix,iy,iz] += 4occ[istate]*dyψ*dyψ

            dzψ = 0.0 # derivative with respect to z 
            for dz in -2:2
                jx = ix 
                jy = iy 
                jz = iz + dz; if !(1 ≤ jz ≤ Nz) continue end 
                j  = (jz-1)*Nx*Ny + (jy-1)*Nx + jx 

                dzψ += first_deriv_coeff(iz, jz, Δz, Nz, Πz)*ψ[j]
            end
            τ[ix,iy,iz] += 4occ[istate]*dzψ*dzψ
        end
    end

end

function test_calc_density!(param) 
    @unpack A, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    ψs, spEs, qnums = initial_states(param)
    ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    occ = similar(spEs)
    calc_occ!(occ, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    τ = similar(ρ)
    @time calc_density!(ρ, τ, param, ψs, spEs, qnums, occ)

    @testset "particle number" begin
        @test sum(ρ)*2Δx*2Δy*2Δz ≈ A rtol=1e-2
    end

        
    plot_density(param, ρ)
    plot_density(param, τ)

end


function calc_potential!(vpot, param, ρ)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, xs, ys, zs = param 

    fill!(vpot, 0)
    @. vpot = (3/4)*t₀*ρ + (α+2)/16*t₃*ρ^(α+1)

    @. vpot *= 2mc²/(ħc*ħc)

    return 
end

function test_calc_potential!(param)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    ψs, spEs, qnums = initial_states(param)
    ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    occ = similar(spEs)
    calc_occ!(occ, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    τ = similar(ρ)
    calc_density!(ρ, τ, param, ψs, spEs, qnums, occ)

    vpot = similar(ρ)
    @time calc_potential!(vpot, param, ρ)

    plot_density(param, vpot)
end







function calc_total_energy(param, ρ, τ)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 

    ε = zeros(Float64, Nx, Ny, Nz) 

    # kinetic term 
    @. ε += ħc^2/2mc²*τ 

    # t₀ term 
    @. ε += (3/8)*t₀*ρ^2 

    # t₃ term 
    @. ε += (1/16)*t₃*ρ^(α+2)

    E = sum(ε)*2Δx*2Δy*2Δz
end


function calc_norm(param, ψ)
    @unpack Δx, Δy, Δz = param 
    sqrt(dot(ψ, ψ)*2Δx*2Δy*2Δz)
end

function calc_sp_energy(param, Hmat, ψ)
    @unpack mc², ħc = param 
    dot(ψ, Hmat, ψ)/dot(ψ, ψ) * (ħc*ħc/2mc²)
end

function imaginary_time_evolution!(ψs, spEs, qnums, occ, ρ, τ, vpot, Hmat, param; Δt=0.1)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    nstates = size(ψs, 2)

    calc_density!(ρ, τ, param, ψs, spEs, qnums, occ)
    calc_potential!(vpot, param, ρ)

    for istate in 1:nstates 
        make_Hamiltonian!(Hmat, param, vpot, qnums[istate])

        U₁ = I - 0.5Δt*Hmat
        U₂ = I + 0.5Δt*Hmat

        @views ψs[:,istate] = U₂\(U₁*ψs[:,istate])

        # gram schmidt orthogonalization 
        for jstate in 1:istate-1
            if qnums[istate] !== qnums[jstate] continue end
            @views ψs[:,istate] .-= ψs[:,jstate] .* (dot(ψs[:,jstate], ψs[:,istate])*2Δx*2Δy*2Δz)
        end

        # normalization 
        @views ψs[:,istate] ./= calc_norm(param, ψs[:,istate])
        @views spEs[istate] = calc_sp_energy(param, Hmat, ψs[:,istate])
    end

    return
end


function HF_calc_with_imaginary_time_step(;Δt=0.1, iter_max=20)
    param = PhysicalParam()
    @unpack Nx, Ny, Nz, xs, ys, zs = param 
    N = Nx*Ny*Nz

    @time ψs, spEs, qnums = initial_states(param)
    @time ψs, spEs, qnums = sort_states(ψs, spEs, qnums)

    occ = similar(spEs)
    calc_occ!(occ, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    τ = similar(ρ)

    vpot = similar(ρ)
    Hmat = spzeros(Float64, N, N)

    Etots = Float64[]
    for iter in 1:iter_max
        @time imaginary_time_evolution!(ψs, spEs, qnums, occ, ρ, τ, vpot, Hmat, param; Δt=Δt)
        ψs, spEs, qnums = sort_states(ψs, spEs, qnums)
        push!(Etots, calc_total_energy(param, ρ, τ))
    end

    p = plot(Etots)
    display(p)

    plot_density(param, ρ)

    @show Etots[end]
    show_states(ψs, spEs, qnums, occ)
end










#=
function initial_density(param)
    @unpack A, Nx, Ny, Nz, xs, ys, zs = param 

    ρ = zeros(Float64, Nx, Ny, Nz) 

    r₀ = 1.2
    R = r₀*A^(1/3) 
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

        @time for i in 1:length(vals) 
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
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 

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

    p = plot(xs, ρ[:,1,1])
    display(p)

    ρ = initial_density(param) 
    calc_potential!(vpot, param, ρ)
    qnum = QuantumNumbers()
    Hmat = make_Hamiltonian(param, vpot, qnum)
    @show calc_sp_energy(param, Hmat, ψs[:,1])

    show_states(ψs, spEs, qnums, occ)
end
=#

    










end # module
