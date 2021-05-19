module SimplifiedTriaxialSkyrmeHF

using Plots
using LinearAlgebra
using Parameters
using SparseArrays
using Arpack

@with_kw struct PhysicalParam{T} @deftype Float64
    ħc = 197.
    mc² = 938.

    Z::Int64 = 8
    N::Int64 = Z
    A::Int64 = Z + N

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
    

param = PhysicalParam()


function second_deriv_coeff(i, j, a, N, Π)
    d = 0.0
    if i == 1
        d += ifelse(j==2,    1, 0)
        d += ifelse(j==1, -2+Π, 0)
    elseif i == N
        d += ifelse(j==N,  -2, 0)
        d += ifelse(j==N-1, 1, 0)
    else
        d += ifelse(j==i+1, 1, 0)
        d += ifelse(j==i,  -2, 0)
        d += ifelse(j==i-1, 1, 0)
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
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        r2 = xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz]
        vpot[ix, iy, iz] = r2
    end

    qnum = QuantumNumbers()

    @time Hmat = make_Hamiltonian(param, vpot, qnum)

    @time vals, vecs = eigs(Hmat, nev=3, which=:SM)
    vals./2
end





end # module
