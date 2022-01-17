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