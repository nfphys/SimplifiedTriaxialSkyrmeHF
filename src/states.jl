"""
    calc_howf(param, n, x)

Calculate harmonic oscillator wave function.
"""
function calc_howf(param, n, x)
    @unpack mc², ħc, ħω₀ = param 
    ξ = sqrt(mc²*ħω₀/(ħc*ħc)) * x 

    1/sqrt(2^n * factorial(n)) * (mc²*ħω₀/(π*ħc*ħc))^(1/4) * exp(-0.5ξ*ξ) * hermite(n,ξ)
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
    
    return SingleParticleStates(nstates, ψs, spEs, qnums, occ)
end

function sort_states!(states)
    @unpack ψs, spEs, qnums = states
    p = sortperm(spEs)

    ψs[:,:] = ψs[:,p]
    spEs[:] = spEs[p]
    qnums[:] = qnums[p]
    return 
end

function calc_occ!(states, param)
    @unpack A = param 
    @unpack nstates, occ = states

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


function show_states(states)
    @unpack nstates, ψs, spEs, qnums, occ = states
    nstates = size(ψs, 2)
    println("")
    for i in 1:nstates
        println("i = ", i, ": ")
        @show spEs[i] occ[i] qnums[i]
    end
end


function test_initial_states(param; Nmax=2, istate=1)
    @unpack ħc, mc², Nx, Ny, Nz, Δx, Δy, Δz, ħω₀, xs, ys, zs = param 

    @time states = initial_states(param; Nmax=Nmax) 
    @unpack nstates, ψs, spEs, qnums, occ = states

    @time sort_states!(states)
    calc_occ!(states, param)

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

    show_states(states)
    
end