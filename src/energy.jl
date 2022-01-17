function calc_total_energy(param, dens, ϕy)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    @unpack ρ, τ = dens

    ε = zeros(Float64, Nx, Ny, Nz) 

    # kinetic term 
    @. ε += ħc^2/2mc²*τ 

    # t₀ term 
    @. ε += (3/8)*t₀*ρ^2 

    # t₃ term 
    @. ε += (1/16)*t₃*ρ^(α+2)

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix
        ε[ix,iy,iz] += (1/2)*ρ[ix,iy,iz]*ϕy[i]
    end

    E = sum(ε)*2Δx*2Δy*2Δz
end


function calc_sp_energy(param, Hmat, ψ)
    @unpack mc², ħc = param 
    dot(ψ, Hmat, ψ)/dot(ψ, ψ) * (ħc*ħc/2mc²)
end


function calc_total_energy_with_spEs(param, dens, states)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    @unpack ρ, τ = dens 
    @unpack spEs, occ = states

    ε = zeros(Float64, Nx, Ny, Nz)

    @. ε += ħc^2/4mc²*τ

    @. ε += -α/32*t₃*ρ^(α+2)

    E = sum(ε)*2Δx*2Δy*2Δz 

    nstates = length(spEs)
    for i in 1:nstates
        E += 4occ[i]*spEs[i]/2
    end

    return E

end