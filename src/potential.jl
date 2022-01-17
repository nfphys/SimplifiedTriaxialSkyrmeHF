function calc_potential!(vpot, param, dens)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, xs, ys, zs = param 
    @unpack ρ = dens

    fill!(vpot, 0)
    @. vpot = (3/4)*t₀*ρ + (α+2)/16*t₃*ρ^(α+1)

    @. vpot *= 2mc²/(ħc*ħc)

    return 
end

function calc_yukawa_potential(param, dens, Lmat)
    @unpack a, V₀ = param
    @unpack ρ = dens
    @views ϕ_yukawa = (-Lmat + a^(-2)*I)\ρ[:]
    @. ϕ_yukawa *= 4π*a*V₀
    return ϕ_yukawa
end

function calc_potential!(vpot, param, dens, ϕ_yukawa)
    @unpack mc², ħc, t₀, t₃, α, Nx, Ny, Nz, xs, ys, zs = param 
    @unpack ρ = dens

    fill!(vpot, 0)
    @. vpot = (3/4)*t₀*ρ + (α+2)/16*t₃*ρ^(α+1) 

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        vpot[ix,iy,iz] += ϕ_yukawa[i]
    end

    @. vpot *= 2mc²/(ħc*ħc)

    return
end

function test_calc_potential!(param)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    states = initial_states(param)
    sort_states!(states)
    calc_occ!(states, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    τ = similar(ρ)
    dens = Densities(ρ, τ)
    calc_density!(dens, param, states)

    Lmat = spzeros(Float64, N, N)
    make_Laplacian!(Lmat, param)
    @time ϕ_yukawa = calc_yukawa_potential(param, dens, Lmat)

    vpot = similar(ρ)
    @time calc_potential!(vpot, param, dens, ϕ_yukawa)

    plot_density(param, vpot)
    p = plot(xs, vpot[:,1,1])
    display(p)
end