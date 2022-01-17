

function calc_density!(dens, param, states)
    @unpack mc², ħc, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    @unpack ψs, spEs, qnums, occ = states
    @unpack ρ, τ = dens
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

function plot_density(param, ρ)
    @unpack xs, ys, zs = param
    p = heatmap(xs, ys, ρ[:,:,1]'; xlabel="x", ylabel="y", ratio=:equal)
    display(p)

    p = heatmap(xs, zs, ρ[:,1,:]'; xlabel="x", ylabel="z", ratio=:equal)
    display(p)

    p = heatmap(ys, zs, ρ[1,:,:]'; xlabel="y", ylabel="z", ratio=:equal)
    display(p)
end

function test_calc_density!(param) 
    @unpack A, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param
    N = Nx*Ny*Nz

    states = initial_states(param)
    sort_states!(states)
    calc_occ!(states, param)

    ρ = zeros(Float64, Nx, Ny, Nz)
    τ = similar(ρ)
    dens = Densities(ρ, τ)
    
    @time calc_density!(dens, param, states)

    @testset "particle number" begin
        @test sum(ρ)*2Δx*2Δy*2Δz ≈ A rtol=1e-2
    end
        
    plot_density(param, ρ)
    plot_density(param, τ)
end