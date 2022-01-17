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
    @unpack a, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    N = Nx*Ny*Nz

    Lmat = spzeros(Float64, N, N)
    @time make_Laplacian!(Lmat, param) 

    ρ = zeros(Float64, Nx, Ny, Nz) # 一様帯電球の電荷密度
    ϕ_exact = zeros(Float64, Nx, Ny, Nz) 
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = (iz-1)*Nx*Ny + (iy-1)*Nx + ix 
        r = sqrt(xs[ix]*xs[ix] + ys[iy]*ys[iy] + zs[iz]*zs[iz])
        if r < 1 
            ρ[ix,iy,iz] = 1
            ϕ_exact[ix,iy,iz] = -r*r/6 + 1/2
        else
            ϕ_exact[ix,iy,iz] = 1/3r 
        end
    end
    @. ϕ_exact -= ϕ_exact[100]

    @time ϕ = -(Lmat)\ρ[:]
    @time ϕ = reshape(ϕ, Nx, Ny, Nz)

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