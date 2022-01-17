function hermite(n, x)
    h₀ = one(x)
    if n == 0
        return h₀
    end
    
    h₁ = 2x
    if n == 1
        return h₁
    end
    
    h₂ = zero(x)
    for i in 2:n
        h₂ = 2x*h₁ - 2(i-1)*h₀ # i-th hermite polynomial
        h₀ = h₁
        h₁ = h₂
    end
    
    return h₂
end

function deriv_hermite(n, x)
    if n == 0
        return zero(x)
    end
    
    return 2n*hermite(n-1, x)
end