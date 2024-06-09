function T1_transform_1e(F, t, o, v)
    t1_exp = zeros(v[end], v[end])
    t1_exp[v, o] = t
    x = I(v[end]) - t1_exp
    y = I(v[end]) + t1_exp'
    return fixed_einsum("pr,rs,qs->pq", x, F, y, optimize="optimal")
end
function T1_transform_2e(g, t, o, v)
    t1_exp = zeros(v[end], v[end])
    t1_exp[v, o] = t
    x = I(v[end]) - t1_exp
    y = I(v[end]) + t1_exp'
    return fixed_einsum("pt,qu,rm,sn,tumn ->pqrs", x, y, x, y, g, optimize="optimal")
end

# Weird things where fixed_einsum --> float instead gives Array{Float64, 0}
function fixed_einsum(args...; kwargs...)
    res = np.einsum(args...; kwargs...)
    if res isa Array && iszero(length(size(res)))
        res[]
    else
        res
    end
end

# function Base.:*(A::T, B::Array{T,0}) where {T}
#     A * B[]
# end

function extract_mat(mat, string, o, v)
    dims = []
    for c in string
        if c == 'a'
            push!(dims, v[1])
        elseif c == 'i'
            push!(dims, o[end])
        elseif c == 'o'
            push!(dims, o)
        elseif c == 'v'
            push!(dims, v)
        else
            throw("Unrecognized character")
        end
    end
    return mat[dims...]
end


function energy(F, g, L, t, γ1, γ2, t2, o, v)
    # Evaluates Energy = <HF|..H..|HF>
    E = 0.0
    E = E .+  +2.00000000  * np.einsum("ii->", extract_mat(F, "oo", o, v), optimize="optimal");
    E = E .+  -1.00000000  * np.einsum("iijj->", extract_mat(L, "oooo", o, v), optimize="optimal");
    # E = E .+  +1.00000000  * np.einsum("ia,ai->", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    E = E .+  +1.00000000  * np.einsum("iajb,aibj->", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # E = E .+  +0.25000000  * np.einsum("iajb,ai,bj->", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
                
    return E
end


function eta_ai(F, g, L, t, γ1, γ2, t2, o, v)
    # Evaluates Energy = <HF|..H..|ai>
    eta = 0.0
    
    return eta
end


function eta_aiai(F, g, L, t, γ1, γ2, t2, o, v)
    # Evaluates Energy = <HF|..H..|ai>
    eta = 0.0
    
    return eta
end


function Omega_0_bj(F, g, L, t, γ1, γ2, t2, o, v)
    # Evaluates Omega_0_bj = <bj|..H..|HF>
    a = v[1] - o[end]
    i = o[end]
    Omega_0_ai = zeros(v[end] - v[1] + 1, o[end])
    Omega_0_ai[:,:] +=  +1.00000000  * extract_mat(F, "vo", o, v);
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.50000000  * np.einsum("ab,bi->ai", extract_mat(F, "vv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("ji,aj->ai", extract_mat(F, "oo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +2.00000000  * np.einsum("jb,aibj->ai", extract_mat(F, "ov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -1.00000000  * np.einsum("jb,ajbi->ai", extract_mat(F, "ov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.50000000  * np.einsum("aijb,bj->ai", extract_mat(L, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.25000000  * np.einsum("jb,aj,bi->ai", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +1.00000000  * np.einsum("abjc,bicj->ai", extract_mat(L, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -1.00000000  * np.einsum("jikb,ajbk->ai", extract_mat(L, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.25000000  * np.einsum("abjc,bi,cj->ai", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.25000000  * np.einsum("jikb,aj,bk->ai", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,aj,bick->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bi,ajck->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +1.00000000  * np.einsum("jbkc,bj,aick->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bj,akci->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.12500000  * np.einsum("jbkc,aj,bi,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
                
    return Omega_0_ai
end

function Omega_0_bjck(F, g, L, t, γ1, γ2, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    a = v[1] - o[end]
    i = o[end]

    Omega_0_aibj = zeros(v[end] - v[1] + 1, o[end], v[end] - v[1] + 1, o[end])
    Omega_0_aibj[:,:,:,:] +=  +1.00000000  * extract_mat(g, "vovo", o, v);
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ai,bj->aibj", extract_mat(F, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bj,ai->aibj", extract_mat(F, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ac,bjci->aibj", extract_mat(F, "vv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bc,aicj->aibj", extract_mat(F, "vv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ki,akbj->aibj", extract_mat(F, "oo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kj,aibk->aibj", extract_mat(F, "oo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("aibc,cj->aibj", extract_mat(g, "vovv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("aikj,bk->aibj", extract_mat(g, "vooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("acbj,ci->aibj", extract_mat(g, "vvvo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bjki,ak->aibj", extract_mat(g, "vooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ac,bj,ci->aibj", extract_mat(F, "vv", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bc,ai,cj->aibj", extract_mat(F, "vv", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ki,ak,bj->aibj", extract_mat(F, "oo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kj,ai,bk->aibj", extract_mat(F, "oo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kk,ai,bj->aibj", extract_mat(F, "oo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("aikc,bjck->aibj", extract_mat(L, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bjkc,aick->aibj", extract_mat(L, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("aikc,bkcj->aibj", extract_mat(g, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("acbd,cidj->aibj", extract_mat(g, "vvvv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ackj,bkci->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bjkc,akci->aibj", extract_mat(g, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bcki,akcj->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kilj,akbl->aibj", extract_mat(g, "oooo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kc,ai,bjck->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ai,bkcj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ak,bjci->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kc,bj,aick->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bj,akci->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bk,aicj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ci,akbj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cj,aibk->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("aikc,bj,ck->aibj", extract_mat(L, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bjkc,ai,ck->aibj", extract_mat(L, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kkll,ai,bj->aibj", extract_mat(L, "oooo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("aikc,bk,cj->aibj", extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("acbd,ci,dj->aibj", extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ackj,bk,ci->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bjkc,ak,ci->aibj", extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bcki,ak,cj->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kilj,ak,bl->aibj", extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kc,ai,bj,ck->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kc,ai,bk,cj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kc,ak,bj,ci->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,bj,cidk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,ci,bjdk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,dk,bjci->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,ai,cjdk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,cj,aidk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,dk,aicj->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,ak,bjcl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,bj,akcl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,cl,akbj->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,ai,bkcl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,bk,aicl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,cl,aibk->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,bk,cidj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,ci,bkdj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,dj,bkci->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,ak,cjdi->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,cj,akdi->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,di,akcj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,ak,blcj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,bl,akcj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,cj,akbl->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,al,bkci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,bk,alci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,ci,albk->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("ackd,bj,ci,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("bckd,ai,cj,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kilc,ak,bj,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kjlc,ai,bk,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("ackd,bk,ci,dj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("bckd,ak,cj,di->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("kilc,ak,bl,cj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("kjlc,al,bk,ci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aibk,cjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aicj,bkdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,aick,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aick,bldj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akbj,cidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akci,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akdl,bjci->aibj", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kcld,akbl,cidj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kcld,akci,bldj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kcld,akdj,blci->aibj", extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ai,bj,ckdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ai,bk,cjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ai,cj,bkdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kcld,ai,ck,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ai,ck,bldj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ak,bj,cidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ak,ci,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ak,dl,bjci->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bj,ci,akdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kcld,bj,ck,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bj,ck,aldi->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bk,cj,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bk,dl,aicj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ci,dl,akbj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cj,dl,aibk->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,bl,cidj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,ci,bldj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,dj,blci->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,bk,cj,aldi->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,bk,di,alcj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ci,dj,akbl->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.06250000  * np.einsum("kcld,ai,bj,ck,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.06250000  * np.einsum("kcld,ai,bk,cj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.06250000  * np.einsum("kcld,ak,bj,ci,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    # Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.06250000  * np.einsum("kcld,ak,bl,ci,dj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
                    
    return Omega_0_aibj
end

function Jacobian_singles(F, g, L, t, γ1, γ2, c1, c2, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    a = v[1] - o[end]
    i = o[end]
    Jacobian_ai = zeros(v[end] - v[1] + 1, o[end])
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("ab,bi->ai", extract_mat(F, "vv", o, v), extract_mat(c1, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("ji,aj->ai", extract_mat(F, "oo", o, v), extract_mat(c1, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +2.00000000  * np.einsum("jb,aibj->ai", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jb,ajbi->ai", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jb,biaj->ai", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +2.00000000  * np.einsum("jb,bjai->ai", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("aijb,bj->ai", extract_mat(L, "voov", o, v), extract_mat(c1, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jb,aj,bi->ai", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jb,bi,aj->ai", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("abjc,bicj->ai", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("abjc,cjbi->ai", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jikb,ajbk->ai", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jikb,bkaj->ai", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +0.50000000  * np.einsum("abjc,bi,cj->ai", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +0.50000000  * np.einsum("abjc,cj,bi->ai", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jikb,aj,bk->ai", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jikb,bk,aj->ai", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jbkc,aj,bick->ai", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jbkc,bi,ajck->ai", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +2.00000000  * np.einsum("jbkc,bj,aick->ai", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jbkc,bj,akci->ai", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("jbkc,aibj,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,ajbi,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,ajck,bi->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,biaj,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bick,aj->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("jbkc,bjai,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bjak,ci->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bjci,ak->ai", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.25000000  * np.einsum("jbkc,aj,bi,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.25000000  * np.einsum("jbkc,bi,aj,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.25000000  * np.einsum("jbkc,bj,ak,ci->ai", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");    
    
    return Jacobian_ai 
end

function Jacobian_doubles(F, g, L, t, γ1, γ2, c1, c2, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    a = v[1] - o[end]
    i = o[end]
    Jacobian_aibj = zeros(v[end] - v[1] + 1, o[end], v[end] - v[1] + 1, o[end])
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ac,bjci->aibj", extract_mat(F, "vv", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ac,cibj->aibj", extract_mat(F, "vv", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bc,aicj->aibj", extract_mat(F, "vv", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bc,cjai->aibj", extract_mat(F, "vv", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ki,akbj->aibj", extract_mat(F, "oo", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ki,bjak->aibj", extract_mat(F, "oo", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kj,aibk->aibj", extract_mat(F, "oo", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kj,bkai->aibj", extract_mat(F, "oo", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ak,bjki->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bk,aikj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ci,acbj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvvo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cj,aibc->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vovv", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ac,ci,bj->aibj", extract_mat(F, "vv", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bc,cj,ai->aibj", extract_mat(F, "vv", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ki,ak,bj->aibj", extract_mat(F, "oo", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kj,bk,ai->aibj", extract_mat(F, "oo", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("aikc,bjck->aibj", extract_mat(L, "voov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("aikc,ckbj->aibj", extract_mat(L, "voov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bjkc,aick->aibj", extract_mat(L, "voov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bjkc,ckai->aibj", extract_mat(L, "voov", o, v), extract_mat(c2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akbl,kilj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "oooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("akci,bjkc->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("akcj,bcki->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkal,kjli->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "oooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bkci,ackj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bkcj,aikc->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ciak,bjkc->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cibk,ackj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cidj,acbd->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvvv", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cjak,bcki->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cjbk,aikc->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjdi,adbc->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvvv", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,ak,bjci->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,bk,aicj->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,ci,akbj->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,cj,aibk->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,aibk,cj->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,aicj,bk->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kc,aick,bj->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,akbj,ci->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,akci,bj->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bjak,ci->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bjci,ak->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kc,bjck,ai->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bkai,cj->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bkcj,ai->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ciak,bj->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cibj,ak->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cjai,bk->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cjbk,ai->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kc,ckai,bj->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kc,ckbj,ai->aibj", extract_mat(F, "ov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("aikc,ck,bj->aibj", extract_mat(L, "voov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bjkc,ck,ai->aibj", extract_mat(L, "voov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ak,bjkc,ci->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ak,bcki,cj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kilj,bl->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bk,aikc,cj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bk,ackj,ci->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kjli,al->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,acbd,dj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ci,ackj,bk->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ci,bjkc,ak->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cj,aikc,bk->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,adbc,di->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cj,bcki,ak->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kc,ak,bj,ci->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kc,bk,ai,cj->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kc,ci,ak,bj->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kc,cj,ai,bk->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kc,ck,ai,bj->aibj", extract_mat(F, "ov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ackd,ci,bjdk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ackd,dk,bjci->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bckd,cj,aidk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bckd,dk,aicj->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kilc,ak,bjcl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kilc,cl,akbj->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kjlc,bk,aicl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kjlc,cl,aibk->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,bjci,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,bjdk,ci->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,cibj,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,cidk,bj->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,dkbj,ci->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,dkci,bj->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,aicj,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,aidk,cj->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,cjai,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,cjdk,ai->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,dkai,cj->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,dkcj,ai->aibj", extract_mat(L, "vvov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,akbj,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,akcl,bj->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,bjak,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,bjcl,ak->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,clak,bj->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,clbj,ak->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,aibk,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,aicl,bk->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,bkai,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,bkcl,ai->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,clai,bk->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,clbk,ai->aibj", extract_mat(L, "ooov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ak,bckd,cjdi->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ak,kilc,blcj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ak,kclj,blci->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bk,ackd,cidj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bk,kjlc,alci->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bk,kcli,alcj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ci,ackd,bkdj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ci,bdkc,akdj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ci,kjlc,albk->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cj,adkc,bkdi->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cj,bckd,akdi->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cj,kilc,akbl->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akbl,kilc,cj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akbl,kclj,ci->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("akci,bdkc,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akci,kclj,bl->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("akcj,bckd,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akcj,kilc,bl->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkal,kjlc,ci->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkal,kcli,cj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bkci,ackd,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkci,kjlc,al->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bkcj,adkc,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkcj,kcli,al->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ciak,bdkc,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ciak,kclj,bl->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cibk,ackd,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cibk,kjlc,al->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cidj,ackd,bk->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cidj,bdkc,ak->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjak,bckd,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cjak,kilc,bl->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjbk,adkc,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cjbk,kcli,al->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjdi,adkc,bk->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjdi,bckd,ak->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ackd,ci,bj,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ackd,dk,bj,ci->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bckd,cj,ai,dk->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bckd,dk,ai,cj->aibj", extract_mat(L, "vvov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kilc,ak,bj,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kilc,cl,ak,bj->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kjlc,bk,ai,cl->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kjlc,cl,ai,bk->aibj", extract_mat(L, "ooov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ak,bckd,cj,di->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ak,kilc,bl,cj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ak,kclj,bl,ci->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bk,ackd,ci,dj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bk,kjlc,al,ci->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bk,kcli,al,cj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ci,ackd,bk,dj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ci,bdkc,ak,dj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ci,kjlc,al,bk->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("cj,adkc,bk,di->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("cj,bckd,ak,di->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cj,kilc,ak,bl->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aibk,cjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aicj,bkdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,aick,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aick,bldj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akbj,cidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akci,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akdl,bjci->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bjak,cidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bjci,akdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,bjck,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bjck,aldi->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bkai,cjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bkcj,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bkdl,aicj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ciak,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cibj,akdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cidl,akbj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cjai,bkdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cjbk,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cjdl,aibk->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,ckai,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckai,bldj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckal,bjdi->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,ckbj,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckbj,aldi->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckbl,aidj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckdi,albj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckdj,aibl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akbl,kcld,cidj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akci,kcld,bldj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akcj,kdlc,bldi->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkal,kcld,cjdi->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkci,kdlc,aldj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkcj,kcld,aldi->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ciak,kcld,bldj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cibk,kdlc,aldj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cidj,kcld,akbl->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjak,kdlc,bldi->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjbk,kcld,aldi->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjdi,kcld,albk->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ak,bj,cidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ak,ci,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ak,dl,bjci->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,bk,ai,cjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,bk,cj,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,bk,dl,aicj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ci,ak,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ci,bj,akdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ci,dl,akbj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,cj,ai,bkdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,cj,bk,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,cj,dl,aibk->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kcld,ck,ai,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,ai,bldj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,al,bjdi->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kcld,ck,bj,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,bj,aldi->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,bl,aidj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,di,albj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,dj,aibl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,aibk,cj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,aicj,bk,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kcld,aick,bj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,aick,bl,dj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,akbj,ci,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,akci,bj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,akdl,bj,ci->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bjak,ci,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bjci,ak,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kcld,bjck,ai,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bjck,al,di->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bkai,cj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bkcj,ai,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bkdl,ai,cj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ciak,bj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cibj,ak,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cidl,ak,bj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cjai,bk,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cjbk,ai,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cjdl,ai,bk->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kcld,ckai,bj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckai,bl,dj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckal,bj,di->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kcld,ckbj,ai,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckbj,al,di->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckbl,ai,dj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckdi,al,bj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckdj,ai,bl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kcld,ckdl,ai,bj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c2, "vovo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kcld,bl,cidj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kcld,ci,bldj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kcld,dj,blci->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kcld,al,cjdi->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kcld,cj,aldi->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kcld,di,alcj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,kcld,ak,bldj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,kcld,bl,akdj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,kcld,dj,akbl->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,kcld,al,bkdi->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,kcld,bk,aldi->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,kcld,di,albk->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akbl,kcld,ci,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akci,kcld,bl,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akcj,kdlc,bl,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkal,kcld,cj,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkci,kdlc,al,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkcj,kcld,al,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ciak,kcld,bl,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cibk,kdlc,al,dj->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cidj,kcld,ak,bl->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjak,kdlc,bl,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjbk,kcld,al,di->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjdi,kcld,al,bk->aibj", extract_mat(c2, "vovo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kcld,ak,bj,ci,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kcld,bk,ai,cj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kcld,ci,ak,bj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kcld,cj,ai,bk,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ck,ai,bj,dl->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kcld,ck,ai,bl,dj->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("kcld,ck,al,bj,di->aibj", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("ak,kcld,bl,ci,dj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("bk,kcld,al,cj,di->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("ci,kcld,ak,bl,dj->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("cj,kcld,al,bk,di->aibj", extract_mat(c1, "vo", o, v), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");

    return Jacobian_aibj 
end

function Jacobian_times_c(F, g, L, t, γ1, γ2, c1, c2, t2, o, v)
    n = o[end] * (v[end] - o[end])
    results = zeros(n+n^2)
    results[1:n] = reshape(Jacobian_singles(F, g, L, t, γ1, γ2, c1, c2, t2, o, v), (n,1))
    results[n+1:end] = reshape(Jacobian_doubles(F, g, L, t, γ1, γ2, c1, c2, t2, o, v), (n^2,1))

    return results
end


function build_omega(F, g, L, t, γ1, γ2, t2, o, v)
    n = o[end] * (v[end] - o[end])
    omega = zeros(n^2+n+2)
    omega[1:n] = reshape(Omega_0_bj(F, g, L, t, γ1, γ2, t2, o, v), (n, 1))
    # omega[v[1], o[end]] = 0.0                   # This to avoid s_ai being changed with E_1
    omega[n+1] = eta_ai(F, g, L, t, γ1, γ2, t2, o, v)
    omega[n+2] = eta_aiai(F, g, L, t, γ1, γ2, t2, o, v)
    omega[n+3:end] = reshape(Omega_0_bjck(F, g, L, t, γ1, γ2, t2, o, v), (n^2, 1))
    return omega
end

function num_gradient_gamma1(F, g, L, t, γ1, γ2, t2, o, v)
    ref = eta_ai(F, g, L, t, γ1, γ2, t2, o, v)
    new = eta_ai(F, g, L, t, γ1+0.00001, γ2, t2, o, v)
    return (new-ref)/0.00001
end

function num_gradient_gamma2(F, g, L, t, γ1, γ2, t2, o, v)
    ref = eta_aiai(F, g, L, t, γ1, γ2, t2, o, v)
    new = eta_aiai(F, g, L, t, γ1, γ2+0.00001, t2, o, v)
    return (new-ref)/0.00001
end



# ENV["PYTHON"]="/home/frossi/.julia/conda/3/x86_64/bin/python3.11"
# ENV["PYTHON"]="/home/federicor/miniconda3/bin/python3.11"
# import Pkg; Pkg.build("PyCall")
using PyCall
using LinearAlgebra

np = pyimport("numpy")
pyscf = pyimport("pyscf")
scf = pyimport("pyscf.scf")
cc = pyimport("pyscf.cc")
qmmm = pyimport("pyscf.qmmm")

# scf = pyscf.scf
# cc = pyscf.cc
# qmmm = pyscf.qmmm

# mol = pyscf.M(atom="H 0 0 0; Li 1.6 0 0; He 1.0 2.0 0.0; He 1.2 -1.5 1.0", basis="ccPVDZ")
# mol = pyscf.M(atom="He 0.0 0.0 0.0; H -0.4097604 1.631325 0; H 0.65 0.0 0.0", basis="ccPVDZ")
# mol = pyscf.M(atom= "O 0, 0, 0; H 0, 1, 0; H 0.0 0.0 1", basis="sto3g")
mol = pyscf.M(atom="H 0.0 0.0 0.0; Li 0 1.606 0", basis="sto-3g")
mf = scf.RHF(mol)
mf.conv_tol = 1e-13
mf.max_cycle = 1000
coords = [(0.35, 0.6, 0.8)]
charges = [0.0]
hf = qmmm.mm_charge(mf, coords, charges).run()

# FCI solution
cisolver = pyscf.fci.FCI(hf)
cisolver.nroots = 5
(e0, e1, e2, e3, e4), _ = cisolver.kernel()
println(e0, e1, e2, e3, e4)

# CCSD solution
mycc = cc.CCSD(hf).run()
@show (e0, e1), (fcivec0, fcivec1) = mycc.eomee_ccsd_singlet(nroots=3)
println("t1: ", mycc.t1)
println("t2: ", mycc.t2)

println(hf.energy_nuc())


C = hf.mo_coeff
pyscf.tools.dump_mat.dump_mo(mol, C)

#hf.analyze()

h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
h = C' * h_ao * C

g_ao = mol.intor("int2e")
g = pyscf.ao2mo.incore.full(g_ao, C)

no = mol.nelectron ÷ 2
nv = mol.nao - no
@show o = 1:no
@show v = no+1:mol.nao
i = o[end]
a = v[1]

L = 2 * g - permutedims(g, [1, 4, 3, 2])
@show F = C' * hf.get_fock() * C
h = F .- fixed_einsum("pqii->pq", L[:, :, o, o], optimize="optimal")

F_t = F
g_t = g
L_t = L


# @show F = Diagonal(hf.mo_energy)
# @show isF = fixed_einsum("pqii->pq", L[:, :, o, o], optimize="optimal")
#Initial guess for s1
# @show s[v, o] = -L[i, a, v, o] / g[a, i, a, i]

# Initial values
n = o[end] * (v[end] - o[end])
amplitudes = zeros(n^2+n + 2)
t = zeros(mol.nao, mol.nao)
t2 = zeros(mol.nao, mol.nao, mol.nao, mol.nao)
γ1 = 0.0
γ2 = 0.0

old = 0.0
E2 = 0.0

# -0.21993579571484237
# t at step 72 : [0.0026519385528204456;;]

restart = false
save_restart = false
if restart
    temp_t = zeros(nv * no)
    open("saved_t_0.65", "r") do inp
        for (pos, line) in enumerate(eachline(inp))
            temp_t[pos] = parse(Float64, line)
        end
        global t[v,o] = reshape(temp_t, (nv, no))
    end
    open("saved_t2_0.65", "r") do inp
        temp_t2 = zeros(nv^2 * no^2)
        for (pos, line) in enumerate(eachline(inp))
            temp_t2[pos] = parse(Float64, line)
        end
        global t2[v, o,v,o] = reshape(temp_t2, (nv, no, nv, no))
    end
    open("saved_gamma_0.65", "r") do inp
        global γ1 = parse(Float64, readline(inp))
        global γ2 = parse(Float64, readline(inp))
    end
end


let
    # Conditioning
    @show pre_γ1 = -2*(F[a, a] - F[i, i])#-2*L[a,i,i,a]
    @show pre_γ2 = -8*(F[a, a] - F[i, i]) #+8*L[a,a,i,i]-4*(g[a,a,a,a]+g[i,i,i,i])

    if abs(pre_γ2) < 0.1
        pre_γ2 = sign(pre_γ2)*0.1
    end

    pre_t = ones(mol.nao, mol.nao)
    for a in v
        for i in o
            pre_t[a, i] = 1 // 2 * (F[a, a] - F[i, i])
        end
    end

    println("Pre_t1: ", pre_t[v,o])

    pre_t2 = ones(mol.nao, mol.nao, mol.nao, mol.nao)

    for b in v
        for j in o
            for c in v
                for k in o
                    pre_t2[b,j,c,k] = F[b,b] + F[c,c] - F[j,j] - F[k,k]
                end
            end
        end
    end

    # for b in v
    #     for j in o
    #         for c in v
    #             for k in o
    #                 if abs(pre_t2[b,j,c,k]) < 0.1
    #                     pre_t2[b,j,c,k] = sign(pre_t2[b,j,c,k])*0.1
    #                 end
    #             end
    #         end
    #     end
    # end


    do_diis = true

    if do_diis
        diis_dim = 0
        diis_max = 12
        diis_vals_t1 = []
        diis_vals_t2 = []
        diis_errors = []
    end

    for k in 1:200
        global E2

        @show global amplitudes = -build_omega(F_t, g_t, L_t, t, γ1, γ2, t2, o, v)

        if do_diis && k > 1
            old_t1 = reshape(t[v,o], (n,1))
            old_t2 = reshape(t2[v,o,v,o], (n^2, 1))
        end

        global t[v, o] += reshape(amplitudes[1:n], (nv, no)) ./ pre_t[v, o]
        # global t[a+1:end, :] .= 0.0
        # global t[:, 1:i-1] .= 0.0
        # global t .= 0.0
        # global t[a,i] = 0.0

        # global γ1 += amplitudes[n+1] / (pre_γ1)
        # global γ2 += amplitudes[n+2] / (pre_γ2)
        global γ1 = 0.0
        global γ2 = 0.0

        global t2[v, o, v, o] += reshape(amplitudes[n+3:end], (nv, no, nv, no)) ./ pre_t2[v, o, v, o]

        if do_diis
            # Check max dimensions
            if diis_dim == diis_max
                popfirst!(diis_vals_t1)
                popfirst!(diis_vals_t2)
                popfirst!(diis_errors)
            else
                diis_dim +=1
            end
            println("DIIS dimension: ", diis_dim)

            push!(diis_vals_t1, reshape(t[v,o], (n,1)))
            push!(diis_vals_t2, reshape(t2[v,o,v,o], (n^2, 1)))
            if k > 1
                errors_t1 = old_t1 - reshape(t[v,o], (n, 1))
                errors_t2 = old_t2 - reshape(t2[v,o,v,o], (n^2, 1))
                push!(diis_errors, [errors_t1; errors_t2])

                B_mat = -1.0*ones(diis_dim, diis_dim)
                B_mat[end,end] = 0.0

                for (i, ei) in enumerate(diis_errors)
                    for (j, ej) in enumerate(diis_errors)
                        ele_ij = ei'*ej
                        B_mat[i,j] = ele_ij[1]
                    end
                end
                if k == 5
                    display(B_mat)
                end

                max_val = maximum([abs(bi) for bi in B_mat])
                B_mat = B_mat / max_val


                res = zeros(diis_dim)
                res[end] = -1.0

                Ci = B_mat \ res

                linear_t1 = zeros(n)
                linear_t2 = zeros(n^2)
                for i in 1:diis_dim-1
                    linear_t1 += Ci[i] * diis_vals_t1[i+1]
                    linear_t2 += Ci[i] * diis_vals_t2[i+1]
                end
                t[v,o] = reshape(linear_t1, (nv, no))
                t2[v,o,v,o] = reshape(linear_t2, (nv, no, nv, no))
            end

        end


        println("gamma1 at step ", k, " : ", γ1)
        println("gamma2 at step ", k, " : ", γ2)

        println("t at step ", k, " : ")
        display(t[v, o])
        println("t2 at step ", k, " : ")
        display(t2[v, o, a, i])
        println("Module of t ", sqrt(sum([abs(e)^2 for e in t[v,o]])))
        println("Module of t2 ", sqrt(sum([abs(e)^2 for e in t2[v,o,v,o]])))

        println("before")
        display(h[v,o])
        # T1 - transform integrals
        global h_t = T1_transform_1e(h, t[v, o], o, v)
        global g_t = T1_transform_2e(g, t[v, o], o, v)
        global L_t = 2 * g_t - permutedims(g_t, [1, 4, 3, 2])
        global F_t = h_t .+ fixed_einsum("pqii->pq", L_t[:, :, o, o], optimize="optimal")
        display(F_t)
        println("after")
        display(h_t[v,o])

        if k % 1 == 0
            println(" --------------Energy at step ", k, " : ", hf.energy_nuc() + energy(F_t, g_t, L_t, t, γ1, γ2, t2, o, v))
            println(" --------------E_2 at step ", k, " : ", hf.energy_nuc() + E2)
            println("E1 - E0 ", E2 - energy(F, g, L, t, γ1, γ2, t2, o, v))
            diff = old - hf.energy_nuc() - energy(F, g, L, t, γ1, γ2, t2, o, v)
            println(" --------------Difference at step ", k, " : ", diff)
            global old = hf.energy_nuc() + energy(F, g, L, t, γ1, γ2, t2, o, v)
            println(" --------------Module of omegas ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes) ]))) #if l != 1 + nv * (no - 1) # if l != n^2+2n+2+(v[end]-o[end])^2*o[end]*(o[end]-1)+(v[end]-o[end])*(o[end]-1)
        end

        
        E2 = 0.0 #-2*amplitudes[n^2+2n+3+(v[end]-o[end])^2*o[end]*(o[end]-1)]    #Needs normalization because proj is not normalized, only orthogonalized

        # updating preconditioning
        if k > 0
            @show pre_γ1 = 1.0 #num_gradient_gamma1(F, g, L, t, γ1, γ2, t2, o, v)
            @show pre_γ2 = 1.0 #num_gradient_gamma2(F, g, L, t, γ1, γ2, t2, o, v)
            if abs(pre_γ1) < 0.1
                pre_γ1 = sign(pre_γ1)*0.1
            end
            if abs(pre_γ2) < 0.3
                pre_γ2 = sign(pre_γ2)*0.3
            end
        end

        println("Eta_ai  ", amplitudes[n+1])
        println("Eta_aiai  ", amplitudes[n+2])
        println("Omega_bj_0 ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes[1:n])]))) #if l == 1 + nv * (no - 1)
        println("Omega_bjck_0 ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes[n+3:end])])))

        module_omega = sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes) if l ∉ [n+1,n+2]]))

        if abs(old > 500) || isnan(old)
            throw("Energy diverges")
            println("Energy diverges")
            break
            # elseif diff < 0
            #     throw("Energy start to increase")
        elseif module_omega < 1e-8
            println("Converged!")
            break
        end
    end


    if save_restart
        open("saved_t1_0.65_CCSD", "w") do inp
            for el in reshape(t[v,o], (nv * no, 1))
                println(inp, el)
            end
        end
        open("saved_t2_0.65_CCSD", "w") do inp
            for el in reshape(t2[v, o,v,o], (nv^2 * no^2, 1))
                println(inp, el)
            end
        end
        open("saved_gamma_0.65_CCSD", "w") do inp
            println(inp, γ1 )
            println(inp, γ2)
        end
    end

    println("Nuclear electrostatic: ", hf.energy_nuc())

    hf_energy = 2 * fixed_einsum("jj->", F[o, o], optimize="optimal")
    hf_energy -= fixed_einsum("iijj->", extract_mat(L, "oooo", o, v), optimize="optimal")
    H_00 = hf_energy[1]
    println("H_00: ", H_00)

    H_02 = 2 * g[i, a, i, a]
    println("H_02: ", g[i, a, i, a])
    H_20 = 4 * g[a, i, a, i]

    H_22 = 0.0
    H_22 += 8 * (F[a, a] - F[i, i])
    H_22 = H_22 .+ 4 * (g[a, a, a, a] + g[i, i, i, i]) - 8 * L[a, a, i, i]
    H_22 += 4 * H_00
    H_22 /= 4
    println("H_22: ", H_22)

    # converged SCF energy = -3.59843575098934
    temp = sqrt((H_22 - H_00)^2 + H_02^2)
    s_plus = ((H_22 - H_00) + temp) / (-2 * H_02)
    s_minus = ((H_22 - H_00) - temp) / (-2 * H_02)
    println("s0 + : ", s_plus)
    println("s0 - : ", s_minus)

    x_plus = (-(H_22 - H_00) + temp) / (-2 * H_02)
    x_minus = (-(H_22 - H_00) - temp) / (-2 * H_02)
    println("x + : ", x_plus)
    println("x - : ", x_minus)

    # println(build_omega(F_t, g_t, L_t, t, s_plus, t_aiai, o, v)[n+1])
    # println(build_omega(F_t, g_t, L_t, t, s_minus, t_aiai, o, v)[n+1])

    # E = E_nuc + <HF|...|HF>
    # println("Energy: ", hf.energy_nuc() + energy(F, g, L, s, t, s0, t_aiai, o, v))

    # # eta_ai = <HF|...|ai>
    # println("Eta_ai: ", eta_ai(F, g, L, s, s0, o, v))

    # # Omega_0_bj = <bj|..H..|HF>
    # println("Omega_0_bj: ", Omega_0_bj(F, g, L, s, s0, o, v))

    # # Omega_ai_bj = <bj|..H..|ai>
    # # Omega_ai_bj[a,i] is E_2 without E_nuc
    # println("Omega_ai_bj: ", Omega_ai_bj(F, g, L, s, s0, o, v))
    # println("E_2: ", hf.energy_nuc() + Excited_energy(F, g, L, s, t, s0, t_aiai, o, v))


    # global s[v, o] += reshape(amplitudes[1:n], (nv, no))
    # global s[v[1], o[end]] = 0
    # println("s", s)


    # global s0 += -amplitudes[n+1]
    # global t += reshape(amplitudes[n+2:end], (nv, no))
    println("t", t)

    # println(L_t[a,i,o,o])

    # Think: together or not?
    # Omega_ai = <ai|...|HF>     (scalar)
    # Omega_bj = <bj|...|HF>     (matrix)
    # and also for
    # E_1 = <ai|...|ai>          (scalar)
    # Omega_bj_ai = <bj|...|ai>  (matrix)

end