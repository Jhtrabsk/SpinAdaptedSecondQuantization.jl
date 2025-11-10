using LinearAlgebra
using Arpack
using LinearMaps: FunctionMap
using LinearAlgebra: checksquare, Symmetric
using TensorCast

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


function energy(F, g, L, t, t2, o, v)
    # Evaluates Energy = <HF|..H..|HF>
    E = 0.0
    E = E .+  +2.00000000  * np.einsum("ii->", extract_mat(F, "oo", o, v), optimize="optimal");
    E = E .+  -1.00000000  * np.einsum("iijj->", extract_mat(L, "oooo", o, v), optimize="optimal");
    E = E .+  +1.00000000  * np.einsum("ia,ai->", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    E = E .+  +1.00000000  * np.einsum("iajb,aibj->", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    E = E .+  +0.25000000  * np.einsum("iajb,ai,bj->", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
                
    return E
end

function HF_energy(F, g, L, t, t2, o, v)

    # Evaluates Energy = <HF|..H..|HF>
    E = 0.0
    E = E .+  +2.00000000  * np.einsum("ii->", extract_mat(F, "oo", o, v), optimize="optimal");
    E = E .+  -1.00000000  * np.einsum("iijj->", extract_mat(L, "oooo", o, v), optimize="optimal");

    return E
end

function eta_bj(F, g, L, t, t2, o, v)
    # Evaluates Energy = <HF|..H..|ai>
    eta_bj = zeros(v[end] - v[1] + 1, o[end])
    eta_bj +=  +2.00000000  * np.einsum("ia->ai", extract_mat(F, "ov", o, v), optimize="optimal")
    eta_bj = eta_bj .+  +1.00000000  * np.einsum("iajb,bj->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    
    return eta_bj
end

function eta_aibj(F, g, L, t, t2, o, v)
    # Evaluates Energy = <HF|..H..|ai>
    eta_aibj = zeros(v[end] - v[1] + 1, o[end], v[end] - v[1] + 1, o[end])
    eta_aibj = eta_aibj .+  +2.00000000  * np.einsum("iajb->aibj", extract_mat(L, "ovov", o, v), optimize="optimal");
    
    return eta_aibj
end

function restricted_eta(F, g, L, t, t2, o, v)
    n = (v[end]-o[end])* o[end]
    eta_restr = zeros(n+Int(n*(n+1)/2))
    eta_restr[1:n] = reshape(eta_bj(F, g, L, 2t, t2, o, v), (n,1))
    eta_restr[n+1:end] = compact_4d_to_linear(eta_aibj(F, g, L, 2t, t2, o, v), o, v)

    return eta_restr
end


function Omega_0_bj(F, g, L, t, t2, o, v)
    # Evaluates Omega_0_bj = <bj|..H..|HF>
    # a = v[1] - o[end]
    # i = o[end]
    Omega_0_ai = zeros(v[end] - v[1] + 1, o[end])
    Omega_0_ai[:,:] +=  +1.00000000  * extract_mat(F, "vo", o, v);
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.50000000  * np.einsum("ab,bi->ai", extract_mat(F, "vv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("ji,aj->ai", extract_mat(F, "oo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +2.00000000  * np.einsum("jb,aibj->ai", extract_mat(F, "ov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -1.00000000  * np.einsum("jb,ajbi->ai", extract_mat(F, "ov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.50000000  * np.einsum("aijb,bj->ai", extract_mat(L, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.25000000  * np.einsum("jb,aj,bi->ai", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +1.00000000  * np.einsum("abjc,bicj->ai", extract_mat(L, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -1.00000000  * np.einsum("jikb,ajbk->ai", extract_mat(L, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.25000000  * np.einsum("abjc,bi,cj->ai", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.25000000  * np.einsum("jikb,aj,bk->ai", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,aj,bick->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bi,ajck->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +1.00000000  * np.einsum("jbkc,bj,aick->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bj,akci->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.12500000  * np.einsum("jbkc,aj,bi,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
                    
    return Omega_0_ai
end

function Omega_0_bjck(F, g, L, t, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    a = v[1] - o[end]
    i = o[end]
    Omega_0_aibj = zeros(v[end] - v[1] + 1, o[end], v[end] - v[1] + 1, o[end])
    Omega_0_aibj[:,:,:,:] +=  +1.00000000  * extract_mat(g, "vovo", o, v);
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ac,bjci->aibj", extract_mat(F, "vv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bc,aicj->aibj", extract_mat(F, "vv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ki,akbj->aibj", extract_mat(F, "oo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kj,aibk->aibj", extract_mat(F, "oo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("aibc,cj->aibj", extract_mat(g, "vovv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("aikj,bk->aibj", extract_mat(g, "vooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("acbj,ci->aibj", extract_mat(g, "vvvo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bjki,ak->aibj", extract_mat(g, "vooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("aikc,bjck->aibj", extract_mat(L, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bjkc,aick->aibj", extract_mat(L, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("aikc,bkcj->aibj", extract_mat(g, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("acbd,cidj->aibj", extract_mat(g, "vvvv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ackj,bkci->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bjkc,akci->aibj", extract_mat(g, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bcki,akcj->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kilj,akbl->aibj", extract_mat(g, "oooo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ak,bjci->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bk,aicj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ci,akbj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cj,aibk->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("aikc,bk,cj->aibj", extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("acbd,ci,dj->aibj", extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ackj,bk,ci->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bjkc,ak,ci->aibj", extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bcki,ak,cj->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kilj,ak,bl->aibj", extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,ci,bjdk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,dk,bjci->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,cj,aidk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,dk,aicj->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,ak,bjcl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,cl,akbj->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,bk,aicl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,cl,aibk->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,bk,cidj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,ci,bkdj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,dj,bkci->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,ak,cjdi->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,cj,akdi->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,di,akcj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,ak,blcj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,bl,akcj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,cj,akbl->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,al,bkci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,bk,alci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,ci,albk->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("ackd,bk,ci,dj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("bckd,ak,cj,di->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("kilc,ak,bl,cj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("kjlc,al,bk,ci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
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
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ak,ci,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ak,dl,bjci->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bk,cj,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bk,dl,aicj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ci,dl,akbj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cj,dl,aibk->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,bl,cidj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,ci,bldj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,dj,blci->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,bk,cj,aldi->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,bk,di,alcj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ci,dj,akbl->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.06250000  * np.einsum("kcld,ak,bl,ci,dj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    
    return scale_diagonal_4d(Omega_0_aibj, 0.5, o, v)
end

function Jacobian_singles_EE_nosym(F, g, L, t, c1, c2, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    # a = v[1] - o[end]
    # i = o[end]
    n = (v[end]-o[end])* o[end]

    c2_4d = linear_to_4d(c2, o, v)
    c2_4d = scale_diagonal_4d(c2_4d, 2.0, o, v)
    
    Jacobian_ai = zeros(v[end] - v[1] + 1, o[end])
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("ab,bi->ai", extract_mat(F, "vv", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("ji,aj->ai", extract_mat(F, "oo", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +2.00000000  * np.einsum("jb,aibj->ai", extract_mat(F, "ov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jb,ajbi->ai", extract_mat(F, "ov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jb,biaj->ai", extract_mat(F, "ov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +2.00000000  * np.einsum("jb,bjai->ai", extract_mat(F, "ov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("aijb,bj->ai", extract_mat(L, "voov", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jb,aj,bi->ai", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jb,bi,aj->ai", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("abjc,bicj->ai", extract_mat(L, "vvov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("abjc,cjbi->ai", extract_mat(L, "vvov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jikb,ajbk->ai", extract_mat(L, "ooov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jikb,bkaj->ai", extract_mat(L, "ooov", o, v), c2_4d, optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +0.50000000  * np.einsum("abjc,bi,cj->ai", extract_mat(L, "vvov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +0.50000000  * np.einsum("abjc,cj,bi->ai", extract_mat(L, "vvov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jikb,aj,bk->ai", extract_mat(L, "ooov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jikb,bk,aj->ai", extract_mat(L, "ooov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jbkc,aj,bick->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jbkc,bi,ajck->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +2.00000000  * np.einsum("jbkc,bj,aick->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -1.00000000  * np.einsum("jbkc,bj,akci->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("jbkc,aibj,ck->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,ajbi,ck->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,ajck,bi->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,biaj,ck->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bick,aj->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  +1.00000000  * np.einsum("jbkc,bjai,ck->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bjak,ci->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bjci,ak->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.25000000  * np.einsum("jbkc,aj,bi,ck->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.25000000  * np.einsum("jbkc,bi,aj,ck->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_ai[:,:] = Jacobian_ai[:,:] .+  -0.25000000  * np.einsum("jbkc,bj,ak,ci->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");    
    
    return Jacobian_ai 
end


function linear_to_4d(c2, o, v)
    n = (v[end]-o[end])* o[end]

    index = 1
    c2_2d = zeros(n,n)
    for in1 in 1:n
        for in2 in in1:n
            c2_2d[in1,in2] = c2[index]
            c2_2d[in2,in1] = c2[index]
            index +=1
        end
    end

    c2_4d = zeros(v[end]-o[end], o[end], v[end]-o[end], o[end])
    @cast  c2_4d[a,i,b,j] = c2_2d[(a,i),(b,j)]
    return c2_4d
end


function compact_4d_to_linear(vec_4d, o, v)
    n = (v[end]-o[end])* o[end]

    vec_2d = zeros(n,n)
    @cast vec_2d[(a,i),(b,j)] := vec_4d[a,i,b,j]

    index = 1
    vec_1d = zeros(Int(n*(n+1)/2))
    for in1 in 1:n
        for in2 in in1:n
            vec_1d[index] = vec_2d[in1,in2]
            index +=1
        end
    end

    return vec_1d
end

function scale_diagonal_tilde(vec, scal, o, v)
    # Take a vector ntilde and scales by half when ai=bj keeping in ntilde.
    n = (v[end]-o[end])* o[end]

    index = 1
    for in1 in 1:n
        for in2 in in1:n
            if in1==in2
                vec[index] = vec[index]*scal
            end
            index +=1
        end
    end
    return vec
end

function scale_diagonal_2d(vec, scal, o, v)
        # Take a vector ntilde and scales by half when ai=bj keeping in ntilde.
        n = (v[end]-o[end])* o[end]

        for in1 in 1:n
            vec[in1,in1] = vec[in1,in1]*scal
        end
        return vec
end

function scale_diagonal_4d(vec_4d, scal, o, v)
    # Take a vector ntilde and scales by half when ai=bj keeping in ntilde.
    n = (v[end]-o[end])* o[end]

    vec_2d = zeros(n,n)
    @cast vec_2d[(a,i),(b,j)] := vec_4d[a,i,b,j]

    for i in 1:n
        vec_2d[i,i] *= scal
    end

    @cast vec_4d[a,i,b,j] = vec_2d[(a,i),(b,j)]
    return vec_4d
end



function Jacobian_doubles_EE_nosym(F, g, L, t, c1, c2, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    # a = v[1] - o[end]
    # i = o[end]
    n = (v[end]-o[end])* o[end]

    c2_4d = linear_to_4d(c2, o, v)
    c2_4d = scale_diagonal_4d(c2_4d, 2.0, o, v)

    Jacobian_aibj = zeros(v[end] - v[1] + 1, o[end], v[end] - v[1] + 1, o[end])
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ac,bjci->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ac,cibj->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bc,aicj->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bc,cjai->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ki,akbj->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ki,bjak->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kj,aibk->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kj,bkai->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ak,bjki->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bk,aikj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ci,acbj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvvo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cj,aibc->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vovv", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("aikc,bjck->aibj", extract_mat(L, "voov", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("aikc,ckbj->aibj", extract_mat(L, "voov", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bjkc,aick->aibj", extract_mat(L, "voov", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bjkc,ckai->aibj", extract_mat(L, "voov", o, v), c2_4d, optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akbl,kilj->aibj", c2_4d, extract_mat(g, "oooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("akci,bjkc->aibj", c2_4d, extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("akcj,bcki->aibj", c2_4d, extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkal,kjli->aibj", c2_4d, extract_mat(g, "oooo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bkci,ackj->aibj", c2_4d, extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bkcj,aikc->aibj", c2_4d, extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ciak,bjkc->aibj", c2_4d, extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cibk,ackj->aibj", c2_4d, extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cidj,acbd->aibj", c2_4d, extract_mat(g, "vvvv", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cjak,bcki->aibj", c2_4d, extract_mat(g, "vvoo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cjbk,aikc->aibj", c2_4d, extract_mat(g, "voov", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjdi,adbc->aibj", c2_4d, extract_mat(g, "vvvv", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,ak,bjci->aibj", extract_mat(F, "ov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,bk,aicj->aibj", extract_mat(F, "ov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,ci,akbj->aibj", extract_mat(F, "ov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kc,cj,aibk->aibj", extract_mat(F, "ov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,aibk,cj->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,aicj,bk->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,akbj,ci->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bjak,ci->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bjci,ak->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bkai,cj->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cibj,ak->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cjai,bk->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ak,bjkc,ci->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ak,bcki,cj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kilj,bl->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bk,aikc,cj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bk,ackj,ci->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kjli,al->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,acbd,dj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ci,ackj,bk->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ci,bjkc,ak->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cj,aikc,bk->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,adbc,di->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cj,bcki,ak->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ackd,ci,bjdk->aibj", extract_mat(L, "vvov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ackd,dk,bjci->aibj", extract_mat(L, "vvov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bckd,cj,aidk->aibj", extract_mat(L, "vvov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bckd,dk,aicj->aibj", extract_mat(L, "vvov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kilc,ak,bjcl->aibj", extract_mat(L, "ooov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kilc,cl,akbj->aibj", extract_mat(L, "ooov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kjlc,bk,aicl->aibj", extract_mat(L, "ooov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kjlc,cl,aibk->aibj", extract_mat(L, "ooov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,bjci,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,bjdk,ci->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,cibj,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,dkbj,ci->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,aicj,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,aidk,cj->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,cjai,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,dkai,cj->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,akbj,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,bjak,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,bjcl,ak->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,clbj,ak->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,aibk,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,aicl,bk->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,bkai,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,clai,bk->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ak,bckd,cjdi->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ak,kilc,blcj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ak,kclj,blci->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bk,ackd,cidj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bk,kjlc,alci->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bk,kcli,alcj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ci,ackd,bkdj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ci,bdkc,akdj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ci,kjlc,albk->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cj,adkc,bkdi->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cj,bckd,akdi->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cj,kilc,akbl->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akbl,kilc,cj->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akbl,kclj,ci->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("akci,bdkc,dj->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akci,kclj,bl->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("akcj,bckd,di->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akcj,kilc,bl->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkal,kjlc,ci->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkal,kcli,cj->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bkci,ackd,dj->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkci,kjlc,al->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bkcj,adkc,di->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkcj,kcli,al->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ciak,bdkc,dj->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ciak,kclj,bl->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cibk,ackd,dj->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cibk,kjlc,al->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cidj,ackd,bk->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cidj,bdkc,ak->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjak,bckd,di->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cjak,kilc,bl->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjbk,adkc,di->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cjbk,kcli,al->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjdi,adkc,bk->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjdi,bckd,ak->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ak,bckd,cj,di->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ak,kilc,bl,cj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ak,kclj,bl,ci->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bk,ackd,ci,dj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bk,kjlc,al,ci->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bk,kcli,al,cj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ci,ackd,bk,dj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ci,bdkc,ak,dj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ci,kjlc,al,bk->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("cj,adkc,bk,di->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("cj,bckd,ak,di->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cj,kilc,ak,bl->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aibk,cjdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aicj,bkdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,aick,bjdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,aick,bldj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akbj,cidl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akci,bjdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,akdl,bjci->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bjak,cidl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bjci,akdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,bjck,aidl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bjck,aldi->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bkai,cjdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bkcj,aidl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,bkdl,aicj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ciak,bjdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cibj,akdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cidl,akbj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cjai,bkdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cjbk,aidl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,cjdl,aibk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,ckai,bjdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckai,bldj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckal,bjdi->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("kcld,ckbj,aidl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckbj,aldi->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckbl,aidj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckdi,albj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kcld,ckdj,aibl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akbl,kcld,cidj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akci,kcld,bldj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akcj,kdlc,bldi->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkal,kcld,cjdi->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkci,kdlc,aldj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkcj,kcld,aldi->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ciak,kcld,bldj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cibk,kdlc,aldj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cidj,kcld,akbl->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjak,kdlc,bldi->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjbk,kcld,aldi->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjdi,kcld,albk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ak,ci,bjdl->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ak,dl,bjci->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,bk,cj,aidl->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,bk,dl,aicj->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ci,ak,bjdl->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ci,dl,akbj->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,cj,bk,aidl->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,cj,dl,aibk->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,al,bjdi->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,bl,aidj->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,di,albj->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kcld,ck,dj,aibl->aibj", extract_mat(L, "ovov", o, v),  reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,aibk,cj,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,aicj,bk,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,aick,bl,dj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,akbj,ci,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bjak,ci,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bjci,ak,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bjck,al,di->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bkai,cj,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cibj,ak,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cjai,bk,dl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckai,bl,dj->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ckbj,al,di->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kcld,bl,cidj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kcld,ci,bldj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ak,kcld,dj,blci->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kcld,al,cjdi->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kcld,cj,aldi->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bk,kcld,di,alcj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,kcld,ak,bldj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,kcld,bl,akdj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ci,kcld,dj,akbl->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,kcld,al,bkdi->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,kcld,bk,aldi->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cj,kcld,di,albk->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akbl,kcld,ci,dj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akci,kcld,bl,dj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akcj,kdlc,bl,di->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkal,kcld,cj,di->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkci,kdlc,al,dj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkcj,kcld,al,di->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ciak,kcld,bl,dj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cibk,kdlc,al,dj->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cidj,kcld,ak,bl->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjak,kdlc,bl,di->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjbk,kcld,al,di->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjdi,kcld,al,bk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("ak,kcld,bl,ci,dj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("bk,kcld,al,cj,di->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("ci,kcld,ak,bl,dj->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_aibj[:,:,:,:] = Jacobian_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("cj,kcld,al,bk,di->aibj",  reshape(c1, (v[end]-o[end], o[end])), extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    
    Jacobian_aibj = scale_diagonal_4d(Jacobian_aibj, 0.5, o, v)
    Jacobian_aibj_1d = compact_4d_to_linear(Jacobian_aibj, o, v)

    return Jacobian_aibj_1d
end


function Jacobian_t_singles_EE_nosym(F, g, L, t, c1, c2, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    # a = v[1] - o[end]
    # i = o[end]
    n = (v[end]-o[end])* o[end]
    c2_4d = linear_to_4d(c2, o, v)

    Jacobian_t_ai = zeros(v[end] - v[1] + 1, o[end])
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ij,aj->ai", extract_mat(F, "oo", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("ba,bi->ai", extract_mat(F, "vv", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("iabj,bj->ai", extract_mat(L, "ovvo", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ib,aj,bj->ai", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ja,bi,bj->ai", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ajbk,ijbk->ai", c2_4d, extract_mat(g, "oovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bicj,bacj->ai", c2_4d, extract_mat(g, "vvvo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("bjak,ikbj->ai", c2_4d, extract_mat(g, "oovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bjci,bjca->ai", c2_4d, extract_mat(g, "vovv", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("iabc,bj,cj->ai", extract_mat(L, "ovvv", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("iajk,bk,bj->ai", extract_mat(L, "ovoo", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ijkb,aj,bk->ai", extract_mat(L, "ooov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bajc,bi,cj->ai", extract_mat(L, "vvov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ib,ajck,bjck->ai", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ib,cjak,bkcj->ai", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ja,bick,bjck->ai", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ja,bkci,bkcj->ai", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +2.00000000  * np.einsum("iajb,ck,bjck->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("iajb,ck,bkcj->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ibjc,ak,bkcj->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("jakb,ci,bkcj->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ajbk,ijbc,ck->ai", c2_4d, extract_mat(g, "oovv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("ajbk,ijlk,bl->ai", c2_4d, extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ajbk,icbk,cj->ai", c2_4d, extract_mat(g, "ovvo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bicj,bacd,dj->ai", c2_4d, extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("bicj,bakj,ck->ai", c2_4d, extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("bicj,cjka,bk->ai", c2_4d, extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("bjak,ikbc,cj->ai", c2_4d, extract_mat(g, "oovv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjak,iklj,bl->ai", c2_4d, extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("bjak,icbj,ck->ai", c2_4d, extract_mat(g, "ovvo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("bjci,bjka,ck->ai", c2_4d, extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjci,bdca,dj->ai", c2_4d, extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("bjci,cakj,bk->ai", c2_4d, extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("iajb,ck,bk,cj->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("ibjc,ak,bk,cj->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("jakb,ci,bk,cj->ai", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("iabc,bjdk,cjdk->ai", extract_mat(L, "ovvv", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("iajk,bkcl,bjcl->ai", extract_mat(L, "ovoo", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("iabc,djbk,ckdj->ai", extract_mat(L, "ovvv", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("iajk,blck,blcj->ai", extract_mat(L, "ovoo", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ijkb,ajcl,bkcl->ai", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ijkb,claj,bkcl->ai", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bajc,bidk,cjdk->ai", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bajc,dkbi,cjdk->ai", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("ajbk,ijlc,blck->ai", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("ajbk,icbd,cjdk->ai", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("ajbk,iclk,blcj->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("bicj,bakd,ckdj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("bicj,cdka,bkdj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bicj,kalj,bkcl->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bjak,iklc,blcj->ai", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("bjak,icbd,ckdj->ai", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bjak,iclj,blck->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("bjci,bdka,ckdj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -1.00000000  * np.einsum("bjci,cakd,bkdj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +1.00000000  * np.einsum("bjci,kalj,blck->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.25000000  * np.einsum("ajbk,ijlc,bl,ck->ai", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("ajbk,icbd,cj,dk->ai", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.25000000  * np.einsum("ajbk,iclk,bl,cj->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("bicj,bakd,ck,dj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("bicj,cdka,bk,dj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.25000000  * np.einsum("bicj,kalj,bk,cl->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.25000000  * np.einsum("bjak,iklc,bl,cj->ai", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("bjak,icbd,ck,dj->ai", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.25000000  * np.einsum("bjak,iclj,bl,ck->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("bjci,bdka,ck,dj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.25000000  * np.einsum("bjci,cakd,bk,dj->ai", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.25000000  * np.einsum("bjci,kalj,bl,ck->ai", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("iajb,ckdl,bk,cjdl->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("iajb,ckdl,bl,ckdj->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("iajb,ckdl,cj,bkdl->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("iajb,ckdl,dj,blck->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ibjc,akdl,bk,cjdl->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ibjc,akdl,cj,bkdl->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ibjc,dkal,bl,cjdk->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("ibjc,dkal,cj,bldk->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("jakb,cidl,bk,cjdl->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("jakb,cidl,cj,bkdl->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("jakb,cldi,bk,cldj->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  -0.50000000  * np.einsum("jakb,cldi,dj,bkcl->ai", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("ajbk,icld,bl,cjdk->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("ajbk,icld,cj,bldk->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("ajbk,icld,dk,blcj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bicj,kald,bk,cldj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bicj,kald,cl,bkdj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bicj,kald,dj,bkcl->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjak,icld,bl,ckdj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjak,icld,ck,bldj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjak,icld,dj,blck->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjci,kald,bl,ckdj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjci,kald,ck,bldj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.50000000  * np.einsum("bjci,kald,dj,blck->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.12500000  * np.einsum("ajbk,icld,bl,cj,dk->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.12500000  * np.einsum("bicj,kald,bk,cl,dj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.12500000  * np.einsum("bjak,icld,bl,ck,dj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_ai[:,:] = Jacobian_t_ai[:,:] .+  +0.12500000  * np.einsum("bjci,kald,bl,ck,dj->ai", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
        
    return Jacobian_t_ai 
end

function Jacobian_t_doubles_EE_nosym(F, g, L, t, c1, c2, t2, o, v)
    # Evaluates Omega_0_bjck = <bjck|..H..|HF>
    # a = v[1] - o[end]
    # i = o[end]
    n = (v[end]-o[end])* o[end]
    c2_4d = linear_to_4d(c2, o, v)

    Jacobian_t_aibj = zeros(v[end] - v[1] + 1, o[end], v[end] - v[1] + 1, o[end])
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("ia,bj->aibj", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ib,aj->aibj", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ja,bi->aibj", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("jb,ai->aibj", extract_mat(F, "ov", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ik,akbj->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ik,bjak->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jk,aibk->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jk,bkai->aibj", extract_mat(F, "oo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ca,bjci->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ca,cibj->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cb,aicj->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cb,cjai->aibj", extract_mat(F, "vv", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("iajk,bk->aibj", extract_mat(L, "ovoo", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("iacb,cj->aibj", extract_mat(L, "ovvv", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ikjb,ak->aibj", extract_mat(L, "ooov", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("jbca,ci->aibj", extract_mat(L, "ovvv", o, v), reshape(c1, (v[end]-o[end], o[end])), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("iack,bjck->aibj", extract_mat(L, "ovvo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("iack,ckbj->aibj", extract_mat(L, "ovvo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("jbck,aick->aibj", extract_mat(L, "ovvo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("jbck,ckai->aibj", extract_mat(L, "ovvo", o, v), c2_4d, optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ajck,ibck->aibj", c2_4d, extract_mat(g, "ovvo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akbl,ikjl->aibj", c2_4d, extract_mat(g, "oooo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("akcj,ikcb->aibj", c2_4d, extract_mat(g, "oovv", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bick,jack->aibj", c2_4d, extract_mat(g, "ovvo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkal,iljk->aibj", c2_4d, extract_mat(g, "oooo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bkci,jkca->aibj", c2_4d, extract_mat(g, "oovv", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cibk,jkca->aibj", c2_4d, extract_mat(g, "oovv", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cidj,cadb->aibj", c2_4d, extract_mat(g, "vvvv", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("cjak,ikcb->aibj", c2_4d, extract_mat(g, "oovv", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjdi,cbda->aibj", c2_4d, extract_mat(g, "vvvv", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ckaj,ibck->aibj", c2_4d, extract_mat(g, "ovvo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ckbi,jack->aibj", c2_4d, extract_mat(g, "ovvo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ic,akbj,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ic,bjak,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jc,aibk,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jc,bkai,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ka,bjci,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ka,cibj,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kb,aicj,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kb,cjai,ck->aibj", extract_mat(F, "ov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("iajc,bk,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("iakb,cj,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("iakc,bj,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ibkc,aj,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("icjb,ak,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jakc,bi,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jbka,ci,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("jbkc,ai,ck->aibj", extract_mat(L, "ovov", o, v), reshape(c1, (v[end]-o[end], o[end])), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("iacd,bjck,dk->aibj", extract_mat(L, "ovvv", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("iakl,bjcl,ck->aibj", extract_mat(L, "ovoo", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("iacd,ckbj,dk->aibj", extract_mat(L, "ovvv", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("iakl,clbj,ck->aibj", extract_mat(L, "ovoo", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("iklc,akbj,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("iklc,bjak,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("jbcd,aick,dk->aibj", extract_mat(L, "ovvv", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jbkl,aicl,ck->aibj", extract_mat(L, "ovoo", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("jbcd,ckai,dk->aibj", extract_mat(L, "ovvv", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jbkl,clai,ck->aibj", extract_mat(L, "ovoo", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jklc,aibk,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("jklc,bkai,cl->aibj", extract_mat(L, "ooov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cakd,bjci,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cakd,cibj,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cbkd,aicj,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cbkd,cjai,dk->aibj", extract_mat(L, "vvov", o, v), c2_4d, extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ajck,ibcd,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ajck,iblk,cl->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akbl,ikjc,cl->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akbl,icjl,ck->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("akcj,iklb,cl->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("akcj,idcb,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bick,jacd,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bick,jalk,cl->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkal,iljc,ck->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkal,icjk,cl->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bkci,jkla,cl->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bkci,jdca,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cibk,jkla,cl->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cibk,jdca,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cidj,cakb,dk->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cidj,dbka,ck->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("cjak,iklb,cl->aibj", c2_4d, extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjak,idcb,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjdi,cbka,dk->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("cjdi,dakb,ck->aibj", c2_4d, extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ckaj,ibcd,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ckaj,iblk,cl->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ckbi,jacd,dk->aibj", c2_4d, extract_mat(g, "ovvv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ckbi,jalk,cl->aibj", c2_4d, extract_mat(g, "ovoo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("iajc,bkdl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("iajc,dkbl,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("iakb,cjdl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("iakb,cldj,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("iakc,bjdl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("iakc,bjdl,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("iakc,dlbj,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("iakc,dlbj,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ibkc,ajdl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ibkc,dlaj,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("icjb,akdl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("icjb,dkal,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ickd,albj,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ickd,bjal,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jakc,bidl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jakc,dlbi,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jbka,cidl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jbka,cldi,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("jbkc,aidl,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jbkc,aidl,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +2.00000000  * np.einsum("jbkc,dlai,ckdl->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jbkc,dlai,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jckd,aibl,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("jckd,blai,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kalc,bjdi,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kalc,dibj,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kblc,aidj,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("kblc,djai,cldk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ajck,ibld,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akbl,icjd,ckdl->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("akcj,idlb,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bick,jald,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkal,icjd,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bkci,jdla,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cibk,jdla,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cidj,kalb,ckdl->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjak,idlb,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("cjdi,kalb,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ckaj,ibld,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("ckbi,jald,cldk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("iakc,bjdl,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("iakc,dlbj,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ickd,albj,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ickd,bjal,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("jbkc,aidl,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("jbkc,dlai,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("jckd,aibl,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("jckd,blai,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kalc,bjdi,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kalc,dibj,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kblc,aidj,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kblc,djai,cl,dk->aibj", extract_mat(L, "ovov", o, v), c2_4d, extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ajck,ibld,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akbl,icjd,ck,dl->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("akcj,idlb,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bick,jald,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkal,icjd,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("bkci,jdla,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cibk,jdla,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cidj,kalb,ck,dl->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjak,idlb,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("cjdi,kalb,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ckaj,ibld,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Jacobian_t_aibj[:,:,:,:] = Jacobian_t_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("ckbi,jald,cl,dk->aibj", c2_4d, extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
        
    Jacobian_t_aibj_1d = compact_4d_to_linear(Jacobian_t_aibj, o, v)

    return Jacobian_t_aibj_1d
end


function Jacobian_times_c(F, g, L, t, c1, c2, t2, o, v)
    n = o[end] * (v[end] - o[end])

    results = zeros(n+Int(n*(n+1)/2))
    results[1:n] = reshape(Jacobian_singles_EE_nosym(F, g, L, 2t, c1, 0.5*c2, t2, o, v), (n,1))
    results[n+1:end] = Jacobian_doubles_EE_nosym(F, g, L, 2t, c1, 0.5*c2, t2, o, v)   #This doesn't need any replacement

    return results
end

function Jacobian_t_times_c(F, g, L, t, c1, c2, t2, o, v)
    n = o[end] * (v[end] - o[end])
    results = zeros(n+Int(n*(n+1)/2))
    results[1:n] = reshape(Jacobian_t_singles_EE_nosym(F, g, L, 2t, c1, 0.5*c2, t2, o, v), (n,1))
    results[n+1:end] = Jacobian_t_doubles_EE_nosym(F, g, L, 2t, c1, 0.5*c2, t2, o, v)   #This doesn't need any replacement
    return results
end

function Jacobian_product_factory(F, g, L, t, t2, o, v)
    function Jacobian_product(x)
        n = o[end] * (v[end] - o[end])
        return Jacobian_times_c(F, g, L, t, x[1:n], x[n+1:end], t2, o, v)
    end
    return Jacobian_product
end

function Jacobian_t_product_factory(F, g, L, t, t2, o, v)
    function Jacobian_t_product(x)
        n = o[end] * (v[end] - o[end])
        return Jacobian_t_times_c(F, g, L, t, x[1:n], x[n+1:end], t2, o, v)
    end
    return Jacobian_t_product
end

function full_matrix_times_c(F, g, L, t, c1, c2, cs, cd, t2, reduced_matrix, r1_vec, l1_vec, mu_tilde_r1, hf_nu_tilde, l1_nu_tilde, Omega_ai, o, v)
    n = o[end] * (v[end] - o[end])
    # E0 = energy(F, g, L, t, t2, o, v)
    # sigma is Z times c

    c = zeros(n+Int(n*(n+1)/2))
    c[1:n] = cs
    c[n+1:end] = cd  # removed 0.5
    sigma = zeros(n+Int(n*(n+1)/2))
    sigma[1:n] = reshape(Jacobian_singles_EE_nosym(F, g, L, 2t, cs, 0.5*cd, t2, o, v), (n,1))
    sigma[n+1:end] = Jacobian_doubles_EE_nosym(F, g, L, 2t, cs, 0.5*cd, t2, o, v)   #This doesn't need any replacement
    sigma -= mu_tilde_r1 * (l1_vec' * c)
    sigma -= r1_vec * (l1_nu_tilde' * c)
    sigma -= r1_vec * (l1_vec' * c) * reduced_matrix[2,2]
    
    sigma_2d = 1*(cs * Omega_ai' + Omega_ai * cs')
    sigma_2d = scale_diagonal_2d(sigma_2d, 0.5, o, v)
    sigma_4d = zeros(v[end]-o[end], o[end], v[end]-o[end], o[end])
    @cast  sigma_4d[a,i,b,j] = sigma_2d[(a,i),(b,j)]
    sigma[n+1:end] += compact_4d_to_linear(sigma_4d, o, v)

    result = zeros(2+n+Int(n*(n+1)/2))
    result[1] = reduced_matrix[1,2]*c2 + hf_nu_tilde' * c
    result[2] = reduced_matrix[2,1]*c1 + (reduced_matrix[2,2] - reduced_matrix[1,1])*c2 + l1_nu_tilde' * c
    result[3:end] = sigma + c2 * mu_tilde_r1

    return result
end

function full_matrix_product_factory(F, g, L, t, t2, reduced_matrix,r1_vec, l1_vec, mu_tilde_r1, hf_nu_tilde, l1_nu_tilde, Omega_ai, o, v)
    function full_matrix_product(x)
        n = o[end] * (v[end] - o[end])
        return full_matrix_times_c(F, g, L, t, x[1], x[2], x[3:n+2], x[n+3:end], t2, reduced_matrix, r1_vec, l1_vec, mu_tilde_r1, hf_nu_tilde, l1_nu_tilde, Omega_ai, o, v)
    end
    return full_matrix_product
end


function build_omega(F, g, L, t, t2, o, v)
    n = o[end] * (v[end] - o[end])
    omega = zeros(n^2+n)
    omega[1:n] = reshape(Omega_0_bj(F, g, L, 2t, t2, o, v), (n, 1))
    omega[n+1:end] = reshape(Omega_0_bjck(F, g, L, 2t, t2, o, v), (n^2, 1))
    return omega
end

function get_eigs(f,m,sym::Bool,n_ev::Int)
    #A_imp = FunctionMap{Float64}(f,nothing, m, ismutating=false, issymmetric=sym)
    A_imp = FunctionMap{Float64}(f, m, issymmetric=sym)
    @show d,V,nconv,niter,nmult,resid = eigs(A_imp,nev=n_ev, which=:SM, tol=1e-12, maxiter=1000)
    return d, V
end

function get_generalized_eigs(f,B,m,sym::Bool,n_ev::Int)
    #A_imp = FunctionMap{Float64}(f,nothing, m, ismutating=false, issymmetric=sym)
    A_imp = FunctionMap{Float64}(f, m, issymmetric=sym)
    @show d,V,nconv,niter,nmult,resid = eigs(A_imp, B; nev=n_ev, which=:SM, tol=1e-8, maxiter=10000, explicittransform=:none)
    return d, V
end

# ENV["PYTHON"]="/home/frossi/.julia/conda/3/x86_64/bin/python3.11"
# ENV["PYTHON"]="/home/federicor/miniconda3/bin/python3.11"
# import Pkg; Pkg.build("PyCall")
using PyCall

np = pyimport("numpy")
pyscf = pyimport("pyscf")
scf = pyimport("pyscf.scf")
cc = pyimport("pyscf.cc")
qmmm = pyimport("pyscf.qmmm")

# mol = pyscf.M(atom="H 0 0 0; Li 1.6 0 0; He 1.0 2.0 0.0; He 1.2 -1.5 1.0", basis="sto-3g")
mol = pyscf.M(atom="H          0.86681        0.60144        0.00000;
H         -0.86681        0.60144        0.00000;
O          0.00000       -0.07579        0.00000", basis="sto-3g")
# mol = pyscf.M(atom="H 0.0 0.0 0.0; H 1.0 0.5 0.2", basis="sto-3g")
mf = scf.RHF(mol)
mf.conv_tol = 1e-15
mf.max_cycle = 1000
coords = [(0.35, 0.6, 0.8)]
charges = [0.0]
hf = qmmm.mm_charge(mf, coords, charges).run()

# FCI solution
cisolver = pyscf.fci.FCI(hf)
cisolver.nroots = 4
(e0, e1, e2, e3), _ = cisolver.kernel()
println(e0, e1, e2, e3)
println("FCI E0: ", e0)
println("FCI E1: ", e1)
println("FCI E2: ", e2)
println("FCI E3: ", e3)


# CCSD solution
mycc = cc.CCSD(hf)
mycc.conv_tol = 1e-12
mycc.max_cycle = 3000
mycc.run()
@show (e0, e1), vec_es = mycc.eomee_ccsd_singlet(nroots=2)
display(vec_es)
println("t1: ", mycc.t1)
println("t2: ", mycc.t2)

println("Nuclear repulsion: ", hf.energy_nuc())


C = hf.mo_coeff
pyscf.tools.dump_mat.dump_mo(mol, C)

hf.analyze()

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

# Initial values
n = o[end] * (v[end] - o[end])
amplitudes = zeros(n^2+n)
t = zeros(mol.nao, mol.nao)
t2 = zeros(mol.nao, mol.nao, mol.nao, mol.nao)

old = 0.0
E2 = 0.0

println("My HF", hf.energy_nuc()+HF_energy(F, g, L, t, t2, o, v))

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
                    if (b==c) && (j==k)
                        pre_t2[b,j,c,k] = pre_t2[b,j,c,k] / 2
                    end
                end
            end
        end
    end

    do_diis = true
    remove_projection = true

    if do_diis
        diis_dim = 0
        diis_max = 8
        diis_vals_t1 = []
        diis_vals_t2 = []
        diis_errors = []
    end

    println()
    println("####### STARTING ITERATIONS #######")

    for k in 1:100
        global E2

        @show global amplitudes = -build_omega(F, g, L, t, t2, o, v)

        # if remove_projection && k>2
        #     global l1_vec, r1_vec
        #     amplitudes_rescaled = deepcopy(amplitudes)
        #     amplitudes_rescaled[1:n] = reshape(reshape(amplitudes[1:n], (nv, no)), (n,1))
        #     amplitudes_rescaled[n+3:end] = reshape(reshape(amplitudes[n+3:end], (nv, no, nv, no)), (n^2, 1))
        #     omega_restr = zeros(n+Int(n*(n+1)/2))
        #     omega_restr[1:n] = amplitudes_rescaled[1:n]
        #     omega_restr[n+1:end] = compact_4d_to_linear(reshape(amplitudes_rescaled[n+3:end], (nv, no, nv ,no)), o ,v)

        #     omega_tilde = omega_restr - (omega_restr' * l1_vec) * r1_vec

        #     amplitudes[1:n] = omega_tilde[1:n]
        #     amplitudes[n+3:end] = reshape(linear_to_4d(omega_tilde, o, v), (n^2, 1))
        # end



        if do_diis && k > 1
            old_t1 = reshape(t[v,o], (n,1))
            old_t2 = reshape(t2[v,o,v,o], (n^2, 1))
        end


        if remove_projection
            omega_restr = zeros(n+Int(n*(n+1)/2))
            omega_restr[1:n] = deepcopy(amplitudes[1:n])
            omega_restr[n+1:end] = compact_4d_to_linear(reshape(amplitudes[n+1:end], (nv, no, nv ,no)), o ,v)
            omega_original_restr = deepcopy(omega_restr)
            if k > 1
                global l1_vec, r1_vec
                omega_restr = omega_restr - (omega_restr' * l1_vec) * r1_vec
                amplitudes[1:n] = omega_restr[1:n]
                amplitudes[n+1:end] = reshape(linear_to_4d(omega_restr[n+1:end], o, v), (n^2, 1))
            end
        end

        @show addition_t1 = reshape(amplitudes[1:n], (nv, no)) ./ pre_t[v, o]
        global t[v, o] += addition_t1

        @show addition_t2 = reshape(amplitudes[n+1:end], (nv, no, nv, no)) ./ pre_t2[v, o, v, o]
        global t2[v, o, v, o] += addition_t2

        if remove_projection
            # Right vectors
            Jacobian_product = Jacobian_product_factory(F, g, L, t, t2, o, v)
            Rλ = zeros(Float64, 2)  # Change the type (Float64) as needed
            Rλ, RVec = get_eigs(Jacobian_product, n+Int(n*(n+1)/2), false, 2)
            r1_vec = real.(RVec[:,1]) * sign(real.(RVec[1,1]))
            println(r1_vec)

            # Left vectors
            Jacobian_t_product = Jacobian_t_product_factory(F, g, L, t, t2, o, v)
            Lλ = zeros(Float64, 2)  # Change the type (Float64) as needed
            Lλ, LVec = get_eigs(Jacobian_t_product, n+Int(n*(n+1)/2), false, 2)
            l1_vec = real.(LVec[:,1]) * sign(real.(LVec[1,1]))
            println(l1_vec)

            # checking stuff
            println("Projection L1|R2: ",  sum(l1_vec .* real.(RVec[:,2])))

            # Enforcing normalization
            @show scalar = l1_vec' * r1_vec
            l1_vec = l1_vec / scalar
            println("Check <L1|R1>: ", l1_vec' * r1_vec)

            # Remove projection on t
            t_vec = zeros(n+Int(n*(n+1)/2))
            t_vec[1:n] = reshape(t[v, o], (n, 1))
            t_vec[n+1:end] = compact_4d_to_linear(t2[v,o,v,o], o ,v)
            println("t before projection")
            display(t_vec)
            proj_t = l1_vec' * t_vec
            t_vec = t_vec - proj_t * r1_vec
            println("t after projection")
            display(t_vec)
            # println("Projection:", sum(t_vec .* full_l1_vec))
            t[v, o] = reshape(t_vec[1:n], (nv, no))
            t2[v, o, v, o] = linear_to_4d(t_vec[n+1:end], o, v)


            println("Omega original restricted")
            display(omega_original_restr)
            amplitudes_rescaled = deepcopy(amplitudes)
            amplitudes_rescaled[1:n] = reshape(reshape(amplitudes[1:n], (nv, no)) ./ pre_t[v, o], (n,1))
            amplitudes_rescaled[n+1:end] = reshape(reshape(amplitudes[n+1:end], (nv, no, nv, no)) ./ pre_t2[v, o, v, o], (n^2, 1))
            omega_restr = zeros(n+Int(n*(n+1)/2))
            omega_restr[1:n] = -amplitudes_rescaled[1:n]
            omega_restr[n+1:end] = compact_4d_to_linear(reshape(-amplitudes_rescaled[n+1:end], (nv, no, nv ,no)), o ,v)
            println("Omega restricted")
            display(omega_restr)
            println("Omega resricted on L2", omega_restr' * real.(LVec[:,2]) * sign(real.(LVec[1,2])))
            println("Omega resricted on L1 (<L1|H|HF>): ", omega_restr' * l1_vec)
            println("Omega resricted on R1 (<R1|H|HF>): ", omega_restr' * l1_vec)

            # Omega_restr_norescal = zeros(n+Int(n*(n+1)/2))
            # Omega_restr_norescal[1:n] = -deepcopy(amplitudes[1:n])
            # Omega_restr_norescal[n+1:end] = compact_4d_to_linear(reshape(-deepcopy(amplitudes[n+1:end]), (nv, no, nv ,no)), o ,v)

            Omega_ai = reshape(Omega_0_bj(F, g, L, 2t, t2, o, v), (n,1))
            out_matr = 0.5*(Omega_ai * r1_vec[1:n]' +  r1_vec[1:n]*Omega_ai')
            l1_vec_2d = zeros(n,n)
            l1_vec_4d = linear_to_4d(l1_vec[n+1:end], o, v)
            @cast l1_vec_2d[(a,i),(b,j)] := l1_vec_4d[a,i,b,j]
            additional_22 = sum(l1_vec_2d .* out_matr) 

            reduced_matrix = zeros(2,2)
            reduced_matrix[1,1] = hf.energy_nuc() + energy(F, g, L, 2t, t2, o, v)
            reduced_matrix[1,2] = r1_vec' * restricted_eta(F, g, L, t, t2, o, v)
            reduced_matrix[2,1] = -omega_original_restr' * l1_vec  # Minus because comes from amplitudes = -omega
            reduced_matrix[2,2] = reduced_matrix[1,1] + l1_vec' * Jacobian_product(r1_vec)  + additional_22
            println("<L1[D]|R1[S]H[S]|HF> additional: ", additional_22 )
            println("<L1|H|R1> with nuclear repulsion : ", reduced_matrix[2,2])
            println("<HF|H|R1>: ",  reduced_matrix[1,2])

            println("L1 * A * R1: ", l1_vec' * Jacobian_product(r1_vec))

            println("Reduced matrix", reduced_matrix)
            println("Matrix")
            display(reduced_matrix)
            new_energies = eigvals(reduced_matrix)
            println("NEW E0 FROM MATRIX: ", new_energies[1])
            println("NEW E1 FROM MATRIX: ", new_energies[2])
            println(eigvecs(reduced_matrix))

            println("Evaluating remaining element <mu_tilde|H|R1>")
            mu_tilde_r1_2d = (r1_vec[1:n] * Omega_ai' + Omega_ai* r1_vec[1:n]')
            mu_tilde_r1_2d = scale_diagonal_2d(mu_tilde_r1_2d, 0.5, o, v)
            # mu_tilde_r1_2d = 1*(r1_vec[1:n] * Omega_ai')   # Trying this
            mu_tilde_r1_4d = zeros(v[end]-o[end], o[end], v[end]-o[end], o[end])
            @cast  mu_tilde_r1_4d[a,i,b,j] = mu_tilde_r1_2d[(a,i),(b,j)]
            mu_tilde_r1 = zeros(n+Int(n*(n+1)/2))
            mu_tilde_r1[n+1:end] = compact_4d_to_linear(mu_tilde_r1_4d, o, v)
            mu_tilde_r1 -= additional_22 * r1_vec  # This was missing
            println("Norm of element <mu_tilde|H|R1>: ", norm(mu_tilde_r1))
            display(mu_tilde_r1)

            println("Evaluating remaining element <HF|H|nu_tilde>")
            hf_nu_tilde = restricted_eta(F, g, L, t, t2, o, v) - reduced_matrix[1,2] * l1_vec
            println("Norm of element <HF|H|nu_tilde>: ", norm(hf_nu_tilde))
            display(hf_nu_tilde)

            println("Evaluating remaining element <L1|H|nu_tilde>")
            l1_nu_tilde = zeros(n+Int(n*(n+1)/2))
            l1_nu_tilde[1:n] = l1_vec_2d * Omega_ai
            l1_nu_tilde -= additional_22 * l1_vec    # This was missing
            println("Norm of element <L1|H|nu_tilde>: ", norm(l1_nu_tilde))
            display(l1_nu_tilde)

            println("Building the whole matrix...")
            full_matrix_product = full_matrix_product_factory(F, g, L, t, t2, reduced_matrix, r1_vec, l1_vec, mu_tilde_r1, hf_nu_tilde, l1_nu_tilde,  Omega_ai, o, v)
            full_eig = zeros(Float64, 2)  # Change the type (Float64) as needed
            full_eigval, full_eigvec = get_eigs(full_matrix_product, 2+n+Int(n*(n+1)/2), false, 2)
            println("The first 2 eigvals are: ",  full_eigval .+ reduced_matrix[1,1])
            println("NEW E0 FROM FULL MATRIX: ", full_eigval[1] + reduced_matrix[1,1])
            println("NEW E1 FROM FULL MATRIX: ", full_eigval[2] + reduced_matrix[1,1])


            build_full_matrix = false
            if build_full_matrix
                # bUILD THE ENTIRE FULL MATRIX
                mat_from_fullmatrix = zeros(2+n+Int(n*(n+1)/2),2+n+Int(n*(n+1)/2))
                for i in 1:2+n+Int(n*(n+1)/2)
                    c_vec = zeros(2+n+Int(n*(n+1)/2))
                    c_vec[i] = 1.0
                    mat_from_fullmatrix[:,i] = full_matrix_product(c_vec)
                    mat_from_fullmatrix[i,i] += reduced_matrix[1,1]
                end

                println("Testing H * r1")
                test_r1 = zeros(2+n+Int(n*(n+1)/2))
                test_r1[3:end] = r1_vec
                display(mat_from_fullmatrix*test_r1)

                println("Testing L1 * H")
                test_l1 = zeros(2+n+Int(n*(n+1)/2))
                test_l1[3:end] = l1_vec
                display(test_l1'*mat_from_fullmatrix)

                S_mat = zeros(2+n+Int(n*(n+1)/2),2+n+Int(n*(n+1)/2))
                for i in 1:2+n+Int(n*(n+1)/2)
                    S_mat[i,i] = 1.0
                end
                for mu in 1:n+Int(n*(n+1)/2)
                    for nu in 1:n+Int(n*(n+1)/2)
                        S_mat[mu+2,nu+2] -= r1_vec[mu] * l1_vec[nu] 
                    end
                end
                println("S mat")
                display(S_mat)
                println("Eigvals S_mat: ", eigvals(S_mat))
                # S_half = S_mat^(-0.5)
                # println("S^-1/2 mat")
                # display(S_half)


                println("Full matrix")
                display(mat_from_fullmatrix)
                println("Eigvals fullmat: ", eigvals(mat_from_fullmatrix))

                # println("Full matrix, S_half M S_half")
                # mat_from_fullmatrix=S_half*mat_from_fullmatrix*S_half
                # display(mat_from_fullmatrix)
                # display(eigvals(mat_from_fullmatrix))

                # if k > 8
                #     println("solving generalized eigen problem")
                #     full_matrix_product = full_matrix_product_factory(F, g, L, t, t2, reduced_matrix, r1_vec, l1_vec, mu_tilde_r1, hf_nu_tilde, l1_nu_tilde,  Omega_ai, o, v)
                #     full_eig = zeros(Float64, 2)  # Change the type (Float64) as needed
                #     full_eigval, full_eigvec = get_generalized_eigs(full_matrix_product, S_mat, 2+n+Int(n*(n+1)/2), false, 2)
                #     println("The first 2 eigvals are: ",  full_eigval .+ reduced_matrix[1,1])
                # end
            end

            omega_tilde = omega_restr - (omega_restr' * l1_vec) * r1_vec
            println("Omega tilde, projection removed")
            display(omega_tilde)
            println("Omega_tilde on L1: ", omega_tilde' * l1_vec)
            println("Omega_tilde on R1: ", omega_tilde' * r1_vec)
        end

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

        # Printing stuff
        println("t at step ", k, " : ")
        display(t[v, o])
        println("t2 at step ", k, " : ")
        display(t2[v, o, v, o])
        println("Module of t ", norm(t[v,o]))
        println("Module of t2 ", norm(t2[v,o,v,o]))

        println(" --------------Energy at step ", k, " : ", hf.energy_nuc() + energy(F, g, L, 2t, t2, o, v))
        println(" --------------E_2 at step ", k, " : ", hf.energy_nuc() + E2)
        println("E1 - E0 ", E2 - energy(F, g, L, 2t, t2, o, v))
        diff = old - hf.energy_nuc() - energy(F, g, L, 2t, t2, o, v)
        println(" --------------Difference at step ", k, " : ", diff)
        global old = hf.energy_nuc() + energy(F, g, L, 2t, t2, o, v)
        println(" --------------Module of omegas ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes) ]))) #if l != 1 + nv * (no - 1) # if l != n^2+2n+2+(v[end]-o[end])^2*o[end]*(o[end]-1)+(v[end]-o[end])*(o[end]-1)

        E2 = 0.0  # not needed this time

        println("Omega_bj_0 ", norm(amplitudes[1:n]))
        println("Omega_bjck_0 ", norm(amplitudes[n+1:end]))

        println()
        println("######## END OF ITERATION ",k, " #########")
        println()

        module_omega = norm(amplitudes)
        if remove_projection
            module_omega_tilde = norm(omega_tilde)
        else
            module_omega_tilde = 1.0
        end

        if abs(old > 500) || isnan(old)
            throw("Energy diverges")
            println("Energy diverges")
            break
            # elseif diff < 0
            #     throw("Energy start to increase")
        elseif module_omega < 1e-12 || module_omega_tilde < 1e-12
            println("Converged!")
            break
        end
    end

    println("########## FINISHED ##########")

    println("T1:")
    display(t[v,o])
    println("T2")
    display(t2[v,o,v,o])

    println("Now that is finished, EOM:")
    println("Right side")
    Jacobian_product = Jacobian_product_factory(F, g, L, t, t2, o, v)
    println(typeof(Jacobian_product(ones(n+Int(n*(n+1)/2)))), Jacobian_product(ones(n+Int(n*(n+1)/2))))
    # Solve the eigenvalue problem using ARPACK
    Rλ = zeros(Float64, 6)  # Change the type (Float64) as needed
    Rλ, RVec = get_eigs(Jacobian_product, n+Int(n*(n+1)/2), false, 6)
    display(RVec)
    display(Rλ)

    # println("Exact right side")
    # Jac = zeros(n+Int(n*(n+1)/2), n+Int(n*(n+1)/2))
    # for k in 1:n+Int(n*(n+1)/2)
    #     vec = zeros(n+Int(n*(n+1)/2))
    #     vec[k] = 1.0
    #     Jac[:,k] = Jacobian_product(vec)
    # end

    # display(Jac)
    # display(eigvals(Jac))
    # display(eigvecs(Jac))

    println("Left side")
    Jacobian_t_product = Jacobian_t_product_factory(F, g, L, t, t2, o, v)
    Lλ = zeros(Float64, 6)  # Change the type (Float64) as needed
    Lλ, LVec = get_eigs(Jacobian_t_product, n+Int(n*(n+1)/2), false, 6)
    display(LVec)
    display(Lλ)

    # println("Exact left side")
    # Jac = zeros(n+Int(n*(n+1)/2), n+Int(n*(n+1)/2))
    # for k in 1:n+Int(n*(n+1)/2)
    #     vec = zeros(n+Int(n*(n+1)/2))
    #     vec[k] = 1.0
    #     Jac[:,k] = Jacobian_t_product(vec)
    # end

    # display(Jac)
    # display(eigvals(Jac))
    # display(eigvecs(Jac))

    println("Checking orthogonality")
    display(LVec' * RVec )
    println(LVec[:,1]' * RVec[:,1] / (norm(LVec[:,1])*norm(RVec[:,1])))

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
end