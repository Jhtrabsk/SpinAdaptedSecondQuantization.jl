## set guess amplitudes to zeros

using LinearAlgebra

function T1_transform_1e(F, t, o, v)
    t1_exp = zeros(v[end], v[end])
    t1_exp[v, o] = t
    x = I(v[end]) - t1_exp
    y = I(v[end]) + t1_exp'
    return fixed_einsum("pr,rs,qs->pq", x, F, y, optimize="optimal")
end

function Gamma_transform_1e(F, Gamma, o, v)
    G1_exp = zeros(v[end], v[end]) # make a zero by zero matrix
    G1_exp[2:end, 1] = Gamma # insert t in each v,o element
    x = I(v[end]) - G1_exp
    y = I(v[end]) + G1_exp'
    return fixed_einsum("pr,rs,qs->pq", x, F, y, optimize="optimal")
end

# Fix the transformation for S1 and Gamma 1

function T1_transform_2e(g, t, o, v)
    t1_exp = zeros(v[end], v[end])
    t1_exp[v, o] = t
    x = I(v[end]) - t1_exp
    y = I(v[end]) + t1_exp'
    return fixed_einsum("pt,qu,rm,sn,tumn ->pqrs", x, y, x, y, g, optimize="optimal")
end

function Gamma_transform_2e(g, t, o, v)
    t1_exp = zeros(v[end], v[end])
    t1_exp[2:end, 1] = t
    x = I(v[end]) - t1_exp
    y = I(v[end]) + t1_exp'
    return fixed_einsum("pt,qu, turs ->pqrs", x, y, g, optimize="optimal")
end

function T1_transform_g_p_2e(g, t, o, v)
    t1_exp = zeros(v[end], v[end])
    t1_exp[v, o] = t
    x = I(v[end]) - t1_exp
    y = I(v[end]) + t1_exp'
    return fixed_einsum("rm ,sn ,pqmn ->pqrs", x, y, g, optimize="optimal")
end

#Change 
# np.einsum  --> fixed_einsum
# Has been done

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

function extract_mat(mat, string, o, v, B=2)
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
        elseif c == 'I'
            push!(dims, 1)
        elseif c == 'A'
            push!(dims, B)
        elseif c == 'V'
            push!(dims, 2:v[end])
        else
            throw("Unrecognized character")
        end
    end
    return mat[dims...]
end



# function calculate_hf_energy(L_oooo, F_oo)
#     E = 0
#     @show E = E .+ +2.00000000 * fixed_einsum("ii->", F_oo, optimize="optimal")
#     @show E = E .+ -1.00000000 * fixed_einsum("iijj->", L_oooo, optimize="optimal")
# end

function energy(F, g, L, s, t, gamma, s2, t2, o, v)
    # Evaluates Energy = <HF|..H..|HF>
    E = 0
    return E
end


# function name_omega(integrals..., amplitudes..., o, v)
#     Omega = 0
#     #Paste everything

#     return Omega

# end


# Correct
function HF_Energy_(F, h, o, v)
    # Evaluates HF_Energy = <HF|..H..|HF>

    B = 2
    E = 0
    E =  E .+  +1.00000000  * fixed_einsum("ii->", extract_mat(F, "oo", o, v, B), optimize="optimal");
    println(E, " F")
    E =  E .+  +1.00000000  * fixed_einsum("ii->", extract_mat(h, "oo", o, v, B), optimize="optimal");
    print(E , " h \n")
 
return E
end


function HF_Energy_p(h_p, g_p, o, v)
    # Evaluates HF_Energy = <HF|..H..|HF>

    B = 2
    
    E = 0
    E +=      +1.00000000  * extract_mat(h_p, "II", o, v);
    #print(E, "g_p \n")
    E = E .+  -2.00000000  * fixed_einsum("ii->", extract_mat(g_p, "IIoo", o, v, B), optimize="optimal");
    print(E, "g_p \n")
    #print(E, " h_p \n")
    #print(h_p, "h_p \n")
    return E
end 

function HF_p(F, h, h_p, g_p, o, v)

    B = 2
    
    E = 0
    E +=  +1.00000000  * extract_mat(h_p, "II", o, v);
    E = E .+  +1.00000000  * fixed_einsum("ii->", extract_mat(F, "oo", o, v, B), optimize="optimal");
    E = E .+  -1.00000000  * fixed_einsum("ii->", extract_mat(g_p, "IIoo", o, v, B), optimize="optimal");
    E = E .+  +1.00000000  * fixed_einsum("ii->", extract_mat(h, "oo", o, v, B), optimize="optimal");
    
    return E
end     
# Correct
function CC_Energy(F, L, g, h_p, g_p, s, u ,o, v)
    # Evaluates CC_Energy = <HF|..H..|HF>
    E = 0
    B = 2

    #E +=  +1.00000000  * extract_mat(h_p, "II", o, v);
    E = E .+  +2.00000000  * fixed_einsum("ii->", extract_mat(F, "oo", o, v, B), optimize="optimal");
    #E = E .+  -2.00000000  * fixed_einsum("ii->", extract_mat(g_p, "IIoo", o, v, B), optimize="optimal");
    E = E .+  -1.00000000  * fixed_einsum("iijj->", extract_mat(L, "oooo", o, v, B), optimize="optimal");
   # E = E .+  -2.00000000  * fixed_einsum("Aia,Aai->", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
    E = E .+  +1.00000000  * fixed_einsum("iajb,aibj->", extract_mat(g, "ovov", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    
    return E
end

function CC_ref_energy(F, g, L, t, t2, o, v)
    # Evaluates Energy = <HF|..H..|HF>
    E = 0.0
    E = E .+  +2.00000000  * np.einsum("ii->", extract_mat(F, "oo", o, v), optimize="optimal");
    E = E .+  -1.00000000  * np.einsum("iijj->", extract_mat(L, "oooo", o, v), optimize="optimal");
    #E = E .+  +1.00000000  * np.einsum("ia,ai->", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    E = E .+  +1.00000000  * np.einsum("iajb,aibj->", extract_mat(L, "ovov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #E = E .+  +0.25000000  * np.einsum("iajb,ai,bj->", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
                
    return E
end

function Omega_0_bj(F, g, L, t, t2, o, v)
    # Evaluates Omega_0_bj = <bj|..H..|HF>
    # a = v[1] - o[end]
    # i = o[end]
    Omega_0_ai = zeros(v[end] - v[1] + 1, o[end])
    Omega_0_ai[:,:] +=  +1.00000000  * extract_mat(F, "vo", o, v);
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.50000000  * np.einsum("ab,bi->ai", extract_mat(F, "vv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("ji,aj->ai", extract_mat(F, "oo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +2.00000000  * np.einsum("jb,aibj->ai", extract_mat(F, "ov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -1.00000000  * np.einsum("jb,ajbi->ai", extract_mat(F, "ov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.50000000  * np.einsum("aijb,bj->ai", extract_mat(L, "voov", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.25000000  * np.einsum("jb,aj,bi->ai", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +1.00000000  * np.einsum("abjc,bicj->ai", extract_mat(L, "vvov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -1.00000000  * np.einsum("jikb,ajbk->ai", extract_mat(L, "ooov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +0.25000000  * np.einsum("abjc,bi,cj->ai", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.25000000  * np.einsum("jikb,aj,bk->ai", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,aj,bick->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bi,ajck->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  +1.00000000  * np.einsum("jbkc,bj,aick->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bj,akci->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_ai[:,:] = Omega_0_ai[:,:] .+  -0.12500000  * np.einsum("jbkc,aj,bi,ck->ai", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
                    
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
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("aibc,cj->aibj", extract_mat(g, "vovv", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("aikj,bk->aibj", extract_mat(g, "vooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("acbj,ci->aibj", extract_mat(g, "vvvo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bjki,ak->aibj", extract_mat(g, "vooo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("aikc,bjck->aibj", extract_mat(L, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("bjkc,aick->aibj", extract_mat(L, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("aikc,bkcj->aibj", extract_mat(g, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("acbd,cidj->aibj", extract_mat(g, "vvvv", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("ackj,bkci->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bjkc,akci->aibj", extract_mat(g, "voov", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -1.00000000  * np.einsum("bcki,akcj->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +1.00000000  * np.einsum("kilj,akbl->aibj", extract_mat(g, "oooo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ak,bjci->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,bk,aicj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,ci,akbj->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kc,cj,aibk->aibj", extract_mat(F, "ov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("aikc,bk,cj->aibj", extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("acbd,ci,dj->aibj", extract_mat(g, "vvvv", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("ackj,bk,ci->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bjkc,ak,ci->aibj", extract_mat(g, "voov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("bcki,ak,cj->aibj", extract_mat(g, "vvoo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    ##Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kilj,ak,bl->aibj", extract_mat(g, "oooo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,ci,bjdk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("ackd,dk,bjci->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,cj,aidk->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("bckd,dk,aicj->aibj", extract_mat(L, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,ak,bjcl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kilc,cl,akbj->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,bk,aicl->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("kjlc,cl,aibk->aibj", extract_mat(L, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,bk,cidj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,ci,bkdj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("ackd,dj,bkci->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,ak,cjdi->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,cj,akdi->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.50000000  * np.einsum("bckd,di,akcj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,ak,blcj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,bl,akcj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kilc,cj,akbl->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,al,bkci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,bk,alci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.50000000  * np.einsum("kjlc,ci,albk->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("ackd,bk,ci,dj->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.12500000  * np.einsum("bckd,ak,cj,di->aibj", extract_mat(g, "vvov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("kilc,ak,bl,cj->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.12500000  * np.einsum("kjlc,al,bk,ci->aibj", extract_mat(g, "ooov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
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
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ak,ci,bjdl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ak,dl,bjci->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bk,cj,aidl->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,bk,dl,aicj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,ci,dl,akbj->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  -0.25000000  * np.einsum("kcld,cj,dl,aibk->aibj", extract_mat(L, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,bl,cidj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,ci,bldj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ak,dj,blci->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,bk,cj,aldi->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,bk,di,alcj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.25000000  * np.einsum("kcld,ci,dj,akbl->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t2, "vovo", o, v), optimize="optimal");
   #Omega_0_aibj[:,:,:,:] = Omega_0_aibj[:,:,:,:] .+  +0.06250000  * np.einsum("kcld,ak,bl,ci,dj->aibj", extract_mat(g, "ovov", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), extract_mat(t, "vo", o, v), optimize="optimal");
    
    return Omega_0_aibj
end

# Correct
function Omega_ai(F, g, h_p, g_p, s, s2, u, o, v)

    Omega_ai= zeros(v[end] - o[end], o[end])
    B = 2

    Omega_ai[:,:] +=  +1.00000000  * extract_mat(F, "vo", o, v, B);
    Omega_ai[:,:] +=  -1.00000000  * extract_mat(g_p, "IIvo", o, v, B);
    Omega_ai[:,:] = Omega_ai[:,:] .+  +1.00000000  * fixed_einsum("A,Aai->ai", extract_mat(h_p, "IV", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  +1.00000000  * fixed_einsum("jb,aibj->ai", extract_mat(F, "ov", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  -1.00000000  * fixed_einsum("Aab,Abi->ai", extract_mat(g_p, "IVvv", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  +1.00000000  * fixed_einsum("Aji,Aaj->ai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  -2.00000000  * fixed_einsum("Ajj,Aai->ai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  -1.00000000  * fixed_einsum("jb,aibj->ai", extract_mat(g_p, "IIov", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  -1.00000000  * fixed_einsum("jikb,ajbk->ai", extract_mat(g, "ooov", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  +1.00000000  * fixed_einsum("jbac,bjci->ai", extract_mat(g, "ovvv", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  -2.00000000  * fixed_einsum("Ajb,Aaibj->ai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
    Omega_ai[:,:] = Omega_ai[:,:] .+  +1.00000000  * fixed_einsum("Ajb,Aajbi->ai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");

    return Omega_ai 
end 


# Correct
function Omega_aibj(F, g, g_p, h_p, s, s2, t, u , o, v)
    # Evaluates Omega_0_bj = <bj|..H..|HF>

    Omega_bjai = zeros(v[end] - o[end], o[end], v[end] - o[end], o[end])
    B = 2

    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("aibj->bjai", extract_mat(g, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("ki,akbj->bjai", extract_mat(F, "oo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kj,aibk->bjai", extract_mat(F, "oo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("ac,cibj->bjai", extract_mat(F, "vv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("bc,cjai->bjai", extract_mat(F, "vv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Aai,Abj->bjai", extract_mat(g_p, "IVvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Abj,Aai->bjai", extract_mat(g_p, "IVvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("ki,akbj->bjai", extract_mat(g_p, "IIoo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kj,aibk->bjai", extract_mat(g_p, "IIoo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("ac,cibj->bjai", extract_mat(g_p, "IIvv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("bc,cjai->bjai", extract_mat(g_p, "IIvv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("A,Aaibj->bjai", extract_mat(h_p, "IV", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kiac,bjck->bjai", extract_mat(g, "oovv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kibc,akcj->bjai", extract_mat(g, "oovv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kilj,akbl->bjai", extract_mat(g, "oooo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kjac,bkci->bjai", extract_mat(g, "oovv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kjbc,aick->bjai", extract_mat(g, "oovv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("acbd,cidj->bjai", extract_mat(g, "vvvv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcai,bjck->bjai", extract_mat(g, "ovvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcbj,aick->bjai", extract_mat(g, "ovvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Aac,Abjci->bjai", extract_mat(g_p, "IVvv", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Abc,Aaicj->bjai", extract_mat(g_p, "IVvv", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Aki,Aakbj->bjai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akj,Aaibk->bjai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -2.00000000  * fixed_einsum("Akk,Aaibj->bjai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Aak,bjci->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Abk,aicj->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Aci,akbj->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("Akc,Acj,aibk->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Akc,Aai,bjck->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("Akc,Abj,aick->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcld,akbl,cidj->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcld,akdj,blci->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcld,aicl,bkdj->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kcld,akbj,cidl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kcld,aibk,cjdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kcld,aicj,bkdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kcld,bjci,akdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  -1.00000000  * fixed_einsum("kcld,bjcl,aidk->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(t, "vovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    Omega_bjai[:,:,:,:] = Omega_bjai[:,:,:,:] .+  +1.00000000  * fixed_einsum("kcld,aick,bjdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(u, "vovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");

 
    return Omega_bjai
end

# Correct
function Omega_AI(F, L, g_p, s, s2, o, v)
    # Evaluates Omega_0_bj = <bj|..H..|HF> # this need to be a vector
    Omega_AI = zeros(v[end] - 1)
    
    for B in 2: v[end]

        Omega_AI[B-1] +=  +0.50000000  * extract_mat(h_p, "AI", o, v, B);
        Omega_AI[B-1] = Omega_AI[B-1] .+  -1.00000000  * fixed_einsum("ii->", extract_mat(g_p, "AIoo", o, v, B), optimize="optimal");
        Omega_AI[B-1] = Omega_AI[B-1] .+  +1.00000000  * fixed_einsum("ia,ai->", extract_mat(F, "ov", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI[B-1] = Omega_AI[B-1] .+  -1.00000000  * fixed_einsum("Bia,Bai->", extract_mat(g_p, "AVov", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI[B-1] = Omega_AI[B-1] .+  +0.50000000  * fixed_einsum("iajb,aibj->", extract_mat(L, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
    
    end 

    Omega_AI = zeros(v[end] - 1)

    return Omega_AI
end

# Correct
function Omega_Ai_ai(F, L, g, h_p, g_p, s, s2, u, o, v)
    # Evaluates Omega_0_bj = <bj|..H..|HF>
    Omega_AI_ai = zeros(v[end] - 1, v[end] - o[end], o[end])

    for B in 2: v[end]

        Omega_AI_ai[B-1,:,:] +=  -0.50000000  * extract_mat(g_p, "AIvo", o, v, B);
        Omega_AI_ai[B-1,:,:] +=  -0.50000000  * extract_mat(h_p, "II", o, v) * extract_mat(s, "AIvo", o, v, B);
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +0.50000000  * fixed_einsum("ab,bi->ai", extract_mat(F, "vv", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("ji,aj->ai", extract_mat(F, "oo", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +1.00000000  * extract_mat(s, "AIvo", o, v, B) * fixed_einsum("jj->", extract_mat(g_p, "IIoo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +0.50000000  * fixed_einsum("B,Bai->ai", extract_mat(h_p, "AV", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +1.00000000  * fixed_einsum("jb,aibj->ai", extract_mat(F, "ov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("jb,ajbi->ai", extract_mat(F, "ov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +0.50000000  * fixed_einsum("aijb,bj->ai", extract_mat(L, "voov", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("Bab,Bbi->ai", extract_mat(g_p, "AVvv", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +0.50000000  * fixed_einsum("Bji,Baj->ai", extract_mat(g_p, "AVoo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -1.00000000  * fixed_einsum("Bjj,Bai->ai", extract_mat(g_p, "AVoo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("jb,aibj->ai", extract_mat(g_p, "AIov", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("jikb,ajbk->ai", extract_mat(L, "ooov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +1.00000000  * fixed_einsum("abjc,bicj->ai", extract_mat(g, "vvov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("abjc,bjci->ai", extract_mat(g, "vvov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -1.00000000  * fixed_einsum("Bjb,Baibj->ai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +0.50000000  * fixed_einsum("Bjb,Bajbi->ai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +1.00000000  * extract_mat(s, "AIvo", o, v, B) * fixed_einsum("Bjb,Bbj->", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  +0.50000000  * fixed_einsum("jbkc,bj,aick->ai", extract_mat(L, "ovov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("jbkc,aj,bick->ai", extract_mat(g, "ovov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_ai[B-1,:,:] = Omega_AI_ai[B-1,:,:].+  -0.50000000  * fixed_einsum("jbkc,bi,ajck->ai", extract_mat(g, "ovov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
   
    end

    Omega_AI_ai = zeros(v[end] - 1, v[end] - o[end], o[end])

    return Omega_AI_ai
end

# Correct
function Omega_AI_aibj(F, L, g, h_p, g_p, s, s2, t, u, o, v)
    # Evaluates Omega_Ai_aibj = < AI | aibj|..H..|HF|0>
    Omega_AI_bjai = zeros(v[end]-1, v[end] - o[end], o[end], v[end] - o[end], o[end])

    # Fix the loop over g_p

    for B in 2:v[end]

        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("ai,bj->bjai", extract_mat(g_p, "IIvo", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("bj,ai->bjai", extract_mat(g_p, "IIvo", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * extract_mat(h_p, "II", o, v) * fixed_einsum("aibj->bjai", extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("ac,bjci->bjai", extract_mat(F, "vv", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("bc,aicj->bjai", extract_mat(F, "vv", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("ki,akbj->bjai", extract_mat(F, "oo", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kj,aibk->bjai", extract_mat(F, "oo", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("aibc,cj->bjai", extract_mat(g, "vovv", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("aikj,bk->bjai", extract_mat(g, "vooo", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("acbj,ci->bjai", extract_mat(g, "vvvo", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("bjki,ak->bjai", extract_mat(g, "vooo", o, v, B), extract_mat(s, "AIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bai,Bbj->bjai", extract_mat(g_p, "AVvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bbj,Bai->bjai", extract_mat(g_p, "AVvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +1.00000000  * fixed_einsum("kk,aibj->bjai", extract_mat(g_p, "IIoo", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("ac,bjci->bjai", extract_mat(g_p, "AIvv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("bc,aicj->bjai", extract_mat(g_p, "AIvv", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("ki,akbj->bjai", extract_mat(g_p, "AIoo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kj,aibk->bjai", extract_mat(g_p, "AIoo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("B,Baibj->bjai", extract_mat(h_p, "AV", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("B,ai,Bbj->bjai", extract_mat(h_p, "IV", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("B,bj,Bai->bjai", extract_mat(h_p, "IV", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("aikc,bjck->bjai", extract_mat(L, "voov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("bjkc,aick->bjai", extract_mat(L, "voov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("aikc,bkcj->bjai", extract_mat(g, "voov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("acbd,cidj->bjai", extract_mat(g, "vvvv", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("ackj,bkci->bjai", extract_mat(g, "vvoo", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("bjkc,akci->bjai", extract_mat(g, "voov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("bcki,akcj->bjai", extract_mat(g, "vvoo", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kilj,akbl->bjai", extract_mat(g, "oooo", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bac,Bbjci->bjai", extract_mat(g_p, "AVvv", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bbc,Baicj->bjai", extract_mat(g_p, "AVvv", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bki,Bakbj->bjai", extract_mat(g_p, "AVoo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkj,Baibk->bjai", extract_mat(g_p, "AVoo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -1.00000000  * fixed_einsum("Bkk,Baibj->bjai", extract_mat(g_p, "AVoo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kc,ak,bjci->bjai", extract_mat(F, "ov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kc,bk,aicj->bjai", extract_mat(F, "ov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kc,ci,akbj->bjai", extract_mat(F, "ov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kc,cj,aibk->bjai", extract_mat(F, "ov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bac,bj,Bci->bjai", extract_mat(g_p, "IVvv", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bbc,ai,Bcj->bjai", extract_mat(g_p, "IVvv", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bki,bj,Bak->bjai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bkj,ai,Bbk->bjai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +1.00000000  * fixed_einsum("Bkk,ai,Bbj->bjai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +1.00000000  * fixed_einsum("Bkk,bj,Bai->bjai", extract_mat(g_p, "IVoo", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s, "VIvo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kc,ai,bjck->bjai", extract_mat(g_p, "IIov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kc,bj,aick->bjai", extract_mat(g_p, "IIov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("ackd,dk,bjci->bjai", extract_mat(L, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("bckd,dk,aicj->bjai", extract_mat(L, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kilc,cl,akbj->bjai", extract_mat(L, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kjlc,cl,aibk->bjai", extract_mat(L, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("ackd,bk,cidj->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("ackd,di,bjck->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("ackd,dj,bkci->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("bckd,ak,cjdi->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("bckd,di,akcj->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("bckd,dj,aick->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kilc,al,bjck->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kilc,bl,akcj->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kilc,cj,akbl->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kjlc,al,bkci->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kjlc,bl,aick->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kjlc,ci,albk->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("ackd,ci,bjdk->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("bckd,cj,aidk->bjai", extract_mat(g, "vvov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kilc,ak,bjcl->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kjlc,bk,aicl->bjai", extract_mat(g, "ooov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +1.00000000  * fixed_einsum("Bkc,ai,Bbjck->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bkc,ai,Bbkcj->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +1.00000000  * fixed_einsum("Bkc,bj,Baick->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bkc,bj,Bakci->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "AIvo", o, v, B), extract_mat(s2, "VIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +1.00000000  * fixed_einsum("Bkc,Bck,aibj->bjai", extract_mat(g_p, "IVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(s2, "AIvovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Bak,bjci->bjai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Bbk,aicj->bjai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Bci,akbj->bjai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Bcj,aibk->bjai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bkc,Bai,bjck->bjai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("Bkc,Bbj,aick->bjai", extract_mat(g_p, "AVov", o, v, B), extract_mat(s, "VIvo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,akdl,bjci->bjai", extract_mat(L, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,bkdl,aicj->bjai", extract_mat(L, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,cidl,akbj->bjai", extract_mat(L, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,cjdl,aibk->bjai", extract_mat(L, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,aick,bjdl->bjai", extract_mat(L, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,bjck,aidl->bjai", extract_mat(L, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,akbl,cidj->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,akdi,bjcl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,akdj,blci->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,bkdi,alcj->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,bkdj,aicl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  +0.50000000  * fixed_einsum("kcld,cidj,akbl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(t, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,aibk,cjdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,aicj,bkdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,akbj,cidl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,akci,bjdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,bjci,akdl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
        Omega_AI_bjai[B-1,:,:,:,:] = Omega_AI_bjai[B-1,:,:,:,:] .+  -0.50000000  * fixed_einsum("kcld,bkcj,aidl->bjai", extract_mat(g, "ovov", o, v, B), extract_mat(s2, "AIvovo", o, v, B), extract_mat(u, "vovo", o, v, B), optimize="optimal");
    
    end
    Omega_AI_bjai = zeros(v[end]-1, v[end] - o[end], o[end], v[end] - o[end], o[end])
    return Omega_AI_bjai
end



#function build_omega(F, g, L, s, t, gamma, s2, t2, o, v)
#   n = o[end] * (v[end] - o[end])
#    omega = zeros(2*n^2+2 * n + 1)
#    omega[1:n] = reshape(Omega_aiai_bj(F, g, L, s, t, gamma, s2, t2, o, v), (n, 1))
#    # omega[v[1], o[end]] = 0.0                   # This to avoid s_ai being changed with E_1
#    omega[n+1] = eta_aiai(F, g, L, s, t, gamma, s2, t2, o, v)
#    omega[n+2:2*n+1] = reshape(Omega_0_bj(F, g, L, s, t, gamma, s2, t2, o, v), (n, 1))
#    omega[2*n+2:n^2+2*n+1] = reshape(Omega_0_bjck(F, g, L, s, t, gamma, s2, t2, o, v), (n^2, 1))
#    omega[n^2+2*n+2:end] = reshape(Omega_aiai_bjck(F, g, L, s, t, gamma, s2, t2, o, v), (n^2, 1))
#
#    return omega
#end

function build_omega(F, g, g_p, h_p, L, s, t, s2, u, o, v)
    n = o[end] * (v[end] - o[end])
    omega = zeros(n^2 + n + v[end]-1 + n*(v[end]-1) + n^2*(v[end]-1))
    omega[1 : (v[end]-1)] = Omega_AI(F, L, g_p, s, s2, o, v)
    omega[v[end]: n + v[end] - 1] = reshape(Omega_ai(F, g, h_p, g_p, s, s2, u, o, v), (n, 1))
    omega[n + v[end]: v[end] - 1 + n + n*(v[end]-1)] = reshape(Omega_Ai_ai(F, L, g, h_p, g_p, s, s2, u,  o, v), (n*(v[end]-1),1))     #
    omega[v[end] + n + n*(v[end]-1):  v[end] - 1 + n + n*(v[end]-1) + n^2] = reshape(Omega_aibj(F, g, g_p, h_p, s, s2, t, u , o, v), (n^2 ,1)) #
    omega[v[end] + n + n*(v[end]-1) + n^2: end] = reshape(Omega_AI_aibj(F, L, g, h_p, g_p, s, s2, t, u, o, v), (n^2*(v[end]-1), 1)) 

    println(omega[1 : (v[end]-1)], " AI \n")
    println(omega[v[end]: n + v[end] - 1], " ai \n")
    println(omega[n + v[end]: v[end] - 1 + n + n*(v[end]-1)], "AI,ai")
    println(omega[v[end] + n + n*(v[end]-1):  v[end] - 1 + n + n*(v[end]-1) + n^2], "aibj")
    println(omega[v[end] + n + n*(v[end]-1) + n^2: end], "Ai_aibj")

    println(norm(omega[1 : (v[end]-1)] ), " AI \n")
    println(norm(omega[v[end]: n + v[end] - 1] ), " ai \n")
    println(norm(omega[n + v[end]: v[end] - 1 + n + n*(v[end]-1)]), " Ai_ai \n ")
    println(norm(omega[v[end] + n + n*(v[end]-1):  v[end] - 1 + n + n*(v[end]-1) + n^2]), " aibj \n" )
    println(norm(omega[v[end] + n + n*(v[end]-1) + n^2: end]), " AI_aibj \n")

    return omega
 end

 function build_omega_ref(F, g, L, t, t2, o, v)
    n = o[end] * (v[end] - o[end])
    omega = zeros(n^2+n)
    omega[1:n] = reshape(Omega_0_bj(F, g, L, t, t2, o, v), (n, 1))
    omega[n+1:end] = reshape(Omega_0_bjck(F, g, L, t, t2, o, v), (n^2, 1))

    println(omega[1:n], "ai") 
    println(omega[n+1:end], "aibj")  
 
    return omega
end

#ENV["PYTHON"]="/opt/homebrew/bin/python3.11"
#import Pkg; Pkg.build("PyCall")

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

## Set up the molecule 
# mol = pyscf.M(atom="H 0 0 0; Li 1.6 0 0; He 1.0 2.0 0.0; He 1.2 -1.5 1.0", basis="ccPVDZ")
# mol = pyscf.M(atom= "H 0, 0, 0; Li 1.606 0.0 0.0", basis="sto3g")

#mol = pyscf.M(atom= "H,          0.86681,        0.60144,        5.00000; 
#H         -0.86681        0.60144        5.00000; 
#O          0.00000       -0.07579        5.00000; 
#He         0.10000       -0.02000        7.53000", basis="sto3g")

#mol = pyscf.M(atom= "O 0, 0, 0; H 0, 1, 0; H 0.0 0.0 1", basis="sto3g")

#
# Specify diffierent basis for different ghost atoms
#


mol = pyscf.M(atom="H 0.0, 0.0, 0.0; Li 1.606, 0.0, 0.0", basis= "sto3g")

mf = scf.RHF(mol)
mf.conv_tol = 1e-10
mf.max_cycle = 1000
coords = [(0.35, 0.6, 0.8)]
charges = [0.0]
hf = qmmm.mm_charge(mf, coords, charges).run()

# FCI solution
# cisolver = pyscf.fci.FCI(hf)
# cisolver.nroots = 5
# (e0, e1, e2, e3, e4), _ = cisolver.kernel()
# println(e0, e1, e2, e3, e4)

# CCSD solution
#mycc = cc.CCSD(hf).run()
#@show (e0, e1), (fcivec0, fcivec1) = mycc.eomee_ccsd_singlet(nroots=3)
#println("t1: ", mycc.t1)
#println("t2: ", mycc.t2)


# hf = scf.RHF(mf)
# hf.run()

## SCF 

println(mol.energy_nuc())
println(hf.energy_nuc())

C = hf.mo_coeff
pyscf.tools.dump_mat.dump_mo(mol, C)


#hf.analyze()

# mol = ...
# solve HF --> C

# mol.intor("int1e_nuc") = - mol.intor("int1e_nuc") 
# (same for g)
# solve HF --> Cp


h_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
# h_ao_p = mol.intor("int1e_kin") - mol.intor("int1e_nuc")

h = C' * h_ao * C
# h_p = C_p' * h_ao_p * C_p

g_ao = mol.intor("int2e")

##
## g_ao_p = mol.intor("int2e")
## g_ao_p = C Cp g_ao_p C Cp
g = pyscf.ao2mo.incore.full(g_ao, C)
## display(g)
## g_test = np.einsum("px, qy, xyzw, rz, sw ->pqrs", C', C', g_ao, C', C', optimize="optimal")
## display(g_test)
## println(sum(g-g_test))
##

no = mol.nelectron ÷ 2
nv = mol.nao - no

print(no, "no \n ")
print(nv, "nv \n ")


o = 1:no
v = no+1:mol.nao
i = o[end]
a = v[1]

L = 2 * g - permutedims(g, [1, 4, 3, 2]) 

F = C' * hf.get_fock() * C
h_ao_p = mol.intor("int1e_kin") - mol.intor("int1e_nuc")


function Overlap()
    S = hf.get_ovlp()
    C_s = eigvecs(S)
    eigen_S = eigvals(S)
    S_half_inv = C_s*diagm([1/sqrt(el) for el in eigen_S]) * C_s' 

    return S_half_inv
end 

function Density(F_HF, F_p, h_ao, h_p_ao, g_ao, F_prev_p, F_prev, o,)
        
        ##
        ##
        ##
    
    S_half_inv = Overlap()
        
    F_HF = S_half_inv*F_HF*S_half_inv
    F_p = S_half_inv*F_p*S_half_inv

    #display(F_HF)

    for _ in 1:1000

        ##
        ## Evaluation
        ##

        ##
        ## F_pq = h_pq + 2 \sum_i g_iipq - g_ipqi - g_IIpq
        ## F_PQ = h_PQ - \sum_i g_PQii
        ##

        C = real.(eigvecs(F_HF))
        C_p = real.(eigvecs(F_p))

        #display(C)

        D = 2*np.einsum("pi, qi -> pq", C[:,o], C[:,o], optimize="optimal")
        D_p = 2*np.einsum("p, q -> pq", C_p[:,1], C_p[:,1], optimize="optimal")

        D = S_half_inv*D*S_half_inv'
        D_p = S_half_inv*D_p*S_half_inv'

        F_HF = h_ao + np.einsum("rs, pqrs -> pq", D, g_ao) - 1/2*np.einsum("rs, psrq -> pq", D, g_ao) #- 1/2*np.einsum("pq, pqrs -> rs", D_p, g_ao) 
        F_p =  h_p_ao - np.einsum("rs, pqrs -> pq", D, g_ao) 

        F_HF = S_half_inv*F_HF*S_half_inv
        F_p = S_half_inv*F_p*S_half_inv

        ##
        ## Converger Checker
        ##

        F_diff_p = sum(abs.(real.(eigvals(F_prev_p))) .- abs.(real.(eigvals(F_p))))
        F_diff = sum(abs.(real.(eigvals(F_prev))) .- abs.(real.(eigvals(F_HF))))

        F_prev = F_HF
        F_prev_p = F_p

        #print(F_diff, F_diff_p,  " F_diff \n")
        
        if abs(F_diff) < 10^(-10) && abs(F_diff_p) < 10^(-10)

            ##
            ## Return value
            ##

            return real.(eigvals(F_p)), S_half_inv*C_p, real.(eigvals(F_HF)), S_half_inv*C
        end
            
    end

end


function construcu_hf_p()

    e_p, C_p, e, C  = Density(h_ao, h_ao_p, h_ao, h_ao_p, g_ao, 1, 1, o)

    F = diagm(e)
    F_p = diagm(e_p)

    display(F)
    display(F_p)

    h = C'*h_ao*C
    E_HF = HF_Energy_(F, h, o, v) 

    #D_in = zeros(v[end])
    #D_in[o] .= 2.0
    #D = diagm(D_in) 

    h_p = C_p' * h_ao_p * C_p
    g_p = np.einsum("xyzw, px, qy, rz, sw ->pqrs", g_ao, C_p', C_p', C', C', optimize="optimal")

    E = HF_Energy_(F, h, v, o)
    E_P = HF_Energy_p(h_p, g_p, o, v) 

    println(E_HF, " E_HF")
    println(E_P, " P ")
    println(E_P + E_HF)
    println(E_P + E_HF + hf.mol.energy_nuc() )
    println(E_HF + hf.mol.energy_nuc(), "HF-energy")
    println(HF_p(F, h, h_p, g_p, o, v) +  hf.mol.energy_nuc())

    return h_p, g_p, F, F_p
end 

h_p, g_p, F, F_p = construcu_hf_p()


## Fix down here 
# h = F .- np.einsum("pqii->pq", L[:, :, o, o], optimize="optimal")

# v = 3:10
# V = 2:10
# V = 2:mol.nao

# @show F = Diagonal(hf.mo_energy)
# @show isF = fixed_einsum("pqii->pq", L[:, :, o, o], optimize="optimal")
#Initial guess for s1
# @show s[v, o] = -L[i, a, v, o] / g[a, i, a, i]

##
## This is not P-CCSD ´
##

## Initial values, chages this. This is the guess


n = o[end] * (v[end] - o[end])   # no * nv
amplitudes = zeros(n+n^2) # dim of the amplitudes, vector

s = zeros(mol.nao, mol.nao, mol.nao, mol.nao)
t = zeros(mol.nao, mol.nao)
s2 = zeros(mol.nao, mol.nao, mol.nao, mol.nao, mol.nao, mol.nao)
t2 = zeros(mol.nao, mol.nao, mol.nao, mol.nao)
gamma = zeros(v[end]-1)

old = 0.0
E2 = 0.0

h_p = zeros(mol.nao, mol.nao)
g_p = zeros(mol.nao, mol.nao, mol.nao, mol.nao)
F_p = zeros(mol.nao, mol.nao)

# This work 

restart = false
save_restart = false
if restart
    temp_t = zeros(nv * no)
    open("saved_t_693", "r") do inp
        for (pos, line) in enumerate(eachline(inp))
            temp_t[pos] = parse(Float64, line)
        end
        global t = reshape(temp_t, (nv, no))
    end
    open("saved_s_693", "r") do inp
        temp_s = zeros(nv * no)
        for (pos, line) in enumerate(eachline(inp))
            temp_s[pos] = parse(Float64, line)
        end
        global s[v, o] = reshape(temp_s, (nv, no))
    end
    open("saved_other_693", "r") do inp
        global gamma = parse(Float64, readline(inp))
        global t_aiai = parse(Float64, readline(inp))
        global s_aiai = parse(Float64, readline(inp))
    end
end

let
    # Conditioning update if needed, 
    pre_t_aiai = (F[a, a] - F[i, i])

    #something wierd with the first element

    pre_gamma = zeros((v[end])-1)
    for A in 2:v[end]
        pre_gamma[A-1] = F_p[1,1] - F_p[A, A]
    end 

    println(pre_gamma[1:(v[end])-1], " pre_gamma")

    pre_t = ones(mol.nao, mol.nao)
    for a in v
        for i in o
            pre_t[a, i] = (F[a, a] - F[i, i])
        end
    end

    println(pre_t[v,o]," pre_t1")

    pre_s = zeros(v[end]-1, mol.nao, mol.nao)

    for A in 2:v[end]
        for a in v 
            for i in o
               pre_s[A-1,a,i] = -(F_p[1,1] - F_p[A, A] + F[i,i] - F[a,a])
            end 
        end 
    end

    println(pre_s[v[end]-1,v,o], " pre_s")

    #pre_smat = ones((v[end]-1), mol.nao, mol.nao)
    #pre_smat[v, o] .= pre_gamma / 2
    #pre_smat[v, o] = pre_smat[v, o] .+ 2*(2*L[a,a,i,i]-g[a,a,a,a]-g[i,i,i,i])
    #pre_smat[v, o] += 4 * pre_t[v, o]
    #println("Before: ", pre_t[v,o])


    #for a in v
    #    for i in o
    #        pre_smat[a, i] += L[a,i,i,a]
    #        pre_t[a,i] += 0.5*L[i,a,a,i]
    #    end
    #end
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

    println(pre_t2[v,o,v,o], " pre_t2")


#    pre_s2 = ones(mol.nao, mol.nao, mol.nao, mol.nao, mol.nao)
#    pre_s2[v,o,v,o] .= 8 * (F[i,i] - F[a,a] + L[a,a,i,i])
#    pre_s2[v,o,v,o] = pre_s2[v[end]-1,v,o,v,o] .- 4* (g[a,a,a,a]+g[i,i,i,i])
#    pre_s2[v,o,v,o] = pre_s2[v[end]-1,v,o,v,o] + 4*pre_t2[v,o,v,o]

 #   for b in v
 #       for j in o
 #         for c in v
 #               for k in o
 #                   if abs(pre_t2[b,j,c,k]) < 0.1
 #                       pre_t2[b,j,c,k] = sign(pre_t2[b,j,c,k])*0.1
 #                   end
 #                   if abs(pre_s2[b,j,c,k]) < 0.1
 #                       pre_s2[b,j,c,k] = sign(pre_s2[b,j,c,k])*0.1
 #                   end
 #               end
 #           end
 #       end
 #   end

    pre_s2 = zeros(v[end]-1, mol.nao, mol.nao, mol.nao, mol.nao)

    for A in 2:v[end]
        for a in v 
            for i in o
                for b in v
                    for k in o 
                        pre_s2[A-1,a,i,b,k] = -(F_p[1,1] - F_p[A, A] + F[i,i] - F[a,a] - F[b,b] + F[k,k])
                    end
                end
            end 
        end 
    end

    println(pre_s2[1:v[end]-1,v, o, v, o], " pre_s2")    

    # pre_t[a, i] = -pre_t[a, i]   # Again, no explanation for this

    for k in 1:100
        global E2


# s = nxn
# s[3:10,1:2] += reshape(Omega(for the s), nv, no)

        #global s[1, 2:v[end], v, o] += reshape(amplitudes[n + v[end]: v[end] - 1 + n + n*(v[end]-1)], (v[end]-1, nv, no)) ./ pre_s[1:v[end]-1, v, o]
        # global s[a+1:end, :] .= 0.0
        # global s[:, 1:i-1] .= 0.0
        # global s[a+1:end, 1:i-1] .= 0.0
        # global s .= 0.0

        #omega = zeros(n^2 + n + v[end]-1 + n*(v[end]-1) + n^2*(v[end]-1))
        #omega[1 : (v[end]-1)] = Omega_AI(F, L, g_p, s, s2, o, v)
        #omega[v[end]: n + v[end] - 1] = reshape(Omega_ai(F, g, h_p, g_p, s, s2, u, o, v), (n, 1))
        #omega[n + v[end]: v[end] - 1 + n + n*(v[end]-1)] = reshape(Omega_Ai_ai(F, L, g, h_p, g_p, s, s2, u,  o, v), (n*(v[end]-1),1))     #
        #omega[v[end] + n + n*(v[end]-1):  v[end] - 1 + n + n*(v[end]-1) + n^2] = reshape(Omega_aibj(F, g, g_p, h_p, s, s2, t, u , o, v), (n^2 ,1)) #
        #omega[v[end] + n + n*(v[end]-1) + n^2: end] = reshape(Omega_AI_aibj(F, L, g, h_p, g_p, s, s2, t, u, o, v), (n^2*(v[end]-1), 1)) 

        #omega[1:n] = reshape(Omega_0_bj(F, g, L, 2t, t2, o, v), (n, 1))
        #omega[n+1:end] = reshape(Omega_0_bjck(F, g, L, 2t, t2, o, v), (n^2, 1))

#exit()
        global t[v, o] += reshape(amplitudes[1 : n], (nv, no)) ./ pre_t[v, o]
        # global t[a+1:end, :] .= 0.0
        # global t[:, 1:i-1] .= 0.0
        # global t .= 0.0
        # global t[a,i] = 0.0

        #global gamma += amplitudes[1:v[end]-1] ./ pre_gamma[1:v[end]-1]
        
        # global gamma = 0.0
        # global t_aiai += amplitudes[n+2] / pre_t_aiai
        # t[a, i] = 0.002651938552278201
        # t_aiai = -0.21993579570956315
        # t[a, i] = 0.0
        # t_aiai = 0.0

        global t2[v, o, v, o] += reshape(amplitudes[n+1: end], (nv, no, nv, no)) ./ pre_t2[v, o, v, o]
        #global s2[1, 2:v[end], v, o, v, o] += reshape(amplitudes[v[end] + n + n*(v[end]-1) + n^2: end], (v[end]-1, nv, no, nv, no)) ./ pre_s2[1:v[end]-1,v, o, v, o]

        println(size(t2), "t2 size", n^2)
        println(size(t), "t1 size", n)
        
        s2[1, 2:v[end], v, o, v, o] = zeros(v[end]-1, v[end] - o[end], o[end], v[end] - o[end], o[end])
        s[1, 2:v[end], v, o] = zeros(v[end]-1, v[end] - o[end], o[end])
        gamma =   zeros(v[end] - 1)
        #global s2[a,i,a,i] = 0.0 
        u = zeros(v[end],v[end],v[end],v[end])
        global u[v, o, v, o] = 2*t2[v, o, v, o] - permutedims(t2[v, o, v, o], [1, 4, 3, 2])  

        println("s at step ", k, " : ")
        display(s[1, 2:v[end], v, o])
        println("gamma at step ", k, " : ", gamma)
        println("t at step ", k, " : ")
        display(t[v, o])
        println("t2 at step ", k, " : ")
        display(t2[v, o, v, o])
        println("s2 at step ", k, " : ")
        display(s2[1, 2:v[end], v, o, v, o])
        println("Module of t ", sqrt(sum([abs(e)^2 for e in t])))

#println("Precond s: ", pre_smat[v,o] )
#  hAO   h   
#  gao   g
#  L
#  Fpq = hpq + \sum_i Lpqii


        # T1 - transform integrals
         global h_t = T1_transform_1e(h, t[v, o], o, v)
         global g_t = T1_transform_2e(g, t[v, o], o, v)
         global L_t = 2 * g_t - permutedims(g_t, [1, 4, 3, 2])
         global F_t = h_t .+ fixed_einsum("pqii->pq", L_t[:, :, o, o], optimize="optimal")

         #global h_pt = Gamma_transform_1e(h_p, gamma, o, v)
         #global g_pt = Gamma_transform_2e(g_p, gamma, o, v)

         #global g_pt = T1_transform_g_p_2e(g_pt, t[v,o], o, v)
         
         #global L_t = 2 * g_t - permutedims(g_t, [1, 4, 3, 2])
         #global F_t = h_t .+ fixed_einsum("pqii->pq", L_t[:, :, o, o], optimize="optimal")


        if k % 1 == 0
            println(" --------------Energy at step ", k, " : ", hf.energy_nuc() + CC_ref_energy(F_t, g_t, L_t, t, t2, o, v))
            println(" --------------E_2 at step ", k, " : ", hf.energy_nuc() + E2)
            println("E1 - E0 ", E2 - energy(F, g, L, s, t, gamma, s2, t2, o, v))
            diff = old - hf.energy_nuc() - energy(F, g, L, s, t, gamma, s2, t2, o, v)
            println(" --------------Difference at step ", k, " : ", diff)
            global old = hf.energy_nuc() + energy(F, g, L, s, t, gamma, s2, t2, o, v)
            println(" --------------Module of omegas ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes) if l != n^2+2*n+2+(v[end]-o[end])^2*o[end]*(o[end]-1)+(v[end]-o[end])*(o[end]-1)]))) #if l != 1 + nv * (no - 1)
        end

        #@show global amplitudes = -build_omega(F, g, L, s, t, gamma, s2, t2, o, v) / 2 build_omega(F, g, g_p, h_p, L, s, t, s2, o, v)

        @show global amplitudes = -build_omega_ref(F_t, g_t, L_t, t, t2, o, v)
        println(size(amplitudes), "size")

        #exit()

        #E2 = -2*amplitudes[n^2+2n+3+(v[end]-o[end])^2*o[end]*(o[end]-1)]    #Needs normalization because proj is not normalized, only orthogonalized
        #println("Eta_aiai  ", amplitudes[n+1])
        #println("Omega_bj_aiai ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes[1:n])]))) #if l == 1 + nv * (no - 1)
        #println("Omega_bj_0 ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes[n+2:2*n+1])]))) #if l == 1 + nv * (no - 1)
        #println("Omega_bjck_0 ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes[2*n+2:n^2+2*n+1])])))
        #println("Omega_bjck_aiai ", sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes[n^2+2*n+2:end]) if l != (v[end]-o[end])^2*o[end]*(o[end]-1)+(v[end]-o[end])*(o[end]-1)+1])))

        if abs(old > 500) || isnan(old)
            throw("Energy diverges")
            println("Energy diverges")
            break
            # elseif diff < 0
            #     throw("Energy start to increase")
        elseif sqrt(sum([abs(e)^2 for (l, e) in enumerate(amplitudes) if l != n^2+2*n+2+(v[end]-o[end])^2*o[end]*(o[end]-1)+(v[end]-o[end])*(o[end]-1)])) < 1e-12
            println("Converged!")
            break
        end
    end

    # \sqrt(sum_i x_i^2)

    if save_restart
        open("saved_t_694", "w") do inp
            for el in reshape(t, (nv * no, 1))
                println(inp, el)
            end
        end
        open("saved_s_694", "w") do inp
            for el in reshape(s[v, o], (nv * no, 1))
                println(inp, el)
            end
        end
        open("saved_other_694", "w") do inp
            println(inp, gamma)
            println(inp, t_aiai)
        end
    end

    println("Nuclear electrostatic: ", hf.energy_nuc())

    B = 2

    hf_energy = 2 * fixed_einsum("jj->", F[o, o], optimize="optimal")
    hf_energy -= fixed_einsum("iijj->", extract_mat(L, "oooo", o, v, B), optimize="optimal")
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
    println("gamma + : ", s_plus)
    println("gamma - : ", s_minus)

    x_plus = (-(H_22 - H_00) + temp) / (-2 * H_02)
    x_minus = (-(H_22 - H_00) - temp) / (-2 * H_02)
    println("x + : ", x_plus)
    println("x - : ", x_minus)

    # println(build_omega(F_t, g_t, L_t, t, s_plus, t_aiai, o, v)[n+1])
    # println(build_omega(F_t, g_t, L_t, t, s_minus, t_aiai, o, v)[n+1])

    # E = E_nuc + <HF|...|HF>
    # println("Energy: ", hf.energy_nuc() + energy(F, g, L, s, t, gamma, t_aiai, o, v))

    # # eta_ai = <HF|...|ai>
    # println("Eta_ai: ", eta_ai(F, g, L, s, gamma, o, v))

    # # Omega_0_bj = <bj|..H..|HF>
    # println("Omega_0_bj: ", Omega_0_bj(F, g, L, s, gamma, o, v))

    # # Omega_ai_bj = <bj|..H..|ai>
    # # Omega_ai_bj[a,i] is E_2 without E_nuc
    # println("Omega_ai_bj: ", Omega_ai_bj(F, g, L, s, gamma, o, v))
    # println("E_2: ", hf.energy_nuc() + Excited_energy(F, g, L, s, t, gamma, t_aiai, o, v))


    # global s[v, o] += reshape(amplitudes[1:n], (nv, no))
    # global s[v[1], o[end]] = 0
    # println("s", s)
    println("s_vo", s[v, o])
    println("s_aj ", s[a, o])
    println("s_vi ", s[v, i])

    # global gamma += -amplitudes[n+1]
    println("gamma ", gamma)
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

#  -7.5223297796196686E-006   4.0763997667575063E-017   7.1680180481994595E-018   2.4776697161891376E-005  -4.7923600307716793E-006  -9.7694289272027226E-018  -1.7429238394301908E-018   1.1386825089939462E-005   1.0617499179460387E-002   4.3824170737388604E-017   4.8755752951630303E-003   7.7104288179383417E-018  -1.3010426069826053E-018   4.8755752951630303E-003  -2.4770834678736480E-003  -1.0378040305248517E-017  -1.8265436978394185E-018   4.3373952784254658E-003  -3.1706084681404399E-004  -2.5740611783475169E-017  -4.5295279615132666E-018   1.2903158056975094E-003   5.7639434447570009E-003  -2.4874618608398153E-017   7.1334678785579236E-003  -8.6736173798840355E-019   3.5579314891171902E-018  -3.7466726059619556E-017   1.1108627927707469E-002  -4.3768588543686935E-018  -8.6736173798840355E-019   7.1334678785579227E-003   6.2507856036841330E-019  -6.5981012336264810E-018  -2.6020852139652106E-018   1.1108627927707469E-002   5.9194855508179901E-004   3.4138003616610268E-018   6.0161910366447301E-019  -7.2779642937813478E-005  -3.3198886857096266E-002  -1.0684404133830268E-016  -1.8778539156148141E-017   6.2273813320342312E-002   1.5675862283140358E-017  -2.1338033893353667E-017  -1.3410995266527337E-006   3.5491977543267161E-005   6.8875780412369636E-008  -3.0382114040258130E-017   1.0971985417071032E-002  -1.0492178441129827E-002  -1.6313468035357910E-018  -7.3071871319506934E-017   2.4881875392252541E-002  -2.3793786324149357E-002   7.9551567917947565E-018   0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        4.1354311965679884E-017  -1.0492178441129827E-002  -1.0971985417071034E-002   2.2199220973730286E-018   9.9470082462289123E-017  -2.3793786324149351E-002  -2.4881875392252544E-002  -1.0832858897010860E-017   0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        8.9547843019425685E-003   3.2783124772333576E-017   5.7650947658158809E-018   9.0427867360835783E-003   1.3373868216497772E-002   5.0252180368604051E-017   8.8289450019720202E-018  -2.7677665505083038E-002   0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000       -1.1029787161917467E-003  -4.2110103726619344E-018  -7.4102803609218644E-019  -1.6761284982117961E-003   1.4863199104501047E-002   4.7299090085723884E-017   8.3139383690873162E-018  -4.8517553175085222E-002   0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        1.0184079419508447E-002   3.8223278841657345E-017   6.7250659798422675E-018  -2.2163182260689792E-003  -3.9626919789444935E-004  -1.5870836469260631E-018  -2.7890675227672070E-019   4.5178957319401882E-003   0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000        0.0000000000000000 