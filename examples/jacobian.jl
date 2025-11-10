using SpinAdaptedSecondQuantization

include("multilevel.jl")

function act_on_bra(ex)
    act_on_ket(ex')'
end

using Printf
function print_code_einsum_withextract_freefixed(t::Term, symbol::String, translation, fixed)
    # Print python np.einsum code for term t added to symbol. With deltas, only works on 0,2 or 4 dimensional symbol
    # fixed = [] or fixed = ["a", "i"]
    scalar_str = @sprintf "%+12.8f" t.scalar
    translation = update_index_translation(t, translation)

    function write_extract(t, a, external, translation)
        # t term, a tensor
        write_str = " extract_mat($(get_symbol(a)), \""
        for b in get_indices(a)
            if t.constraints[b] ∉ [VirtualOrbital, OccupiedOrbital]
                throw("Space not supported")
            elseif b ∈ t.sum_indices || sprint(SASQ.print_mo_index, t.constraints, translation, b)[1] in external
                write_str *= t.constraints[b] == VirtualOrbital ? "v" : "o"
            else
                write_str *= t.constraints[b] == VirtualOrbital ? "a" : "i"
            end

        end
        return write_str * "\", o, v)"
    end

    # Finding deltas. Only δ_ab and/or δ_ac. No δ_bc
    fix_b = false
    fix_c = false
    fix_j = false
    fix_k = false
    if length(t.deltas) > 0
        for d in t.deltas
            delta_ind = sprint(SASQ.print_mo_index, t.constraints, translation, d.indices...)
            if 'b' in delta_ind
                fix_b = true
            end
            if 'c' in delta_ind
                fix_c = true
            end
            if 'j' in delta_ind
                fix_j = true
            end
            if 'k' in delta_ind
                fix_k = true
            end
        end
    end

    external_int = get_external_indices(t)
    external = sprint(SASQ.print_mo_index, t.constraints, translation, external_int...)

    # Remove a and i  from external
    external = join([a for a in external if a ∉ fixed])

    # Determining actual externals, after the deltas
    new_ext = ""
    if length(external) >= 2
        if length(fixed) == 0
            new_ext *= fix_b ? "" : "a"
            new_ext *= fix_j ? "" : "i"
        else
            new_ext *= fix_b ? "" : "b"
            new_ext *= fix_j ? "" : "j"
        end
    end
    if length(external) == 4
        if length(fixed) == 0
            new_ext *= fix_c ? "" : "b"
            new_ext *= fix_k ? "" : "j"
        else
            new_ext *= fix_c ? "" : "c"
            new_ext *= fix_k ? "" : "k"
        end
    end

    temp_color = index_color
    disable_color()

    # Make einsum_str and find not-summed tensors
    einsum_str = "\""
    notsum_str = ""
    not_summed_tensors = []
    print_einsum = false
    for a in t.tensors
        # if (No indices) or (nothing summed  and 0 external)                  
        if length(get_indices(a)) == 0 || (length(t.sum_indices) == 0 && sum([1 for b in get_indices(a) if sprint(SASQ.print_mo_index, t.constraints, translation, b)[1] in new_ext]) == 0)
            push!(not_summed_tensors, a)
        else
            indices = []
            for b in get_indices(a)
                if b ∈ t.sum_indices || sprint(SASQ.print_mo_index, t.constraints, translation, b)[1] in new_ext
                    push!(indices, sprint(SASQ.print_mo_index, t.constraints, translation, b))
                end
            end
            if join(indices) == new_ext
                push!(not_summed_tensors, a)
                new_ext = ""
            elseif join(indices) == ""
                    push!(not_summed_tensors, a)
            else
                einsum_str *= join(indices)
                einsum_str *= ","
                print_einsum = true
            end
        end
    end

    if print_einsum
        einsum_str = einsum_str[begin:end-1] * "->" * new_ext * "\""
    end

    if (temp_color)
        enable_color()
    end

    # Make tensor_str
    tensor_str = ""
    for a in t.tensors 
        if a ∉ not_summed_tensors
            tensor_str *= ","
            tensor_str *= write_extract(t, a, external, translation)
        end
    end

    # Make string for not-summed tensors
    for a in not_summed_tensors
        if length(get_indices(a)) > 0
            notsum_str *= " *"
            notsum_str *= write_extract(t, a, external, translation)
        else
            notsum_str *= " * $(get_symbol(a))"
        end
    end

    # Printing depending on deltas
    if length(external) >= 2
        pre_string = "$(symbol)_$(external)"
        pre_string *= fix_b ? "[a" : "[:"
        pre_string *= fix_j ? ",i" : ",:"
        if length(external) == 4
            pre_string *= fix_c ? ",a" : ",:"
            pre_string *= fix_k ? ",i" : ",:"
        end
        pre_string *= "]"
    else
        pre_string = "$(symbol)"
    end

    if length(tensor_str) > 0
        return "$pre_string = $pre_string .+ $scalar_str $notsum_str * np.einsum($einsum_str$tensor_str, optimize=\"optimal\");"
    else
        return "$pre_string += $scalar_str $notsum_str;"
    end
end

function cc_projection(order)
    # Biorthogonal projection operators
    El(i, a) = E(i, a) * virtual(a) * occupied(i)
    a = 1
    b = 3
    c = 5
    i = 2
    j = 4
    k = 6
    if order == 0
        P = SASQ.Expression(1)
    elseif order == 1
        P = 1 // 2 * El(i, a)
    elseif order == 2
        P = 1 // 6 * (2 * El(i, a) * El(j, b) + El(i, b) * El(j, a))
    elseif order == 3
        P = 1 // 120 * (17 * El(i, a) * El(j, b) * El(k, c)
                        -
                        1 * El(i, a) * El(j, c) * El(k, b)
                        -
                        1 * El(i, b) * El(j, a) * El(k, c)
                        -
                        7 * El(i, b) * El(j, c) * El(k, a)
                        -
                        7 * El(i, c) * El(j, a) * El(k, b)
                        -
                        1 * El(i, c) * El(j, b) * El(k, a))
    else
        throw("order not supported")
    end
    return P
end

function old_project_on_bra(start, operator, order)
    P = SASQ.Expression(1)
    for i in 1:order
        P = P + 1 // i * prod(operator for j = 1:i)
    end
    newbra = start * P |> act_on_bra |> simplify_heavy
    return newbra
end

function project_on_bra(start, operator, order)
    # <start|(1 + operator + 1/2 operator^2 +...) expanded until order
    # s_ai and exceding E_ia are filtered out
    newbra = start
    temp = start
    for i in 1:order
        temp = temp * 1 // i * operator |> act_on_bra
        temp = filter_ai(temp)
        newbra = newbra + temp
    end
    return newbra
end

function project_on_bra_fixed(start, operator, order, fixed_order)
    # Like project_on_bra but only keeps terms of a fixed final order
    newbra = start
    temp = start
    temp = filter_min_order(temp, fixed_order)
    for i in 1:order
        temp = temp * 1 // i * operator |> act_on_bra |> simplify_heavy
        temp = filter_ai(temp)
        temp = filter_min_order(temp, fixed_order)
        newbra = newbra + temp
    end
    terms = [length(t.operators) == fixed_order for t in newbra.terms]
    return SASQ.Expression(newbra[terms])
end

function filter_min_order(exp, min_order)
    terms = [length(t.operators) >= min_order for t in exp.terms]
    if all(iszero(terms))
        return SASQ.Expression(0)
    end
    return SASQ.Expression(exp[terms])
end

function filter_term_ai(term)
    # Check if there are too many i+ or a
    tot_i = 0
    tot_a = 0
    for op in term.operators
        if op.p == 2
            tot_i = tot_i + 1
        end
        if op.q == 1
            tot_a = tot_a + 1
        end
    end
    if ((tot_i <= 2 || 2 in term.sum_indices) && (tot_a <= 2 || 1 in term.sum_indices))
        return true
    else
        return false
    end
end

function filter_s_ai(term)
    # Check if there is s_ai in that term
    if 1 in term.sum_indices || 2 in term.sum_indices
        return true
    end
    for tens in term.tensors
        if (tens.indices == [1, 2] && tens.symbol == "s")
            return false
        end
    end
    return true
end

function filter_s_aiai(term)
    # Check if there is s_ai in that term
    if 1 in term.sum_indices || 2 in term.sum_indices
        return true
    end
    for tens in term.tensors
        if (tens.indices == [1, 2, 1, 2] && tens.symbol == "s2")
            return false
        end
    end
    return true
end


function filter_ai(expression)
    terms = [filter_term_ai(t) && filter_s_aiai(t) for t in expression.terms]
    return SASQ.Expression(expression[terms])
end


function latex_file(name, expression, trans, renaming)
    open(name, "w") do output_file
        i=0
        io = IOBuffer()
        print(io, "& = ")
        for t in expression.terms
            i += 1
            SASQ.print_latex(io, (t, trans), renaming)
            if (i%5 == 0 && position(io) > 80) || ((position(io) > 150) || (i==length(expression.terms)))
                print(io, "\\\\ \n")
                write(output_file, String(take!(io)))
                io = IOBuffer()
                print(io, "& ")
            end
        end
    end
end

trans = translate(OccupiedOrbital => 2:2:10, VirtualOrbital => 1:2:10)

# h = ∑(real_tensor("h", 1, 2) * E(1, 2), 1:2) |> simplify
# g = 1 // 2  ∑(psym_tensor("g", 1:4...)  e(1:4...), 1:4) |> simplify
# H = h+g

#Defining Hamiltonian
Φ = 1 // 2 * ∑(psym_tensor("g", 1, 2, 3, 4) * e(1, 2, 3, 4), 1:4) +∑((-2 * psym_tensor("g", 1, 2, 3, 3) + psym_tensor("g", 1, 3, 3, 2)) * occupied(3) * E(1, 2), 1:3)
F = ∑(real_tensor("F", 1, 2) * E(1, 2), 1:2)
H = F + Φ |> simplify

# Definition of other operators
Eai = E(1, 2) * virtual(1) * occupied(2)
Eia = E(2, 1) * virtual(1) * occupied(2)

t_one = 1//2 * summation(real_tensor("t", 1, 2) *  E(1, 2) * virtual(1) * occupied(2), [1, 2]) 
t_two = 1 // 2 * summation(psym_tensor("t2", 1, 2, 3, 4) * E(1, 2) * E(3, 4) * virtual(1) * occupied(2) * virtual(3) * occupied(4), 1:4)

c_one = summation(real_tensor("c1", 1, 2) * E(1, 2) * virtual(1) * occupied(2), 1:2)
c_two = summation(psym_tensor("c2", 1, 2, 3, 4) * E(1, 2) * E(3, 4) * virtual(1) * occupied(2) * virtual(3) * occupied(4), 1:4)

@time begin
    println("Evaluating <bj|...H...")
    proj = project_on_bra(cc_projection(1), -1 * t_two, 1)
    # println("After -taiai")
    # println((proj, trans))

    proj = project_on_bra(proj, -1 * t_one, 2)
    # println("After -t1")
    # println((proj, trans))

    # grad_proj = act_on_bra(proj  commutator(H, E(3, 4)  virtual(3) * occupied(4))) |> simplify_heavy
    # grad_proj = filter_ai(grad_proj)
    proj = act_on_bra(proj * commutator(H, c_one+c_two)) |> simplify_heavy
    proj = filter_ai(proj)
    # println("After H")
    # println((proj, trans))

    proj = project_on_bra(proj, t_one, 8) #Checked, it is max order 3
    # grad_proj = project_on_bra(grad_proj, t_one, 6) #Checked, it is max order 3
    # println("After T1")
    # println((proj, trans))

    proj = project_on_bra(proj, t_two, 4)
    # grad_proj = project_on_bra(grad_proj, t_two, 1)
    # println("After taiai")
    # println((proj, trans))
end

@time begin
    println("Evaluating <bj|...|HF> ")
    bj_all_hf = act_on_ket(proj, 0) |> simplify_heavy
    bj_all_hf = filter_ai(bj_all_hf)
    println("Total number of uncombined terms on |HF> : ", length(bj_all_hf.terms))
end


println("...replacements...")
@time begin
    bj_all_hf = look_for_tensor_replacements_smart(bj_all_hf, make_exchange_transformer("g", "L"))
    # bj_all_hf = look_for_tensor_replacements_smart(bj_all_hf, make_exchange_transformer("t2", "u"))
    # bj_all_hf = look_for_tensor_replacements_smart(bj_all_hf, make_exchange_transformer("s2", "v"))
end

output_file = open("bj_all_hf.txt", "w")
write(output_file, string((bj_all_hf, trans)))
close(output_file)

renaming = Dict(key => val for (key, val) in [["s0", "s_0"], ["s2","s"], ["t2","t"]])
# latex_file("bj_all_hf_latex.txt", bj_all_hf, trans, renaming)

open("file_omega_ai:.py", "w") do output_file
    for t in bj_all_hf.terms
        println(output_file, print_code_einsum_testing(t, "Jacobian", trans, ['k']))
    end
end


#open("bj_all_hf_code.py", "w") do output_file
#    for t in bj_all_hf.terms
#        println(output_file, SASQ.print_code_einsum_withextract_freefixed(t, "Jacobian", trans, []))
#    end
#end

@time begin
    println("Evaluating <bj|...|ai> ")
    bj_all_ai = act_on_ket(proj * Eai, 0) |> simplify_heavy
    bj_all_ai = filter_ai(bj_all_ai)
    println("Total number of uncombined terms on |ai> : ", length(bj_all_ai.terms))
end

println("...replacements...")
@time begin
    bj_all_ai = look_for_tensor_replacements_smart(bj_all_ai, make_exchange_transformer("g", "L"))
end

output_file = open("bj_all_ai.txt", "w")
write(output_file, string((bj_all_ai, trans)))
close(output_file)

# latex_file("bj_all_ai_latex.txt", bj_all_ai, trans, renaming)

#open("bj_all_ai_code.py", "w") do output_file
#    for t in bj_all_ai.terms
#        println(output_file, SASQ.print_code_einsum_withextract_general(t, "Omega_ai", trans))
#    end
#end

@time begin
    println("Evaluating <bj|...|aiai> ")
    bj_all_aiai = act_on_ket(proj * Eai*Eai, 0) |> simplify_heavy
    bj_all_aiai = filter_ai(bj_all_aiai)

    println("Total number of uncombined terms on |ai> : ", length(bj_all_ai.terms))
end

# grad_bj_all_aiai = act_on_ket(grad_proj * Eai*Eai, 0) |> simplify_heavy
# grad_bj_all_aiai = filter_ai(grad_bj_all_aiai)

println("...replacements...")
@time begin
    bj_all_aiai = look_for_tensor_replacements_smart(bj_all_aiai, make_exchange_transformer("g", "L"))
end
# grad_bj_all_aiai = look_for_tensor_replacements_smart(grad_bj_all_aiai, make_exchange_transformer("g", "L"))


output_file = open("bj_all_aiai.txt", "w")
write(output_file, string((bj_all_aiai, trans)))
close(output_file)

# output_file = open("grad_bj_all_aiai.txt", "w")
# write(output_file, string((grad_bj_all_aiai, trans)))
# close(output_file)

# latex_file("bj_all_aiai_latex.txt", bj_all_aiai, trans, renaming)
# latex_file("grad_bj_all_aiai_latex.txt", grad_bj_all_aiai, trans, renaming)


#open("bj_all_aiai_code.py", "w") do output_file
#    for t in bj_all_aiai.terms
#        println(output_file, SASQ.print_code_einsum_withextract_general(t, "Omega_aiai", trans))
#    end
#end

# open("grad_bj_all_aiai_code.py", "w") do output_file
#     for t in grad_bj_all_aiai.terms
#         println(output_file, SASQ.print_code_einsum_withextract(t, "Grad_omega_aiai", trans))
#     end
# end


# REQUIRED ORDERS ###############

# hf   order 0
# s_two order 0
# s_one order 0
# s_zero order 2
# H
# s_zero order 2
# s_one order 3
# s_two order 3 (for |ai>, 2 for |aiai>)

# bj   order 1
# s_two order 0
# s_one order 1
# s_zero order 2
# H
# s_zero order 2
# s_one order 3
# s_two order 3 (for |ai>, 2 for |aiai>)

# bjck   order 2
# s_two order 1
# s_one order 3
# s_zero order 2
# H
# s_zero order 2
# s_one order 3
# s_two order 4 (for |ai>, 3 for |aiai>)