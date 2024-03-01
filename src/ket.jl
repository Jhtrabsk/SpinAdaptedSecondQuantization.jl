export act_on_ket, act_on_ket_positrons 

function act_on_ket(ex::Expression{T}, max_ops=Inf) where {T}
    nth = Threads.nthreads()
    terms = [Term{T}[] for _ in 1:nth]
    Threads.@threads for id in 1:nth
        for i in id:nth:length(ex.terms)
            append!(terms[id], act_on_ket(ex[i], max_ops).terms)
        end
    end

    all_terms, rest = Iterators.peel(terms)
    for other_terms in rest
        append!(all_terms, other_terms)
    end

    Expression(all_terms)
end

function act_on_ket_positrons(ex::Expression{T}, max_ops=Inf) where {T}
    nth = Threads.nthreads()
    terms = [Term{T}[] for _ in 1:nth]
    Threads.@threads for id in 1:nth
        for i in id:nth:length(ex.terms)
            append!(terms[id], act_on_ket_positrons(ex[i], max_ops).terms)
        end
    end

    all_terms, rest = Iterators.peel(terms)
    for other_terms in rest
        append!(all_terms, other_terms)
    end

    Expression(all_terms)
end

function act_on_ket(t::Term{A}, max_ops) where {A<:Number}
    if iszero(t.scalar)
        return Expression(zero(A))
    end
    if isempty([op for op in t.operators if typeof(op) == SASQ.SingletExcitationOperator])
        return Expression([t])
    end

    elec_op = copy([op for  op in t.operators if typeof(op) == SASQ.SingletExcitationOperator])
    pos_op = copy([op for  op in t.operators if typeof(op) == SASQ.SingletExcitationOperatorP])   # Try != Singlet
    copyt = SASQ.Term(t.scalar, t.sum_indices, t.deltas, t.tensors, elec_op, t.constraints, t.max_simplified, true)

    right_op = pop!(copyt.operators)
    right_op_act = act_on_ket(right_op)
    print(right_op)
    copyt_act = act_on_ket(copyt,
        max_ops - minimum(length(t1.operators) for t1 in right_op_act.terms))

    terms = Term{A}[]
    for r in right_op_act.terms
        Γ, comm = reductive_commutator_fuse(copyt, r)

        ele_operators = sum([1 for op in r.operators if typeof(op) == SASQ.SingletExcitationOperator])
        if ele_operators <= max_ops
            new_max = max_ops - length(r.operators)
            append!(terms, Γ * fuse(r, ter)
                           for ter in copyt_act.terms
                           if length(ter.operators) <= new_max)
        end

        append!(terms, act_on_ket(comm, max_ops).terms)
    end
    terms = [SASQ.Term(t1.scalar, t1.sum_indices, t1.deltas, t1.tensors, [pos_op; t1.operators], t1.constraints, t1.max_simplified, true) for t1 in terms]
    Expression(terms)
end

function act_on_ket_positrons(t::Term{A}, max_ops) where {A<:Number}
    if iszero(t.scalar)
        return Expression(zero(A))
    end
    if isempty([op for op in t.operators if typeof(op) == SASQ.SingletExcitationOperatorP])
        return Expression([t])
    end

    elec_op = copy([op for  op in t.operators if typeof(op) == SASQ.SingletExcitationOperatorP])
    pos_op = copy([op for  op in t.operators if typeof(op) == SASQ.SingletExcitationOperator])   # Try != Singlet
    copyt = SASQ.Term(t.scalar, t.sum_indices, t.deltas, t.tensors, elec_op, t.constraints, t.max_simplified, true)

    right_op = pop!(copyt.operators)
    print(right_op)
    right_op_act = act_on_ket_positrons(right_op)
    copyt_act = act_on_ket_positrons(copyt,
        max_ops - minimum(length(t1.operators) for t1 in right_op_act.terms))

    terms = Term{A}[]
    for r in right_op_act.terms
        Γ, comm = reductive_commutator_fuse(copyt, r)

        ele_operators = sum([1 for op in r.operators if typeof(op) == SASQ.SingletExcitationOperator])

        if ele_operators <= max_ops
            new_max = max_ops - length(r.operators)
            append!(terms, Γ * fuse(r, ter)
                           for ter in copyt_act.terms
                           if length(ter.operators) <= new_max)
        end

        append!(terms, act_on_ket_positrons(comm, max_ops).terms)
    end
    terms = [SASQ.Term(t1.scalar, t1.sum_indices, t1.deltas, t1.tensors, [pos_op; t1.operators], t1.constraints, t1.max_simplified, true) for t1 in terms]
    Expression(terms)
end

export hf_expectation_value
hf_expectation_value(ex::Expression) = act_on_ket(ex, 0)
