export E_p, e_p

"""
    SingletExcitationOperator

The basic E_pq type operator.
"""
struct SingletExcitationOperatorP <: Operator
    p::Int
    q::Int
end

function Base.show(io::IO,
    (
        e_p, constraints, translation
    )::Tuple{SingletExcitationOperatorP,Constraints,IndexTranslation})
    print(io, "E_")
    print_mo_index(io, constraints, translation, e_p.p, e_p.q)
end

function exchange_indices(e_p::SingletExcitationOperatorP, mapping)
    SingletExcitationOperatorP(
        exchange_index(e_p.p, mapping),
        exchange_index(e_p.q, mapping)
    )
end

function get_all_indices(e::SingletExcitationOperatorP)
    (e_p.p, e_p.q)
end

function Base.:(==)(a::SingletExcitationOperatorP, b::SingletExcitationOperatorP)
    (a.p, a.q) == (b.p, b.q)
end

function Base.isless(a::SingletExcitationOperatorP, b::SingletExcitationOperatorP)
    (a.p, a.q) < (b.p, b.q)
end

"""
    E(p, q)

Constructs an expression containing a single excitation operator.
"""
E_p(p, q) = Expression(SingletExcitationOperatorP(p, q))

"""
    e(p, q, r, s) = E(p, q) * E(r, s) - δ(r, q) * E(p, s)

Alias for the two electron singlet excitation operator.
```
"""
e_p(p, q, r, s) = E_p(p, q) * E_p(r, s) - δ(r, q) * E_p(p, s)

function convert_to_elementary_operators(o::SingletExcitationOperatorP)
    Expression(
        [(fermiondag(o.p, spin)*fermion(o.q, spin))[1] for spin in (α, β)]
    )
end

function act_on_ket(op::SingletExcitationOperatorP)
    p = op.p
    q = op.q
    E_p(p, q) * virtual(p) * occupied(q) +
    2 * δ(p, q) * occupied(p, q)
end

function Base.adjoint(op::SingletExcitationOperatorP)
    SingletExcitationOperatorP(op.q, op.p)
end
