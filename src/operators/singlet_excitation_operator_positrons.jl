export E, e

"""
    SingletExcitationOperator Positrons

The basic EP_pq type operator.
"""
struct SingletExcitationOperator <: Operator
    p::Int
    q::Int
end

function Base.show(io::IO,
    (
        e, constraints, translation
    )::Tuple{SingletExcitationOperator,Constraints,IndexTranslation})
    print(io, "EP_")
    print_mo_index(io, constraints, translation, e.p, e.q)
end

function exchange_indices(e::SingletExcitationOperator, mapping)
    SingletExcitationOperator(
        exchange_index(e.p, mapping),
        exchange_index(e.q, mapping)
    )
end

function get_all_indices(e::SingletExcitationOperator)
    (e.p, e.q)
end

function Base.:(==)(a::SingletExcitationOperator, b::SingletExcitationOperator)
    (a.p, a.q) == (b.p, b.q)
end

function Base.isless(a::SingletExcitationOperator, b::SingletExcitationOperator)
    (a.p, a.q) < (b.p, b.q)
end

"""
    P(p, q)

Constructs an expression containing a single excitation operator.
"""
EP(p, q) = Expression(SingletExcitationOperator(p, q))

"""
    EP(p, q, r, s) = EP(p, q) * EP(r, s) - δ(r, q) * EP(p, s)

Alias for the two electron singlet excitation operator. 
```
"""
ep(p, q, r, s) = P(p, q) * P(r, s) - δ(r, q) * P(p, s)

function convert_to_elementary_operators(o::SingletExcitationOperator)
    Expression(
        [(fermiondag(o.p, spin)*fermion(o.q, spin))[1] for spin in (α, β)]
    )
end

function act_on_ket(op::SingletExcitationOperator)
    p = op.p
    q = op.q
    EP(p, q) * virtual(p) * occupied(q) +
    2 * δ(p, q) * occupied(p, q)
end

function Base.adjoint(op::SingletExcitationOperator)
    SingletExcitationOperator(op.q, op.p)
end