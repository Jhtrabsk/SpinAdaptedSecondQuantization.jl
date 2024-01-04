export EP, ep

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
        ep, constraints, translation
    )::Tuple{SingletExcitationOperator,Constraints,IndexTranslation})
    print(io, "EP_")
    print_mo_index(io, constraints, translation, ep.p, ep.q)
end

function exchange_indices(ep::SingletExcitationOperator, mapping)
    SingletExcitationOperator(
        exchange_index(ep.p, mapping),
        exchange_index(ep.q, mapping)
    )
end

function get_all_indices(ep::SingletExcitationOperator)
    (ep.p, ep.q)
end

function Base.:(==)(ap::SingletExcitationOperator, bp::SingletExcitationOperator)
    (ap.p, a.q) == (bp.p, bp.q)
end

function Base.isless(ap::SingletExcitationOperator, bp::SingletExcitationOperator)
    (ap.p, ap.q) < (bp.p, bp.q)
end

"""
    EP(p, q)

Constructs an expression containing a single excitation operator.
"""
EP(p, q) = Expression(SingletExcitationOperator(p, q))

"""
    ep(p, q, r, s) = EP(p, q) * EP(r, s) - δ(r, q) * EP(p, s)

Alias for the two electron singlet excitation operator. 
```
"""
ep(p, q, r, s) = EP(p, q) * EP(r, s) - δ(r, q) * EP(p, s)

function convert_to_elementary_operators(op::SingletExcitationOperator)
    Expression(
        [(fermiondag(op.p, spin)*fermion(op.q, spin))[1] for spin in (α, β)]
    )
end

function act_on_ket(op::SingletExcitationOperator)
    EP = op.p
    q = op.q
    EP(p, q) * virtual(p) * occupied(q) +
    2 * δ(p, q) * occupied(p, q)
end

function Base.adjoint(op::SingletExcitationOperator)
    SingletExcitationOperator(op.q, op.p)
end