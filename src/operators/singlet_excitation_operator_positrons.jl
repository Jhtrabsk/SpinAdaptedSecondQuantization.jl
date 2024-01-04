export EP, ep

"""
    SingletExcitationOperatorPositrons Positrons

The basic EP_pq type operator.
"""
struct SingletExcitationOperatorPositronsPositrons <: Operator
    p::Int
    q::Int
end

function Base.show(io::IO,
    (
        e, constraints, translation
    )::Tuple{SingletExcitationOperatorPositrons,Constraints,IndexTranslation})
    print(io, "EP_")
    print_mo_index(io, constraints, translation, e.p, e.q)
end

function exchange_indices(e::SingletExcitationOperatorPositrons, mapping)
    SingletExcitationOperatorPositrons(
        exchange_index(e.p, mapping),
        exchange_index(e.q, mapping)
    )
end

function get_all_indices(e::SingletExcitationOperatorPositrons)
    (e.p, e.q)
end

function Base.:(==)(a::SingletExcitationOperatorPositrons, b::SingletExcitationOperatorPositrons)
    (a.p, a.q) == (b.p, b.q)
end

function Base.isless(a::SingletExcitationOperatorPositrons, b::SingletExcitationOperatorPositrons)
    (a.p, a.q) < (b.p, b.q)
end

"""
    EP(p, q)

Constructs an expression containing a single excitation operator.
"""
EP(p, q) = Expression(SingletExcitationOperatorPositrons(p, q))

"""
    ep(p, q, r, s) = EP(p, q) * EP(r, s) - δ(r, q) * EP(p, s)

Alias for the two electron singlet excitation operator. 
```
"""
ep(p, q, r, s) = EP(p, q) * EP(r, s) - δ(r, q) * EP(p, s)

function convert_to_elementary_operators(o::SingletExcitationOperatorPositrons)
    Expression(
        [(fermiondag(o.p, spin)*fermion(o.q, spin))[1] for spin in (α, β)]
    )
end

function act_on_ket(o::SingletExcitationOperatorPositrons)
    EP = o.p
    q = o.q
    EP(p, q) * virtual(p) * occupied(q) +
    2 * δ(p, q) * occupied(p, q)
end

function Base.adjoint(o::SingletExcitationOperatorPositrons)
    SingletExcitationOperatorPositrons(o.q, o.p)
end