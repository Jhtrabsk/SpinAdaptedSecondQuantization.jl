# Implement commutation relations here

# To implement a commutation relation you implement the function
# `reductive_commutator(a, b)` between the two operator types.
# The function should return (∓1, [a, b]±), where the first integer should
# be the sign change of commutation (1 for commutator, -1 for anticommutator)

# When implementing a commutation relation between two different operator types
# only one order is required, as a generic function takes care of the other
# i.e. implement only [a, b]± and not also [b, a]±

function reductive_commutator(
    a::SingletPositronsExcitationOperator,
    b::SingletPositronsExcitationOperator
)
    p = a.p
    q = a.q
    r = b.p
    s = b.q

    (1, δ(q, r) * E(p, s) - δ(p, s) * E(r, q))
end

function reductive_commutator(a::Positron, b::Positron)
    (-1, δ(a.p, b.p) * (a.spin == b.spin) * (a.dag != b.dag))
end

function reductive_commutator(e::SingletPositronsExcitationOperator, a::Positron)
    p = e.p
    q = e.q
    r = a.p

    (1, if a.dag
        δ(q, r) * fermiondag(p, a.spin)
    else
        -δ(p, r) * fermion(q, a.spin)
    end)
end

function reductive_commutator(a::PositronOperator, b::PositronOperator)
    if a.dag && !b.dag
        return (1, Expression(-1))
    elseif !a.dag && b.dag
        return (1, Expression(1))
    else
        return (1, zero(Expression{Int64}))
    end
end

function reductive_commutator(::SingletExcitationOperator, ::PositronOperator)
    return (1, zero(Expression{Int64}))
end

function reductive_commutator(::FermionOperator, ::PositronOperator)
    return (1, zero(Expression{Int64}))
end

function reductive_commutator(::TripletExcitationOperator, ::PositronOperator)
    return (1, zero(Expression{Int64}))
end

function reductive_commutator(t::TripletPositronExcitationOperator,
    e::SingletPositronExcitationOperator)

    p = t.p
    q = t.q

    r = e.p
    s = e.q

    (1, δ(r, q) * τ(p, s) - δ(p, s) * τ(r, q))
end

function reductive_commutator(t1::TripletPositronExcitationOperator,
    t2::TripletPositronExcitationOperator)

    p = t1.p
    q = t1.q

    r = t2.p
    s = t2.q

    (1, δ(r, q) * E(p, s) - δ(p, s) * E(r, q))
end

function reductive_commutator(e::TripletPositronExcitationOperator, a::Positron)
    p = e.p
    q = e.q
    r = a.p

    (1, if a.dag
        δ(q, r) * fermiondag(p, a.spin)
    else
        -δ(p, r) * fermion(q, a.spin)
    end * (a.spin == α ? 1 : -1))
end