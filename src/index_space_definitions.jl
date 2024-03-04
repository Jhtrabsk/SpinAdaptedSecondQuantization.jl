export GeneralOrbital, OccupiedOrbital, VirtualOrbital

const GeneralOrbital = new_space(:GeneralOrbital, "g", "pqrstuv")
const VirtualOrbital = new_space(:VirtualOrbital, "v", "abcdefg")
const OccupiedOrbital = new_space(:OccupiedOrbital, "o", "ijklmno")

add_subspace_relation(GeneralIndex, GeneralOrbital)

add_space_sum(OccupiedOrbital, VirtualOrbital, GeneralOrbital)

export occupied, virtual, electron, occupiedP, virtualP, positron

function occupied(indices...)
    constrain(p => OccupiedOrbital for p in indices)
end

function virtual(indices...)
    constrain(p => VirtualOrbital for p in indices)
end

function electron(indices...)
    constrain(p => GeneralOrbital for p in indices)
end

##
## Positron
##

const GeneralOrbitalP = new_space(:GeneralOrbital, "gp", "PQRSTUV")
const VirtualOrbitalP = new_space(:VirtualOrbital, "vp", "ABCDEFG")
const OccupiedOrbitalP = new_space(:OccupiedOrbital, "op", "IJKLMNO")

add_space_sum(OccupiedOrbitalP, VirtualOrbitalP, GeneralOrbitalP)

function virtualP(indices...)
    constrain(p => VirtualOrbitalP for p in indices)
end

function occupiedP(indices...)
    constrain(p => OccupiedOrbitalP for p in indices)
end

function positron(indices...)
    constrain(p => GeneralOrbitalP for p in indices)
end 