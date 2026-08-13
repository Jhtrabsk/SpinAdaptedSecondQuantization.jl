E_AA = E_AA .+  +1.00000000  * extract_mat(L1, "AI", o, v) * fixed_einsum("C->CB", extract_mat(R1, "VI", o, v), optimize="optimal");
