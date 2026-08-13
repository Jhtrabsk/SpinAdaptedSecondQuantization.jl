E_IA = E_IA .+  +2.00000000  * fixed_einsum("Aaibj,aibj->A", extract_mat(L3, "AIvovo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
