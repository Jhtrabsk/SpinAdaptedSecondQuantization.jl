E_II = E_II .+  +2.00000000  * fixed_einsum("aibj,aibj->", extract_mat(l2, "vovo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
E_IA = E_IA .+  +2.00000000  * fixed_einsum("Aaibj,aibj->A", extract_mat(L3, "AIvovo", o, v), extract_mat(r2, "vovo", o, v), optimize="optimal");
