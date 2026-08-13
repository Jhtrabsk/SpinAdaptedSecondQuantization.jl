E_IA = E_IA .+  +2.00000000  * fixed_einsum("aibj,ai,bj->A", extract_mat(l2, "vovo", o, v), extract_mat(r1, "vo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
