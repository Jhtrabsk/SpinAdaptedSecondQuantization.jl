E = E .+  +0.50000000  * fixed_einsum("iajb,aibj->", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
