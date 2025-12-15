E = E .+  -2.00000000  * fixed_einsum("aibj,ac,bjci->", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IAvv", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E = E .+  +2.00000000  * fixed_einsum("aibj,ki,akbj->", extract_mat(c2, "vovo", o, v), extract_mat(g_p, "IAoo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
