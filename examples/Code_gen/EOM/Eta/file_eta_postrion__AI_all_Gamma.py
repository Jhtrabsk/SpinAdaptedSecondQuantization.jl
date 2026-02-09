E +=  +1.00000000  * extract_mat(h_p, "IA", o, v);
E = E .+  -2.00000000  * fixed_einsum("ii->", extract_mat(g_p, "IAoo", o, v), optimize="optimal");
