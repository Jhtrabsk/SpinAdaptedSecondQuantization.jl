E = E .+  -1.00000000  * fixed_einsum("ai,ia->", extract_mat(c1, "vo", o, v), extract_mat(g_p, "AIov", o, v), optimize="optimal");
E = E .+  +1.00000000  * fixed_einsum("iajb,ai,bj->", extract_mat(L, "ovov", o, v), extract_mat(c1, "vo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
