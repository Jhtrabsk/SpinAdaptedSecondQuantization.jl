E +=  +1.00000000  * extract_mat(h_p, "II", o, v);
E = E .+  +2.00000000  * fixed_einsum("ii->", extract_mat(F, "oo", o, v), optimize="optimal");
E = E .+  -2.00000000  * fixed_einsum("ii->", extract_mat(g_p, "IIoo", o, v), optimize="optimal");
E = E .+  -1.00000000  * fixed_einsum("iijj->", extract_mat(L, "oooo", o, v), optimize="optimal");
E = E .+  -2.00000000  * fixed_einsum("Aia,Aai->", extract_mat(g_p, "IVov", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E = E .+  +1.00000000  * fixed_einsum("iajb,aibj->", extract_mat(g, "ovov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
