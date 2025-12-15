E = E .+  +1.00000000  * fixed_einsum("ia,ai->", extract_mat(F, "ov", o, v), extract_mat(p, "AIvo", o, v), optimize="optimal");
E = E .+  -1.00000000  * fixed_einsum("Bia,Bai->", extract_mat(g_p, "AVov", o, v), extract_mat(p, "VIvo", o, v), optimize="optimal");
