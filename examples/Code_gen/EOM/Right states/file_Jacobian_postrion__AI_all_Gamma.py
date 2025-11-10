E +=  -0.50000000  * extract_mat(h_p, "II", o, v) * extract_mat(p3, "AI", o, v);
E = E .+  +1.00000000  * extract_mat(p3, "AI", o, v) * fixed_einsum("ii->", extract_mat(g_p, "IIoo", o, v), optimize="optimal");
E = E .+  +0.50000000  * fixed_einsum("B,B->", extract_mat(h_p, "AV", o, v), extract_mat(p3, "VI", o, v), optimize="optimal");
E = E .+  -1.00000000  * fixed_einsum("Bii,B->", extract_mat(g_p, "AVoo", o, v), extract_mat(p3, "VI", o, v), optimize="optimal");
E = E .+  +1.00000000  * extract_mat(p3, "AI", o, v) * fixed_einsum("Bia,Bai->", extract_mat(g_p, "IVov", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
