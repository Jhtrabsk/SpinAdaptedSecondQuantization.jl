Omega_AIai__ai[:,:] +=  -0.50000000  * extract_mat(g_p, "AIvo", o, v);
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +0.50000000  * np.einsum("ab,bi->ai", extract_mat(F, "vv", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("ji,aj->ai", extract_mat(F, "oo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +0.50000000  * np.einsum("B,Bai->ai", extract_mat(h_p, "AV", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +1.00000000  * np.einsum("jb,aibj->ai", extract_mat(F, "ov", o, v), extract_mat(s2, "AIvovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("jb,ajbi->ai", extract_mat(F, "ov", o, v), extract_mat(s2, "AIvovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +0.50000000  * np.einsum("aijb,bj->ai", extract_mat(L, "voov", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("Bab,Bbi->ai", extract_mat(g_p, "AVvv", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +0.50000000  * np.einsum("Bji,Baj->ai", extract_mat(g_p, "AVoo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -1.00000000  * np.einsum("Bjj,Bai->ai", extract_mat(g_p, "AVoo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("jb,aibj->ai", extract_mat(g_p, "AIov", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("jikb,ajbk->ai", extract_mat(L, "ooov", o, v), extract_mat(s2, "AIvovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +1.00000000  * np.einsum("abjc,bicj->ai", extract_mat(g, "vvov", o, v), extract_mat(s2, "AIvovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("abjc,bjci->ai", extract_mat(g, "vvov", o, v), extract_mat(s2, "AIvovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -1.00000000  * np.einsum("Bjb,Baibj->ai", extract_mat(g_p, "AVov", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +0.50000000  * np.einsum("Bjb,Bajbi->ai", extract_mat(g_p, "AVov", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  +0.50000000  * np.einsum("jbkc,bj,aick->ai", extract_mat(L, "ovov", o, v), extract_mat(s, "AIvo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("jbkc,aj,bick->ai", extract_mat(g, "ovov", o, v), extract_mat(s, "AIvo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
Omega_AIai__ai[:,:] = Omega_AIai__ai[:,:] .+  -0.50000000  * np.einsum("jbkc,bi,ajck->ai", extract_mat(g, "ovov", o, v), extract_mat(s, "AIvo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
