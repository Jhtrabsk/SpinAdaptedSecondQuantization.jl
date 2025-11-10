E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * extract_mat(h_p, "II", o, v) * fixed_einsum("aibj->bjai", extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] +=  -0.25000000 * extract_mat(h_p, "II", o, v) * extract_mat(p2, "AIvovo", o, v);
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("ki,akbj->bjai", extract_mat(F, "oo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("ki,bjak->bjai", extract_mat(F, "oo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kj,aibk->bjai", extract_mat(F, "oo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kj,bkai->bjai", extract_mat(F, "oo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("ac,cibj->bjai", extract_mat(F, "vv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("ac,bjci->bjai", extract_mat(F, "vv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("bc,cjai->bjai", extract_mat(F, "vv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("bc,aicj->bjai", extract_mat(F, "vv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("kk,aibj->bjai", extract_mat(g_p, "IIoo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * extract_mat(p2, "AIvovo", o, v) * fixed_einsum("kk->", extract_mat(g_p, "IIoo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("B,Baibj->bjai", extract_mat(h_p, "AV", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("B,Bbjai->bjai", extract_mat(h_p, "AV", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcai,ckbj->bjai", extract_mat(L, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcai,bjck->bjai", extract_mat(L, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcbj,ckai->bjai", extract_mat(L, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcbj,aick->bjai", extract_mat(L, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcai,cjbk->bjai", extract_mat(g, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcai,bkcj->bjai", extract_mat(g, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcbj,ciak->bjai", extract_mat(g, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcbj,akci->bjai", extract_mat(g, "ovvo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kilj,akbl->bjai", extract_mat(g, "oooo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kilj,blak->bjai", extract_mat(g, "oooo", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kibc,cjak->bjai", extract_mat(g, "oovv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kibc,akcj->bjai", extract_mat(g, "oovv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kjac,cibk->bjai", extract_mat(g, "oovv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kjac,bkci->bjai", extract_mat(g, "oovv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("acbd,cidj->bjai", extract_mat(g, "vvvv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("acbd,djci->bjai", extract_mat(g, "vvvv", o, v), extract_mat(p2, "AIvovo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("Bkk,Baibj->bjai", extract_mat(g_p, "AVoo", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.50000000  * fixed_einsum("Bkk,Bbjai->bjai", extract_mat(g_p, "AVoo", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("Bki,Bakbj->bjai", extract_mat(g_p, "AVoo", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("Bki,Bbjak->bjai", extract_mat(g_p, "AVoo", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("Bkj,Baibk->bjai", extract_mat(g_p, "AVoo", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("Bkj,Bbkai->bjai", extract_mat(g_p, "AVoo", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bac,Bcibj->bjai", extract_mat(g_p, "AVvv", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bac,Bbjci->bjai", extract_mat(g_p, "AVvv", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bbc,Bcjai->bjai", extract_mat(g_p, "AVvv", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bbc,Baicj->bjai", extract_mat(g_p, "AVvv", o, v), extract_mat(p2, "VIvovo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,aibj,Bck->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * extract_mat(p2, "AIvovo", o, v) * fixed_einsum("Bkc,Bck->", extract_mat(g_p, "IVov", o, v), extract_mat(s, "VIvo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Bckai,bj->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Bckbj,ai->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bkc,Bciak,bj->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bkc,Bcjbk,ai->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bkc,Bakci,bj->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Baick,bj->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("Bkc,Bbkcj,ai->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.50000000  * fixed_einsum("Bkc,Bbjck,ai->bjai", extract_mat(g_p, "IVov", o, v), extract_mat(p2, "VIvovo", o, v), extract_mat(s, "AIvo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,ckdi,albj->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,ckdj,aibl->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,ckal,dibj->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,ckbl,djai->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,cidl,akbj->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,cjdl,aibk->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,akdl,cibj->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,bkdl,cjai->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,ckai,dlbj->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,ckbj,dlai->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,aick,dlbj->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,bjck,dlai->bjai", extract_mat(L, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,cidj,akbl->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,cial,dkbj->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,cibl,djak->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,cjdi,albk->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,cjal,dibk->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,cjbl,dkai->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,akdi,clbj->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,akdj,cibl->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,akbl,cidj->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,bkdi,cjal->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,bkdj,clai->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  +0.25000000  * fixed_einsum("kcld,bkal,cjdi->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");

E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,ciak,dlbj->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,cibj,dlak->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,cjai,dlbk->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,cjbk,dlai->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,akci,dlbj->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,akbj,cidl->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,aicj,dlbk->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,aibk,cjdl->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,bkcj,dlai->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,bkai,cjdl->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,bjci,dlak->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");
E_bjai[:,:,:,:] = E_bjai[:,:,:,:] .+  -0.25000000  * fixed_einsum("kcld,bjak,cidl->bjai", extract_mat(g, "ovov", o, v), extract_mat(p2, "AIvovo", o, v), extract_mat(u, "vovo", o, v), optimize="optimal");