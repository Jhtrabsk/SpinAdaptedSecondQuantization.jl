E_ckbjai[:,:] = E_ckbjai[:,:] .+  -0.50000000  * fixed_einsum("ljia,blck->ckbjai", extract_mat(L, "ooov", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ckbjai[:,:] = E_ckbjai[:,:] .+  -0.50000000  * fixed_einsum("lkia,bjcl->ckbjai", extract_mat(L, "ooov", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ckbjai[:,:] = E_ckbjai[:,:] .+  +0.50000000  * fixed_einsum("iabd,djck->ckbjai", extract_mat(L, "ovvv", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ckbjai[:,:] = E_ckbjai[:,:] .+  +0.50000000  * fixed_einsum("iacd,dkbj->ckbjai", extract_mat(L, "ovvv", o, v), extract_mat(t, "vovo", o, v), optimize="optimal");
E_ckbjai[:,:] = E_ckbjai[:,:] .+  -1.00000000  * fixed_einsum("Aia,Abjck->ckbjai", extract_mat(g_p, "IVov", o, v), extract_mat(s2, "VIvovo", o, v), optimize="optimal");
