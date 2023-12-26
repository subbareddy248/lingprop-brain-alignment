for i in {244..244} {249..249} {254..269}; do
    cd "sub-"$i/func || continue
    datalad install -g *21styear_space-fsaverage6_hemi-L_desc-clean.func.gii
    datalad install -g *21styear_space-fsaverage6_hemi-R_desc-clean.func.gii
    cd "../../"
done
