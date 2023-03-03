n_steps="90"       # 100 150 200
orders="3"         
gpus="2"           
n_samples="100"
ds="ffhq-64x64" # afhqv2-64x64, ffhq-64x64, cifar10-32x32, imagenet-64x64
fr="vp"            # vp, ve, adm for imagenet
guid="uncond"      # uncond for afhqv2 ffhq - cond for cifar and imagenet
sched="vp"         # vp, ve, linear
scale="vp"         # vp, none
solver="etd-serk"  # rk, etd-erk, etd-serk
disc="edm"         # vp, ve, iddpm, edm
noise_pred="True"  # boolean

for o in $orders; do
	for n in $n_steps; do
		OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$gpus generate.py --outdir=output \
        --steps=$n \
        --solver=$solver \
        --order=$o \
        --noise_pred=False \
        --disc=$disc \
        --schedule=$sched \
        --scaling=$scale \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-$ds-$guid-$fr.pkl \
        --seeds=1-$n_samples 
	    
        OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$gpus fid-is.py calc --images=output \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/$ds.npz \
        --num $n_samples
        
#	    rm -r output
	done
done


###################### OPTIONS ########################
# --butcher_type=3a
# --batch=64
# --disc iddpm
# --rho=$o \
# --sigma_min=0.002 \
# --sigma_max=81 \
# --S_churn=30 \
# --S_min=0.01 \
# --S_max=1 \
# --S_noise=1.007 \
########################################################
