n_steps="4"       # 100 150 200
orders="3"         
gpus="4"           
n_samples="50000"  
ds="cifar10-32x32" # afhqv2-64x64, ffhq-64x64, cifar10-32x32, , imagenet-64x64
fr="vp"            # vp, ve, adm for imagenet
guid="uncond"      # uncond for afhqv2 ffhq - cond for cifar and imagenet
solv="etd-serk"
discr="edm"
sched="linear"
scal="none"

for o in $orders; do
	for n in $n_steps; do
        rm -r fid-tmp
        
		OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$gpus generate.py --outdir=fid-tmp \
        --steps=$n \
        --solver=$solv \
        --order=$o \
        --noise_pred=False \
        --disc=$discr \
        --schedule=$sched \
        --scaling=$scal \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-$ds-$guid-$fr.pkl \
        --seeds=1-$n_samples \
        
       OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$gpus fid.py calc --images=fid-tmp \
       --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/$ds.npz \
       --num $n_samples
	done
done