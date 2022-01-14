from gridtools import *

@param('grid.log_dir')
@param('grid.out_file')
def main(log_dir, out_file):
    out_dir = Path(log_dir) / str(uuid4())
    out_dir.mkdir(exist_ok=True, parents=True)

    starts = []
    wds = [Parameters(wd=k) for k in [1e-4]]
    lrs = [Parameters(lr=float(k)) for k in [1.7]]
    res = [Parameters(min_res=160, max_res=a, val_res=b) for a, b in
    [(192, 256), (160, 224)]]

    epochs = []
    for e in [16, 24, 32, 40, 56, 88]:
        fifth = int(e // 8)
        start_ramp = e - fifth * 2 - 1
        end_ramp = e - fifth - 1
        epochs.append(Parameters(
            epochs=e,
            start_ramp=start_ramp,
            end_ramp=end_ramp,
            workers=12
        ))

    base_dir = '/home/ubuntu/' if os.path.exists('/home/ubuntu/') else '/mnt/cfs/home/engstrom/store/ffcv/'
    archs = [
        Parameters(train_dataset=base_dir + 'train_500_0.5_90.ffcv',
                   val_dataset=base_dir + 'val_500_0.5_90.ffcv',
                   batch_size=512,
                   arch='resnet50',
                   distributed=1,
                   logs=log_dir,
                   world_size=8),
    ]

    axes = [archs, wds, lrs, res, epochs]

    rn18_base = 'imagenet_configs/resnet50_base.yaml'
    design_command(axes, out_dir, out_file, rn18_base, cuda_preamble="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", jobs=1)

if __name__ == '__main__':
    Section('grid', 'data related stuff').params(
        log_dir=Param(str, 'out directory', default=str(Path('~/store/ffcv_rn50_8gpu/').expanduser())),
        out_file=Param(str, 'out file', default=str(Path('~/store/ffcv_rn50_8gpu/jobs_18.txt').expanduser()))
    )

    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()

