# CDeFuse
## Environment
> python 3.9
> pytorch 2.1.0
## Test
### Data preparation
Download MSRS and RoadScene and put them in dataset folder respectively.
[MSRS](https://github.com/Linfeng-Tang/MSRS)
[Roadscene](https://github.com/jiayi-ma/RoadScene)

### Start to test
run this command to test:
~~~
python test.py --config ./configs/UF-Base.yaml --ir your_ir_test_datapath --vi your_vi_test_datapath
~~~
the fusion result will be saved in 'fused_images' folder.

## Train

Training Code will be publicly available after Oct 01.