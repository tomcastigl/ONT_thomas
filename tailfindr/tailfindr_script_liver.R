library(tailfindr)

print('running')
find_tails(fast5_dir = '/scratch/tcastigl/ONT_liver/0/',
             save_dir = '/scratch/tcastigl/data/tail_findr_liver_res/0/',
             csv_filename = 'out_liver_0.csv',
             basecall_group = 'Basecall_1D_001',
             num_cores = 14,
          )
print('done')