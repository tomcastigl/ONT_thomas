library(tailfindr)
args = commandArgs(trailingOnly=TRUE)
print(args)
folder <-paste0('/work/upnae/ONT_modif/data/h5mC/',args)
save_dir <- '/work/upnae/thomas_trna/data'
print(folder)
csv_name <- paste0(substr(folder,28,nchar(folder)),'.csv')
csv_name <- gsub('/','_',csv_name)
print(csv_name)
print(paste('in',folder))
find_tails(fast5_dir = folder,
             save_dir = save_dir,
             csv_filename = csv_name,
             basecall_group = 'Basecall_1D_001',
             num_cores = 14,
          )