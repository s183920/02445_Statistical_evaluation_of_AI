data_fos <- get(load("fosfor_data.Rdata"))
data_fos
plot(data_fos)
plot(data_fos$yield ~data_fos$olsenP)
