options(stringsAsFactors = F, encoding = "UTF-8")

#################
# package
#################

suppressMessages({
    
    library(GSVA)                        ## GSVA
    
    })

##################
# Output dir
##################

mydir <- "../demo_result"
dir.create(mydir, recursive = T)

##################
# load expression profile
##################

exp <- read.csv("../demo_data/exp.csv.gz", header = T, row.names = 1, check.names = F)

##################################
# OS geneset
#################################

O.genes <- read.csv("../demo_data/O.csv", header = T)
R.genes <- read.csv("../demo_data/R.csv", header = T)
OS.genes <- read.csv("../demo_data/OS.csv", header = T)

pathwaysList.all <- list()

pathwaysList.all[["O.score"]] <- unique(O.genes$symbol)
pathwaysList.all[["R.score"]] <- unique(R.genes$symbol)
pathwaysList.all[["OS.score"]] <- unique(OS.genes$symbol)

#############
# Preprocessing expression profile
#############

## log2-transformed TPM values
exp <- log2(exp + 1)
exp <- as.matrix(exp)


####################
# calculating the gsva score
####################

gsvapar <- gsvaParam(exp, pathwaysList.all, kcdf = "Gaussian") #"Gaussian" for logCPM, logRPKM, logTPM, "Poisson" for counts
gsva_score <- gsva(gsvapar, verbose=T) 

write.csv(gsva_score, file = paste0(mydir, "/01.oxidative.stress.score.csv"), row.names=T, quote=F)
