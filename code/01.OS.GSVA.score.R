options(stringsAsFactors = F, encoding = "UTF-8")

#################
# package
#################

suppressMessages({
    
    library(clusterProfiler)            ## enrichment analysis
    
    library(GSVA)                        ## GSVA
    library(openxlsx)                    ## openxlsx
    
    })

##################
# Output dir
##################

mydir <- "Demo"
dir.create(mydir, recursive = T)


##################
# load expression profile
##################

load("01.tpm.exp.rdata")

##################################
# OS geneset
#################################

O.genes <- read.xlsx("O.xlsx")
R.genes <- read.xlsx("R.xlsx")
OS.genes <- read.xlsx("OS.xlsx")

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



  
