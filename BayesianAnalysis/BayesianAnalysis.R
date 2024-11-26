setwd(normalizePath(dirname(rstudioapi::getActiveDocumentContext()$path)))
df <- read.csv('data_for_R.csv')

# BEST package is no longer active, but can still be downloaded from CRAN archive
# Download package tarball from CRAN archive
#url <- "https://cran.r-project.org/src/contrib/Archive/BEST/BEST_0.5.4.tar.gz"
#pkgFile <- "BEST_0.5.4.tar.gz"
#download.file(url = url, destfile = pkgFile)
# make sure the dependencies coda and rjags are installed
# Install package from downloaded file
#install.packages(pkgs=pkgFile, type="source", repos=NULL)

library(BEST)
library(tidyverse)

# NMI----------------------------

df_NMI <- df %>% 
  filter(entropy_scores == 'NMI')

df_NMI_fine <- df_NMI %>% 
  filter(condition == 'fine')

agg_NMI_fine <- df_NMI_fine %>%
  summarize(mean = mean(value),
            sd = sd(value))

df_NMI_mixed <- df_NMI %>% 
  filter(condition == 'mixed')

agg_NMI_mixed <- df_NMI_mixed %>%
  summarize(mean = mean(value),
            sd = sd(value))

df_NMI_coarse <- df_NMI %>% 
  filter(condition == 'coarse')

agg_NMI_coarse <- df_NMI_coarse %>%
  summarize(mean = mean(value),
            sd = sd(value))

grand_mean <- (agg_NMI_fine$mean + agg_NMI_mixed$mean + agg_NMI_coarse$mean)/3
grand_sd <- (agg_NMI_fine$sd + agg_NMI_mixed$sd + agg_NMI_coarse$sd)/3

# calculate a region of practical equivalence with zero according to recommendation by Kruschke (2018)
rope <- c(-0.1*grand_sd, 0.1*grand_sd)

priors <- list(muM = grand_mean, muSD = grand_sd)

# either load or generate models
#load("BEST_NMI_fine.Rda")
#load("BEST_NMI_coarse.Rda")
BEST_NMI_fine <- BESTmcmc(df_NMI_fine$value, df_NMI_mixed$value, priors=priors, parallel=TRUE)
BEST_NMI_coarse <- BESTmcmc(df_NMI_coarse$value, df_NMI_mixed$value, priors=priors, parallel=TRUE)

# check for convergence
print(BEST_NMI_fine)
print(BEST_NMI_coarse)
# -> all models converged

Diff_fine <- (BEST_NMI_fine$mu1 - BEST_NMI_fine$mu2)
meanDiff_fine <- round(mean(Diff_fine), 3)
hdiDiff_fine <- hdi(BEST_NMI_fine$mu1 - BEST_NMI_fine$mu2)
plotAll(BEST_NMI_fine)
plot(BEST_NMI_fine,ROPE=rope)
summary(BEST_NMI_fine)
# CrI does not include 0
# 100% probability that the difference in means is larger than 0 (pd)
# 0% in ROPE

Diff_coarse <- (BEST_NMI_coarse$mu1 - BEST_NMI_coarse$mu2)
meanDiff_coarse <- round(mean(Diff_coarse), 3)
hdiDiff_coarse <- hdi(BEST_NMI_coarse$mu1 - BEST_NMI_coarse$mu2)
plotAll(BEST_NMI_coarse)
plot(BEST_NMI_coarse,ROPE=rope)
summary(BEST_NMI_coarse)
# CrI does not include 0
# 100% probability that the difference in means is larger than 0 (pd)
# 0% in ROPE

# save all models for reproducibility
write.csv(BEST_NMI_fine, "BEST_NMI_fine.csv", row.names=FALSE, quote=FALSE) 
save(BEST_NMI_fine,file="BEST_NMI_fine.Rda")
write.csv(BEST_NMI_coarse, "BEST_NMI_coarse.csv", row.names=FALSE, quote=FALSE) 
save(BEST_NMI_coarse,file="BEST_NMI_coarse.Rda")

# effectiveness----------------------------

df_effectiveness <- df %>% 
  filter(entropy_scores == 'effectiveness')

df_eff_fine <- df_effectiveness %>% 
  filter(condition == 'fine')

agg_eff_fine <- df_eff_fine %>%
  summarize(mean = mean(value),
            sd = sd(value))

df_eff_mixed <- df_effectiveness %>% 
  filter(condition == 'mixed')

agg_eff_mixed <- df_eff_mixed %>%
  summarize(mean = mean(value),
            sd = sd(value))

df_eff_coarse <- df_effectiveness %>% 
  filter(condition == 'coarse')

agg_eff_coarse <- df_eff_coarse %>%
  summarize(mean = mean(value),
            sd = sd(value))

grand_mean <- (agg_eff_fine$mean + agg_eff_mixed$mean + agg_eff_coarse$mean)/3
grand_sd <- (agg_eff_fine$sd + agg_eff_mixed$sd + agg_eff_coarse$sd)/3

# calculate a region of practical equivalence with zero according to recommendation by Kruschke (2018)
rope <- c(-0.1*grand_sd, 0.1*grand_sd)

priors <- list(muM = grand_mean, muSD = grand_sd)

# either load or generate models
#load("BEST_eff_fine.Rda")
#load("BEST_eff_coarse.Rda")
BEST_eff_fine <- BESTmcmc(df_eff_fine$value, df_eff_mixed$value, priors=priors, parallel=TRUE)
BEST_eff_coarse <- BESTmcmc(df_eff_coarse$value, df_eff_mixed$value, priors=priors, parallel=TRUE)

# check for convergence
print(BEST_eff_fine)
print(BEST_eff_coarse)
# -> all models converged

Diff_fine <- (BEST_eff_fine$mu1 - BEST_eff_fine$mu2)
meanDiff_fine <- round(mean(Diff_fine), 3)
hdiDiff_fine <- hdi(BEST_eff_fine$mu1 - BEST_eff_fine$mu2)
plotAll(BEST_eff_fine)
plot(BEST_eff_fine,ROPE=rope)
summary(BEST_eff_fine)
# CrI includes 0
# 85.1% probability that the difference in means is larger than 0 (pd)
# 15% in ROPE

Diff_coarse <- (BEST_eff_coarse$mu1 - BEST_eff_coarse$mu2)
meanDiff_coarse <- round(mean(Diff_coarse), 3)
hdiDiff_coarse <- hdi(BEST_eff_coarse$mu1 - BEST_eff_coarse$mu2)
plotAll(BEST_eff_coarse)
plot(BEST_eff_coarse,ROPE=rope)
summary(BEST_eff_coarse)
# CrI does not include 0
# 100% probability that the difference in means is larger than 0 (pd)
# 0% in ROPE

# save all models for reproducibility
write.csv(BEST_eff_fine, "BEST_eff_fine.csv", row.names=FALSE, quote=FALSE) 
save(BEST_eff_fine,file="BEST_eff_fine.Rda")
write.csv(BEST_eff_coarse, "BEST_eff_coarse.csv", row.names=FALSE, quote=FALSE) 
save(BEST_eff_coarse,file="BEST_eff_coarse.Rda")

# consistency----------------------------

df_consistency <- df %>% 
  filter(entropy_scores == 'consistency')

df_cons_fine <- df_consistency %>% 
  filter(condition == 'fine')

agg_cons_fine <- df_cons_fine %>%
  summarize(mean = mean(value),
            sd = sd(value))

df_cons_mixed <- df_consistency %>% 
  filter(condition == 'mixed')

agg_cons_mixed <- df_cons_mixed %>%
  summarize(mean = mean(value),
            sd = sd(value))

df_cons_coarse <- df_consistency %>% 
  filter(condition == 'coarse')

agg_cons_coarse <- df_cons_coarse %>%
  summarize(mean = mean(value),
            sd = sd(value))

grand_mean <- (agg_cons_fine$mean + agg_cons_mixed$mean + agg_cons_coarse$mean)/3
grand_sd <- (agg_cons_fine$sd + agg_cons_mixed$sd + agg_cons_coarse$sd)/3

# calculate a region of practical equivalence with zero according to recommendation by Kruschke (2018)
rope <- c(-0.1*grand_sd, 0.1*grand_sd)

priors <- list(muM = grand_mean, muSD = grand_sd)

# either load or generate models
#load("BEST_cons_fine.Rda")
#load("BEST_cons_coarse.Rda")
BEST_cons_fine <- BESTmcmc(df_cons_fine$value, df_cons_mixed$value, priors=priors, parallel=TRUE)
BEST_cons_coarse <- BESTmcmc(df_cons_coarse$value, df_cons_mixed$value, priors=priors, parallel=TRUE)

# check for convergence
print(BEST_cons_fine)
print(BEST_cons_coarse)
# -> all models converged

Diff_fine <- (BEST_cons_fine$mu1 - BEST_cons_fine$mu2)
meanDiff_fine <- round(mean(Diff_fine), 3)
hdiDiff_fine <- hdi(BEST_cons_fine$mu1 - BEST_cons_fine$mu2)
plotAll(BEST_cons_fine)
plot(BEST_cons_fine,ROPE=rope)
summary(BEST_cons_fine)
# CrI does not include 0
# 100% probability that the difference in means is larger than 0 (pd)
# 0% in ROPE

Diff_coarse <- (BEST_cons_coarse$mu1 - BEST_cons_coarse$mu2)
meanDiff_coarse <- round(mean(Diff_coarse), 3)
hdiDiff_coarse <- hdi(BEST_cons_coarse$mu1 - BEST_cons_coarse$mu2)
plotAll(BEST_cons_coarse)
plot(BEST_cons_coarse,ROPE=rope)
summary(BEST_cons_coarse)
# CrI does not include 0
# 100% probability that the difference in means is larger than 0 (pd)
# 0% in ROPE

# save all models for reproducibility
write.csv(BEST_cons_fine, "BEST_cons_fine.csv", row.names=FALSE, quote=FALSE) 
save(BEST_cons_fine,file="BEST_cons_fine.Rda")
write.csv(BEST_cons_coarse, "BEST_cons_coarse.csv", row.names=FALSE, quote=FALSE) 
save(BEST_cons_coarse,file="BEST_cons_coarse.Rda")
