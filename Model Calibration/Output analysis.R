
library(tidyverse)
library(raster)
library(viridis)
library(rgdal)
library(ncdf4)


setwd('C:/Users/Oli/Documents/PhD/wham/Model Calibration/Outputs 03102021')
Correct   <- nc_open('Total_new_unoc_pars.nc')
AFTs      <- read.csv('AFTs_2015.csv')
Fire      <- read.csv('Fire_dat_by_AFT.csv')


lon     <- ncvar_get(Correct, "lon")
lat     <- ncvar_get(Correct, "lat", verbose = F)
Correct    <- aperm(ncvar_get(Correct, "value"), c(2, 1, 3))

for(i in 1:25) {
  
  Correct.rast[[i]]         <- raster(Correct[, , i])
  extent(Correct.rast[[i]]) <- c(-180, 180, -90, 90)
  #Correct.rast[[i]][JULES.mask <= 0] <- NA

}

#Fire.rast <- brick(unlist(Fire.rast))
cor.test(log1p(MOD.JULES)[], log1p(Fire.rast[[25]])[])
cor.test(log1p(MOD.JULES)[], log1p(Correct.rast[[25]])[])
cor.test(MOD.JULES[], (Fire.rast[[25]][]), method = 'kendall')
cor.test(MOD.JULES[], (Correct.rast[[25]][]), method = 'kendall')

plot(Fire.rast[[25]]*100, col = inferno(11))
plot(MOD.JULES*100, breaks = c(0, 0.5,1, 2, 4, 7.5, 12.5, 20, 30, 50, 90), col = inferno(11))


#### Time series
fire.ts <- data.frame(Corrected = unlist(lapply(Correct.rast, function(x) {mean(x[], na.rm = T)})),
                      Old     = unlist(lapply(Fire.rast, function(x) {mean(x[], na.rm = T)})))

fire.ts$Year <- 1989 + seq(1, nrow(fire.ts)) 

fire.ts %>% 
  pivot_longer(c('Corrected', 'Old')) %>%
  ggplot(aes(x = Year, y = value * 100, colour = name)) + geom_line(size = 1.5) +
  theme_classic() + scale_colour_viridis_d() + ylab('Burned area %') + 
  geom_smooth(method = "lm", se = F, aes(colour = name), linetype = 'dashed') +
  theme(text = element_text(size = 14))


########################################################################################

### Compare MODIS and WHAM

########################################################################################

fdf <- data.frame(MODIS = MOD.JULES[], WHAM = Correct.rast[[25]][], 
                  ET = ET[[25]][], HDI = HDI[[25]][], HDI_GDP = (HDI[[25]] * log(GDP[[25]]))[],
                  Soil = JULES.PFTs$`bare soil`[], Crop  = CMIP6_Cropland[[25]][], 
                  Livestock = (CMIP6_Pasture[[25]] + CMIP6_Rangeland[[25]])[], 
                  Pastoralist = AFTs$Pastoralist, Ext_L = AFTs$Ext_LF_p + AFTs$Ext_LF_r, 
                  HG = AFTs$Hunter_gatherer, pastoralist_fire = Fire$pastoralist, 
                  extensive_fire = Fire$extensive, hg_fire = Fire$hg, pyrome = Fire$pyrome)

f_neg <- fdf[fdf$MODIS > fdf$WHAM & fdf$MODIS > 0.1, ]
Sahel <- fdf[12768:12788, ]

### Okay, what if we redo the soil constraint to account for BA?
setwd('C:/Users/Oli/Documents/PhD/wham/Model Calibration/Outputs 03102021')
Soil_reconfigure <- read.csv('Soil_constraint_reconfigure.csv')
Soil_constraint  <- data.frame(JULES.PFTs$`bare soil`[])
Soil_constraint  <- pivot_longer(Soil_constraint, colnames(Soil_constraint))[, 2]
Soil_constraint$value <- ifelse(Soil_constraint$value >= 0.1325, 1 - Soil_constraint$value, 1)


Soil_reconfigure$Constraint <- Soil_constraint$value
Soil_reconfigure$New_fire   <- (Soil_reconfigure$Vegetation + Soil_reconfigure$Pasture) / Soil_reconfigure$Constraint
Soil_reconfigure$New_Constraint <- 1 - ifelse((Soil_reconfigure$Soil - fdf$MODIS) < 0.1325, 0, Soil_reconfigure$Soil - fdf$MODIS)
Soil_reconfigure$New_fire       <- Soil_reconfigure$New_fire * Soil_reconfigure$New_Constraint
Soil_reconfigure$New_fire       <-  Soil_reconfigure$New_fire + Soil_reconfigure$Arable

fdf$New_fire <- Soil_reconfigure$New_fire

