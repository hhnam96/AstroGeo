# (1) load the Astrochron package
library(astrochron);
setwd("/obs/nhoanghoai/AstroGeo")

# (2) Obtain the Xiamaling Cu/Al dataset from the Astrochron server
CuAl=getData("Xiamaling-CuAl");
write.csv(CuAl, "Xiamaling/CuAl.csv", row.names=FALSE)

# (3) Interpolate the data to the median sampling interval of 0.012 m.
CuAl_0.012=linterp(CuAl);
# (4) Determine nominal precession and eccentricity periods,
#then conduct nominal timeOpt analysis (see Table S2).
targetTot=calcPeriods(g=c(5.525000,7.455000,17.300000,17.850000,4.257455),k=78,output=2);
targetE=sort(targetTot[1:5],decreasing=T);
targetP=sort(targetTot[6:10],decreasing=T);
# (5) run nominal timeOpt
# output sedimentation rate grid and fit
res1=timeOpt(CuAl_0.012,sedmin=.1,sedmax=.6,numsed=100,targetE=targetE,
             targetP=targetP,flow=1/19,fhigh=1/12,roll=10^7,limit=T,output=1);
# output optimal time series, bandpassed series, amplitude envelope and
# TimeOpt-reconstructed eccentricity
res2=timeOpt(CuAl_0.012,sedmin=.1,sedmax=.6,numsed=100,targetE=targetE,
             targetP=targetP,flow=1/19,fhigh=1/12,roll=10^7,limit=T,output=2);
# (6) perform nominal timeOpt significance testing (this function call uses
#(this function call uses 7 cores for parallel processing)
simres=timeOptSim(CuAl_0.012,sedmin=.1,sedmax=0.6,numsed=100,
                  targetE=targetE,targetP=targetP,flow=1/19,fhigh=1/12,roll=10^7,
                  numsim=2000,output=2,ncores=7);
# (7) plot summary figure
timeOptPlot(CuAl_0.012,res1,res2,simres,flow=1/19,fhigh=1/12,roll=10^7,
            targetE=targetE,targetP=targetP,xlab="Height(m)",ylab="Cu/Al",
            fitR=0.2996136,verbose=T)