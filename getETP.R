library(astrochron)
setwd("/obs/nhoanghoai/AstroGeo")
t0 = 0 
tf = 10000
ecc = etp(tmin=t0,tmax=tf,dt=1, eWt=1, oWt=0, pWt=0)
prec = etp(tmin=t0,tmax=tf,dt=1, eWt=0, oWt=0, pWt=1)
obl = etp(tmin=t0,tmax=tf,dt=1, eWt=0, oWt=1, pWt=0)
write.csv(ecc, "ETP/eccentricity.csv", row.names=FALSE)
write.csv(prec, "ETP/precession.csv", row.names=FALSE)
write.csv(obl, "ETP/obliquity.csv", row.names=FALSE)

