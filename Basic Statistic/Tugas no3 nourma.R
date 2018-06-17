
library(readxl)
credit <- read.csv("C:/Users/nourma059258/Documents/Tugas1/creditability.csv")

set.seed(26)
summary(credit)

p1 <- subset(credit, Sex.Marital.Status == 3 , select=c(Creditability))
p2 <- subset(credit, Sex.Marital.Status != 3, select=c(Creditability))
simulasi <- 2000


t.test(p1$Creditability,p2$Creditability,  alternative="greater", conf.level=0.95)
