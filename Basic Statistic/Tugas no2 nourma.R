
library(readxl)
credit <- read.csv("C:/Users/nourma059258/Documents/Tugas1/creditability.csv")

summary(credit)

p1 <- subset(credit, Account.Balance == 4 , select=c(Creditability))
p2 <- subset(credit, Account.Balance < 4 , select=c(Creditability))
simulasi <- 2000

re_p1 <- replicate(simulasi,sum(sample(p1$Creditability, length(p1$Creditability), replace=TRUE)))
#re_p1
p1_persen <- re_p1/length(p1$Creditability)*100
#p1_persen
re_p2 <- replicate(simulasi,sum(sample(p2$Creditability, length(p2$Creditability), replace=TRUE)))
#re_p2
p2_persen <- re_p2/length(p2$Creditability)*100

summary(re_p1)

sel_persen = p1_persen-p2_persen >= 25
#sel_persen
sum(sel_persen)/2000

selisih = p1_persen - p2_persen
t.test(selisih, mu = 25,  alternative="greater", conf.level=0.95)
