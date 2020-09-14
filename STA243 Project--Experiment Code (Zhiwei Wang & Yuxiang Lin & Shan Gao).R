#-------------------------------------------------------------------------
#Data Simulation  
#Parameters are chosen in order to satisfy the requirements in section 3.3
#-------------------------------------------------------------------------
library(glmnet)
set.seed(20)
X1 <- matrix(runif(1800,-sqrt(1/3),sqrt(1/3)),600,3)  
beta <- matrix(c(1,1,1),3,1)
epsilon <- matrix(rnorm(600,0,0.1),600,1)
y1 <- X1%*%beta+epsilon
M = cbind(X1,y1)
index<-sample(1:600,600,replace = FALSE)
shuffle_M <- M[index,]
Xnew<-shuffle_M[,1:3]
ynew<-shuffle_M[,4]  
#-------------------------------------------------------------------------
#Find out the best beta to calculate the minimal loss function value
#-------------------------------------------------------------------------
ridge_cv <- cv.glmnet(X1, y1, alpha = 0, lambda = seq(0.001, 0.02, 0.0001))
best_lambda <- ridge_cv$lambda.min
best_lambda
best_ridge <- glmnet(X1, y1, alpha = 0, lambda = 0.0142)
coef(best_ridge)
#-------------------------------------------------------------------------
#Algorithm: SVRG without replacement
#-------------------------------------------------------------------------
svrg1 <- function(X1,y1,T,S,eta){
  n<-nrow(Xnew)
  d<-ncol(Xnew)
  Omega <- matrix(0,d,S*30+1)
  omega<-matrix(0,d,S+1)
  beta1<-matrix(0,d,T*S+1)
  for (i in 1:S){
    beta1[,(i-1)*T+1]=omega[,i]  
    n_strange <- matrix(0,d,S)  
    grad<-matrix(0,d,n)
    for (j in 1:n){
      grad[,j]<-omega[,i]%*%Xnew[j,]%*%t(Xnew[j,])-ynew[j]*t(Xnew[j,])+0.0142*omega[,i]
    }
    n_strange = rowSums(grad)/n
    for (t in seq((i-1)*T+1,i*T,1)) {
      beta1[,t+1]<-beta1[,t] - eta*(beta1[,t]%*%Xnew[t-((i-1)*T),]%*%t(Xnew[t-((i-1)*T),])-ynew[t-((i-1)*T)]*t(Xnew[t-((i-1)*T),])+0.0142*beta1[,t]-grad[,t-((i-1)*T)]+n_strange)
      if (t%%20 == 0){
        Omega[,30*(i-1) + 1 + ((t-(i-1)*T)%/%20)] <- beta1[,t] - eta*(beta1[,t]%*%Xnew[t-((i-1)*T),]%*%t(Xnew[t-((i-1)*T),])-ynew[t-((i-1)*T)]*t(Xnew[t-((i-1)*T),])+0.0142*beta1[,t]-grad[,t-((i-1)*T)]+n_strange)
      }
    }   
    omega[,i+1]<-rowSums(beta1[,((i-1)*T+1):(i*T)])/T
  }
  return(Omega)
}
omega1 <- svrg1(Xnew,ynew,600,10,0.01)
omega1
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#Algorithm: SVRG with replacement
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
svrg2 <- function(X1,y1,m,S,eta){
  n<-nrow(Xnew)
  d<-ncol(Xnew)
  omega<-matrix(0,d,S*30+1)
  w_iter = rep(0, 3)
  for (i in 1:S){
    w_0 = w_iter
    w_mid = w_iter
    grad<-matrix(0,d,n)
    for (j in 1:n){  
      grad[,j]<-w_0%*%Xnew[j,]%*%t(Xnew[j,])-ynew[j]*t(Xnew[j,])+0.0142*w_0
    }
    mu = rowSums(grad)/n
    for (t in 1:m) {
      index = sample(1:n, 1)
      w_mid = w_mid - eta*((w_mid%*%Xnew[index,]%*%t(Xnew[index,])-ynew[index]*t(Xnew[index,])+0.0142*w_mid) - (w_0%*%Xnew[index,]%*%t(Xnew[index,])-ynew[index]*t(Xnew[index,])+0.0142*w_0) + mu)
      if (t%%20 == 0){
        omega[,30*(i-1) + 1 + (t%/%20)] = w_mid
      }
    }
    w_iter = w_mid
  }
  return(omega)
}
omega2 = svrg2(Xnew,ynew,600,10,0.01)
omega2
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#Algorithm: distributed learning of SVRG without replacement
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
svrg3 <- function(X,y,S,K,T,eta){  
  n<-nrow(X)
  d<-ncol(X)  
  
  V<-list()  
  for (g in seq(1,S+1,1)) {
    V[[g]]<-matrix(0,d,n/T+1)
  }  
  omega<-matrix(0,d,n/T+1)  
  ind <- sample(1:n,n,FALSE)
  Xnew<-X[ind,]
  ynew<-y[ind]  
  beta2<-matrix(0,d,T+1)  
  n_strange <- matrix(0,d,1)  
  grad<-matrix(0,d,n)
  for (y in 1:S){
    ### K:number of machines
    ### T:number of batch size
    ### M:batch
    for (k in 1:K){
      for (i in seq(1,600/K/T,1)){  
        omega[,1]<-V[[y]][,1]
        for (j in 1:n){  
          grad[,j]<-omega[,i]%*%Xnew[j,]%*%t(Xnew[j,])-ynew[j]*t(Xnew[j,])+0.0142*omega[,i]
        }  
        n_strange<-rowSums(grad)/n
        beta2[,1]<-omega[,(k-1)*(n/K/T)+i]
        for (t in 1:T){
          beta2[,t+1]<-beta2[,t] - eta*(beta2[,t]%*%Xnew[(k-1)*(n/K)+(i-1)*T+t,]%*%t(Xnew[(k-1)*(n/K)+(i-1)*T+t,])-ynew[(k-1)*(n/K)+(i-1)*T+t]*t(Xnew[(k-1)*(n/K)+(i-1)*T+t,])+0.0142*beta2[,t]-grad[,(k-1)*(n/K)+(i-1)*T+t]+n_strange)
        } 
        omega[,(k-1)*(n/K/T)+i+1]<-rowSums(beta2[,1:(T+1)])/(T+1)
      }
    }
    V[[y]]<-omega
    V[[y+1]][,1]<-omega[,n/T+1]  
  }
  return(V)
}

V <- svrg3(Xnew,ynew,10,3,20,0.01)
omega3 <- matrix(0, 3, 301)
for(i in 1:10){
  omega3[, (1 + 30*(i - 1)):(1 + 30*i)] <- as.matrix(V[[i]])
}

loss_calculation <- function(Xnew, ynew, beta){
  return(mean((ynew - Xnew%*%beta)^2 + 0.0053*beta^2))
}
  
#-------------------------------------------------------------------------
#Parameter values are stored in omega, we calculate the corresponding loss function value
#-------------------------------------------------------------------------
Expected_loss_difference1 <- rep(0, ncol(omega1))
for (i in 1:ncol(omega1)) {
  Expected_loss_difference1[i] <- loss_calculation(Xnew, ynew, omega1[,i]) - loss_calculation(Xnew, ynew, coef(best_ridge)[2:4])
}
Expected_loss_difference1


Expected_loss_difference2 <- rep(0, ncol(omega2))
for (i in 1:ncol(omega2)) {
  Expected_loss_difference2[i] <- loss_calculation(Xnew, ynew, omega2[,i]) - loss_calculation(Xnew, ynew, coef(best_ridge)[2:4])
}
Expected_loss_difference2

Expected_loss_difference3 <- rep(0, ncol(omega3))
for (i in 1:ncol(omega3)) {
  Expected_loss_difference3[i] <- loss_calculation(Xnew, ynew, omega3[,i]) - loss_calculation(Xnew, ynew, coef(best_ridge)[2:4])
}
Expected_loss_difference3

batch_num <- 1:301  
#-------------------------------------------------------------------------
#Draw plots for comparisons
#-------------------------------------------------------------------------
par(mfrow=c(1,2))
plot(batch_num, Expected_loss_difference1, type = 'l', col = 'blue', xlab="Batch number", ylab="Loss Difference", main = 'Comparison of with and w/o \nrepalcement SVRG algorithm', cex.main=0.8)
lines(batch_num, Expected_loss_difference2, type = 'l', col = 'red')
legend(100, 0.25, legend=c("SVRG w/o replacement", "SVRG with replacement"), lty=1:1, col=c("blue", "red"), cex=0.6)

plot(batch_num, Expected_loss_difference1, type = 'l', col = 'blue', xlab="Batch number", ylab="Loss Difference", main = 'Comparison of w/o replacement SVRG and \ndistributed-learning w/o relacement SVRG algorithm', cex.main=0.8)
lines(batch_num, Expected_loss_difference3, type = 'l', col = 'red')
legend(60, 0.26, legend=c("SVRG w/o replacement", "Distributed-learning SVRG w/o replacement"), lty=1:1, col=c("blue", "red"), cex=0.6)