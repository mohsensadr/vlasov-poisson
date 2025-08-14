
/*
int crossMED_match(double *U1,double *U2,double *U3, double *W, double *Wold, double *Wprior, struct GAS *gas, struct CELLS *cells,  struct BOX *box, int *index, int num, double Wthreshold, int *Ncol){
  double tol=1e-10; // tolerance for gradient getting to zero
  int Nm=18; // number of moments
  int d1[Nm] = {1,0,0,1,0,1,2,0,0, 3,1,1, 2,0,0, 2,0,0};
  int d2[Nm] = {0,1,0,1,1,0,0,2,0, 0,2,0, 1,3,1, 0,2,0};
  int d3[Nm] = {0,0,1,0,1,1,0,0,2, 0,0,2, 0,0,2, 1,1,3};

  double p[Nm];// p: array of moments
  double g[Nm];// g: gradient of Newton-minimizor
  double g_norm[Nm];
  double lam[Nm];// lam: array of lagrange multipliers
  double *x = (double *) malloc( Nm * sizeof(double) );// array for storing del(lam)
  double **hess;// hess: Hessian of Newton-minimizor
  hess=new double *[Nm];
  for(int i=0;i<Nm;i++)
      hess[i]=new double [Nm];

  double av_st=0;
  int max_st = 0, st=0;
  double Ski, Sji,norm_grad,SkiSji;
  double dummy;
  int conv=0,i,j,k,ii;

  double maxW, maxW2, sumW, sumW2, sumWold;

    // set initial guess of lam
    for(j=0; j<Nm; j++)
      lam[j]=0.0;
    // compute moments of weights
    for(j=0; j<Nm; j++){
      p[j]=0.0;
      //peq[j]=0.0;
    }
    sumW = 0.0;
    for(ii=0;ii<cells[num].num_inside;ii++){
      i = cells[num].indices_inside[ii];
      sumW +=  W[i];
      for(j=0; j<Nm; j++){
        p[j] += (pow(U1[i],d1[j]) * pow(U2[i],d2[j]) * pow(U3[i],d3[j])) * W[i];
      }
    }
    for(j=0; j<Nm; j++){
      p[j] = p[j]/ sumW;
    }
    for(j=0; j<Nm; j++)
      g[j]=0.0;
    sumW2 = 0.0;
    for(ii=0;ii<cells[num].num_inside;ii++){
      i = cells[num].indices_inside[ii];
      sumW2 += Wprior[i];//W0;
      for(j=0; j<Nm; j++){
        g[j] += (pow(U1[i],d1[j]) * pow(U2[i],d2[j]) * pow(U3[i],d3[j])) *Wprior[i];//W0;// / (1.*cells[num].num_inside);
      }
    }
    for(j=0; j<Nm; j++){
        g[j] = g[j] / sumW2 - p[j];
    }
    for(j=0; j<Nm; j++){
      g_norm[j] = g[j];
      if(j>=3){
        g_norm[j] = g_norm[j]/p[j];
      }
    }
    norm_grad = norm1(g_norm,Nm);
    if(norm_grad>tol){
      sumWold = 0.0;
      for(ii=0;ii<cells[num].num_inside;ii++){
        i = cells[num].indices_inside[ii];
        Wold[i] = W[i];
        W[i] = Wprior[i];//W0;
        sumWold += Wold[i];
      }

    conv = 0; st = 0;
    while(conv==0){
      sumW = 0.0;
        for(j=0; j<Nm; j++)
          g[j]=0.0;
          for(ii=0;ii<cells[num].num_inside;ii++){
            i = cells[num].indices_inside[ii];
            sumW +=  W[i];
            for(j=0; j<Nm; j++){
              g[j] += (pow(U1[i],d1[j]) * pow(U2[i],d2[j]) * pow(U3[i],d3[j]) ) * W[i];// / (1.*cells[num].num_inside);
            }
          }
          for(j=0; j<Nm; j++){
            g[j] = g[j] / sumW - p[j];
          }

          for(j=0; j<Nm; j++){
            g_norm[j] = g[j];
            if(j>=3){
              g_norm[j] = g_norm[j]/p[j];
            }
          }
          norm_grad = norm1(g_norm,Nm);

      if(norm_grad<tol){
        conv = 1;
        maxW2 = 0.0;
        sumW2 = 0.0;
        for(ii=0;ii<cells[num].num_inside;ii++){
          sumW2 += W[i];
          i = cells[num].indices_inside[ii];
          if(fabs(W[i])>maxW2)
            maxW2 = fabs(W[i]);
        }

          for(ii=0;ii<cells[num].num_inside;ii++){
            i = cells[num].indices_inside[ii];
            Wold[i] = W[i]*sumWold/sumW2;
          }
          cells[num].Wmax = maxW2;
      }
      else{
        av_st+=1.0;
        st += 1;
        for(k=0;k<Nm;k++){
          for(j=0;j<Nm;j++){
            hess[k][j] = 0.0;
          }
        }
        for(k=0;k<Nm;k++){
          for(j=k;j<Nm;j++){
            Ski = 0.0;
            Sji = 0.0;
            SkiSji = 0.0;
            for(ii=0;ii<cells[num].num_inside;ii++){
              i = cells[num].indices_inside[ii];
              Ski += pow(U1[i],d1[k]) * pow(U2[i],d2[k]) * pow(U3[i],d3[k])*W[i];
              Sji += pow(U1[i],d1[j]) * pow(U2[i],d2[j]) * pow(U3[i],d3[j])*W[i];
              SkiSji += pow(U1[i],d1[k]) * pow(U2[i],d2[k]) * pow(U3[i],d3[k])*pow(U1[i],d1[j]) * pow(U2[i],d2[j]) * pow(U3[i],d3[j])*W[i];
            }
            hess[k][j] =SkiSji/sumW - Ski/sumW*p[j] - Sji/sumW*p[k]+p[j]*p[k];
          }
        }
        for(k=0;k<Nm;k++){
          for(j=0;j<k;j++){
            hess[k][j] = hess[j][k];
          }

          for(j=0; j<Nm; j++)
            lam[j]= lam[j] - x[j];
          // correct the weights
          for(ii=0;ii<cells[num].num_inside;ii++){
            dummy = 0.0;
            i = cells[num].indices_inside[ii];
            for(j=0; j<Nm; j++){
              Sji = (pow(U1[i],d1[j]) * pow(U2[i],d2[j]) * pow(U3[i],d3[j]) - p[j]);
              //if(peq[j]>1.0)
              //  Sji = Sji/peq[j];
              dummy += lam[j]*Sji;
            }
            dummy = exp(-dummy);
            W[i] = (1./dummy)*Wprior[i];//W0;//Wold[i];
          }
        }
        if(st>max_st)
          max_st = st;
        }
      }

  for(ii=0;ii<cells[num].num_inside;ii++){
    i = cells[num].indices_inside[ii];
    W[i] = Wold[i];
  }
  return st;
}
*/