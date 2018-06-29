settings.outformat="pdf";

bool hybrid=true;
int N=1024;

int ceilquotient(int a, int b)
{
  return (a+b-1) # b;
}

pair[] fft0(pair[] f, int sign=1)
{
  int N=f.length;
  if(N == 1) return new pair[] {f[0]};
  if(N == 2) 
    return new pair[] {f[0]+f[1],f[0]-f[1]};

  pair zeta=expi(2*pi*sign/N);
  int h=N # 2;

  pair[] even=fft0(f[0:h]+f[h:N]);

  for(int k=0; k < h; ++k)
    f[k]=(f[k]-f[k+h])*zeta^k;

  pair[] odd=fft0(f[0:h]);

  for(int l=0; l < h; ++l)
    f[2*l]=even[l];
  for(int l=0; l < h; ++l)
    f[2*l+1]=odd[l];

  return f;
}

// Compute h_j=sum_k f_k g_{j-k}
pair[] convolve0(pair[] f, pair[] g)
{
  pair[] u=new pair[f.length];
  pair[] v=new pair[f.length];
  
  int m=f.length;
  int n=2m;

  pair zeta=exp(2*pi*I/n);

  for(int k=0; k < m; ++k) {
    pair zetak=zeta^k;
    u[k]=zetak*f[k];
    v[k]=zetak*g[k];
  }
  
  f=fft(fft(f,1)*fft(g,1),-1);
  u=fft(fft(u,1)*fft(v,1),-1);

  for(int k=0; k < m; ++k) {
    f[k]=(f[k]+zeta^(-k)*u[k])/n;
  }
  return f;
}

// Compute h_j=sum_k f_k g_{j-k} where g[k]=expi(-pi*a*k^2)
pair[] convolve0(pair[] f, real a)
{
  static int lastm=0;
  static real lasta=0;
  static pair[] v,g;
  int m=f.length;
  
  pair zeta=exp(pi*I/m);

  if(m != lastm || a != lasta) {
    lastm=m;
    lasta=a;
    g=new pair[m];
    v=new pair[m];

    v[0]=g[0]=1.0;
    for(int k=1; k < m; ++k) {
      pair gk=expi(-pi*a*k^2);
      g[k]=gk;
      v[k]=gk*zeta^k;
    }

    g=fft(g,-1);
    v=fft(v,-1);
  }

  pair[] u=new pair[m];
  u[0]=f[0];

  for(int k=1; k < m; ++k)
    u[k]=zeta^k*f[k];
  
  u=fft(fft(u,-1)*v,1);
  f=fft(fft(f,-1)*g,1);

  for(int k=0; k < m; ++k)
    f[k]=(f[k]+zeta^(-k)*u[k])/(2*m);

  u.delete();
  return f;
}

// Special case of convolution used for fractional FFT
// g[k]=expi(-pi*a*k^2) with symmetry g(k-j)=g(j-k).
pair[] convolve0sym(pair[] f, real a)
{
  static int lastm=0;
  static real lasta=0;
  static pair[] v,g;
  int m=f.length;
  pair[] u=new pair[m];

  if(m != lastm || a != lasta) {
    lastm=m;
    lasta=a;
    g=new pair[m];
    v=new pair[m];

    v[0]=1;
    for(int k=1; k < m; ++k)
      v[k]=expi(-pi*a*k^2);

    int h=m # 2;
    g[0]=1;
    for(int k=1; k <= h; ++k)
      g[k]=v[k]+v[m-k];
    for(int k=1; k <= h; ++k) 
      g[m-k]=g[k]; // TODO: optimize FFT

    for(int k=1; k <= h; ++k)
      v[k]=expi(pi*k/m)*(v[k]-v[m-k]);

    for(int k=1; k <= h; ++k)
      v[m-k]=expi(-2*pi*k/m)*v[k]; // TODO: optimize FFT

    g=fft(g,-1);
    v=fft(v,-1);
  }

  pair[] u=new pair[m];
  u[0]=f[0];

  for(int k=1; k < m; ++k)
    u[k]=expi(pi*k/m)*f[k];
  
  u=fft(fft(u,-1)*v,1);
  f=fft(fft(f,-1)*g,1);

  for(int k=0; k < m; ++k)
    f[k]=(f[k]+expi(-pi*k/m)*u[k])/(2*m);

  u.delete();
  return f;
}

// Direct partial FFT
pair[] direct(pair[] f, int N=f.length)
{
  pair zeta=expi(2*pi/N);

  pair[] F=new pair[N];

  for(int j=0; j < f.length; ++j) {
    pair sum=0;
    for(int k=0; k <= j; ++k) sum += zeta^(j*k)*f[k];
    F[j]=sum;
  }
  return F;
}	

pair[] direct(pair[] f, real a, real b, int N=f.length)
{
  pair zeta=expi(2*pi/N);

  pair[] F=new pair[N];

  for(int j=0; j < f.length; ++j) {
    pair sum=0;
    for(int k=0; k <= min(a*j+b,f.length-1); ++k) sum += zeta^(j*k)*f[k];
    F[j]=sum;
  }
  return F;
}	

pair[] fractdirect(pair[] f, real a)
{
  int N=f.length;
  pair zeta=expi(2*pi*a);

  pair[] F=new pair[N];

  for(int j=0; j < f.length; ++j) {
    pair sum=0;
    for(int k=0; k < N; ++k) sum += zeta^(j*k)*f[k];
    F[j]=sum;
  }
  return F;
}	

// Return the fractional Fourier transform of f (a=1/N is usual FFT).
pair[] fractfft(pair[] f, real a)
{
  int N=f.length;
  pair[] z=sequence(new pair(int k) {return expi(pi*a*k^2);},N);
  return z*convolve0sym(z*f,a);
}

// Return the rectangular M x pM fractional Fourier transform of f
pair[] fractfftV(pair[] f, int p, real a)
{
  int pM=f.length;
  int M=pM # p;
  assert(M*p == pM);
  
  pair[] F=array(M,0.0);
  pair[] g=new pair[M];

  for(int r=0; r < p; ++r) {
    for(int m=0; m < M; ++m)
      g[m]=f[p*m+r];
    g=fractfft(g,a*p);
    for(int j=0; j < M; ++j)
      F[j] += expi(2*pi*a*j*r)*g[j];
  }
  return F;
}

// Return the rectangular qN x N fractional Fourier transform of f
pair[] fractfftH(pair[] f, int q, real a)
{
  int N=f.length;
  
  pair[] F=new pair[q*N];
  pair[] g=new pair[N];

  for(int r=0; r < q; ++r) {
    for(int k=0; k < N; ++k)
      g[k]=expi(2*pi*a*r*k)*f[k];
    g=fractfft(g,a*q);
    for(int m=0; m < N; ++m)
      F[q*m+r]=g[m];
  }
  return F;
}

// Return the rectangular M x N fractional Fourier transform of f
pair[] fractfft(pair[] f, int M, real a)
{
  int N=f.length;
  if(N > M) {
    int p=ceilquotient(N,M);
    return fractfftV(concat(f,array(p*M-N,0.0)),p,a);
  } else {
    int q=ceilquotient(M,N);
    return fractfftH(f,q,a)[0:M];
  }
}

// Return the partial Fourier transform of f for c(j)=pj+s for s >= 0.
pair[] partfftU(pair[] f, int M, real a, int p, int s)
{
  int N=f.length;
  if(p == 0) return fractfft(f[0:s+1],M,a);
  pair zeta=expi(2*pi*a);
  pair[] g=sequence(new pair(int k) {return zeta^(k^2/(2*p));},N);
  pair[] h=convolve0(sequence(new pair(int k) {
        return g[k]*zeta^(-s*k/p)*f[k];},
      N),a/p);
  return sequence(new pair(int j) {int n=p*j+s; return g[n]*h[n];},M);
}


// Return the partial Fourier transform of f for c(j)=(j+s)/q for s >= 0.
pair[] partfftL(pair[] f, int M, real a, int q, int s)
{
  int sign=sgn(q);
  q=abs(q);
  int N=f.length;
  pair zeta=expi(2*pi*a);
  pair[] g=sequence(new pair(int k) {return zeta^(sign*q*k^2/2);},N);
  int R=min(M,q); // Number of remainders
  pair[][] h=new pair[R][N];
  for(int j=0; j < R; ++j) {
    //    int w=sign*((sign*j) % q-s);
    int w=j+(j*sign >= 0 ? -sign*s : s-q);
    h[j]=convolve0(sequence(new pair(int k) {
          return g[k]*zeta^(w*k)*f[k];},
        N),sign*a*q);
  }

  return sequence(new pair(int j)
                  {int n=(sign*j+s) # q; return g[n]*h[j % q][n];},M);
}

pair[] f=1+sequence(N);
precision(6);

int[] c=new int[N];
for(int j=0; j < N; ++j)
  // c[j]=min(floor(101j#100),N-1);
  //  c[j]=N-1;
  //  c[j]=N-1-j;
  //     c[j]=quotient(N,3);
     c[j]=j;
//  c[j]=floor((N-1)*sin(pi*j/(N-1)));

//for(int j=0; j < 8; ++j)
//  c[j]=quotient(14*j,7);
//for(int j=8; j < N; ++j)
//  c[j]=quotient(14*(15-j),7);

//for(int j=0; j < quotient(N+1,2); ++j)
//  c[j]=2*j;
//for(int j=quotient(N+1,2); j < N; ++j)
//  c[j]=2*(N-1-j);

//write(c);

pair[] direct(pair[] f, int N=f.length)
{
  pair zeta=expi(2*pi/N);

  pair[] F=new pair[N];

  for(int j=0; j < f.length; ++j) {
    pair sum=0;
    for(int k=0; k <= c[j]; ++k) sum += zeta^(j*k)*f[k];
    F[j]=sum;
  }
  return F;
}

cputime();

write("Direct:");
pair[] d;
d=direct(f);
//write(d);
write(cputime());

//int Q=1;
//Q=min(Q,N # 2);

write();
write("Partfft:");

size(1000);

int count=0;

pen fullpen=blue;
pen straightpen=red;

pair zeta=expi(2*pi/N);

pair[] F=array(N,0.0);

void Rectangle(int x0, int x1, int y0, int y1) {
  draw(box((x0,y0),(x1-1,y1-1)),fullpen);
  ++count;
  int K=y1-y0;
  int M=x1-x0;
  pair[] g=new pair[K];
  for(int k=0; k < K; ++k)
    g[k]=zeta^(x0*k)*f[y0+k];
  pair G[]=fractfft(g,M,1/N);
  for(int j=0; j < M; ++j)
    F[x0+j] += zeta^((x0+j)*y0)*G[j];
}

void Trapeozoid(int x0, int x1, int y0, int y1, int p, int q) {
  //  if(x1 == x0+1 && y1 == y0+1) {// Single point optimization
  //        F[x0] += zeta^(x0*y0)*f[y0];
  //        return;
  //  }
  bool frac=x1 == x0+1 || y1 == y0+1 || p == 0;
  if(frac) draw(box((x0,y0),(x1-1,y1-1)),fullpen);
  else
    draw((x0,y0)--(x0,c[x0])--(x1-1,c[x1-1])--(x1-1,y0)--cycle,straightpen);
  //     label((string) (p,q),(x0,y0),NE,straightpen);
  ++count;
  int K=y1-y0;
  int M=x1-x0;
  pair[] g=new pair[K];
  for(int k=0; k < K; ++k)
    g[k]=zeta^(x0*k)*f[y0+k];
  pair G[];

  if(p == 1)
    G=partfftL(g,M,1/N,q,abs(q)*(c[x0]-y0));
  else
    G=partfftU(g,M,1/N,p,c[x0]-y0);

  for(int j=0; j < M; ++j)
    F[x0+j] += zeta^((x0+j)*y0)*G[j];
}

void partition(int x0, int x1, int y0, int y1)
{
  bool straight=hybrid;
  bool empty=true;
  int left=x0;
  int right=x1-1;
  
  for(int j=x0; j < x1; ++j) {
    int cj=c[j];
    if(y0 <= cj && cj < y1) {
      left=j;
      empty=false;
      break;
    }   
  }

  for(int j=x1-1; j >= left; --j) {
    int cj=c[j];
    if(y0 <= cj && cj < y1) {
      right=j;
      break;
    }
  }

  int p=c[x1-1]-c[x0];
  int q=x1-1-x0;
  if(abs(p) > q) {p=p#q; q=1;}
  else if(q >= abs(p) && p != 0) {q=q#p; p=1;}
  int Q=abs(q);
  int P=sgn(q)*p;

  if(straight) {
    for(int j=left; j <= right; ++j) {
      int r=c[left]*Q+(j-left)*P-c[j]*Q;
      if(r < 0 || r >= Q) {
        straight=false;
        break;
      }
    }
  } else {
    bool Below=false;
    bool Above=false;

    for(int j=x0; j < x1; ++j) {
      int cj=c[j];
      if(y0 <= cj) Below=true;
      if(cj < y1) Above=true;
    }
    if(Below && Above) empty=false;
  }
  
  if(straight || empty || x0 >= x1-1 || y0 >= y1-1) {
    if(x0 < x1 && y0 < y1) {
      if(empty) {
        if(y1 <= c[x0])
          Rectangle(x0,x1,y0,y1);
      } else {
        if(x0 < left) {
          if(y1 <= c[x0])
            Rectangle(x0,left,y0,y1);
        }
        Trapeozoid(left,right+1,y0,min(max(c[left],c[right])+1,y1),p,q);
        if(right+1 < x1 && y1-1 <= c[right+1])
          Rectangle(right+1,x1,y0,y1);

        if(false)
        for(int j=x0; j < x1; ++j)
          for(int k=y0; k < y1; ++k)
            if(k <= c[j])
              draw((j,k),darkgreen+1mm+opacity(0.1));
      }
    }
    return;
  }
  

  if(y1-y0 <= x1-x0) {
    int xm=(x0+x1)#2;
    partition(x0,xm,y0,y1);
    partition(xm,x1,y0,y1);
  } else {
    int ym=(y0+y1)#2;
    partition(x0,x1,y0,ym);
    partition(x0,x1,ym,y1);
  }
}

// for(int j=0; j < N; ++j)
for(int j=0; j < N; j += 16)
  draw((j,c[j]),black+4mm);

cputime();
partition(0,N,0,N);
write();
write(cputime());
write();
//write(F);
write();

write();
write("L2 norm:");

//for(int j=0; j < N; ++j)
//  write((string) j," ",format("%.3f",abs(d[j]-F[j])));
write(sqrt(sum(abs(d-F)^2)));
write();
write("Count:");
write(count);
