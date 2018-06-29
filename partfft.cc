unsigned int N;
//unsigned int N=1024;
//unsigned int N=8192;
//unsigned int N=65536;
//unsigned int N=1048576;

double factorseconds=0.01; // Time limit for factorization tests
double testseconds=1.0; // Time limit for timing tests

#include <inttypes.h>

#include "Complex.h"
#include "convolution.h"
#include "tests/utils.h"
#include "seconds.h"

using namespace std;
using namespace utils;
using namespace fftwpp;

const double pi=M_PI;
const Complex I(0.0,1.0);

inline uint32_t log2(uint32_t n)
{
  return 31-__builtin_clz(n);
}

inline uint32_t ceillog2(uint32_t n)
{
  uint32_t m=log2(n);
  return n == (1u << m) ? m : m+1;
}

unsigned int BuildZeta(unsigned int m, double alpha,
                       Complex *&ZetaH, Complex *&ZetaL,
                       bool factor=true)
{
  unsigned int s=(int) sqrt((double) m);
  unsigned int t=m/s;
  if(s*t < m) ++t;
  double arg=twopi*alpha;
  if(factor) {
    ZetaH=ComplexAlign(t);
  
//#ifndef FFTWPP_SINGLE_THREAD
//#pragma omp parallel for num_threads(threads)
//#endif    
    for(unsigned int a=0; a < t; ++a) {
      double theta=s*a*arg;
      ZetaH[a]=Complex(cos(theta),sin(theta));
    }
  } else {
    ZetaH=NULL;
    s=m;
  }
  ZetaL=ComplexAlign(s);
//#ifndef FFTWPP_SINGLE_THREAD
//#pragma omp parallel for num_threads(threads)
//#endif
  for(unsigned int b=0; b < s; ++b) {
    double theta=b*arg;
    ZetaL[b]=Complex(cos(theta),sin(theta));
  }
  return s;
}

fft1d **FFT;

// Return the gcd of a >= b;
inline unsigned int gcd(unsigned int a, unsigned int b)
{
  do {  
    int r=a % b;
    a=b;
    b=r;
  } while(b != 0);
  return a;
}

template<class K, class T, class L>
class Table {
public:
  typedef std::map<K,T,L> M;
  M table;
  T Lookup(K key) {
    typename M::iterator p=table.find(key);
    if(p == table.end()) {
      key.Reduce();
      T e(key);
      table[key]=e;
      return e;
    }
    return p->second;
  }
  T Lookup(K key, unsigned int m) {
    typename M::iterator p=table.find(key);
    if(p == table.end()) {
      key.Reduce();
      T e(key,m);
      table[key]=e;
      return e;
    }
    return p->second;
  }
};

struct Keyii {
  int a;
  unsigned int b;
  Keyii(int a, unsigned int b) : a(a), b(b) {}
  void Reduce() {
    unsigned int d=gcd(b,abs(a));
    a /= d;
    b /= d;
  }
};
  
struct Lessii {
  bool operator()(const Keyii& a, const Keyii& b) const {
    return a.a*b.b < a.b*b.a;
  }
};

class Exp {
public:  
  unsigned int b;
  Complex *ZetaH,*ZetaL;
  unsigned int s;
  bool factor;
  
  Exp() {ZetaH=ZetaL=NULL;}
  Exp(Keyii key) {
    b=key.b;
    
    s=BuildZeta(b,(double) key.a/b,ZetaH,ZetaL,true);
    double T=time();
    
    Complex *ZetaH0=ZetaH;
    Complex *ZetaL0=ZetaL;
    unsigned int s0=s;
    
    s=BuildZeta(b,(double) key.a/b,ZetaH,ZetaL,false);
    double F=time();
    
    factor=T <= F;
    if(factor) {
      Deallocate(ZetaH,ZetaL);
      ZetaH=ZetaH0;
      ZetaL=ZetaL0;
      s=s0;
    } else
      Deallocate(ZetaH0,ZetaL0);
  }
  
  double time() {
    unsigned int C=0;
    double sum=0.0;
    static Complex Sum=0.0;
    for(;sum < factorseconds;++C) {
      double start=utils::totalseconds();
      for(unsigned int k=0; k < b; ++k)
        Sum += exp(k*k);
      double seconds=totalseconds()-start;
      sum += seconds;
    }
    return sum/C;
  }

  inline Complex exp(unsigned int k) {
    if(k >= b) k %= b;
    if(k < s) return ZetaL[k];
    unsigned int H=k/s;
    unsigned int L=k-H*s;
    return ZetaH[H]*ZetaL[L];
  }
  
  inline Vec Vexp(unsigned int k) {
    if(k >= b) k %= b;
    if(k < s) return LOAD(ZetaL+k);
    unsigned int H=k/s;
    unsigned int L=k-H*s;
    return ZMULT(LOAD(ZetaH+H),LOAD(ZetaL+L));
  }
  
  inline Vec Vexp1(unsigned int k) {
    return LOAD(ZetaL+(k % b));
  }
  
  void Deallocate(Complex *ZetaH, Complex *ZetaL) {
    if(ZetaL) deleteAlign(ZetaL);
    if(ZetaH) deleteAlign(ZetaH);
  }
};
  
Table<Keyii,Exp,Lessii> Exptable;

class Frac {
public:  
  Complex *g,*v;
  
  Frac() {}
  Frac(Keyii key, unsigned int m) {
    g=ComplexAlign(m);
    v=ComplexAlign(m);
  
    unsigned int l=ceillog2(m);
    if(!FFT[l])
      FFT[l]=new fft1d(m,1,g);
  
    v[0]=g[0]=1.0;
    unsigned int h=m/2;
    Exp V=Exptable.Lookup(Keyii(key.a,2*key.b));
    for(unsigned int k=1; k < m; k++)
      v[k]=conj(V.exp(k*k));
    
    for(unsigned int k=1; k <= h; ++k)
      g[k]=v[k]+v[m-k];
    for(unsigned int k=1; k <= h; ++k) 
      g[m-k]=g[k];

    Exp W=Exptable.Lookup(Keyii(1,2*m));
    for(unsigned int k=1; k <= h; ++k) {
      Complex V=W.exp(k);
      v[k]=V*(v[k]-v[m-k]);
      v[m-k]=conj(V*V)*v[k];
    }

    FFT[l]->fft(g);
    FFT[l]->fft(v);
  }
};

class Part : public Frac {
public:  
  
  Part() {}
  Part(Keyii key, unsigned int m) {
    g=ComplexAlign(m);
    v=ComplexAlign(m);
  
    unsigned int l=ceillog2(m);
    if(!FFT[l])
      FFT[l]=new fft1d(m,1,g);
  
    v[0]=g[0]=1.0;

    Exp G=Exptable.Lookup(Keyii(key.a,2*key.b));
    Exp V=Exptable.Lookup(Keyii(1,2*m));
    for(unsigned int k=1; k < m; k++) {
      Complex gk=conj(G.exp(k*k));
      g[k]=gk;
      v[k]=gk*V.exp(k);
    }
    
    FFT[l]->fft(g);
    FFT[l]->fft(v);
  }
};
  
typedef Table<Keyii,Part,Lessii> PartTable;
PartTable **Parttable;

Frac P;

typedef Table<Keyii,Frac,Lessii> FracTable;
FracTable **Fractable;

void init(Complex *f, unsigned int m) 
{
  for(unsigned int k=0; k < m; k++)
    f[k]=k+1;
}

void mult(Complex **F, unsigned int m,
          const unsigned int indexsize,
          const unsigned int *index,
          unsigned int r, unsigned int threads)
{
  Complex* F1=r == 0 ? P.g : P.v;
  Complex* F0=F[0];
  
#ifdef __SSE2__
  PARALLEL(
    for(unsigned int j=0; j < m; ++j) {
      Complex *p=F0+j;
      STORE(p,ZMULT(LOAD(p),LOAD(F1+j)));
    }
    );
#else
  PARALLEL(
    for(unsigned int j=0; j < m; ++j)
      F0[j] *= F1[j];
    );
#endif
}

ImplicitConvolution **C;

unsigned int N0=0;
unsigned int N1=0;
unsigned int N2=0;
unsigned int N3=0;

Complex *g0;
Complex *g1;
Complex *g2;
Complex *g3;

inline Complex *alloc(Complex *&g, unsigned int& N0, unsigned int N)
{
  if(N <= N0) return g;
  deleteAlign(g);
  N0=N;
  g=ComplexAlign(N0);
  return g;
}

// Return the fractional Fourier transform of f (a=1/N is usual FFT) in F.
void fractfft(Complex *f, unsigned int N, Complex *F, int a, unsigned int b)
{
  unsigned int m=ceillog2(N);
  unsigned int L=1 << m;
  if(!C[m])
    C[m]=new ImplicitConvolution(L,1,1);
  Complex *g=alloc(g0,N0,L);
  Exp E=Exptable.Lookup(Keyii(a,2*b));
  if(E.factor) {
    for(unsigned int k=0; k < N; ++k) {
      Vec gk=E.Vexp(k*k);
      STORE(F+k,gk);
      STORE(g+k,ZMULT(gk,LOAD(f+k)));
    }
  } else {
    for(unsigned int k=0; k < N; ++k) {
      Vec gk=E.Vexp1(k*k);
      STORE(F+k,gk);
      STORE(g+k,ZMULT(gk,LOAD(f+k)));
    }
  }
  for(unsigned int k=N; k < L; ++k)
    g[k]=0.0;
  Complex *H[]={g};
// Phase fraction is A/B (1/N for full FFT).
  int A=a;
  unsigned int B=b;
  if(!Fractable[m])
    Fractable[m]=new FracTable;
  P=Fractable[m]->Lookup(Keyii(A,B),L);
  
  C[m]->convolve(H,mult);
  for(unsigned int k=0; k < N; ++k) {
    Complex *p=F+k;
    STORE(p,ZMULT(LOAD(p),LOAD(g+k)));
  }
}

// Return the rectangular M x N fractional Fourier transform of f.
void fractfft(Complex *f, unsigned int N, Complex *F, unsigned int M, int a,
              unsigned int b)
{
  Exp E=Exptable.Lookup(Keyii(a,b));
  unsigned int m=ceillog2(N);
  unsigned int L=1 << m;
  if(N > M) {
    unsigned int p=ceilquotient(N,M);
    Complex *G=alloc(g2,N2,M);
    Complex *g=alloc(g0,N0,max(L,M));
    
    unsigned int ap=a*p;
    unsigned int mstop=min(M,ceilquotient(N,p));
    for(unsigned int m=0; m < mstop; ++m)
      g[m]=f[p*m];
    for(unsigned int m=mstop; m < M; ++m)
      g[m]=0.0;
    fractfft(g,M,G,ap,b);
    for(unsigned int j=0; j < M; ++j)
      F[j]= G[j];

    for(unsigned int r=1; r < p; ++r) {
      unsigned int mstop=min(M,ceilquotient(N-r,p));
      for(unsigned int m=0; m < mstop; ++m)
        g[m]=f[p*m+r];
      for(unsigned int m=mstop; m < M; ++m)
        g[m]=0.0;
      fractfft(g,M,G,ap,b);
      if(E.factor)
        for(unsigned int j=0; j < M; ++j) {
          Complex *p=F+j;
          STORE(p,LOAD(p)+ZMULT(E.Vexp(j*r),LOAD(G+j)));
        }
      else
        for(unsigned int j=0; j < M; ++j) {
          Complex *p=F+j;
          STORE(p,LOAD(p)+ZMULT(E.Vexp1(j*r),LOAD(G+j)));
        }
    }
  } else {
    unsigned int q=ceilquotient(M,N);
    Complex *G=alloc(g2,N2,N);
    Complex *g=alloc(g0,N0,L);

    for(unsigned int k=0; k < N; ++k)
      g[k]=f[k];
    fractfft(g,N,G,a*q,b);
    unsigned int mstop=min(N,ceilquotient(M,q));
    for(unsigned int m=0; m < mstop; ++m)
      F[q*m]=G[m];
    for(unsigned int r=1; r < q; ++r) {
      if(E.factor)
        for(unsigned int k=0; k < N; ++k)
          STORE(g+k,ZMULT(E.Vexp(r*k),LOAD(f+k)));
      else
        for(unsigned int k=0; k < N; ++k)
          STORE(g+k,ZMULT(E.Vexp1(r*k),LOAD(f+k)));
      fractfft(g,N,G,a*q,b);
      unsigned int mstop=min(N,ceilquotient(M-r,q));
      for(unsigned int m=0; m < mstop; ++m)
        F[q*m+r]=G[m];
    }
  }
}

// Return the partial Fourier transform of f for c(j)=pj+s for s >= 0.
void partfftU(Complex *f, unsigned int N, Complex *F, unsigned int M,
              int a, unsigned int b, int p, unsigned int s)
{
  if(p == 0) {
    fractfft(f,s+1,F,M,a,b);
    return;
  }
  unsigned int m=ceillog2(N);
  unsigned int L=1 << m;
  if(!C[m])
    C[m]=new ImplicitConvolution(L,1,1);
  int A=p >= 0 ? a : -a;
  unsigned int B=b*abs(p);
  
  if(!Parttable[m])
    Parttable[m]=new PartTable;
  P=Parttable[m]->Lookup(Keyii(A,B),L);
  
  Complex *h0=alloc(g2,N2,L);
  Exp V=Exptable.Lookup(Keyii(A,2*B));
  if(V.factor) {
    for(unsigned int k=0; k < N; ++k) {
      int arg=((int) k-2*s)*k;
      if(arg > 0)
        STORE(h0+k,ZMULT(V.Vexp(abs(arg)),LOAD(f+k)));
      else
        STORE(h0+k,ZMULTC(V.Vexp(abs(arg)),LOAD(f+k)));
    }
  } else {
    for(unsigned int k=0; k < N; ++k) {
      int arg=((int) k-2*s)*k;
      if(arg > 0)
        STORE(h0+k,ZMULT(V.Vexp1(abs(arg)),LOAD(f+k)));
      else
        STORE(h0+k,ZMULTC(V.Vexp1(abs(arg)),LOAD(f+k)));
    }
  }
  for(unsigned int k=N; k < L; ++k)
    h0[k]=0.0;
  Complex *G[]={h0};
  
  C[m]->convolve(G,mult);
  
  Exp E=Exptable.Lookup(Keyii(A,2*B));
  if(E.factor) {
    for(unsigned int j=0; j < M; ++j) {
      int n=p*j+s;
      STORE(F+j,ZMULT(E.Vexp(n*n),LOAD(h0+n)));
    }
  } else {
    for(unsigned int j=0; j < M; ++j) {
      int n=p*j+s;
      STORE(F+j,ZMULT(E.Vexp1(n*n),LOAD(h0+n)));
    }
  }
}

// Return the partial Fourier transform of f for c(j)=(j+s)/q for s >= 0.
void partfftL(Complex *f, unsigned int N, Complex *F, unsigned int M,
              int a, unsigned int b, int q, unsigned int s)
{
  int sign=q > 0.0 ? 1 : -1;
  int Q=abs(q);
  unsigned int m=ceillog2(N);
  unsigned int L=1 << m;
  
  unsigned int R=min(M,Q); // Number of remainders
  Complex *h=alloc(g2,N2,L*R);
  if(!C[m])
    C[m]=new ImplicitConvolution(L,1,1);
  int A=a*q;
  unsigned int B=b;
  if(!Parttable[m])
    Parttable[m]=new PartTable;
  P=Parttable[m]->Lookup(Keyii(A,B),L);
  
  Exp V=Exptable.Lookup(Keyii(a,2*b));
  for(unsigned int j=0; j < R; ++j) {
    Complex *hj=h+L*j;
    int w=2*(j-sign*s-((int) j*sign < 0)*Q);
    if(V.factor) {
      for(unsigned int k=0; k < N; ++k) {
        int arg=(q*k+w)*k;
        if(arg > 0)
          STORE(hj+k,ZMULT(V.Vexp(abs(arg)),LOAD(f+k)));
        else
          STORE(hj+k,ZMULTC(V.Vexp(abs(arg)),LOAD(f+k)));
      } 
    } else {
      for(unsigned int k=0; k < N; ++k) {
        int arg=(q*k+w)*k;
         if(arg > 0)
          STORE(hj+k,ZMULT(V.Vexp1(abs(arg)),LOAD(f+k)));
        else
          STORE(hj+k,ZMULTC(V.Vexp1(abs(arg)),LOAD(f+k)));
      }
    }
    for(unsigned int k=N; k < L; ++k)
      hj[k]=0.0;
    Complex *G[]={hj};
    C[m]->convolve(G,mult);
  }

  Exp E=Exptable.Lookup(Keyii(a*q,2*b));
  if(E.factor) {
    for(unsigned int j=0; j < M; ++j) {
      int p=(sign*j+s);
      int n=(p > 0 ? p : p-(Q-1))/Q;
      STORE(F+j,ZMULT(E.Vexp(n*n),LOAD(h+(j % Q)*L+n)));
    }
  } else {
    for(unsigned int j=0; j < M; ++j) {
      int p=(sign*j+s);
      int n=(p > 0 ? p : p-(Q-1))/Q;
      STORE(F+j,ZMULT(E.Vexp1(n*n),LOAD(h+(j % Q)*L+n)));
    }
  }
}

unsigned int *c;
Complex *ZetaL;

void direct(Complex *f, Complex *F)
{
  for(unsigned int j=0; j < N; ++j) {
    Vec sum=LOAD(0.0);
    for(unsigned int k=0; k <= c[j]; ++k)
      sum += ZMULT(LOAD(ZetaL+((j*k) % N)),LOAD(f+k));
    STORE(F+j,sum);
  }
}

Complex *f;
Complex *F;
Exp E;

unsigned int Count;

Complex factor=I*2.0*pi/N;

void Rectangle(int x0, int x1, int y0, int y1)
{
  ++Count;
  unsigned int K=y1-y0;
  unsigned int M=x1-x0;
  Complex *g=alloc(g1,N1,K);
  Complex *f0=f+y0;
  if(E.factor)
    for(unsigned int k=0; k < K; ++k)
      STORE(g+k,ZMULT(E.Vexp(x0*k),LOAD(f0+k)));
  else 
    for(unsigned int k=0; k < K; ++k)
      STORE(g+k,ZMULT(E.Vexp1(x0*k),LOAD(f0+k)));
  Complex *G=alloc(g3,N3,M);
  
  fractfft(g,K,G,M,1,N);
  
  Complex *F0=F+x0;
  if(E.factor)
    for(unsigned int j=0; j < M; ++j) {
      Complex *p=F0+j;
      STORE(p,LOAD(p)+ZMULT(E.Vexp(((x0+j)*y0)),LOAD(G+j)));
    }
  else
    for(unsigned int j=0; j < M; ++j) {
      Complex *p=F0+j;
      STORE(p,LOAD(p)+ZMULT(E.Vexp1(((x0+j)*y0)),LOAD(G+j)));
    }
}

void Trapezoid(int x0, int x1, int y0, int y1, int p, int q)
{
  ++Count;
  unsigned int K=y1-y0;
  unsigned int M=x1-x0;
  Complex *g=alloc(g1,N1,K);
  Complex *f0=f+y0;
  if(E.factor)
    for(unsigned int k=0; k < K; ++k)
      STORE(g+k,ZMULT(E.Vexp(x0*k),LOAD(f0+k)));
  else 
    for(unsigned int k=0; k < K; ++k)
      STORE(g+k,ZMULT(E.Vexp1(x0*k),LOAD(f0+k)));
  Complex *G=alloc(g3,N3,M);

  if(p == 1)
    partfftL(g,K,G,M,1.0,N,q,abs(q)*(c[x0]-y0));
  else
    partfftU(g,K,G,M,1.0,N,p,c[x0]-y0);
  
  Complex *F0=F+x0;
  if(E.factor)
    for(unsigned int j=0; j < M; ++j) {
      Complex *p=F0+j;
      STORE(p,LOAD(p)+ZMULT(E.Vexp(((x0+j)*y0)),LOAD(G+j)));
     }
  else
    for(unsigned int j=0; j < M; ++j) {
      Complex *p=F0+j;
      STORE(p,LOAD(p)+ZMULT(E.Vexp1(((x0+j)*y0)),LOAD(G+j)));
    }
}

bool hybrid=true;

void partition(unsigned int x0, unsigned int x1, unsigned int y0,
               unsigned int y1)
{
  bool straight=hybrid;
  bool empty=true;
  bool Below=false;
  bool Above=false;
  unsigned int left=x0;
  unsigned int right=x1-1;

  for(unsigned int j=x0; j < x1; ++j) {
    unsigned int cj=c[j];
    bool below=y0 <= cj;
    bool above=cj < y1;
    if(below) Below=true;
    if(above) Above=true;
    if(below && above) {
      left=j;
      empty=false;
      break;
    }
  }

  int p=c[x1-1]-c[x0];
  int q=x1-1-x0;
  if(empty) {
    if(Below && Above) {empty=straight=false;}
  } else {
    for(unsigned int j=x1; j-- > left;) { // Loop from x1-1 to left
      unsigned int cj=c[j];
      if(y0 <= cj && cj < y1) {
        right=j;
        break;
      }
    }

    // Round slope to nearest integer or reciprocal integer.
    if(abs(p) > q) {p=(p+(p > 0 ? q : -q)/2)/q; q=1;}
    else if(q >= abs(p) && p != 0) {q=(q+(p > 0 ? p : -p)/2)/p; p=1;}
    int Q=abs(q);
    int P=q > 0 ? p : -p;

    if(straight) {
      for(unsigned int j=left; j <= right; ++j) {
        int r=c[left]*Q+(j-left)*P-c[j]*Q;
        if(r < 0 || r >= Q) {
          straight=false;
          break;
        }
      }
    }
  }
  
  if(straight || empty || x0 >= x1-1 || y0 >= y1-1) {
    if(x0 < x1 && y0 < y1) {
      if(empty) {
        if(y1 <= c[x0])
          Rectangle(x0,x1,y0,y1);
      } else {
        if(x0 < left && y1 <= c[x0])
          Rectangle(x0,left,y0,y1);
        Trapezoid(left,right+1,y0,min(max(c[left],c[right])+1,y1),p,q);
        if(right+1 < x1 && y1-1 <= c[right+1])
          Rectangle(right+1,x1,y0,y1);
      }
    }
    return;
  }

  if(y1-y0 <= x1-x0) {
    unsigned int xm=(x0+x1)/2;
    partition(x0,xm,y0,y1);
    partition(xm,x1,y0,y1);
  } else {
    unsigned int ym=(y0+y1)/2;
    partition(x0,x1,y0,ym);
    partition(x0,x1,ym,y1);
  }
}

inline void partition0(unsigned int x0, unsigned int x1, unsigned int y0,
                unsigned int y1)
{
  Count=0;
  for(unsigned int j=0; j < N; ++j)
    F[j]=0.0;
  partition(x0,x1,y0,y1);
}

double time()
{
  unsigned int C=0;
  double sum=0.0;
  for(;sum < testseconds;++C) {
    double start=utils::totalseconds();
    partition0(0,N,0,N);
    double seconds=totalseconds()-start;
    sum += seconds;
  }
  return sum/C;
}

double timeFFT()
{
  fft1d Backward(N,1);
  unsigned int C=0;
  double sum=0.0;
  for(;sum < testseconds;++C) {
    double start=utils::totalseconds();
    Backward.fft(f);
    double seconds=totalseconds()-start;
    sum += seconds;
  }
  return sum/C;
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=1;//get_max_threads();

  cout << "1d partial FFT:" << endl;
  
  cout << "N\tTr\t Th\t Cr\t Ch\t Td/Th\t Tr/Th\tTh/Tf" << endl;    
    
  for(N=1024; N <= 1048576; N *= 2) {
  
#ifndef __SSE2__
    fftw::effort |= FFTW_NO_SIMD;
#endif  
  
    unsigned int L=ceillog2(N);  
  
    C=new ImplicitConvolution*[L];
    for(unsigned int i=0; i < L; ++i)
      C[i]=NULL;
  
    FFT=new fft1d*[L];
    for(unsigned int i=0; i < L; ++i)
      FFT[i]=NULL;

    Parttable=new PartTable*[L];
    for(unsigned int i=0; i < L; ++i)
      Parttable[i]=NULL;
  
    Fractable=new FracTable*[L];
    for(unsigned int i=0; i < L; ++i)
      Fractable[i]=NULL;
  
    c=new unsigned int[N];
    for(unsigned int j=0; j < N; ++j)
//        c[j]=j;
      //  c[j]=N-1-j;
      //     c[j]=quotient(N,3);
      c[j]=floor((N-1)*sin(j*pi/(N-1)));

    // allocate arrays:
    f=ComplexAlign(N);
  
    init(f,N);
  
//    cout << "\ninput:" << endl;
//  for(unsigned int i=0; i < N; i++) 
//    cout << f[i] << endl;
//  cout << endl;
  
    F=ComplexAlign(N);
  
    for(unsigned int j=0; j < N; ++j)
      F[j]=0.0;
  
    E=Exptable.Lookup(Keyii(1,N));

    hybrid=false;
    partition0(0,N,0,N);
  
    double Tr=time();
    unsigned int Cr=Count;
    
    hybrid=true;
    partition0(0,N,0,N);
    
    double Th=time();
    unsigned int Ch=Count;
  
#if 0
    cout << "\noutput:" << endl;
    for(unsigned int j= 0; j < N; ++j)
      cout << F[j] << endl;
    cout << endl;
#endif  
  
//    cout << "Th=" << Th << endl;
  
    double Td=0.0;
    if(N <= 8192 || true) {
      Complex *d=ComplexAlign(N);
      Complex *ZetaH;
      BuildZeta(N,1.0/N,ZetaH,ZetaL,false);
      seconds();
      direct(f,d);
      Td=seconds();
      deleteAlign(ZetaL);

//      cout << "Td=" << Td << endl;
//      cout << "Td/Th=" << (Th > 0.0 ? Td/Th : 0.0) << endl;
  
#if 0
      double sum=0.0;
      for(unsigned int j=0; j < N; ++j)
        sum += abs2(d[j]-F[j]);
      cout << endl;
      cout << "L2 norm=" << sqrt(sum) << endl;
#endif      
    }
  
//    cout << endl;
  
    double Tf=timeFFT();
//    cout << "Tf=" << Tf << endl;
//    cout << endl;
  
//    cout << "Count=" << Count << endl;
//    cout << "Th/Tf=" << (Tf > 0.0 ? Th/Tf : 0.0) << endl;
  
    cout.precision(3);
    cout << N << "\t" << Tr << "\t" << Th << "\t";
    cout << Cr << "\t" << Ch << "\t";
    cout << Td/Th << "\t" << Tr/Th << "\t" << Th/Tf << endl;
  }
  
  deleteAlign(f);

  return 0;
}
