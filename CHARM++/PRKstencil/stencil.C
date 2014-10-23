#include "stencil.decl.h"
#define EPSILON       1.e-8
#define COEFX         1.0
#define COEFY         1.0
#define TINDEX(i,j)   (i+RADIUS+(width+2*RADIUS)*(j+RADIUS))
#define temp(i,j)     temperature[TINDEX(i,j)]
#define TNINDEX(i,j)  (i+width*(j))
#define temp_new(i,j) new_temperature[TNINDEX(i,j)]
#define WEIGHT(i,j)   weight[i+RADIUS+(j+RADIUS)*(2*RADIUS+1)]
#define MAX(i,j)      ((i)>(j) ? (i) : (j))
#define MIN(i,j)      ((i)<(j) ? (i) : (j))
#define ABS(x)        ((x)<0.0 ? (-1.0*(x)) : (x))
#define LEFT          1111 
#define RIGHT         2222
#define BOTTOM        3333
#define TOP           4444

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int n; // array size
/*readonly*/ int overdecomposition; 
/*readonly*/ int maxiterations;
/*readonly*/ double weight[(2*RADIUS+1)*(2*RADIUS+1)];

// specify the number of worker chares in each dimension
/*readonly*/ int num_chare_rows;
/*readonly*/ int num_chare_cols;
/*readonly*/ double startTime, endTime;

class ghostMsg : public CMessage_ghostMsg {
public:
  int dir;
  int size;
  double *edge;
  ghostMsg(int _d, int _s) : dir(_d), size(_s) {
  }
};

class Main : public CBase_Main
{

public:
    CProxy_Stencil array;

    Main(CkArgMsg* m) {
        if (m->argc != 4) {
          CkPrintf("%s <maxiterations> <grid_size> <overdecomposition factor>\n", m->argv[0]);
          CkAbort("Abort");
        }

        // store the main proxy
        mainProxy = thisProxy;

        maxiterations = atoi(m->argv[1]);
        if (maxiterations < 1) {
          CkAbort("ERROR: maxiterations must be positive");
        }
        n = atoi(m->argv[2]);
        if (n < CkNumPes()) 
          CkAbort("ERROR: Grid size must be larger than  #PEs");

        overdecomposition = atoi(m->argv[3]);
        if (n < overdecomposition)
          CkAbort("ERROR: Grid size must be larger than overdecomposition");

        if (RADIUS < 0) {
          CkAbort("ERROR: Stencil radius %d should be non-negative\n");
        }
  
        if (2*RADIUS +1 > n) {
          CkAbort("ERROR: Stencil diameter exceeds grid size\n");
        }

        int min_size = n/MAX(CkNumPes(),overdecomposition);
        if (min_size<RADIUS) 
          CkAbort("Some tiles smaller than radius of difference stencil\n");

        num_chare_rows = CkNumPes();
        num_chare_cols = overdecomposition;

        // print info
        CkPrintf("Charm++ stencil execution on 2D grid\n");
        CkPrintf("Number of processes  = %d\n", CkNumPes());
        CkPrintf("Overdecomposition    = %d\n", overdecomposition);
        CkPrintf("Grid size            = %d\n", n);
        CkPrintf("Radius of stencil    = %d\n", RADIUS);
#ifdef STAR
        CkPrintf("Type of stencil      = star\n");
#else
        CkPrintf("Type of stencil      = compact\n");
        CkAbort("ERROR: Compact stencil not (yet) supported\n");
#endif
        CkPrintf("Number of iterations = %d\n", maxiterations);

        // Create new array of worker chares
        array = CProxy_Stencil::ckNew(num_chare_cols, num_chare_rows);

        /* fill the stencil weights to reflect a discrete divergence operator         */
        for (int j=-RADIUS; j<=RADIUS; j++) for (int i=-RADIUS; i<=RADIUS; i++)
					      WEIGHT(i,j) = 0.0;
        #ifdef STAR
          for (int i=1; i<=RADIUS; i++) {
            WEIGHT(0, i) = WEIGHT( i,0) =  (1.0/(2.0*i*RADIUS));
            WEIGHT(0,-i) = WEIGHT(-i,0) = -(1.0/(2.0*i*RADIUS));
          }
        #else
          stencil_size = (2*RADIUS+1)*(2*RADIUS+1);
          for (int j=1; j<=RADIUS; j++) {
            for (int i=-j+1; i<j; i++) {
              WEIGHT(i,j)  =  (1.0/(4.0*j*(2.0*j-1)*RADIUS));
              WEIGHT(i,-j) = -(1.0/(4.0*j*(2.0*j-1)*RADIUS));
              WEIGHT(j,i)  =  (1.0/(4.0*j*(2.0*j-1)*RADIUS));
              WEIGHT(-j,i) = -(1.0/(4.0*j*(2.0*j-1)*RADIUS));      
            }
            WEIGHT(j,j)    =  (1.0/(4.0*j*RADIUS));
            WEIGHT(-j,-j)  = -(1.0/(4.0*j*RADIUS));
          }
        #endif  

        //Start the computation
	array.run();
    }

    // Each worker reports back to here when it completes an iteration
    void report(double result) {
      double totalTime, flops;
      double reference_norm = (COEFX+COEFY)*(maxiterations+1);
      result /= (n-2.0*RADIUS)*(n-2.0*RADIUS);
      totalTime = endTime - startTime;
      flops = (double) (2*(4*RADIUS+1)+1) * (n-2*RADIUS)*(n-2*RADIUS)*maxiterations;
      CkPrintf("Solution norm: %lf\n", result);
      if (ABS(result-reference_norm) < EPSILON) {
        CkPrintf("Solution validates; reference norm: %lf\n", reference_norm);
        CkPrintf("Flops: %e\n", flops/totalTime);
      }
      else {
        CkPrintf("Solution does not validate; reference norm: %lf\n", reference_norm);
        CkPrintf("ABS(result-reference_norm) = %lf\n", ABS(result-reference_norm));
      }
      CkExit();
    }

};

class Stencil: public CBase_Stencil {
  Stencil_SDAG_CODE

public:
  int messages_due, max_messages_due;
  int iterations;

  double local_norm;
  int    istart, iend, jstart, jend, height, width; // global grid indices of tile
  double *temperature, *new_temperature;

  // Constructor, initialize values
  Stencil() {
    int i,j, iloc, jloc, leftover;
      
    /* compute amount of space required for input and solution arrays             */
    width = n/overdecomposition;
    leftover = n%overdecomposition;
    if (thisIndex.x < leftover) {
      istart = (width+1) * thisIndex.x; 
      iend = istart + width;
    }
    else {
      istart = (width+1) * leftover + width * (thisIndex.x-leftover);
      iend = istart + width - 1;
    }
    width = iend - istart + 1;

    height = n/CkNumPes();
    leftover = n%CkNumPes();
    if (thisIndex.y < leftover) {
      jstart = (height+1) * thisIndex.y; 
      jend = jstart + height;
    }
    else {
      jstart = (height+1) * leftover + height * (thisIndex.y-leftover);
      jend = jstart + height - 1;
    }
    height = jend - jstart + 1;

    // allocate two dimensional array
    temperature = new double[(height+2*RADIUS)*(width+2*RADIUS)];
    new_temperature = new double[height*width];
    max_messages_due = 4;
    if (thisIndex.x == 0               ) {max_messages_due--; }
    if (thisIndex.y == 0               ) {max_messages_due--; }
    if (thisIndex.x == num_chare_cols-1) {max_messages_due--; }
    if (thisIndex.y == num_chare_rows-1) {max_messages_due--; }
    messages_due = max_messages_due;

    for(j=jstart,jloc=0;j<=jend;j++,jloc++){
      for(i=istart,iloc=0;i<=iend;i++,iloc++){
        temp(iloc,jloc) = COEFX*i+COEFY*j;
        temp_new(iloc,jloc) = 0.0;
      }
    }
  }

  // a necessary function, which we ignore now. If we were to use load balancing 
  // and migration this function might become useful
    Stencil(CkMigrateMessage* m) {}

    ~Stencil() { 
      delete [] temperature; 
      delete [] new_temperature; 
    }

    // Perform one iteration of work
    // The first step is to send the local state to the neighbors
    void begin_iteration(void) {
      int k;

        // Send my left edge
        if (thisIndex.x > 0) {
          ghostMsg *msg = new (height*RADIUS) ghostMsg(LEFT, height);
          CkSetRefNum(msg, iterations);
          for(int j=0, k=0;j<height;++j) for (int i=0; i<RADIUS; i++)
					    msg->edge[k++]    = temp(i,j);
          thisProxy(thisIndex.x-1, thisIndex.y).receiveGhosts(msg);
        }

	// Send my right edge
        if (thisIndex.x < num_chare_cols-1) {
          ghostMsg *msg = new (height*RADIUS) ghostMsg(RIGHT, height);
          CkSetRefNum(msg, iterations);
          for(int j=0, k=0;j<height;++j) for (int i=0; i<RADIUS; i++)
					    msg->edge[k++]   = temp(width-RADIUS+i,j);
          thisProxy(thisIndex.x+1, thisIndex.y).receiveGhosts(msg);
        }

	// Send my bottom edge
        if (thisIndex.y > 0) {
          ghostMsg *msg = new (width*RADIUS) ghostMsg(BOTTOM, width);
          CkSetRefNum(msg, iterations);
          double *edge = new double[width*RADIUS];
          for (int j=0, k=0; j<RADIUS; j++) for(int i=0;i<width;i++)
					 msg->edge[k++]   = temp(i,j);
          thisProxy(thisIndex.x, thisIndex.y-1).receiveGhosts(msg);
        }

	// Send my top edge
        if (thisIndex.y < num_chare_rows-1) {
          ghostMsg *msg = new (width*RADIUS) ghostMsg(TOP, width);
          CkSetRefNum(msg, iterations);
          double *edge = new double[width*RADIUS];
          for (int j=0, k=0; j<RADIUS; j++) for(int i=0;i<width;i++)
					 msg->edge[k++]  = temp(i,height-RADIUS+j);
          thisProxy(thisIndex.x, thisIndex.y+1).receiveGhosts(msg);
        }
    }

  void processGhosts(ghostMsg *msg) {
      int k; k=0; 
      int size = msg->size;

      switch(msg->dir) {
      case LEFT:
        for(int j=0;j<size;++j) for (int i=0; i<RADIUS; i++)
	  temp(width+i,j) = msg->edge[k++];
        break;

      case RIGHT:
        for(int j=0;j<size;++j) for (int i=0; i<RADIUS; i++)
	  temp(-RADIUS+i,j) = msg->edge[k++];
        break;

      case BOTTOM:
        for (int j=0; j<RADIUS; j++) for(int i=0;i<size;++i){
	  temp(i,height+j) = msg->edge[k++];
        }
        break;

      case TOP:
        for (int j=0; j<RADIUS; j++) for(int i=0;i<size;++i)
	  temp(i,-RADIUS+j) = msg->edge[k++];
        break;

      default: CkAbort("ERROR: invalid direction\n");
      }
      delete msg;
  }
    
    void compute() {

      for (int j=MAX(jstart,RADIUS); j<=MIN(n-1-RADIUS,jend); j++) {
        for (int i=MAX(istart,RADIUS); i<=MIN(n-1-RADIUS,iend); i++) {

          double start =  temp_new(i-istart,j-jstart);
          for (int jj=-RADIUS; jj<=RADIUS; jj++) {
            temp_new(i-istart,j-jstart) += WEIGHT(0,jj)*temp(i-istart,j-jstart+jj);
	  }
          for (int ii=-RADIUS; ii<0; ii++) {
            temp_new(i-istart,j-jstart) += WEIGHT(ii,0)*temp(i-istart+ii,j-jstart);
	  }
          for (int ii=1; ii<=RADIUS; ii++) {
            temp_new(i-istart,j-jstart) += WEIGHT(ii,0)*temp(i-istart+ii,j-jstart);
	  }
        }
      }
    }

    void compute_local_norm() {

      local_norm = 0.0;
      for (int j=MAX(jstart,RADIUS); j<=MIN(n-1-RADIUS,jend); j++) {
        for (int i=MAX(istart,RADIUS); i<=MIN(n-1-RADIUS,iend); i++) {
           local_norm += temp_new(i-istart,j-jstart);
         }
       }
    }
};

#include "stencil.def.h"
